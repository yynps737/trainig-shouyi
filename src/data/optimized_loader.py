#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.distributed as dist
from typing import List, Dict, Optional, Union, Callable, Any, Iterable, Tuple
import threading
import queue
import time
from functools import partial
import random
import math


class PrefetchLoader:
    """
    数据预取加载器 - 在GPU计算时在后台预取下一批次数据
    这可以显著减少CPU和GPU之间的等待时间
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device)
        self.next_batch = None
        self._iterator = iter(self.loader)
        self.preload()

    def preload(self):
        """预加载下一批次到专用CUDA流"""
        try:
            self.next_batch = next(self._iterator)
        except StopIteration:
            self.next_batch = None
            return

        # 预取到专用CUDA流
        with torch.cuda.stream(self.stream):
            if isinstance(self.next_batch, torch.Tensor):
                self.next_batch = self.next_batch.to(self.device, non_blocking=True)
            elif isinstance(self.next_batch, (list, tuple)):
                self.next_batch = [
                    x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x
                    for x in self.next_batch
                ]
            elif isinstance(self.next_batch, dict):
                self.next_batch = {
                    k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in self.next_batch.items()
                }

    def __iter__(self):
        return self

    def __next__(self):
        # 等待当前流中的所有操作完成
        torch.cuda.current_stream().wait_stream(self.stream)

        if self.next_batch is None:
            raise StopIteration

        # 获取预加载的批次
        batch = self.next_batch

        # 预加载下一批次
        self.preload()

        return batch

    def __len__(self):
        return len(self.loader)


class CachedDataset(Dataset):
    """
    缓存数据集，将数据保存在内存或共享内存中以加快访问速度
    特别适合反复使用的小型数据集
    """

    def __init__(self, dataset: Dataset, cache_size: int = -1,
                 shared_memory: bool = False,
                 transform: Optional[Callable] = None,
                 cache_type: str = "auto"):
        """
        初始化缓存数据集

        参数:
            dataset: 原始数据集
            cache_size: 缓存大小 (-1表示全部缓存)
            shared_memory: 是否使用共享内存
            transform: 应用于数据项的变换
            cache_type: 缓存类型 ('ram', 'disk', 'auto')
        """
        self.dataset = dataset
        self.cache_size = cache_size if cache_size > 0 else len(dataset)
        self.cache_size = min(self.cache_size, len(dataset))
        self.shared_memory = shared_memory
        self.transform = transform
        self.cache_type = cache_type

        # 缓存数据
        self.cache = {}
        self.cache_indices = set()

        # 填充缓存
        self._fill_cache()

    def _fill_cache(self):
        """填充缓存"""
        # 确定缓存策略
        total_bytes = 0
        available_ram = psutil.virtual_memory().available if self.cache_type != "disk" else 0
        items_to_cache = min(self.cache_size, len(self.dataset))

        print(f"正在缓存 {items_to_cache} 个数据项...")

        # 随机选择要缓存的索引
        all_indices = list(range(len(self.dataset)))
        random.shuffle(all_indices)
        indices_to_cache = all_indices[:items_to_cache]

        # 填充缓存
        for idx in indices_to_cache:
            item = self.dataset[idx]

            # 估计内存占用
            if isinstance(item, torch.Tensor):
                item_bytes = item.element_size() * item.nelement()
            elif isinstance(item, (tuple, list)) and all(isinstance(x, torch.Tensor) for x in item):
                item_bytes = sum(x.element_size() * x.nelement() for x in item)
            else:
                # 对于复杂对象，使用近似值
                import sys
                item_bytes = sys.getsizeof(item)

            # 检查是否超出可用RAM
            total_bytes += item_bytes
            if self.cache_type == "auto" and total_bytes > available_ram * 0.7:
                # 如果超出可用RAM的70%，停止缓存
                print(f"警告：缓存大小超出可用RAM的70%，停止缓存。已缓存 {len(self.cache_indices)} 个数据项")
                break

            # 存储到缓存
            if self.shared_memory and isinstance(item, torch.Tensor):
                # 使用共享内存
                cache_tensor = item.share_memory_()
                self.cache[idx] = cache_tensor
            else:
                # 普通内存缓存
                self.cache[idx] = item

            self.cache_indices.add(idx)

        print(f"缓存完成，共缓存 {len(self.cache_indices)} 个数据项，占用 {total_bytes / (1024 ** 2):.2f} MB")

    def __getitem__(self, idx):
        """获取数据项"""
        if idx in self.cache_indices:
            # 从缓存获取
            item = self.cache[idx]
        else:
            # 从原始数据集获取
            item = self.dataset[idx]

        # 应用变换
        if self.transform is not None:
            if isinstance(item, tuple):
                # 如果是(输入, 标签)形式，只变换输入
                data, label = item
                data = self.transform(data)
                item = (data, label)
            else:
                item = self.transform(item)

        return item

    def __len__(self):
        return len(self.dataset)


class FastDataLoader(DataLoader):
    """
    优化的数据加载器，结合了多种技术来加速数据加载
    - 预取机制
    - 高效的内存管理
    - 增强的多进程处理
    - CPU与GPU重叠计算
    """

    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4,
                 pin_memory=True, drop_last=False, prefetch=True,
                 persistent_workers=True, device=None, **kwargs):
        """
        初始化快速数据加载器

        参数:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否打乱
            num_workers: 工作线程数
            pin_memory: 是否将数据固定到内存中
            drop_last: 是否丢弃最后不完整的批次
            prefetch: 是否启用预取
            persistent_workers: 是否保持工作线程始终存活
            device: 目标设备
            **kwargs: 其他DataLoader参数
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers and num_workers > 0,
            **kwargs
        )

        self.prefetch = prefetch

        # 确定设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def __iter__(self):
        """返回数据迭代器"""
        iterator = super().__iter__()

        # 如果启用预取并且目标是CUDA设备
        if self.prefetch and self.device.type == 'cuda':
            return PrefetchLoader(iterator, self.device)
        else:
            return iterator


class MixupTransform:
    """
    Mixup数据增强
    将两个样本按一定比例混合，创建新样本
    对于分类任务特别有效
    """

    def __init__(self, alpha=0.2):
        """
        初始化Mixup变换

        参数:
            alpha: Beta分布参数
        """
        self.alpha = alpha

    def __call__(self, batch, target):
        """应用Mixup变换"""
        if self.alpha <= 0:
            return batch, target

        # 获取混合比例
        lam = np.random.beta(self.alpha, self.alpha)

        # 获取批次大小
        batch_size = batch.size(0)

        # 随机排列索引
        index = torch.randperm(batch_size, device=batch.device)

        # 混合输入
        mixed_x = lam * batch + (1 - lam) * batch[index, :]

        # 混合目标
        y_a, y_b = target, target[index]

        return mixed_x, y_a, y_b, lam


class CutMixTransform:
    """
    CutMix数据增强
    将一个样本的一部分区域替换为另一个样本的相应区域
    在图像分类任务中表现优异
    """

    def __init__(self, alpha=1.0):
        """
        初始化CutMix变换

        参数:
            alpha: Beta分布参数
        """
        self.alpha = alpha

    def __call__(self, batch, target):
        """应用CutMix变换"""
        if self.alpha <= 0:
            return batch, target

        # 获取混合比例
        lam = np.random.beta(self.alpha, self.alpha)

        # 获取批次大小和图像尺寸
        batch_size = batch.size(0)
        W, H = batch.size(2), batch.size(3)

        # 随机排列索引
        index = torch.randperm(batch_size, device=batch.device)

        # 计算切割区域大小
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)

        # 随机选择中心点
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # 确定区域边界
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # 创建混合图像
        mixed_x = batch.clone()
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = batch[index, :, bbx1:bbx2, bby1:bby2]

        # 调整lambda值以匹配实际混合比例
        lam = 1. - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        # 混合目标
        y_a, y_b = target, target[index]

        return mixed_x, y_a, y_b, lam


class BatchTransform:
    """
    批次级别的变换
    一次性应用于整个批次而不是单个样本
    可以更高效地实现一些数据增强
    """

    def __init__(self, transforms=None):
        """
        初始化批次变换

        参数:
            transforms: 要应用的变换列表
        """
        self.transforms = transforms or []

    def __call__(self, batch, target):
        """应用批次变换"""
        for transform in self.transforms:
            if callable(transform):
                result = transform(batch, target)

                # 处理不同的返回值形式
                if isinstance(result, tuple) and len(result) >= 2:
                    batch = result[0]
                    if len(result) == 2:
                        target = result[1]
                    else:
                        # 对于Mixup/CutMix，需要特殊处理
                        # 即 (batch, target_a, target_b, lam)
                        batch, target = result[0], (result[1], result[2], result[3])
                else:
                    batch = result

        return batch, target


class SubsetRandomSampler(Sampler):
    """
    子集随机采样器
    从数据集中随机采样而不需要数据复制，有效减少内存使用
    """

    def __init__(self, indices, generator=None):
        """
        初始化子集随机采样器

        参数:
            indices: 要采样的索引
            generator: 随机数生成器
        """
        self.indices = indices
        self.generator = generator

    def __iter__(self):
        """迭代采样"""
        return (self.indices[i] for i in torch.randperm(len(self.indices), generator=self.generator))

    def __len__(self):
        """采样器长度"""
        return len(self.indices)


class GradientAccumulationLoader:
    """
    梯度累积加载器
    模拟大批量训练，同时节省显存
    """

    def __init__(self, loader, accumulation_steps=1):
        """
        初始化梯度累积加载器

        参数:
            loader: 基础数据加载器
            accumulation_steps: 梯度累积步数
        """
        self.loader = loader
        self.accumulation_steps = accumulation_steps

    def __iter__(self):
        """返回累积迭代器"""
        iterator = iter(self.loader)

        # 如果不需要累积，直接返回基础迭代器
        if self.accumulation_steps <= 1:
            return iterator

        # 否则，返回累积迭代器
        try:
            while True:
                # 收集小批量
                micro_batches = []
                for _ in range(self.accumulation_steps):
                    try:
                        micro_batches.append(next(iterator))
                    except StopIteration:
                        # 当到达数据集末尾时，如果有累积的批次，返回它们
                        if micro_batches:
                            yield (micro_batches, self.accumulation_steps, len(micro_batches))
                        raise StopIteration

                # 返回小批量、累积步数和实际步数
                yield (micro_batches, self.accumulation_steps, len(micro_batches))
        except StopIteration:
            return

    def __len__(self):
        """返回累积后的批次数"""
        return math.ceil(len(self.loader) / self.accumulation_steps)


def create_optimized_dataloaders(
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        batch_size: int = 32,
        val_batch_size: int = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = True,
        prefetch: bool = True,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        distributed: bool = False,
        seed: int = 42,
        accumulation_steps: int = 1,
        device: Optional[torch.device] = None
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    创建优化的数据加载器

    参数:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        batch_size: 训练批次大小
        val_batch_size: 验证批次大小
        num_workers: 工作线程数
        pin_memory: 是否将数据固定到内存中
        shuffle: 是否打乱
        drop_last: 是否丢弃最后不完整的批次
        persistent_workers: 是否保持工作线程始终存活
        prefetch: 是否启用预取
        mixup_alpha: Mixup参数，0表示禁用
        cutmix_alpha: CutMix参数，0表示禁用
        distributed: 是否使用分布式训练
        seed: 随机种子
        accumulation_steps: 梯度累积步数
        device: 目标设备

    返回:
        如果提供了验证数据集，则返回(train_loader, val_loader)，否则返回train_loader
    """
    # 设置随机种子
    generator = torch.Generator()
    generator.manual_seed(seed)

    # 确定验证批次大小
    if val_batch_size is None:
        val_batch_size = batch_size * 2  # 验证时通常可以用更大的批次

    # 创建采样器
    train_sampler = None
    val_sampler = None

    if distributed and dist.is_initialized():
        # 分布式采样器
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            seed=seed
        )

        if val_dataset is not None:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
                seed=seed
            )

        # 分布式模式下不进行批次洗牌
        shuffle = False

    # 创建训练数据加载器
    train_loader = FastDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle and train_sampler is None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=train_sampler,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch=prefetch,
        device=device,
        generator=generator
    )

    # 应用数据增强
    if mixup_alpha > 0 or cutmix_alpha > 0:
        transforms = []
        if mixup_alpha > 0:
            transforms.append(MixupTransform(alpha=mixup_alpha))
        if cutmix_alpha > 0:
            transforms.append(CutMixTransform(alpha=cutmix_alpha))

        # 实现批次变换需要自定义训练循环
        print(f"注意：启用了批次级数据增强 (Mixup={mixup_alpha}, CutMix={cutmix_alpha})，"
              f"需要在训练循环中显式应用这些变换")

    # 梯度累积
    if accumulation_steps > 1:
        train_loader = GradientAccumulationLoader(train_loader, accumulation_steps)
        print(f"启用梯度累积: 步数 = {accumulation_steps}，"
              f"有效批次大小 = {batch_size * accumulation_steps}")

    # 如果没有验证集，直接返回训练加载器
    if val_dataset is None:
        return train_loader

    # 创建验证数据加载器
    val_loader = FastDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,  # 验证集通常不需要打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # 验证时通常不丢弃最后一批
        sampler=val_sampler,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch=prefetch,
        device=device
    )

    return train_loader, val_loader


def apply_batch_transforms(batch, target, transforms):
    """
    应用批次级变换

    参数:
        batch: 输入批次
        target: 目标标签
        transforms: 变换列表

    返回:
        变换后的(batch, target)
    """
    if not transforms:
        return batch, target

    # 应用所有变换
    batch_transform = BatchTransform(transforms)
    return batch_transform(batch, target)


def apply_mixup(batch, target, alpha=0.2):
    """
    应用Mixup数据增强

    参数:
        batch: 输入批次
        target: 目标标签
        alpha: Mixup参数

    返回:
        (mixed_batch, target_a, target_b, lam)
    """
    transform = MixupTransform(alpha=alpha)
    return transform(batch, target)


def apply_cutmix(batch, target, alpha=1.0):
    """
    应用CutMix数据增强

    参数:
        batch: 输入批次
        target: 目标标签
        alpha: CutMix参数

    返回:
        (mixed_batch, target_a, target_b, lam)
    """
    transform = CutMixTransform(alpha=alpha)
    return transform(batch, target)


if __name__ == "__main__":
    # 测试数据加载优化

    # 创建一个简单的测试数据集
    class DummyDataset(Dataset):
        def __init__(self, size=1000, dim=224):
            self.size = size
            self.dim = dim

        def __getitem__(self, idx):
            # 生成随机图像和标签
            img = torch.randn(3, self.dim, self.dim)
            label = torch.tensor(idx % 10)
            return img, label

        def __len__(self):
            return self.size


    # 创建数据集
    train_dataset = DummyDataset(size=1000)
    val_dataset = DummyDataset(size=100)

    # 测试不同配置的数据加载器
    for num_workers in [0, 4]:
        for batch_size in [32, 64]:
            print(f"\n测试配置: num_workers={num_workers}, batch_size={batch_size}")

            # 创建普通数据加载器（基准）
            standard_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )

            # 创建优化数据加载器
            optimized_loader = create_optimized_dataloaders(
                train_dataset,
                val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch=True,
                mixup_alpha=0.2
            )

            if isinstance(optimized_loader, tuple):
                train_loader, val_loader = optimized_loader
            else:
                train_loader = optimized_loader
                val_loader = None

            # 测量数据加载时间
            start_time = time.time()
            for _ in range(3):  # 预热
                next(iter(standard_loader))

            # 测量标准加载器
            start_time = time.time()
            for i, (inputs, targets) in enumerate(standard_loader):
                if i >= 10:  # 只测量前10个批次
                    break
            standard_time = time.time() - start_time

            # 测量优化加载器
            start_time = time.time()
            for i, (inputs, targets) in enumerate(train_loader):
                if i >= 10:  # 只测量前10个批次
                    break
            optimized_time = time.time() - start_time

            # 打印结果
            print(f"标准加载器加载10个批次时间: {standard_time:.4f}秒")
            print(f"优化加载器加载10个批次时间: {optimized_time:.4f}秒")
            print(f"加速比: {standard_time / optimized_time:.2f}x")

            # 测试批次变换
            batch, target = next(iter(train_loader))
            mixed_batch, target_a, target_b, lam = apply_mixup(batch, target)
            print(f"Mixup后的批次形状: {mixed_batch.shape}")