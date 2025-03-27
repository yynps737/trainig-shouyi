#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Optional, Union, Any, Callable


def get_dataloader(dataset: Dataset,
                   batch_size: int = 32,
                   shuffle: bool = True,
                   num_workers: int = 4,
                   pin_memory: bool = True,
                   persistent_workers: bool = True,
                   drop_last: bool = False,
                   prefetch_factor: int = 2) -> DataLoader:
    """
    获取数据加载器

    参数:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作线程数
        pin_memory: 是否将数据固定到内存中
        persistent_workers: 是否保持工作线程始终存活
        drop_last: 是否丢弃最后不完整的批次
        prefetch_factor: 预取因子

    返回:
        DataLoader对象
    """
    # 确保CUDA环境下的优化
    if torch.cuda.is_available():
        # 对于较小的数据集，不需要太多工作线程
        if len(dataset) < 10000:
            num_workers = min(2, num_workers)

        # 对于单GPU训练，pin_memory效果更好
        pin_memory = True

        # 对于小批次，prefetch_factor可以增大
        if batch_size < 16:
            prefetch_factor = max(2, prefetch_factor)

        # 启用持久工作线程避免创建销毁开销
        persistent_workers = num_workers > 0
    else:
        # CPU模式下，减少资源占用
        num_workers = min(2, num_workers)
        pin_memory = False
        persistent_workers = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2
    )


def get_train_val_dataloaders(dataset: Dataset,
                              batch_size: int = 32,
                              val_split: float = 0.1,
                              seed: int = 42,
                              **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    获取训练和验证数据加载器

    参数:
        dataset: 数据集
        batch_size: 批次大小
        val_split: 验证集比例
        seed: 随机种子
        **kwargs: 其他数据加载器参数

    返回:
        训练数据加载器和验证数据加载器的元组
    """
    # 设置随机种子
    torch.manual_seed(seed)

    # 拆分数据集
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs
    )

    val_loader = get_dataloader(
        val_dataset,
        batch_size=batch_size * 2,  # 验证时可以用更大的批次
        shuffle=False,
        drop_last=False,
        **kwargs
    )

    return train_loader, val_loader


def get_balanced_sampler(dataset: Dataset) -> torch.utils.data.Sampler:
    """
    获取平衡采样器，用于不平衡数据集

    参数:
        dataset: 数据集

    返回:
        平衡采样器
    """
    if not hasattr(dataset, 'get_class_counts') or not hasattr(dataset, 'labels'):
        raise ValueError("数据集必须实现get_class_counts方法和labels属性")

    # 获取类别计数
    class_counts = dataset.get_class_counts()

    # 计算采样权重
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[dataset.labels]

    # 创建平衡采样器
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )

    return sampler