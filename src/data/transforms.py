#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as T
from typing import List, Optional, Tuple, Union, Dict, Any


def get_train_transforms(input_size: Tuple[int, int] = (224, 224),
                         mean: List[float] = [0.485, 0.456, 0.406],
                         std: List[float] = [0.229, 0.224, 0.225],
                         augmentation_level: str = 'medium') -> T.Compose:
    """
    获取训练数据增强变换

    参数:
        input_size: 输入尺寸 (高度, 宽度)
        mean: 归一化均值
        std: 归一化标准差
        augmentation_level: 增强等级 ('light', 'medium', 'heavy')

    返回:
        组合变换
    """
    # 基础变换
    transforms = [
        T.RandomResizedCrop(input_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ]

    # 根据增强等级添加额外变换
    if augmentation_level == 'medium':
        # 插入到ToTensor之前
        transforms.insert(2, T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
        transforms.insert(2, T.RandomAffine(degrees=10, translate=(0.1, 0.1)))

    elif augmentation_level == 'heavy':
        # 更强的数据增强
        transforms.insert(2, T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        transforms.insert(2, T.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15)))
        transforms.insert(2, T.RandomPerspective(distortion_scale=0.2, p=0.5))
        transforms.insert(2, T.RandomAutocontrast(p=0.3))
        transforms.insert(2, T.RandomGrayscale(p=0.1))
        transforms.insert(2, T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))

    return T.Compose(transforms)


def get_val_transforms(input_size: Tuple[int, int] = (224, 224),
                       mean: List[float] = [0.485, 0.456, 0.406],
                       std: List[float] = [0.229, 0.224, 0.225]) -> T.Compose:
    """
    获取验证数据变换

    参数:
        input_size: 输入尺寸 (高度, 宽度)
        mean: 归一化均值
        std: 归一化标准差

    返回:
        组合变换
    """
    return T.Compose([
        T.Resize(int(input_size[0] * 1.14)),  # 稍大一些以模拟CenterCrop
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def get_mixup_transform(alpha: float = 0.2):
    """
    获取Mixup数据增强

    参数:
        alpha: Beta分布参数

    返回:
        Mixup变换函数
    """

    def mixup_transform(batch, targets):
        """Mixup数据增强"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = batch.size(0)
        index = torch.randperm(batch_size)

        mixed_x = lam * batch + (1 - lam) * batch[index, :]
        y_a, y_b = targets, targets[index]

        return mixed_x, y_a, y_b, lam

    return mixup_transform


def get_cutmix_transform(alpha: float = 1.0):
    """
    获取CutMix数据增强

    参数:
        alpha: Beta分布参数

    返回:
        CutMix变换函数
    """

    def cutmix_transform(batch, targets):
        """CutMix数据增强"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = batch.size(0)
        index = torch.randperm(batch_size)

        # 生成cut区域
        W, H = batch.size(2), batch.size(3)
        cut_w = int(W * np.sqrt(1. - lam))
        cut_h = int(H * np.sqrt(1. - lam))

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # 计算区域范围
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)

        # 混合图像
        mixed_x = batch.clone()
        mixed_x[:, :, x1:x2, y1:y2] = batch[index, :, x1:x2, y1:y2]

        # 调整lambda以匹配区域大小
        lam = 1. - float((x2 - x1) * (y2 - y1)) / (W * H)

        y_a, y_b = targets, targets[index]

        return mixed_x, y_a, y_b, lam

    return cutmix_transform