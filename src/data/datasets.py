#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """图像数据集基类"""

    def __init__(self, images_dir: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 is_training: bool = True):
        """
        初始化图像数据集

        参数:
            images_dir: 图像目录
            transform: 图像变换
            target_transform: 标签变换
            is_training: 是否为训练模式
        """
        self.images_dir = images_dir
        self.transform = transform
        self.target_transform = target_transform
        self.is_training = is_training

        # 查找所有图像文件
        self.image_files = []
        self.labels = []
        self.class_names = []

        # 遍历目录结构
        for class_idx, class_name in enumerate(sorted(os.listdir(images_dir))):
            class_dir = os.path.join(images_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            self.class_names.append(class_name)

            # 查找该类别的所有图像
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.image_files.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """获取数据项"""
        # 加载图像
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        # 应用变换
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_class_counts(self) -> List[int]:
        """获取每个类别的样本数量"""
        counts = [0] * len(self.class_names)
        for label in self.labels:
            counts[label] += 1
        return counts


class TextDataset(Dataset):
    """文本数据集基类"""

    def __init__(self, data_file: str,
                 tokenizer: Any,
                 max_length: int = 512,
                 is_training: bool = True):
        """
        初始化文本数据集

        参数:
            data_file: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            is_training: 是否为训练模式
        """
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training

        # 读取数据
        self.texts = []
        self.labels = []
        self.label_names = []

        # 读取文本文件
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    text = parts[0]
                    label = int(parts[1])

                    self.texts.append(text)
                    self.labels.append(label)

        # 确定标签名称
        if self.labels:
            num_classes = max(self.labels) + 1
            self.label_names = [str(i) for i in range(num_classes)]

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据项"""
        text = self.texts[idx]
        label = self.labels[idx]

        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 处理输出
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze()

        return item

    def get_class_counts(self) -> List[int]:
        """获取每个类别的样本数量"""
        counts = [0] * len(self.label_names)
        for label in self.labels:
            counts[label] += 1
        return counts


class TabularDataset(Dataset):
    """表格数据集基类"""

    def __init__(self, data_file: str,
                 categorical_cols: Optional[List[str]] = None,
                 numerical_cols: Optional[List[str]] = None,
                 target_col: str = 'target',
                 transform: Optional[Callable] = None):
        """
        初始化表格数据集

        参数:
            data_file: 数据文件路径
            categorical_cols: 分类特征列名
            numerical_cols: 数值特征列名
            target_col: 目标列名
            transform: 数据变换
        """
        import pandas as pd

        self.data_file = data_file
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        self.target_col = target_col
        self.transform = transform

        # 读取数据
        self.df = pd.read_csv(data_file)

        # 提取特征和标签
        self.features = self.df[self.categorical_cols + self.numerical_cols]
        self.targets = self.df[target_col]

        # 处理分类特征
        self.categorical_encoders = {}
        for col in self.categorical_cols:
            unique_values = self.features[col].unique()
            self.categorical_encoders[col] = {v: i for i, v in enumerate(unique_values)}

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """获取数据项"""
        # 提取特征
        cat_features = []
        for col in self.categorical_cols:
            value = self.features.iloc[idx][col]
            encoded = self.categorical_encoders[col].get(value, 0)
            cat_features.append(encoded)

        num_features = []
        for col in self.numerical_cols:
            value = self.features.iloc[idx][col]
            num_features.append(value)

        # 转换为张量
        cat_tensor = torch.tensor(cat_features, dtype=torch.long) if cat_features else torch.tensor([])
        num_tensor = torch.tensor(num_features, dtype=torch.float) if num_features else torch.tensor([])

        # 提取标签
        target = self.targets.iloc[idx]
        target_tensor = torch.tensor(target, dtype=torch.float)

        # 应用变换
        features = {
            'categorical': cat_tensor,
            'numerical': num_tensor
        }

        if self.transform:
            features = self.transform(features)

        return features, target_tensor