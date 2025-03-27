#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import io
import base64
from pathlib import Path
from PIL import Image
import seaborn as sns
from matplotlib.figure import Figure


class Visualizer:
    """训练可视化工具类"""

    def __init__(self, save_dir: str = "visualizations"):
        """
        初始化可视化工具

        参数:
            save_dir: 可视化结果保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_training_curves(self, metrics: Dict[str, List[float]],
                             steps: Optional[List[int]] = None,
                             title: str = "训练曲线",
                             save_path: Optional[str] = None) -> Figure:
        """
        绘制训练曲线

        参数:
            metrics: 指标字典，键为指标名，值为指标列表
            steps: 步骤列表，如果为None则使用索引
            title: 图表标题
            save_path: 保存路径，如果为None则自动生成

        返回:
            matplotlib Figure对象
        """
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))

        # 如果未提供步骤，则使用索引
        if steps is None:
            steps = list(range(len(next(iter(metrics.values())))))

        # 绘制每个指标
        for name, values in metrics.items():
            ax.plot(steps, values, label=name)

        # 添加标签和图例
        ax.set_title(title)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}.png")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)

        return fig

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                              title: str = "混淆矩阵",
                              normalize: bool = True,
                              save_path: Optional[str] = None) -> Figure:
        """
        绘制混淆矩阵

        参数:
            cm: 混淆矩阵
            class_names: 类别名称
            title: 图表标题
            normalize: 是否归一化
            save_path: 保存路径，如果为None则自动生成

        返回:
            matplotlib Figure对象
        """
        # 归一化
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)

        # 创建图表
        plt.figure(figsize=(10, 8))

        # 使用seaborn绘制热图
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )

        # 添加标签
        plt.title(title)
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')

        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}.png")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        fig = plt.gcf()

        return fig

    def visualize_model_predictions(self, images: torch.Tensor,
                                    true_labels: List[int],
                                    pred_labels: List[int],
                                    class_names: List[str],
                                    title: str = "模型预测可视化",
                                    max_images: int = 16,
                                    save_path: Optional[str] = None) -> Figure:
        """
        可视化模型预测结果

        参数:
            images: 图像张量
            true_labels: 真实标签
            pred_labels: 预测标签
            class_names: 类别名称
            title: 图表标题
            max_images: 最大图像数量
            save_path: 保存路径，如果为None则自动生成

        返回:
            matplotlib Figure对象
        """
        # 限制图像数量
        num_images = min(len(images), max_images)

        # 创建网格图表
        rows = int(np.ceil(num_images / 4))
        cols = min(num_images, 4)

        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # 处理图像并绘制
        for i in range(num_images):
            # 处理图像
            img = images[i].cpu().numpy()
            if img.shape[0] == 1:  # 灰度图像
                img = img[0]
                cmap = 'gray'
            else:  # RGB图像
                img = np.transpose(img, (1, 2, 0))
                # 如果图像是规范化的，恢复到0-1范围
                if img.min() < 0:
                    img = (img + 1) / 2.0
                img = np.clip(img, 0, 1)
                cmap = None

            # 绘制图像
            axes[i].imshow(img, cmap=cmap)

            # 设置标题，正确预测为绿色，错误预测为红色
            correct = true_labels[i] == pred_labels[i]
            color = 'green' if correct else 'red'
            axes[i].set_title(
                f"真实: {class_names[true_labels[i]]}\n预测: {class_names[pred_labels[i]]}",
                color=color
            )
            axes[i].axis('off')

        # 隐藏多余的子图
        for i in range(num_images, len(axes)):
            axes[i].axis('off')

        plt.suptitle(title)
        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}.png")

        plt.savefig(save_path, dpi=300)

        return fig

    def plot_class_activation_map(self, image: torch.Tensor,
                                  cam: np.ndarray,
                                  title: str = "类激活映射",
                                  save_path: Optional[str] = None, skimage=None) -> Figure:
        """
        绘制类激活映射

        参数:
            image: 原始图像
            cam: 类激活映射
            title: 图表标题
            save_path: 保存路径，如果为None则自动生成

        返回:
            matplotlib Figure对象
        """
        # 处理图像
        img = image.cpu().numpy()
        if img.shape[0] == 1:  # 灰度图像
            img = img[0]
        else:  # RGB图像
            img = np.transpose(img, (1, 2, 0))
            # 如果图像是规范化的，恢复到0-1范围
            if img.min() < 0:
                img = (img + 1) / 2.0
            img = np.clip(img, 0, 1)

        # 调整CAM大小以匹配图像
        cam_resized = np.uint8(255 * cam)
        if cam_resized.shape != img.shape[:2]:
            from skimage.transform import resize
            cam_resized = resize(cam, img.shape[:2], preserve_range=True)

        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        if len(img.shape) == 2:  # 灰度图像
            axes[0].imshow(img, cmap='gray')
        else:  # RGB图像
            axes[0].imshow(img)
        axes[0].set_title("原始图像")
        axes[0].axis('off')

        # 热力图
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title("热力图")
        axes[1].axis('off')

        # 叠加图
        if len(img.shape) == 2:  # 灰度图像转RGB
            img_rgb = np.stack([img] * 3, axis=2)
        else:
            img_rgb = img

        heatmap = np.uint8(255 * plt.cm.jet(cam_resized / 255)[:, :, :3])
        superimposed = np.uint8(0.6 * heatmap + 0.4 * img_rgb * 255)

        axes[2].imshow(superimposed / 255)
        axes[2].set_title("叠加图")
        axes[2].axis('off')

        plt.suptitle(title)
        plt.tight_layout()

        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}.png")

        plt.savefig(save_path, dpi=300)

        return fig

    def plot_learning_rate(self, learning_rates: List[float],
                           steps: Optional[List[int]] = None,
                           title: str = "学习率变化",
                           save_path: Optional[str] = None) -> Figure:
        """
        绘制学习率变化

        参数:
            learning_rates: 学习率列表
            steps: 步骤列表，如果为None则使用索引
            title: 图表标题
            save_path: 保存路径，如果为None则自动生成

        返回:
            matplotlib Figure对象
        """
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))

        # 如果未提供步骤，则使用索引
        if steps is None:
            steps = list(range(len(learning_rates)))

        # 绘制学习率
        ax.plot(steps, learning_rates)

        # 添加标签
        ax.set_title(title)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Learning Rate")
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.7)

        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"{title.replace(' ', '_')}.png")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)

        return fig

    def figure_to_base64(self, fig: Figure) -> str:
        """
        将matplotlib图表转换为base64编码

        参数:
            fig: matplotlib Figure对象

        返回:
            base64编码的图像字符串
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str

    def close_figure(self, fig: Figure) -> None:
        """关闭图表"""
        plt.close(fig)


def get_visualizer(save_dir: str = "visualizations") -> Visualizer:
    """
    获取可视化工具的便捷函数

    参数:
        save_dir: 可视化结果保存目录

    返回:
        Visualizer对象
    """
    return Visualizer(save_dir)