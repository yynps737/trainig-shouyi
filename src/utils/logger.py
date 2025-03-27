#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """训练日志记录工具"""

    def __init__(self, log_dir: str, project_name: str = "aixunliancang",
                 enable_tensorboard: bool = True, log_level: str = "INFO"):
        """
        初始化日志记录器

        参数:
            log_dir: 日志保存目录
            project_name: 项目名称
            enable_tensorboard: 是否启用TensorBoard
            log_level: 日志级别
        """
        # 创建日志目录
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 配置文件日志
        self.logger = logging.getLogger(project_name)
        self.logger.setLevel(getattr(logging, log_level))

        # 避免重复添加处理器
        if not self.logger.handlers:
            # 添加控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(console_handler)

            # 添加文件处理器
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"{project_name}_{timestamp}.log")
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(file_handler)

        # TensorBoard
        self.enable_tensorboard = enable_tensorboard
        if enable_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
        else:
            self.tb_writer = None

        # 指标记录
        self.metrics = {}
        self.step = 0

        # 记录训练配置和系统信息
        self.system_info = self._collect_system_info()
        self.info(f"初始化日志记录器 - 日志目录: {log_dir}")
        self.info(f"系统信息: {self.system_info}")

    def _collect_system_info(self) -> Dict[str, Any]:
        """收集系统信息"""
        info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pytorch_version": torch.__version__,
            "python_version": torch.__version__,  # 简化起见复用PyTorch版本
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory"] = f"{props.total_memory / (1024 ** 3):.2f} GB"

        return info

    def log_config(self, config: Dict[str, Any]) -> None:
        """记录训练配置"""
        self.info("训练配置:")
        # 记录到文本日志
        for key, value in config.items():
            if isinstance(value, dict):
                self.info(f"  {key}:")
                for k, v in value.items():
                    self.info(f"    {k}: {v}")
            else:
                self.info(f"  {key}: {value}")

        # 保存为JSON
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 记录到TensorBoard
        if self.tb_writer:
            # 将配置转换为扁平化的字典
            flat_config = {}
            for key, value in config.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        flat_config[f"{key}/{k}"] = v
                else:
                    flat_config[key] = value

            self.tb_writer.add_hparams(flat_config, {})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = "") -> None:
        """
        记录训练指标

        参数:
            metrics: 指标字典
            step: 当前步骤
            prefix: 指标前缀
        """
        if step is None:
            step = self.step
            self.step += 1

        # 更新指标记录
        for key, value in metrics.items():
            metric_name = f"{prefix}/{key}" if prefix else key
            self.metrics[metric_name] = value

            # 记录到TensorBoard
            if self.tb_writer:
                self.tb_writer.add_scalar(metric_name, value, step)

        # 打印主要指标
        main_metrics = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"步骤 {step}: {main_metrics}")

    def log_model_summary(self, model: torch.nn.Module) -> None:
        """记录模型结构摘要"""
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.info(f"模型总参数: {total_params:,}")
        self.info(f"可训练参数: {trainable_params:,}")

        # 记录到TensorBoard
        if self.tb_writer and hasattr(model, 'forward'):
            try:
                # 尝试记录模型图
                device = next(model.parameters()).device
                dummy_shape = (1, 3, 224, 224)  # 假设是图像模型
                dummy_input = torch.zeros(dummy_shape, device=device)
                self.tb_writer.add_graph(model, dummy_input)
            except Exception as e:
                self.warning(f"记录模型图失败: {e}")

    def log_batch_images(self, images: torch.Tensor, tag: str = "images",
                         step: Optional[int] = None, max_images: int = 16) -> None:
        """记录批次图像到TensorBoard"""
        if self.tb_writer:
            if step is None:
                step = self.step

            # 限制图像数量
            num_images = min(images.shape[0], max_images)
            images = images[:num_images]

            # 添加到TensorBoard
            self.tb_writer.add_images(tag, images, step)

    def log_confusion_matrix(self, cm: np.ndarray, classes: List[str],
                             tag: str = "confusion_matrix", step: Optional[int] = None) -> None:
        """记录混淆矩阵到日志文件"""
        if step is None:
            step = self.step

        # 保存为CSV
        cm_path = os.path.join(self.log_dir, f"{tag}_{step}.csv")
        with open(cm_path, 'w', encoding='utf-8') as f:
            f.write("," + ",".join(classes) + "\n")
            for i, row in enumerate(cm):
                f.write(classes[i] + "," + ",".join(map(str, row)) + "\n")

        self.info(f"混淆矩阵已保存至: {cm_path}")

    def debug(self, message: str) -> None:
        """记录调试信息"""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """记录普通信息"""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """记录警告信息"""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """记录错误信息"""
        self.logger.error(message)

    def close(self) -> None:
        """关闭日志记录器"""
        if self.tb_writer:
            self.tb_writer.close()

        # 记录最终指标摘要
        if self.metrics:
            summary_path = os.path.join(self.log_dir, "metrics_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2)

            self.info(f"指标摘要已保存至: {summary_path}")

        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def get_logger(log_dir: str = "logs", project_name: str = "aixunliancang",
               enable_tensorboard: bool = True) -> Logger:
    """
    获取日志记录器的便捷函数

    参数:
        log_dir: 日志保存目录
        project_name: 项目名称
        enable_tensorboard: 是否启用TensorBoard

    返回:
        Logger对象
    """
    return Logger(log_dir, project_name, enable_tensorboard)