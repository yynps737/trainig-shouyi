#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable


class Callback:
    """回调函数基类"""

    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer, history):
        pass

    def on_epoch_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer, epoch_metrics):
        pass

    def on_batch_begin(self, trainer, batch_idx):
        pass

    def on_batch_end(self, trainer, batch_idx, batch_metrics):
        pass


class EarlyStopping(Callback):
    """早停回调"""

    def __init__(self, monitor: str = 'val_loss', patience: int = 10,
                 min_delta: float = 0.0, mode: str = 'min'):
        """
        初始化早停回调

        参数:
            monitor: 监控指标
            patience: 耐心值
            min_delta: 最小变化量
            mode: 模式，'min'表示越小越好，'max'表示越大越好
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else -float('inf')
        self.early_stop = False

    def on_epoch_end(self, trainer, epoch_metrics):
        """epoch结束时检查早停条件"""
        prefix, metric = self.monitor.split('_', 1) if '_' in self.monitor else ('', self.monitor)

        if prefix:
            current = epoch_metrics.get(f"{prefix}_{metric}", None)
            if current is None:
                current = epoch_metrics.get(metric, None)
        else:
            current = epoch_metrics.get(metric, None)

        if current is None:
            return

        if self.mode == 'min':
            score = -current
            delta = -self.min_delta
        else:
            score = current
            delta = self.min_delta

        if score < self.best_score + delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                trainer.logger.info(f"早停: {self.patience} epochs未改善")


class ModelCheckpoint(Callback):
    """模型检查点回调"""

    def __init__(self, filepath: str, monitor: str = 'val_loss',
                 save_best_only: bool = True, save_freq: int = 1,
                 mode: str = 'min'):
        """
        初始化模型检查点回调

        参数:
            filepath: 保存路径
            monitor: 监控指标
            save_best_only: 是否只保存最佳模型
            save_freq: 保存频率 (epochs)
            mode: 模式，'min'表示越小越好，'max'表示越大越好
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.mode = mode

        self.best_score = float('inf') if mode == 'min' else -float('inf')

    def on_epoch_end(self, trainer, epoch_metrics):
        """epoch结束时保存模型"""
        # 检查是否满足保存频率
        if (trainer.epoch + 1) % self.save_freq != 0 and not self.save_best_only:
            return

        # 获取监控指标
        prefix, metric = self.monitor.split('_', 1) if '_' in self.monitor else ('', self.monitor)

        if prefix:
            current = epoch_metrics.get(f"{prefix}_{metric}", None)
            if current is None:
                current = epoch_metrics.get(metric, None)
        else:
            current = epoch_metrics.get(metric, None)

        # 如果指标不存在，则不保存
        if current is None and self.save_best_only:
            return

        # 确定是否需要保存
        save_path = self.filepath

        # 填充文件名中的变量
        if '{epoch}' in save_path:
            save_path = save_path.replace('{epoch}', f"{trainer.epoch + 1:03d}")

        if '{val_loss}' in save_path and 'val_loss' in epoch_metrics:
            save_path = save_path.replace('{val_loss}', f"{epoch_metrics['val_loss']:.4f}")

        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 决定是否保存
        if self.save_best_only:
            if self.mode == 'min':
                if current < self.best_score:
                    trainer.logger.info(f"{self.monitor} 改善: {self.best_score:.6f} -> {current:.6f}")
                    self.best_score = current
                    trainer.save_checkpoint(save_path)
            else:
                if current > self.best_score:
                    trainer.logger.info(f"{self.monitor} 改善: {self.best_score:.6f} -> {current:.6f}")
                    self.best_score = current
                    trainer.save_checkpoint(save_path)
        else:
            trainer.save_checkpoint(save_path)


class LearningRateScheduler(Callback):
    """学习率调度器回调"""

    def __init__(self, scheduler: Any, monitor: str = 'val_loss',
                 mode: str = 'min'):
        """
        初始化学习率调度器回调

        参数:
            scheduler: 学习率调度器
            monitor: 监控指标 (用于ReduceLROnPlateau)
            mode: 模式，'min'表示越小越好，'max'表示越大越好
        """
        self.scheduler = scheduler
        self.monitor = monitor
        self.mode = mode

    def on_epoch_end(self, trainer, epoch_metrics):
        """epoch结束时更新学习率"""
        if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            # 获取监控指标
            prefix, metric = self.monitor.split('_', 1) if '_' in self.monitor else ('', self.monitor)

            if prefix:
                current = epoch_metrics.get(f"{prefix}_{metric}", None)
                if current is None:
                    current = epoch_metrics.get(metric, None)
            else:
                current = epoch_metrics.get(metric, None)

            if current is not None:
                self.scheduler.step(current)
        else:
            self.scheduler.step()

        # 记录新的学习率
        lr = trainer.get_lr()
        trainer.logger.log_metrics({'learning_rate': lr}, step=trainer.global_step)


class TensorBoardCallback(Callback):
    """TensorBoard回调"""

    def __init__(self, log_dir: str, histogram_freq: int = 0):
        """
        初始化TensorBoard回调

        参数:
            log_dir: 日志目录
            histogram_freq: 直方图频率
        """
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq

        # 初始化TensorBoard写入器
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_train_begin(self, trainer):
        """训练开始时记录模型图"""
        try:
            # 获取一个批次的数据
            batch = next(iter(trainer.train_loader))

            # 处理不同类型的批次
            if isinstance(batch, tuple) and len(batch) >= 1:
                inputs = batch[0]

                # 移动到CPU
                if isinstance(inputs, torch.Tensor):
                    dummy_input = inputs[:1].to('cpu')

                    # 记录模型图
                    self.writer.add_graph(
                        trainer.model.to('cpu'),
                        dummy_input
                    )

                    # 移回原始设备
                    trainer.model.to(trainer.device)
        except Exception as e:
            trainer.logger.warning(f"记录模型图出错: {e}")

    def on_epoch_end(self, trainer, epoch_metrics):
        """epoch结束时记录指标"""
        for key, value in epoch_metrics.items():
            self.writer.add_scalar(f'epoch/{key}', value, trainer.epoch)

        # 记录学习率
        lr = trainer.get_lr()
        self.writer.add_scalar('epoch/learning_rate', lr, trainer.epoch)

        # 记录参数直方图
        if self.histogram_freq > 0 and (trainer.epoch + 1) % self.histogram_freq == 0:
            for name, param in trainer.model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'parameters/{name}', param.data.cpu().numpy(), trainer.epoch)

                    if param.grad is not None:
                        self.writer.add_histogram(f'gradients/{name}', param.grad.data.cpu().numpy(), trainer.epoch)

    def on_train_end(self, trainer, history):
        """训练结束时关闭写入器"""
        self.writer.close()