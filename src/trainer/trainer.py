#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

from torch.utils.data import DataLoader
import torch.nn.functional as F

from ..utils.logger import Logger
from ..config.config import Config


class Trainer:
    """模型训练器"""

    def __init__(self, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: Callable,
                 config: Config,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 scheduler: Optional[Any] = None,
                 logger: Optional[Logger] = None,
                 callbacks: Optional[List[Any]] = None,
                 device: Optional[str] = None):
        """
        初始化训练器

        参数:
            model: 模型
            optimizer: 优化器
            loss_fn: 损失函数
            config: 配置
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            scheduler: 学习率调度器
            logger: 日志记录器
            callbacks: 回调函数列表
            device: 设备
        """
        # 设置设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 模型和优化器
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 配置和日志
        self.config = config

        if logger is None:
            from ..utils.logger import get_logger
            logger = get_logger(log_dir=config.get('output_dir', 'logs'))
        self.logger = logger

        # 混合精度
        self.use_amp = config.get('optimization.mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # 梯度累积
        self.grad_accum_steps = config.get('optimization.gradient_accumulation_steps', 1)

        # 梯度裁剪
        self.grad_clip = config.get('training.gradient_clip', 0.0)

        # 记录步骤和epoch
        self.global_step = 0
        self.epoch = 0

        # 训练状态
        self.best_val_metric = float('inf')
        self.early_stop_counter = 0
        self.early_stop_patience = config.get('training.early_stopping_patience', 0)

        # 回调函数
        self.callbacks = callbacks or []

        # 记录配置
        self.logger.log_config(config.config)

        # 记录模型摘要
        self.logger.log_model_summary(model)

        # 打印训练配置摘要
        self._print_config_summary()

    def _print_config_summary(self) -> None:
        """打印配置摘要"""
        self.logger.info("=" * 50)
        self.logger.info("训练配置摘要")
        self.logger.info("=" * 50)
        self.logger.info(f"模型: {self.model.__class__.__name__}")
        self.logger.info(f"优化器: {self.optimizer.__class__.__name__}")

        if hasattr(self.train_loader.dataset, '__len__'):
            self.logger.info(f"训练集大小: {len(self.train_loader.dataset)}")
            if self.val_loader:
                self.logger.info(f"验证集大小: {len(self.val_loader.dataset)}")

        self.logger.info(f"批次大小: {self.train_loader.batch_size}")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"混合精度: {'启用' if self.use_amp else '禁用'}")
        self.logger.info(f"梯度累积步骤: {self.grad_accum_steps}")
        self.logger.info("=" * 50)

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch

        返回:
            训练指标字典
        """
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        # 使用tqdm显示进度
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")

        # 记录开始时间
        start_time = time.time()

        for batch_idx, batch in enumerate(pbar):
            # 处理批次
            loss, batch_metrics = self.train_batch(batch, batch_idx)

            # 累积损失
            epoch_loss += loss.item()

            # 对于分类任务，计算准确率
            if isinstance(batch, tuple) and len(batch) == 2:
                # 假设数据格式为 (inputs, targets)
                _, targets = batch
                if hasattr(batch_metrics, 'get') and 'logits' in batch_metrics:
                    logits = batch_metrics['logits']
                    if isinstance(targets, torch.Tensor) and len(targets.shape) == 1:
                        # 计算预测值
                        if logits.shape[-1] > 1:  # 多分类
                            preds = torch.argmax(logits, dim=1)
                            correct += (preds == targets.to(self.device)).sum().item()
                            total += targets.size(0)

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{epoch_loss / (batch_idx + 1):.4f}",
                'lr': f"{self.get_lr():.6f}"
            })

            # 回调函数 - 批次结束
            for callback in self.callbacks:
                if hasattr(callback, 'on_batch_end'):
                    callback.on_batch_end(self, batch_idx, batch_metrics)

            # 更新全局步骤
            self.global_step += 1

            # 定期日志记录
            if self.global_step % self.config.get('training.log_interval', 50) == 0:
                metrics = {'loss': loss.item()}

                if total > 0:
                    metrics['accuracy'] = correct / total

                self.logger.log_metrics(metrics, step=self.global_step, prefix='train')

        # 计算epoch平均指标
        epoch_metrics = {
            'loss': epoch_loss / len(self.train_loader)
        }

        if total > 0:
            epoch_metrics['accuracy'] = correct / total

        # 记录epoch训练时间
        epoch_time = time.time() - start_time
        epoch_metrics['time'] = epoch_time

        # 日志记录
        self.logger.info(f"Epoch {self.epoch + 1} 训练完成，"
                         f"损失: {epoch_metrics['loss']:.4f}, "
                         f"耗时: {epoch_time:.2f}秒")

        # 回调函数 - epoch结束
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(self, epoch_metrics)

        return epoch_metrics

    def train_batch(self, batch: Any, batch_idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        训练一个批次

        参数:
            batch: 数据批次
            batch_idx: 批次索引

        返回:
            损失和批次指标
        """
        # 检查是否需要梯度累积
        do_optimizer_step = (batch_idx + 1) % self.grad_accum_steps == 0

        # 处理不同类型的批次
        if isinstance(batch, tuple) and len(batch) == 2:
            # 假设数据格式为 (inputs, targets)
            inputs, targets = batch

            # 移动到设备
            inputs = self._to_device(inputs)
            targets = self._to_device(targets)

            # 混合精度训练
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(inputs)
                    loss = self.loss_fn(logits, targets)

                # 缩放梯度
                self.scaler.scale(loss / self.grad_accum_steps).backward()

                if do_optimizer_step:
                    # 梯度裁剪
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    # 更新学习率调度器
                    if self.scheduler:
                        self.scheduler.step()
            else:
                # 标准训练
                logits = self.model(inputs)
                loss = self.loss_fn(logits, targets)

                # 反向传播
                (loss / self.grad_accum_steps).backward()

                if do_optimizer_step:
                    # 梯度裁剪
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # 更新学习率调度器
                    if self.scheduler:
                        self.scheduler.step()

            return loss, {'logits': logits}

        elif isinstance(batch, dict):
            # 假设数据格式为字典
            batch = {k: self._to_device(v) for k, v in batch.items()}

            # 提取标签
            targets = batch.pop('label', batch.pop('labels', None))

            # 混合精度训练
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)

                    # 处理不同类型的输出
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    elif isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                    else:
                        logits = outputs

                    loss = self.loss_fn(logits, targets)

                # 缩放梯度
                self.scaler.scale(loss / self.grad_accum_steps).backward()

                if do_optimizer_step:
                    # 梯度裁剪
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    # 更新学习率调度器
                    if self.scheduler:
                        self.scheduler.step()
            else:
                # 标准训练
                outputs = self.model(**batch)

                # 处理不同类型的输出
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs

                loss = self.loss_fn(logits, targets)

                # 反向传播
                (loss / self.grad_accum_steps).backward()

                if do_optimizer_step:
                    # 梯度裁剪
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # 更新学习率调度器
                    if self.scheduler:
                        self.scheduler.step()

            return loss, {'logits': logits}

        else:
            raise ValueError(f"不支持的批次格式: {type(batch)}")

    def validate(self) -> Dict[str, float]:
        """
        验证模型

        返回:
            验证指标字典
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # 处理不同类型的批次
                if isinstance(batch, tuple) and len(batch) == 2:
                    # 假设数据格式为 (inputs, targets)
                    inputs, targets = batch

                    # 移动到设备
                    inputs = self._to_device(inputs)
                    targets = self._to_device(targets)

                    # 前向传播
                    outputs = self.model(inputs)

                    # 计算损失
                    loss = self.loss_fn(outputs, targets)
                    val_loss += loss.item()

                    # 计算准确率
                    if len(targets.shape) == 1:  # 单标签分类
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == targets).sum().item()
                        total += targets.size(0)

                        # 记录所有预测和标签
                        all_targets.append(targets.cpu().numpy())
                        all_preds.append(preds.cpu().numpy())

                elif isinstance(batch, dict):
                    # 假设数据格式为字典
                    batch = {k: self._to_device(v) for k, v in batch.items()}

                    # 提取标签
                    targets = batch.pop('label', batch.pop('labels', None))

                    # 前向传播
                    outputs = self.model(**batch)

                    # 处理不同类型的输出
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    elif isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                    else:
                        logits = outputs

                    # 计算损失
                    loss = self.loss_fn(logits, targets)
                    val_loss += loss.item()

                    # 计算准确率
                    if len(targets.shape) == 1:  # 单标签分类
                        preds = torch.argmax(logits, dim=1)
                        correct += (preds == targets).sum().item()
                        total += targets.size(0)

                        # 记录所有预测和标签
                        all_targets.append(targets.cpu().numpy())
                        all_preds.append(preds.cpu().numpy())

        # 计算验证指标
        val_metrics = {
            'loss': val_loss / len(self.val_loader)
        }

        # 如果是分类任务，计算准确率
        if total > 0:
            val_metrics['accuracy'] = correct / total

        # 尝试计算F1分数等
        if all_targets and all_preds:
            try:
                from sklearn.metrics import f1_score, precision_score, recall_score

                all_targets = np.concatenate(all_targets)
                all_preds = np.concatenate(all_preds)

                # 多分类F1
                val_metrics['f1_score'] = f1_score(all_targets, all_preds, average='weighted')
                val_metrics['precision'] = precision_score(all_targets, all_preds, average='weighted')
                val_metrics['recall'] = recall_score(all_targets, all_preds, average='weighted')
            except:
                pass

        # 日志记录
        self.logger.log_metrics(val_metrics, step=self.global_step, prefix='val')
        self.logger.info(f"验证完成，损失: {val_metrics['loss']:.4f}, "
                         f"准确率: {val_metrics.get('accuracy', 0):.4f}")

        return val_metrics

    def train(self, num_epochs: int) -> Dict[str, Any]:
        """
        训练模型

        参数:
            num_epochs: 训练轮数

        返回:
            训练历史记录
        """
        # 初始化历史记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        # 初始化优化器状态
        self.optimizer.zero_grad()

        # 回调函数 - 训练开始
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(self)

        # 训练循环
        for epoch in range(num_epochs):
            self.epoch = epoch

            # 回调函数 - epoch开始
            for callback in self.callbacks:
                if hasattr(callback, 'on_epoch_begin'):
                    callback.on_epoch_begin(self)

            # 训练一个epoch
            train_metrics = self.train_epoch()

            # 记录学习率
            history['learning_rate'].append(self.get_lr())

            # 记录训练指标
            history['train_loss'].append(train_metrics['loss'])
            if 'accuracy' in train_metrics:
                history['train_accuracy'].append(train_metrics['accuracy'])

            # 验证
            if self.val_loader:
                if (epoch + 1) % self.config.get('training.eval_interval', 1) == 0:
                    val_metrics = self.validate()

                    # 记录验证指标
                    history['val_loss'].append(val_metrics['loss'])
                    if 'accuracy' in val_metrics:
                        history['val_accuracy'].append(val_metrics['accuracy'])

                    # 早停检查
                    if self.early_stop_patience > 0:
                        current_val_metric = val_metrics['loss']

                        if current_val_metric < self.best_val_metric:
                            self.best_val_metric = current_val_metric
                            self.early_stop_counter = 0

                            # 保存最佳模型
                            if self.config.get('training.save_best_model', True):
                                self.save_checkpoint(os.path.join(
                                    self.config.get('output_dir', 'logs'),
                                    'best_model.pth'
                                ))
                        else:
                            self.early_stop_counter += 1

                            if self.early_stop_counter >= self.early_stop_patience:
                                self.logger.info(f"早停: {self.early_stop_patience} epochs未改善")
                                break

            # 保存检查点
            if (epoch + 1) % self.config.get('training.checkpoint_interval', 10) == 0:
                self.save_checkpoint(os.path.join(
                    self.config.get('output_dir', 'logs'),
                    f'checkpoint_epoch_{epoch + 1}.pth'
                ))

        # 回调函数 - 训练结束
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(self, history)

        # 保存最终模型
        self.save_checkpoint(os.path.join(
            self.config.get('output_dir', 'logs'),
            'final_model.pth'
        ))

        return history

    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config.config
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        self.logger.info(f"模型已保存至 {path}")

    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_metric = checkpoint.get('best_val_metric', float('inf'))

        self.logger.info(f"模型已从 {path} 加载")

    def get_lr(self) -> float:
        """获取当前学习率"""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0

    def _to_device(self, x: Any) -> Any:
        """将数据移动到设备"""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [self._to_device(i) for i in x]
        elif isinstance(x, tuple):
            return tuple(self._to_device(i) for i in x)
        return x