#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import shutil
import datetime
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import threading
import logging
import uuid
from collections import defaultdict
import copy


class ExperimentManager:
    """
    实验管理器
    管理和跟踪深度学习训练实验的完整生命周期
    """

    def __init__(self, base_dir="experiments", experiment_name=None, config=None):
        """
        初始化实验管理器

        参数:
            base_dir: 实验基础目录
            experiment_name: 实验名称
            config: 配置字典
        """
        # 创建实验ID和名称
        self.experiment_id = str(int(time.time())) + "_" + str(uuid.uuid4())[:8]
        self.experiment_name = experiment_name or f"experiment_{self.experiment_id}"

        # 创建实验目录
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / self.experiment_name
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.log_dir = self.exp_dir / "logs"
        self.artifacts_dir = self.exp_dir / "artifacts"

        # 创建目录
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)

        # 初始化日志
        self.logger = self._setup_logger()

        # 配置
        self.config = config or {}

        # 保存配置文件
        self._save_config()

        # 实验指标
        self.metrics = {}
        self.epoch_metrics = defaultdict(list)
        self.batch_metrics = defaultdict(list)

        # 状态跟踪
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = float('inf')
        self.best_metric_name = 'loss'  # 默认监视损失
        self.start_time = None
        self.end_time = None

        # 可视化工具
        self.visualization = ExperimentVisualization(self.artifacts_dir)

        # 运行状态
        self.is_running = False

        # 自动保存配置
        self.auto_save_interval = 10  # 默认每10个批次保存一次
        self.last_auto_save = 0

        # 打印实验信息
        self.logger.info(f"实验 '{self.experiment_name}' 已创建")
        self.logger.info(f"实验目录: {self.exp_dir}")

    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)

        # 防止重复处理程序
        if not logger.handlers:
            # 文件处理程序
            file_handler = logging.FileHandler(self.log_dir / "experiment.log")
            file_handler.setLevel(logging.INFO)

            # 控制台处理程序
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 创建格式化程序
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # 添加处理程序
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger

    def _save_config(self):
        """保存配置文件"""
        config_path = self.exp_dir / "config.yaml"

        # 保存配置为YAML
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        self.logger.info(f"配置已保存至 {config_path}")

    def start(self, config_update=None):
        """
        开始实验

        参数:
            config_update: 配置更新
        """
        # 更新配置
        if config_update:
            self.config.update(config_update)
            self._save_config()

        # 记录开始时间
        self.start_time = time.time()
        self.is_running = True

        # 记录系统信息
        system_info = self._get_system_info()
        self.logger.info(f"系统信息: {system_info}")

        # 记录开始消息
        self.logger.info(f"实验 '{self.experiment_name}' 已开始")

        # 保存系统信息
        with open(self.exp_dir / "system_info.json", 'w') as f:
            json.dump(system_info, f, indent=4)

    def end(self):
        """结束实验"""
        # 检查是否已经开始
        if not self.is_running:
            self.logger.warning("实验尚未开始，无法结束")
            return

        # 记录结束时间
        self.end_time = time.time()

        # 计算总时间
        duration = self.end_time - self.start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)

        # 记录结束消息
        self.logger.info(f"实验 '{self.experiment_name}' 已结束")
        self.logger.info(f"总运行时间: {int(hours)}h {int(minutes)}m {int(seconds)}s")

        # 保存最终指标
        self._save_metrics()

        # 创建实验摘要
        self._create_experiment_summary()

        # 更新状态
        self.is_running = False

    def log_metric(self, name, value, step=None, epoch=None):
        """
        记录指标

        参数:
            name: 指标名称
            value: 指标值
            step: 步骤索引
            epoch: 轮次索引
        """
        # 更新当前步骤
        if step is not None:
            self.current_step = step

        # 更新当前轮次
        if epoch is not None:
            self.current_epoch = epoch

        # 记录指标
        if name not in self.metrics:
            self.metrics[name] = []

        # 如果值是张量，转换为Python标量
        if isinstance(value, torch.Tensor):
            value = value.item()

        # 添加指标
        self.metrics[name].append((self.current_step, value))

        # 添加到相应的轮次/批次指标
        if epoch is not None:
            self.epoch_metrics[name].append((epoch, value))
        else:
            self.batch_metrics[name].append((self.current_step, value))

        # 记录日志
        self.logger.info(f"指标 '{name}': {value:.6f} (步骤: {self.current_step}, 轮次: {self.current_epoch})")

        # 检查是否是最佳指标
        if name == self.best_metric_name:
            if self.is_better(value, self.best_metric):
                # 更新最佳指标
                previous_best = self.best_metric
                self.best_metric = value

                # 记录日志
                self.logger.info(f"最佳 '{name}' 已更新: {previous_best:.6f} -> {value:.6f}")

                # 自动保存最佳检查点
                if self.config.get('save_best_checkpoint', True):
                    self.save_checkpoint(is_best=True)

        # 自动保存
        if self.current_step - self.last_auto_save >= self.auto_save_interval:
            self._auto_save()
            self.last_auto_save = self.current_step

    def log_metrics(self, metrics_dict, step=None, epoch=None):
        """
        记录多个指标

        参数:
            metrics_dict: 指标字典
            step: 步骤索引
            epoch: 轮次索引
        """
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step, epoch)

    def log_hyperparameters(self, params):
        """
        记录超参数

        参数:
            params: 超参数字典
        """
        # 更新配置
        if 'hyperparameters' not in self.config:
            self.config['hyperparameters'] = {}

        self.config['hyperparameters'].update(params)

        # 保存配置
        self._save_config()

        # 记录日志
        self.logger.info(f"已记录超参数: {params}")

    def save_checkpoint(self, model=None, optimizer=None, scheduler=None,
                        filename=None, additional_data=None, is_best=False):
        """
        保存检查点

        参数:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            filename: 文件名
            additional_data: 额外数据
            is_best: 是否是最佳模型
        """
        # 创建检查点数据
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'metrics': dict(self.metrics),
            'best_metric': self.best_metric,
            'best_metric_name': self.best_metric_name,
            'config': self.config,
        }

        # 添加模型状态
        if model is not None:
            if isinstance(model, torch.nn.DataParallel) or hasattr(model, 'module'):
                # 如果是DataParallel模型，保存其module属性
                checkpoint['model_state_dict'] = model.module.state_dict()
            else:
                checkpoint['model_state_dict'] = model.state_dict()

        # 添加优化器状态
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # 添加调度器状态
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # 添加额外数据
        if additional_data:
            checkpoint.update(additional_data)

        # 确定文件名
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch}_step_{self.current_step}.pt"

        # 保存检查点
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        # 如果是最佳模型，复制一份
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copyfile(checkpoint_path, best_path)
            self.logger.info(f"已保存最佳模型至 {best_path}")

        self.logger.info(f"已保存检查点至 {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, model=None, optimizer=None, scheduler=None,
                        path=None, load_best=False, map_location=None):
        """
        加载检查点

        参数:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            path: 检查点路径
            load_best: 是否加载最佳模型
            map_location: 张量加载位置

        返回:
            加载的检查点数据
        """
        # 确定要加载的检查点
        if load_best:
            path = self.checkpoint_dir / "best_model.pt"
        elif path is None:
            # 查找最新的检查点
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))

            if not checkpoints:
                self.logger.warning("未找到检查点")
                return None

            # 按修改时间排序
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            path = latest_checkpoint

        # 检查文件是否存在
        if not os.path.exists(path):
            self.logger.error(f"检查点文件不存在: {path}")
            return None

        # 加载检查点
        self.logger.info(f"加载检查点: {path}")
        checkpoint = torch.load(path, map_location=map_location)

        # 加载模型状态
        if model is not None and 'model_state_dict' in checkpoint:
            if isinstance(model, torch.nn.DataParallel) or hasattr(model, 'module'):
                # 如果是DataParallel模型，加载到其module属性
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            self.logger.info("已加载模型参数")

        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("已加载优化器状态")

        # 加载调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("已加载学习率调度器状态")

        # 更新实验状态
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']

        if 'step' in checkpoint:
            self.current_step = checkpoint['step']

        if 'metrics' in checkpoint:
            self.metrics = checkpoint['metrics']

        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']

        if 'best_metric_name' in checkpoint:
            self.best_metric_name = checkpoint['best_metric_name']

        # 更新配置
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])

        self.logger.info(f"检查点加载完成，当前轮次: {self.current_epoch}, 步骤: {self.current_step}")

        return checkpoint

    def save_model(self, model, filename=None, optimize_for_inference=False):
        """
        保存模型

        参数:
            model: 要保存的模型
            filename: 文件名
            optimize_for_inference: 是否优化模型用于推理

        返回:
            保存路径
        """
        # 确定文件名
        if filename is None:
            filename = f"model_epoch_{self.current_epoch}.pt"

        # 确保模型处于评估模式
        model.eval()

        # 模型保存路径
        model_path = self.artifacts_dir / filename

        # 如果启用推理优化
        if optimize_for_inference:
            try:
                # 尝试使用TorchScript优化
                if isinstance(model, torch.nn.DataParallel) or hasattr(model, 'module'):
                    # 如果是DataParallel模型，优化其module属性
                    scripted_model = torch.jit.script(model.module)
                else:
                    scripted_model = torch.jit.script(model)

                # 保存优化后的模型
                scripted_model.save(model_path)

                self.logger.info(f"已保存推理优化模型至 {model_path}")
            except Exception as e:
                self.logger.warning(f"TorchScript优化失败: {e}，将保存标准模型")

                # 回退到标准保存
                if isinstance(model, torch.nn.DataParallel) or hasattr(model, 'module'):
                    torch.save(model.module.state_dict(), model_path)
                else:
                    torch.save(model.state_dict(), model_path)
        else:
            # 标准模型保存
            if isinstance(model, torch.nn.DataParallel) or hasattr(model, 'module'):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)

            self.logger.info(f"已保存模型至 {model_path}")

        return model_path

    def is_better(self, current, best, mode='min'):
        """
        检查当前指标是否优于最佳指标

        参数:
            current: 当前指标
            best: 最佳指标
            mode: 比较模式 ('min'或'max')

        返回:
            布尔值指示是否更好
        """
        if mode == 'min':
            return current < best
        else:  # mode == 'max'
            return current > best

    def set_best_metric(self, name, mode='min'):
        """
        设置要监视的最佳指标

        参数:
            name: 指标名称
            mode: 比较模式 ('min'或'max')
        """
        self.best_metric_name = name

        # 重置最佳值
        if mode == 'min':
            self.best_metric = float('inf')
        else:  # mode == 'max'
            self.best_metric = float('-inf')

        # 更新配置
        self.config['best_metric'] = {
            'name': name,
            'mode': mode
        }

        self.logger.info(f"最佳指标已设置为 '{name}' (模式: {mode})")

    def log_artifact(self, artifact, name=None, artifact_type=None):
        """
        记录工件

        参数:
            artifact: 要保存的工件
            name: 工件名称
            artifact_type: 工件类型

        返回:
            保存路径
        """
        # 确定工件名称
        if name is None:
            # 根据类型生成名称
            if artifact_type:
                name = f"{artifact_type}_{uuid.uuid4()}"
            else:
                name = f"artifact_{uuid.uuid4()}"

        # 确定文件扩展名
        ext = ""
        if isinstance(artifact, plt.Figure):
            ext = ".png"
        elif isinstance(artifact, dict) or isinstance(artifact, list):
            ext = ".json"
        elif isinstance(artifact, str) and (artifact.endswith('.png') or artifact.endswith('.jpg')):
            # 图像路径
            ext = ""  # 已包含在名称中

        # 构建完整文件名
        if ext and not name.endswith(ext):
            filename = name + ext
        else:
            filename = name

        # 构建保存路径
        artifact_path = self.artifacts_dir / filename

        # 保存工件
        if isinstance(artifact, plt.Figure):
            # 保存Matplotlib图表
            artifact.savefig(artifact_path, dpi=300, bbox_inches='tight')
            plt.close(artifact)
        elif isinstance(artifact, dict) or isinstance(artifact, list):
            # 保存JSON数据
            with open(artifact_path, 'w') as f:
                json.dump(artifact, f, indent=4)
        elif isinstance(artifact, np.ndarray):
            # 保存NumPy数组
            np.save(artifact_path, artifact)
        elif isinstance(artifact, str) and os.path.exists(artifact):
            # 复制文件
            shutil.copy(artifact, artifact_path)
        else:
            # 其他类型，尝试保存为文本
            try:
                with open(artifact_path, 'w') as f:
                    f.write(str(artifact))
            except Exception as e:
                self.logger.error(f"无法保存工件: {e}")
                return None

        self.logger.info(f"已保存工件至 {artifact_path}")

        return artifact_path

    def visualize_metrics(self, metric_names=None, save=True, show=False):
        """
        可视化指标

        参数:
            metric_names: 要可视化的指标名称列表
            save: 是否保存图表
            show: 是否显示图表

        返回:
            图表对象列表
        """
        # 使用可视化类
        return self.visualization.plot_metrics(
            self.epoch_metrics if len(self.epoch_metrics) > 0 else self.batch_metrics,
            metric_names=metric_names,
            save=save,
            show=show,
            save_dir=self.artifacts_dir
        )

    def _auto_save(self):
        """自动保存状态"""
        # 保存指标
        self._save_metrics()

        # 自动可视化
        if self.config.get('auto_visualize', True):
            self.visualize_metrics(save=True, show=False)

    def _save_metrics(self):
        """保存指标数据"""
        # 保存JSON格式的指标
        metrics_path = self.exp_dir / "metrics.json"

        # 转换为可序列化格式
        serializable_metrics = {}
        for name, values in self.metrics.items():
            serializable_metrics[name] = [(int(step), float(value)) for step, value in values]

        # 保存
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)

        self.logger.debug(f"已保存指标至 {metrics_path}")

    def _create_experiment_summary(self):
        """创建实验摘要"""
        # 计算运行时间
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            hours, remainder = divmod(duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        else:
            duration_str = "未知"

        # 提取最终指标
        final_metrics = {}
        for name, values in self.metrics.items():
            if values:
                final_metrics[name] = values[-1][1]

        # 创建摘要
        summary = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "start_time": datetime.datetime.fromtimestamp(self.start_time).strftime(
                '%Y-%m-%d %H:%M:%S') if self.start_time else None,
            "end_time": datetime.datetime.fromtimestamp(self.end_time).strftime(
                '%Y-%m-%d %H:%M:%S') if self.end_time else None,
            "duration": duration_str,
            "epochs": self.current_epoch,
            "steps": self.current_step,
            "best_metric": {
                "name": self.best_metric_name,
                "value": self.best_metric
            },
            "final_metrics": final_metrics,
            "hyperparameters": self.config.get('hyperparameters', {})
        }

        # 保存摘要
        summary_path = self.exp_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        self.logger.info(f"已创建实验摘要至 {summary_path}")

        # 创建可读摘要
        readable_summary = f"""实验摘要: {self.experiment_name}
=================================================
ID: {self.experiment_id}
开始时间: {summary['start_time']}
结束时间: {summary['end_time']}
持续时间: {duration_str}
轮次数: {self.current_epoch}
步骤数: {self.current_step}

最佳指标:
  {self.best_metric_name}: {self.best_metric:.6f}

最终指标:
"""
        # 添加最终指标
        for name, value in final_metrics.items():
            readable_summary += f"  {name}: {value:.6f}\n"

        # 添加超参数
        readable_summary += "\n超参数:\n"
        for name, value in self.config.get('hyperparameters', {}).items():
            readable_summary += f"  {name}: {value}\n"

        # 保存可读摘要
        readable_path = self.exp_dir / "summary.txt"
        with open(readable_path, 'w') as f:
            f.write(readable_summary)

    def _get_system_info(self):
        """获取系统信息"""
        info = {
            "python_version": ".".join(map(str, list(map(int, torch.__version__.split('.'))))),
            "torch_version": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

        # 如果有CUDA，添加GPU信息
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_0_name"] = torch.cuda.get_device_name(0)

            # 尝试获取显存信息
            try:
                info["gpu_0_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB"
            except:
                pass

        # 添加时间戳
        info["timestamp"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return info


class ExperimentVisualization:
    """
    实验可视化工具
    创建和管理实验结果的可视化
    """

    def __init__(self, save_dir="visualizations"):
        """
        初始化可视化工具

        参数:
            save_dir: 保存目录
        """
        self.save_dir = Path(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        # 设置样式
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                pass

    def plot_metrics(self, metrics, metric_names=None, save=True, show=False, save_dir=None):
        """
        绘制指标图表

        参数:
            metrics: 指标字典
            metric_names: 要绘制的指标名称
            save: 是否保存图表
            show: 是否显示图表
            save_dir: 保存目录

        返回:
            图表对象列表
        """
        # 如果未指定指标名称，使用所有指标
        if metric_names is None:
            metric_names = list(metrics.keys())

        # 对于每个指标创建一个图表
        figures = []

        for name in metric_names:
            if name not in metrics or not metrics[name]:
                continue

            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))

            # 提取数据
            steps, values = zip(*metrics[name])

            # 绘制曲线
            ax.plot(steps, values, 'o-', label=name)

            # 设置标题和标签
            ax.set_title(f"{name} vs. {'Epoch' if 'epoch' in str(steps[0]).lower() else 'Step'}")
            ax.set_xlabel('Epoch' if 'epoch' in str(steps[0]).lower() else 'Step')
            ax.set_ylabel(name)

            # 启用网格
            ax.grid(True, linestyle='--', alpha=0.7)

            # 添加图例
            ax.legend()

            # 紧凑布局
            fig.tight_layout()

            # 保存图表
            if save:
                save_path = (save_dir or self.save_dir) / f"{name}_plot.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')

            # 添加到列表
            figures.append(fig)

            # 是否显示
            if not show:
                plt.close(fig)

        # 如果请求显示，显示所有图表
        if show:
            plt.show()

        return figures

    def plot_loss_accuracy(self, train_loss, val_loss=None, train_acc=None, val_acc=None,
                           save=True, show=False, save_dir=None):
        """
        绘制损失和准确率图表

        参数:
            train_loss: 训练损失
            val_loss: 验证损失
            train_acc: 训练准确率
            val_acc: 验证准确率
            save: 是否保存图表
            show: 是否显示图表
            save_dir: 保存目录

        返回:
            图表对象列表
        """
        figures = []

        # 创建损失图表
        if train_loss:
            fig, ax = plt.subplots(figsize=(10, 6))

            # 提取训练损失数据
            train_epochs, train_losses = zip(*train_loss)
            ax.plot(train_epochs, train_losses, 'o-', label='Train Loss')

            # 如果有验证损失，也绘制
            if val_loss:
                val_epochs, val_losses = zip(*val_loss)
                ax.plot(val_epochs, val_losses, 'o-', label='Validation Loss')

            # 设置标题和标签
            ax.set_title("Loss vs. Epoch")
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')

            # 启用网格
            ax.grid(True, linestyle='--', alpha=0.7)

            # 添加图例
            ax.legend()

            # 紧凑布局
            fig.tight_layout()

            # 保存图表
            if save:
                save_path = (save_dir or self.save_dir) / "loss_plot.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')

            # 添加到列表
            figures.append(fig)

            # 是否显示
            if not show:
                plt.close(fig)

        # 创建准确率图表
        if train_acc or val_acc:
            fig, ax = plt.subplots(figsize=(10, 6))

            # 绘制训练准确率
            if train_acc:
                train_epochs, train_accs = zip(*train_acc)
                ax.plot(train_epochs, train_accs, 'o-', label='Train Accuracy')

            # 绘制验证准确率
            if val_acc:
                val_epochs, val_accs = zip(*val_acc)
                ax.plot(val_epochs, val_accs, 'o-', label='Validation Accuracy')

            # 设置标题和标签
            ax.set_title("Accuracy vs. Epoch")
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')

            # 启用网格
            ax.grid(True, linestyle='--', alpha=0.7)

            # 添加图例
            ax.legend()

            # 紧凑布局
            fig.tight_layout()

            # 保存图表
            if save:
                save_path = (save_dir or self.save_dir) / "accuracy_plot.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')

            # 添加到列表
            figures.append(fig)

            # 是否显示
            if not show:
                plt.close(fig)

        # 如果请求显示，显示所有图表
        if show:
            plt.show()

        return figures

    def plot_learning_rate(self, learning_rates, save=True, show=False, save_dir=None):
        """
        绘制学习率图表

        参数:
            learning_rates: 学习率列表
            save: 是否保存图表
            show: 是否显示图表
            save_dir: 保存目录

        返回:
            图表对象
        """
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))

        # 提取数据
        steps, lrs = zip(*learning_rates)

        # 绘制曲线
        ax.plot(steps, lrs, 'o-', label='Learning Rate')

        # 设置标题和标签
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')

        # 使用对数刻度
        ax.set_yscale('log')

        # 启用网格
        ax.grid(True, linestyle='--', alpha=0.7)

        # 添加图例
        ax.legend()

        # 紧凑布局
        fig.tight_layout()

        # 保存图表
        if save:
            save_path = (save_dir or self.save_dir) / "learning_rate_plot.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        # 是否显示
        if not show:
            plt.close(fig)

        return fig

    def plot_confusion_matrix(self, cm, classes=None, normalize=False,
                              title='Confusion Matrix', cmap=plt.cm.Blues,
                              save=True, show=False, save_dir=None):
        """
        绘制混淆矩阵

        参数:
            cm: 混淆矩阵
            classes: 类别标签
            normalize: 是否归一化
            title: 图表标题
            cmap: 颜色映射
            save: 是否保存图表
            show: 是否显示图表
            save_dir: 保存目录

        返回:
            图表对象
        """
        # 如果输入是张量，转换为NumPy数组
        if isinstance(cm, torch.Tensor):
            cm = cm.cpu().numpy()

        # 归一化
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            print("显示归一化的混淆矩阵")
        else:
            print("显示混淆矩阵，不归一化")

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8))

        # 显示混淆矩阵
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        # 我们希望在图表上显示所有标记
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            ylabel='True label',
            xlabel='Predicted label'
        )

        # 设置类别标签
        if classes is not None:
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)

        # 旋转x轴标签并在上方对齐
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # 循环遍历数据并创建文本注释
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        ax.set_title(title)
        fig.tight_layout()

        # 保存图表
        if save:
            save_path = (save_dir or self.save_dir) / "confusion_matrix.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        # 是否显示
        if not show:
            plt.close(fig)

        return fig

    def plot_roc_curve(self, fpr, tpr, roc_auc, save=True, show=False, save_dir=None):
        """
        绘制ROC曲线

        参数:
            fpr: 假正例率
            tpr: 真正例率
            roc_auc: ROC曲线下面积
            save: 是否保存图表
            show: 是否显示图表
            save_dir: 保存目录

        返回:
            图表对象
        """
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制ROC曲线
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')

        # 绘制对角线
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        # 设置范围和标签
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")

        # 启用网格
        ax.grid(True, linestyle='--', alpha=0.7)

        # 紧凑布局
        fig.tight_layout()

        # 保存图表
        if save:
            save_path = (save_dir or self.save_dir) / "roc_curve.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        # 是否显示
        if not show:
            plt.close(fig)

        return fig

    def plot_precision_recall_curve(self, precision, recall, average_precision,
                                    save=True, show=False, save_dir=None):
        """
        绘制精度-召回率曲线

        参数:
            precision: 精度
            recall: 召回率
            average_precision: 平均精度
            save: 是否保存图表
            show: 是否显示图表
            save_dir: 保存目录

        返回:
            图表对象
        """
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制精度-召回率曲线
        ax.plot(recall, precision, color='darkorange', lw=2,
                label=f'Precision-Recall curve (AP = {average_precision:.2f})')

        # 设置范围和标签
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")

        # 启用网格
        ax.grid(True, linestyle='--', alpha=0.7)

        # 紧凑布局
        fig.tight_layout()

        # 保存图表
        if save:
            save_path = (save_dir or self.save_dir) / "precision_recall_curve.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        # 是否显示
        if not show:
            plt.close(fig)

        return fig


class ExperimentTracker:
    """
    实验跟踪器
    跟踪当前实验的进度，并在控制台显示一个好看的进度条
    """

    def __init__(self, total_epochs, total_steps_per_epoch=None,
                 desc="Training",
                 metrics_to_display=None):
        """
        初始化进度跟踪器

        参数:
            total_epochs: 总轮次
            total_steps_per_epoch: 每轮次的总步骤数
            desc: 描述
            metrics_to_display: 要显示的指标列表
        """
        self.total_epochs = total_epochs
        self.total_steps_per_epoch = total_steps_per_epoch
        self.desc = desc
        self.metrics_to_display = metrics_to_display or ["loss"]

        self.current_epoch = 0
        self.current_step = 0
        self.epoch_start_time = None
        self.total_start_time = None

        # 指标存储
        self.metrics = defaultdict(list)
        self.current_metrics = {}

        # 尝试获取终端宽度
        try:
            self.term_width = os.get_terminal_size().columns
        except:
            self.term_width = 80

    def start(self):
        """开始跟踪"""
        # 记录开始时间
        self.total_start_time = time.time()
        self.epoch_start_time = time.time()

        # 打印初始消息
        self._print_header()

    def update(self, step=None, metrics=None):
        """
        更新跟踪器

        参数:
            step: 当前步骤
            metrics: 指标字典
        """
        # 更新步骤
        if step is not None:
            self.current_step = step

        # 更新指标
        if metrics:
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()

                self.current_metrics[name] = value
                self.metrics[name].append(value)

        # 打印进度
        self._print_progress()

    def next_epoch(self):
        """进入下一轮次"""
        # 完成当前轮次
        elapsed = time.time() - self.epoch_start_time

        # 打印轮次摘要
        self._print_epoch_summary(elapsed)

        # 增加轮次计数
        self.current_epoch += 1

        # 重置步骤和指标
        self.current_step = 0
        self.current_metrics = {}

        # 如果已完成所有轮次，打印总结
        if self.current_epoch >= self.total_epochs:
            self._print_final_summary()
        else:
            # 重置轮次开始时间
            self.epoch_start_time = time.time()

            # 打印新轮次头部
            self._print_header()

    def _print_header(self):
        """打印头部"""
        print("\n" + "=" * self.term_width)
        print(f"{self.desc} - Epoch {self.current_epoch + 1}/{self.total_epochs}")
        print("-" * self.term_width)

        # 打印指标头部
        header = f"{'Step':>6} {'Progress':>10} {'Time':>10}"

        for metric in self.metrics_to_display:
            header += f" {metric:>10}"

        print(header)
        print("-" * self.term_width)

    def _print_progress(self):
        """打印进度"""
        # 清除当前行
        print("\r", end="")

        # 计算进度
        if self.total_steps_per_epoch:
            progress = min(1.0, self.current_step / self.total_steps_per_epoch)
            progress_bar = self._get_progress_bar(progress, width=10)
        else:
            progress_bar = "N/A       "

        # 计算经过的时间
        elapsed = time.time() - self.epoch_start_time
        time_str = self._format_time(elapsed)

        # 构建行
        progress_line = f"{self.current_step:6d} {progress_bar} {time_str:>10}"

        # 添加指标
        for metric in self.metrics_to_display:
            if metric in self.current_metrics:
                value = self.current_metrics[metric]
                progress_line += f" {value:10.4f}"
            else:
                progress_line += f" {'N/A':>10}"

        # 打印
        print(progress_line, end="")

    def _print_epoch_summary(self, elapsed):
        """
        打印轮次摘要

        参数:
            elapsed: 经过的时间
        """
        # 清除当前行
        print("\r", end="")

        # 计算平均指标
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = sum(values) / len(values)

        # 打印摘要
        print("\n" + "-" * self.term_width)
        print(f"Epoch {self.current_epoch + 1}/{self.total_epochs} 完成")
        print(f"时间: {self._format_time(elapsed)}")

        # 打印指标
        for name, value in summary.items():
            print(f"{name}: {value:.6f}")

        print("-" * self.term_width)

    def _print_final_summary(self):
        """打印最终摘要"""
        # 计算总时间
        elapsed = time.time() - self.total_start_time

        print("\n" + "=" * self.term_width)
        print(f"{self.desc} 完成")
        print(f"总时间: {self._format_time(elapsed)}")
        print("=" * self.term_width)

    def _get_progress_bar(self, progress, width=10, fill='█', empty='░'):
        """
        获取进度条字符串

        参数:
            progress: 进度 (0-1)
            width: 进度条宽度
            fill: 填充字符
            empty: 空字符

        返回:
            进度条字符串
        """
        filled_width = int(width * progress)
        bar = fill * filled_width + empty * (width - filled_width)
        return bar

    def _format_time(self, seconds):
        """
        格式化时间

        参数:
            seconds: 秒数

        返回:
            格式化时间字符串
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes, seconds = divmod(seconds, 60)
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours)}h {int(minutes)}m"


def load_experiment(experiment_dir):
    """
    加载现有实验

    参数:
        experiment_dir: 实验目录

    返回:
        ExperimentManager对象
    """
    # 检查目录是否存在
    if not os.path.exists(experiment_dir):
        raise ValueError(f"实验目录不存在: {experiment_dir}")

    # 获取实验名称
    experiment_name = os.path.basename(experiment_dir)

    # 尝试加载配置
    config_path = os.path.join(experiment_dir, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # 创建实验管理器
    base_dir = os.path.dirname(experiment_dir)
    manager = ExperimentManager(base_dir=base_dir, experiment_name=experiment_name, config=config)

    # 尝试加载指标
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # 添加指标
        for name, values in metrics.items():
            manager.metrics[name] = values

            # 分类为轮次或批次指标
            is_epoch_metric = any('epoch' in str(step).lower() for step, _ in values)

            if is_epoch_metric:
                manager.epoch_metrics[name] = values
            else:
                manager.batch_metrics[name] = values

    # 尝试加载摘要
    summary_path = os.path.join(experiment_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        # 提取信息
        if 'epochs' in summary:
            manager.current_epoch = summary['epochs']

        if 'steps' in summary:
            manager.current_step = summary['steps']

        if 'best_metric' in summary:
            manager.best_metric_name = summary['best_metric']['name']
            manager.best_metric = summary['best_metric']['value']

    return manager


def list_experiments(base_dir="experiments"):
    """
    列出所有实验

    参数:
        base_dir: 实验基础目录

    返回:
        实验摘要列表
    """
    # 检查目录是否存在
    if not os.path.exists(base_dir):
        return []

    # 获取所有实验目录
    exp_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # 收集摘要
    summaries = []

    for exp_dir in exp_dirs:
        summary_path = os.path.join(base_dir, exp_dir, "summary.json")

        if os.path.exists(summary_path):
            # 加载摘要
            with open(summary_path, 'r') as f:
                summary = json.load(f)

            # 添加目录信息
            summary['directory'] = os.path.join(base_dir, exp_dir)

            summaries.append(summary)
        else:
            # 尝试从配置构建简单摘要
            config_path = os.path.join(base_dir, exp_dir, "config.yaml")

            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # 创建简单摘要
                summaries.append({
                    'experiment_name': exp_dir,
                    'directory': os.path.join(base_dir, exp_dir),
                    'config': config
                })

    # 按开始时间排序（如果有）
    summaries.sort(key=lambda x: x.get('start_time', ''), reverse=True)

    return summaries


def compare_experiments(experiment_dirs, metric_names=None, save=True, show=False, save_path=None):
    """
    比较多个实验

    参数:
        experiment_dirs: 实验目录列表
        metric_names: 要比较的指标名称
        save: 是否保存图表
        show: 是否显示图表
        save_path: 保存路径

    返回:
        图表对象列表
    """
    # 加载实验
    experiments = []
    for exp_dir in experiment_dirs:
        try:
            exp = load_experiment(exp_dir)
            experiments.append(exp)
        except Exception as e:
            print(f"无法加载实验 {exp_dir}: {e}")

    if not experiments:
        print("未找到有效实验")
        return []

    # 如果未指定指标名称，使用所有常见指标
    if metric_names is None:
        # 收集所有实验的指标
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp.metrics.keys())

        # 选择常见指标
        metric_names = [name for name in all_metrics if
                        any(name in common for common in ['loss', 'acc', 'precision', 'recall', 'f1'])]

        if not metric_names:
            metric_names = list(all_metrics)

    # 创建图表
    figures = []

    for name in metric_names:
        # 检查哪些实验有此指标
        valid_exps = [exp for exp in experiments if name in exp.metrics and exp.metrics[name]]

        if not valid_exps:
            continue

        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 8))

        # 为每个实验绘制曲线
        for exp in valid_exps:
            # 获取数据 - 优先使用轮次数据
            if name in exp.epoch_metrics and exp.epoch_metrics[name]:
                steps, values = zip(*exp.epoch_metrics[name])
                x_label = 'Epoch'
            else:
                steps, values = zip(*exp.metrics[name])
                x_label = 'Step'

            # 绘制曲线
            ax.plot(steps, values, 'o-', label=exp.experiment_name)

        # 设置标题和标签
        ax.set_title(f"{name} Comparison")
        ax.set_xlabel(x_label)
        ax.set_ylabel(name)

        # 启用网格
        ax.grid(True, linestyle='--', alpha=0.7)

        # 添加图例
        ax.legend()

        # 紧凑布局
        fig.tight_layout()

        # 保存图表
        if save:
            # 确定保存路径
            if save_path is None:
                save_path = os.path.join("comparisons", f"{name}_comparison.png")

            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        # 添加到列表
        figures.append(fig)

        # 是否显示
        if not show:
            plt.close(fig)

    # 如果请求显示，显示所有图表
    if show:
        plt.show()

    return figures


def create_experiment_report(experiment_dir, output_path=None):
    """
    创建实验报告

    参数:
        experiment_dir: 实验目录
        output_path: 输出路径

    返回:
        报告路径
    """
    # 加载实验
    exp = load_experiment(experiment_dir)

    # 确定输出路径
    if output_path is None:
        output_path = os.path.join(experiment_dir, "report.md")

    # 创建报告
    report = f"""# 实验报告: {exp.experiment_name}

## 基本信息

- **实验ID**: {exp.experiment_id}
- **创建时间**: {datetime.datetime.fromtimestamp(exp.start_time).strftime('%Y-%m-%d %H:%M:%S') if exp.start_time else 'N/A'}
- **完成时间**: {datetime.datetime.fromtimestamp(exp.end_time).strftime('%Y-%m-%d %H:%M:%S') if exp.end_time else 'N/A'}
- **总轮次**: {exp.current_epoch}
- **总步骤**: {exp.current_step}

## 指标摘要

| 指标 | 最终值 | 最佳值 |
|------|--------|--------|
"""
    # 添加指标
    for name, values in exp.metrics.items():
        if values:
            final_value = values[-1][1]

            # 查找最佳值
            if name == exp.best_metric_name:
                best_value = exp.best_metric
            else:
                # 假设较小值更好
                best_value = min([value for _, value in values])

            report += f"| {name} | {final_value:.6f} | {best_value:.6f} |\n"

    # 添加超参数
    report += "\n## 超参数\n\n"

    if 'hyperparameters' in exp.config:
        for name, value in exp.config['hyperparameters'].items():
            report += f"- **{name}**: {value}\n"
    else:
        report += "*无超参数记录*\n"

    # 添加系统信息
    report += "\n## 系统信息\n\n"

    system_info_path = os.path.join(experiment_dir, "system_info.json")
    if os.path.exists(system_info_path):
        with open(system_info_path, 'r') as f:
            system_info = json.load(f)

        for name, value in system_info.items():
            report += f"- **{name}**: {value}\n"
    else:
        report += "*无系统信息记录*\n"

    # 添加图表
    report += "\n## 指标可视化\n\n"

    # 可视化所有指标
    exp.visualize_metrics(save=True, show=False)

    # 添加图表链接
    for name in exp.metrics.keys():
        image_path = os.path.join("artifacts", f"{name}_plot.png")
        if os.path.exists(os.path.join(experiment_dir, image_path)):
            report += f"### {name}\n\n"
            report += f"![{name}]({image_path})\n\n"

    # 保存报告
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"实验报告已保存至: {output_path}")

    return output_path


if __name__ == "__main__":
    """测试实验管理器和可视化功能"""

    # 测试实验管理器
    exp_manager = ExperimentManager(
        experiment_name="test_experiment",
        config={
            "model": "cnn",
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 64,
                "optimizer": "adamw"
            }
        }
    )

    # 开始实验
    exp_manager.start()

    # 模拟训练循环
    for epoch in range(10):
        for step in range(100):
            # 记录批次指标
            exp_manager.log_metric(
                "loss",
                1.0 - 0.1 * epoch - 0.001 * step + 0.1 * np.random.randn(),
                step=step,
                epoch=epoch
            )

            exp_manager.log_metric(
                "accuracy",
                0.5 + 0.05 * epoch + 0.001 * step + 0.05 * np.random.randn(),
                step=step,
                epoch=epoch
            )

        # 记录轮次指标
        exp_manager.log_metric(
            "val_loss",
            0.8 - 0.07 * epoch + 0.05 * np.random.randn(),
            epoch=epoch
        )

        exp_manager.log_metric(
            "val_accuracy",
            0.6 + 0.04 * epoch + 0.03 * np.random.randn(),
            epoch=epoch
        )

        # 保存检查点
        exp_manager.save_checkpoint(filename=f"checkpoint_epoch_{epoch}.pt")

    # 可视化指标
    figures = exp_manager.visualize_metrics(save=True, show=True)

    # 结束实验
    exp_manager.end()

    # 测试实验比较
    experiments = list_experiments()
    if len(experiments) > 1:
        exp_dirs = [exp['directory'] for exp in experiments[:2]]
        compare_experiments(exp_dirs, show=True)

    # 测试创建报告
    if experiments:
        create_experiment_report(experiments[0]['directory'])

    # 测试进度跟踪器
    print("\n测试进度跟踪器:")

    tracker = ExperimentTracker(
        total_epochs=5,
        total_steps_per_epoch=100,
        metrics_to_display=["loss", "accuracy"]
    )

    tracker.start()

    for epoch in range(5):
        for step in range(100):
            # 随机指标
            loss = 1.0 - 0.1 * epoch - 0.001 * step + 0.1 * np.random.randn()
            accuracy = 0.5 + 0.05 * epoch + 0.001 * step + 0.05 * np.random.randn()

            # 更新跟踪器
            tracker.update(step, {"loss": loss, "accuracy": accuracy})

            # 模拟训练时间
            time.sleep(0.01)

        # 下一轮次
        tracker.next_epoch()