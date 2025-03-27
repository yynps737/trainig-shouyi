#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import torch
import numpy as np
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# 导入自定义模块
from src.config.config import load_config
from src.models.advanced_models import RTXEfficientNet, FastRTXTransformer, build_rtx_optimized_model
from src.utils.advanced_optimizers import get_advanced_optimizer, get_advanced_scheduler, mixed_precision_context
from src.utils.memory_optimization import MemoryOptimizer, MemoryTracker, profile_memory_usage
from src.utils.experiment_manager import ExperimentManager, ExperimentTracker
from src.utils.distributed import setup_distributed, get_data_parallel_model, spawn_processes, cleanup_distributed, \
    is_main_process
from src.data.optimized_loader import create_optimized_dataloaders, apply_mixup, apply_cutmix
from src.utils.logger import get_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AiXunLianCang优化训练脚本")
    parser.add_argument("--config", type=str, default="config/rtx4070_config.yaml", help="配置文件路径")
    parser.add_argument("--model", type=str, default="efficient_net", help="模型类型 (efficient_net, transformer)")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="数据目录")
    parser.add_argument("--output-dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--experiment-name", type=str, help="实验名称")
    parser.add_argument("--checkpoint", type=str, help="加载检查点")
    parser.add_argument("--resume", action="store_true", help="从检查点恢复训练")
    parser.add_argument("--distributed", action="store_true", help="启用分布式训练")
    parser.add_argument("--profile", action="store_true", help="启用性能分析")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--batch-size", type=int, help="批次大小")
    parser.add_argument("--learning-rate", type=float, help="学习率")
    parser.add_argument("--workers", type=int, help="数据加载器工作线程数")
    parser.add_argument("--optimization-level", type=str, choices=["light", "medium", "max", "auto"], help="优化级别")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--mixed-precision", action="store_true", help="启用混合精度训练")
    parser.add_argument("--gradient-checkpoint", action="store_true", help="启用梯度检查点")
    parser.add_argument("--cpu-offload", action="store_true", help="启用CPU卸载")
    parser.add_argument("--compile", action="store_true", help="启用PyTorch编译")
    parser.add_argument("--memory-efficient", action="store_true", help="启用内存高效模式")
    parser.add_argument("--log-interval", type=int, default=10, help="日志记录间隔")
    parser.add_argument("--save-interval", type=int, default=1000, help="保存间隔")
    parser.add_argument("--eval-interval", type=int, default=1, help="评估间隔")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的本地排名")

    return parser.parse_args()


def get_dataset(data_dir, config, dataset_type='CIFAR10'):
    """获取数据集"""
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from src.data.transforms import get_train_transforms, get_val_transforms

    # 获取变换
    train_transform = get_train_transforms(
        input_size=(224, 224),
        augmentation_level=config.get("data.augmentation_level", "medium")
    )

    val_transform = get_val_transforms(
        input_size=(224, 224)
    )

    # 根据数据集类型加载数据
    if dataset_type == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )

        val_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=val_transform
        )
    elif dataset_type == 'CIFAR100':
        train_dataset = datasets.CIFAR100(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )

        val_dataset = datasets.CIFAR100(
            root=data_dir,
            train=False,
            download=True,
            transform=val_transform
        )
    elif dataset_type == 'ImageNet':
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')

        train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=train_transform
        )

        val_dataset = datasets.ImageFolder(
            root=val_dir,
            transform=val_transform
        )
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")

    return train_dataset, val_dataset


def build_model(model_type, config, num_classes=10):
    """构建模型"""
    if model_type == "efficient_net":
        model = RTXEfficientNet(
            in_channels=3,
            num_classes=num_classes,
            width_multiplier=config.get("model.width_multiplier", 1.0),
            depth_multiplier=config.get("model.depth_multiplier", 1.0),
            dropout_rate=config.get("model.dropout_rate", 0.2)
        )
    elif model_type == "transformer":
        model = FastRTXTransformer(
            vocab_size=config.get("model.vocab_size", 30000),
            max_seq_len=config.get("model.max_seq_len", 512),
            d_model=config.get("model.d_model", 512),
            nhead=config.get("model.nhead", 8),
            num_layers=config.get("model.num_layers", 6),
            dim_feedforward=config.get("model.dim_feedforward", 2048),
            dropout=config.get("model.dropout_rate", 0.1),
            num_classes=num_classes
        )
    elif model_type.startswith("rtx_"):
        # 动态构建RTX优化模型
        model_name = model_type[4:]  # 删除"rtx_"前缀
        model = build_rtx_optimized_model(
            model_name,
            num_classes=num_classes,
            **config.get("model", {})
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return model


def optimize_model_for_rtx4070(model, config, input_size=(3, 224, 224), batch_size=32):
    """为RTX 4070优化模型"""
    print("正在为RTX 4070优化模型...")

    # 创建内存优化器
    mem_optimizer = MemoryOptimizer(
        model,
        max_vram_gb=config.get("hardware.vram_gb", 12),
        config=config
    )

    # 执行优化
    optimization_level = config.get("optimization_level", "auto")
    if optimization_level != "auto":
        # 应用特定优化级别
        if optimization_level == "light":
            mem_optimizer._apply_light_optimizations()
        elif optimization_level == "medium":
            mem_optimizer._apply_medium_optimizations()
        elif optimization_level == "max":
            mem_optimizer._apply_all_optimizations()
    else:
        # 自动优化
        mem_optimizer.optimize(input_size, batch_size)

    # 获取优化配置
    return mem_optimizer.config


def create_criterion_and_metrics(config, num_classes=10):
    """创建损失函数和评估指标"""
    # 创建损失函数
    loss_type = config.get("training.loss", "cross_entropy")

    if loss_type == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_type == "label_smoothing":
        smoothing = config.get("training.label_smoothing", 0.1)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
    elif loss_type == "focal":
        from torch.nn.functional import cross_entropy

        # 创建Focal Loss
        def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
            ce_loss = cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = alpha * (1 - pt) ** gamma * ce_loss
            return loss.mean()

        alpha = config.get("training.focal_alpha", 0.25)
        gamma = config.get("training.focal_gamma", 2.0)
        criterion = lambda x, y: focal_loss(x, y, alpha, gamma)
    else:
        raise ValueError(f"不支持的损失函数: {loss_type}")

    # 评估指标函数
    def compute_metrics(outputs, targets):
        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total

        # 计算损失
        loss = criterion(outputs, targets).item()

        metrics = {
            'loss': loss,
            'accuracy': accuracy
        }

        # 添加多分类指标
        if num_classes > 2:
            # 计算Top-5准确率 (如果类别数大于5)
            if num_classes > 5:
                _, top5_predicted = outputs.topk(5, 1, True, True)
                top5_predicted = top5_predicted.t()
                top5_correct = top5_predicted.eq(targets.view(1, -1).expand_as(top5_predicted))
                top5_correct = top5_correct.reshape(-1).float().sum(0)
                top5_accuracy = top5_correct.item() / total
                metrics['top5_accuracy'] = top5_accuracy

        return metrics

    return criterion, compute_metrics


@profile_memory_usage
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                config, experiment, device, metrics_fn=None, local_rank=0, distributed=False):
    """训练模型"""
    # 开始实验
    experiment.start()

    # 创建内存跟踪器
    memory_tracker = MemoryTracker(
        log_interval=config.get("training.log_interval", 10),
        enabled=config.get("training.track_memory", True)
    )
    memory_tracker.start(model)

    # 创建进度跟踪器
    progress_tracker = ExperimentTracker(
        total_epochs=config.get("training.epochs", 100),
        total_steps_per_epoch=len(train_loader),
        desc="Training",
        metrics_to_display=["loss", "accuracy"]
    )
    progress_tracker.start()

    # 获取训练参数
    num_epochs = config.get("training.epochs", 100)
    grad_accum_steps = config.get("optimization.gradient_accumulation_steps", 1)
    clip_grad_norm = config.get("training.gradient_clip", 0.0)
    log_interval = config.get("training.log_interval", 10)
    save_interval = config.get("training.save_interval", 1000)
    eval_interval = config.get("training.eval_interval", 1)

    # 启用混合精度
    use_amp = config.get("optimization.mixed_precision", False)
    amp_dtype = getattr(torch, config.get("optimization.mixed_precision_dtype", "float16"))
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 启用梯度累积
    if grad_accum_steps > 1:
        print(f"启用梯度累积，步数: {grad_accum_steps}")

    # 数据增强配置
    mixup_alpha = config.get("data.mixup_alpha", 0.0)
    cutmix_alpha = config.get("data.cutmix_alpha", 0.0)

    # 编译模型（如果启用）
    if config.get("optimization.compiler_enabled", False) and hasattr(torch, 'compile'):
        print("使用PyTorch编译模型...")
        model = torch.compile(model, mode=config.get("optimization.compile_mode", "default"))

    # 全局步骤计数器
    global_step = 0
    start_epoch = 0

    # 主训练循环
    for epoch in range(start_epoch, num_epochs):
        # 设置为训练模式
        model.train()

        # 重置数据加载器采样器（分布式训练）
        if distributed and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # 轮次指标
        epoch_loss = 0.0
        epoch_metrics = {}

        # 批次循环
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 更新全局步骤
            global_step += 1

            # 是否执行梯度步骤
            do_optimizer_step = (batch_idx + 1) % grad_accum_steps == 0

            # 移动数据到设备
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # 应用Mixup/CutMix数据增强
            if mixup_alpha > 0 and np.random.random() < 0.5:
                inputs, targets_a, targets_b, lam = apply_mixup(inputs, targets, mixup_alpha)
                mixup_applied = True
            elif cutmix_alpha > 0:
                inputs, targets_a, targets_b, lam = apply_cutmix(inputs, targets, cutmix_alpha)
                mixup_applied = True
            else:
                mixup_applied = False

            # 使用混合精度
            with mixed_precision_context(enabled=use_amp, dtype=amp_dtype):
                # 前向传播
                outputs = model(inputs)

                # 计算损失
                if mixup_applied:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets)

                # 根据梯度累积缩放损失
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps

            # 反向传播
            if scaler is not None:
                # 使用梯度缩放器
                scaler.scale(loss).backward()

                # 执行优化步骤
                if do_optimizer_step:
                    # 梯度裁剪
                    if clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                    # 参数更新
                    scaler.step(optimizer)
                    scaler.update()

                    # 清零梯度
                    optimizer.zero_grad()

                    # 更新学习率
                    if scheduler is not None:
                        scheduler.step()
            else:
                # 标准反向传播
                loss.backward()

                # 执行优化步骤
                if do_optimizer_step:
                    # 梯度裁剪
                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                    # 参数更新
                    optimizer.step()

                    # 清零梯度
                    optimizer.zero_grad()

                    # 更新学习率
                    if scheduler is not None:
                        scheduler.step()

            # 计算批次指标
            batch_metrics = {}
            if not mixup_applied and metrics_fn is not None:
                with torch.no_grad():
                    batch_metrics = metrics_fn(outputs, targets)

            # 如果使用Mixup/CutMix，仅记录损失
            if mixup_applied:
                batch_metrics = {'loss': loss.item() * (grad_accum_steps if grad_accum_steps > 1 else 1)}

            # 累积轮次损失
            epoch_loss += loss.item() * (grad_accum_steps if grad_accum_steps > 1 else 1)

            # 更新轮次指标
            for k, v in batch_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = 0.0
                epoch_metrics[k] += v

            # 更新进度跟踪器
            progress_tracker.update(batch_idx, batch_metrics)

            # 更新内存跟踪器
            memory_tracker.step()

            # 记录日志
            if is_main_process() and batch_idx % log_interval == 0:
                # 当前学习率
                current_lr = optimizer.param_groups[0]['lr']

                # 日志消息
                log_msg = (f"Epoch: {epoch}/{num_epochs - 1}, Batch: {batch_idx}/{len(train_loader) - 1}, "
                           f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

                # 添加指标
                for k, v in batch_metrics.items():
                    if k != 'loss':
                        log_msg += f", {k}: {v:.4f}"

                print(log_msg)

                # 记录到实验
                experiment.log_metric("train_loss", loss.item(), step=global_step, epoch=epoch)
                experiment.log_metric("learning_rate", current_lr, step=global_step, epoch=epoch)

                for k, v in batch_metrics.items():
                    if k != 'loss':
                        experiment.log_metric(f"train_{k}", v, step=global_step, epoch=epoch)

            # 定期保存
            if is_main_process() and global_step % save_interval == 0:
                experiment.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    filename=f"checkpoint_epoch_{epoch}_step_{global_step}.pt"
                )

                # 可视化指标
                experiment.visualize_metrics(save=True, show=False)

        # 轮次结束，计算平均指标
        epoch_loss /= len(train_loader)

        for k in epoch_metrics:
            epoch_metrics[k] /= len(train_loader)

        # 记录轮次指标
        if is_main_process():
            print(f"Epoch {epoch} 训练完成，平均损失: {epoch_loss:.4f}")

            # 记录到实验
            experiment.log_metric("epoch_train_loss", epoch_loss, epoch=epoch)

            for k, v in epoch_metrics.items():
                if k != 'loss':
                    print(f"Epoch {epoch} 训练 {k}: {v:.4f}")
                    experiment.log_metric(f"epoch_train_{k}", v, epoch=epoch)

        # 执行验证
        if epoch % eval_interval == 0:
            val_metrics = evaluate_model(model, val_loader, criterion, metrics_fn, device, config)

            # 记录验证指标
            if is_main_process():
                for k, v in val_metrics.items():
                    print(f"Epoch {epoch} 验证 {k}: {v:.4f}")
                    experiment.log_metric(f"val_{k}", v, epoch=epoch)

                    # 如果是损失，检查是否为最佳模型
                    if k == 'loss' and (epoch == 0 or v < experiment.best_metric):
                        experiment.save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            filename="best_model.pt",
                            is_best=True
                        )
                        print(f"发现新的最佳模型，验证损失: {v:.4f}")

        # 更新进度跟踪器
        progress_tracker.next_epoch()

        # 保存检查点
        if is_main_process():
            experiment.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                filename=f"checkpoint_epoch_{epoch}.pt"
            )

    # 停止内存跟踪器
    memory_tracker.stop()
    memory_tracker.plot(os.path.join(experiment.artifacts_dir, "memory_usage.png"))

    # 训练结束
    if is_main_process():
        print("训练完成!")

        # 保存最终模型
        experiment.save_model(model, "final_model.pt")

        # 保存优化模型
        experiment.save_model(model, "optimized_model.pt", optimize_for_inference=True)

        # 结束实验
        experiment.end()

    return model, global_step


def evaluate_model(model, val_loader, criterion, metrics_fn, device, config):
    """评估模型"""
    # 设置为评估模式
    model.eval()

    # 启用混合精度
    use_amp = config.get("optimization.mixed_precision", False)
    amp_dtype = getattr(torch, config.get("optimization.mixed_precision_dtype", "float16"))

    # 初始化指标
    metrics_sum = {}

    # 禁用梯度计算
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # 移动数据到设备
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # 使用混合精度
            with mixed_precision_context(enabled=use_amp, dtype=amp_dtype):
                # 前向传播
                outputs = model(inputs)

                # 计算损失
                loss = criterion(outputs, targets)

            # 计算指标
            batch_metrics = {'loss': loss.item()}

            if metrics_fn is not None:
                batch_metrics.update(metrics_fn(outputs, targets))

            # 累积指标
            for k, v in batch_metrics.items():
                if k not in metrics_sum:
                    metrics_sum[k] = 0.0
                metrics_sum[k] += v

    # 计算平均指标
    metrics_avg = {k: v / len(val_loader) for k, v in metrics_sum.items()}

    return metrics_avg


def train_distributed(rank, world_size, config, args):
    """分布式训练入口点"""
    # 设置分布式环境
    local_rank, _ = setup_distributed(rank=rank, world_size=world_size)

    # 设置设备
    device = torch.device(f"cuda:{local_rank}")

    # 加载配置
    config_obj = load_config(args.config)

    # 更新配置
    if args.epochs:
        config_obj.set("training.epochs", args.epochs)
    if args.batch_size:
        config_obj.set("model.batch_size", args.batch_size)
    if args.learning_rate:
        config_obj.set("optimization.learning_rate", args.learning_rate)
    if args.workers:
        config_obj.set("data.num_workers", args.workers)

    # 合并配置
    config = config_obj.config

    # 设置随机种子
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # 获取数据集
    train_dataset, val_dataset = get_dataset(args.data_dir, config_obj)

    # 创建数据加载器
    train_loader, val_loader = create_optimized_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config.get("model.batch_size", 64),
        num_workers=config.get("data.num_workers", 4),
        pin_memory=config.get("data.pin_memory", True),
        distributed=True,
        seed=args.seed
    )

    # 确定类别数量
    num_classes = 10  # 默认CIFAR10
    if hasattr(train_dataset, 'classes'):
        num_classes = len(train_dataset.classes)

    # 创建模型
    model = build_model(args.model, config, num_classes=num_classes)

    # 优化模型
    config = optimize_model_for_rtx4070(model, config)

    # 移动模型到设备
    model = model.to(device)

    # 封装为分布式模型
    model = get_data_parallel_model(model, use_ddp=True)

    # 创建损失函数和指标
    criterion, metrics_fn = create_criterion_and_metrics(config, num_classes)
    criterion = criterion.to(device)

    # 创建优化器和调度器
    optimizer = get_advanced_optimizer(
        model,
        optimizer_type=config.get("optimization.optimizer", "adamw"),
        lr=config.get("optimization.learning_rate", 1e-4),
        weight_decay=config.get("optimization.weight_decay", 0.01),
        use_8bit=config.get("optimization.use_8bit", False)
    )

    # 学习率调度器
    scheduler = get_advanced_scheduler(
        optimizer,
        scheduler_type=config.get("optimization.scheduler", "cosine_warmup"),
        num_warmup_steps=config.get("optimization.warmup_steps", 100),
        num_training_steps=len(train_loader) * config.get("training.epochs", 100)
    )

    # 创建实验管理器
    experiment_name = args.experiment_name or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment = ExperimentManager(
        base_dir=args.output_dir,
        experiment_name=experiment_name,
        config=config
    )

    # 记录超参数
    experiment.log_hyperparameters({
        "model_type": args.model,
        "num_classes": num_classes,
        "batch_size": config.get("model.batch_size", 64),
        "learning_rate": config.get("optimization.learning_rate", 1e-4),
        "weight_decay": config.get("optimization.weight_decay", 0.01),
        "optimizer": config.get("optimization.optimizer", "adamw"),
        "scheduler": config.get("optimization.scheduler", "cosine_warmup"),
        "epochs": config.get("training.epochs", 100),
    })

    # 加载检查点（如果有）
    if args.checkpoint and args.resume:
        experiment.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=args.checkpoint
        )

    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        experiment=experiment,
        device=device,
        metrics_fn=metrics_fn,
        local_rank=local_rank,
        distributed=True
    )

    # 清理分布式环境
    cleanup_distributed()


def main():
    """主函数入口点"""
    # 解析参数
    args = parse_args()

    # 加载配置
    config_obj = load_config(args.config)

    # 更新配置
    if args.epochs:
        config_obj.set("training.epochs", args.epochs)
    if args.batch_size:
        config_obj.set("model.batch_size", args.batch_size)
    if args.learning_rate:
        config_obj.set("optimization.learning_rate", args.learning_rate)
    if args.workers:
        config_obj.set("data.num_workers", args.workers)
    if args.optimization_level:
        config_obj.set("optimization_level", args.optimization_level)
    if args.mixed_precision:
        config_obj.set("optimization.mixed_precision", True)
    if args.gradient_checkpoint:
        config_obj.set("model.gradient_checkpointing", True)
    if args.cpu_offload:
        config_obj.set("optimization.cpu_offload", True)
    if args.compile:
        config_obj.set("optimization.compiler_enabled", True)
    if args.memory_efficient:
        config_obj.set("optimization.memory_efficient", True)
    if args.log_interval:
        config_obj.set("training.log_interval", args.log_interval)
    if args.save_interval:
        config_obj.set("training.save_interval", args.save_interval)
    if args.eval_interval:
        config_obj.set("training.eval_interval", args.eval_interval)

    # 合并配置
    config = config_obj.config

    # 检查是否启用分布式训练
    if args.distributed:
        # 检查CUDA可用性
        if not torch.cuda.is_available():
            raise RuntimeError("分布式训练需要CUDA支持")

        # 获取GPU数量
        world_size = torch.cuda.device_count()

        # 生成进程
        spawn_processes(
            train_distributed,
            world_size,
            'nccl',
            config=config,
            args=args
        )
    else:
        # 单机训练

        # 设置随机种子
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # 检查CUDA可用性
        if torch.cuda.is_available():
            device = torch.device("cuda:0")

            # 设置cuDNN
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device("cpu")
            print("警告: 未检测到CUDA支持，将使用CPU训练")

        # 获取数据集
        train_dataset, val_dataset = get_dataset(args.data_dir, config_obj)

        # 创建数据加载器
        train_loader, val_loader = create_optimized_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=config.get("model.batch_size", 64),
            num_workers=config.get("data.num_workers", 4),
            pin_memory=config.get("data.pin_memory", True),
            shuffle=True,
            seed=args.seed
        )

        # 确定类别数量
        num_classes = 10  # 默认CIFAR10
        if hasattr(train_dataset, 'classes'):
            num_classes = len(train_dataset.classes)

        # 创建模型
        model = build_model(args.model, config, num_classes=num_classes)

        # 优化模型
        config = optimize_model_for_rtx4070(model, config)

        # 移动模型到设备
        model = model.to(device)

        # 创建损失函数和指标
        criterion, metrics_fn = create_criterion_and_metrics(config, num_classes)
        criterion = criterion.to(device)

        # 创建优化器和调度器
        optimizer = get_advanced_optimizer(
            model,
            optimizer_type=config.get("optimization.optimizer", "adamw"),
            lr=config.get("optimization.learning_rate", 1e-4),
            weight_decay=config.get("optimization.weight_decay", 0.01),
            use_8bit=config.get("optimization.use_8bit", False)
        )

        # 学习率调度器
        scheduler = get_advanced_scheduler(
            optimizer,
            scheduler_type=config.get("optimization.scheduler", "cosine_warmup"),
            num_warmup_steps=config.get("optimization.warmup_steps", 100),
            num_training_steps=len(train_loader) * config.get("training.epochs", 100)
        )

        # 创建实验管理器
        experiment_name = args.experiment_name or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment = ExperimentManager(
            base_dir=args.output_dir,
            experiment_name=experiment_name,
            config=config
        )

        # 记录超参数
        experiment.log_hyperparameters({
            "model_type": args.model,
            "num_classes": num_classes,
            "batch_size": config.get("model.batch_size", 64),
            "learning_rate": config.get("optimization.learning_rate", 1e-4),
            "weight_decay": config.get("optimization.weight_decay", 0.01),
            "optimizer": config.get("optimization.optimizer", "adamw"),
            "scheduler": config.get("optimization.scheduler", "cosine_warmup"),
            "epochs": config.get("training.epochs", 100),
        })

        # 加载检查点（如果有）
        if args.checkpoint and args.resume:
            experiment.load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                path=args.checkpoint
            )

        # 训练模型
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            experiment=experiment,
            device=device,
            metrics_fn=metrics_fn
        )


if __name__ == "__main__":
    main()