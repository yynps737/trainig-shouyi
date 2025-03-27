#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.config.config import load_config
from src.utils.logger import get_logger
from src.trainer.trainer import Trainer
from src.trainer.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from src.utils.optimizers import get_optimizer_for_rtx4070
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.cnn_model import OptimizedCNN, FastTransformer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AiXunLianCang训练脚本")
    parser.add_argument("--config", type=str, default="config/rtx4070_config.yaml", help="配置文件路径")
    parser.add_argument("--model", type=str, default="cnn", help="模型类型 (cnn, transformer)")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="数据目录")
    parser.add_argument("--output-dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--checkpoint", type=str, help="加载检查点")
    parser.add_argument("--resume", action="store_true", help="从检查点恢复训练")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--batch-size", type=int, help="批次大小")
    parser.add_argument("--learning-rate", type=float, help="学习率")

    return parser.parse_args()


def get_model(model_type, config, num_classes=10):
    """获取模型"""
    if model_type == "cnn":
        model = OptimizedCNN(in_channels=3, num_classes=num_classes)
    elif model_type == "transformer":
        model = FastTransformer(vocab_size=30000, num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 返回模型
    return model


def get_loss_fn(config, task_type="classification"):
    """获取损失函数"""
    if task_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
    elif task_type == "regression":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")

    return loss_fn


def get_dataloaders(config, data_dir):
    """获取数据加载器"""
    # 获取变换
    train_transform = get_train_transforms(
        input_size=(224, 224),
        augmentation_level=config.get("data.augmentation_level", "medium")
    )

    val_transform = get_val_transforms(
        input_size=(224, 224)
    )

    # 加载CIFAR-10数据集 (示例)
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

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("model.batch_size", 64),
        shuffle=True,
        num_workers=config.get("data.num_workers", 4),
        pin_memory=config.get("data.pin_memory", True),
        prefetch_factor=config.get("data.prefetch_factor", 2),
        persistent_workers=config.get("data.persistent_workers", True) if config.get("data.num_workers",
                                                                                     4) > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("model.batch_size", 64) * 2,
        shuffle=False,
        num_workers=config.get("data.num_workers", 4),
        pin_memory=config.get("data.pin_memory", True),
        prefetch_factor=config.get("data.prefetch_factor", 2),
        persistent_workers=config.get("data.persistent_workers", True) if config.get("data.num_workers",
                                                                                     4) > 0 else False
    )

    return train_loader, val_loader


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 更新配置
    if args.output_dir:
        config.set("output_dir", args.output_dir)

    if args.epochs:
        config.set("training.epochs", args.epochs)

    if args.batch_size:
        config.set("model.batch_size", args.batch_size)

    if args.learning_rate:
        config.set("optimization.learning_rate", args.learning_rate)

    # 创建输出目录
    os.makedirs(config.get("output_dir"), exist_ok=True)

    # 初始化日志
    logger = get_logger(
        log_dir=os.path.join(config.get("output_dir"), "logs"),
        project_name=config.get("project_name", "aixunliancang")
    )

    # 记录配置
    logger.info(f"加载配置: {args.config}")
    logger.log_config(config.config)

    # 检查CUDA
    if torch.cuda.is_available():
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        # 设置TensorCore加速
        torch.backends.cuda.matmul.allow_tf32 = True
        # 设置cuDNN
        torch.backends.cudnn.benchmark = True

        device = "cuda"
    else:
        logger.info("CUDA不可用，使用CPU")
        device = "cpu"

    # 加载数据集
    logger.info("加载数据集...")
    train_loader, val_loader = get_dataloaders(config, args.data_dir)
    logger.info(f"训练集大小: {len(train_loader.dataset)}, 批次: {len(train_loader)}")
    logger.info(f"验证集大小: {len(val_loader.dataset)}, 批次: {len(val_loader)}")

    # 创建模型
    logger.info(f"创建模型: {args.model}")
    model = get_model(args.model, config, num_classes=10)  # CIFAR-10有10个类别

    # 获取损失函数
    loss_fn = get_loss_fn(config)

    # 获取优化器和调度器
    optimizer_config = {
        "learning_rate": config.get("optimization.learning_rate", 2e-4),
        "weight_decay": config.get("optimization.weight_decay", 0.01),
        "use_8bit": config.get("optimization.use_8bit", False),
        "scheduler": config.get("optimization.scheduler", "cosine_with_warmup"),
        "warmup_steps": config.get("optimization.warmup_steps", 100),
        "total_steps": len(train_loader) * config.get("training.epochs", 100)
    }

    optimizer, scheduler = get_optimizer_for_rtx4070(model, optimizer_config)

    # 创建回调函数
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.get("training.early_stopping_patience", 10)
        ),
        ModelCheckpoint(
            filepath=os.path.join(config.get("output_dir"), "models/best_model.pth"),
            monitor="val_loss",
            save_best_only=True
        ),
        LearningRateScheduler(scheduler, monitor="val_loss")
    ]

    # 创建训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        logger=logger,
        callbacks=callbacks,
        device=device
    )

    # 从检查点恢复
    if args.checkpoint and args.resume:
        logger.info(f"从检查点恢复: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # 训练模型
    logger.info("开始训练...")
    trainer.train(config.get("training.epochs", 100))

    # 保存最终模型
    final_model_path = os.path.join(config.get("output_dir"), "models/final_model.pth")
    trainer.save_checkpoint(final_model_path)
    logger.info(f"最终模型已保存: {final_model_path}")

    logger.info("训练完成!")


if __name__ == "__main__":
    main()