#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union


class Config:
    """配置管理类，加载和管理训练配置"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        参数:
            config_path: 配置文件路径，如果为None则使用默认RTX 4070配置
        """
        self.config = {}

        # 如果未指定配置文件，使用默认RTX 4070配置
        if config_path is None:
            # 获取项目根目录
            root_dir = Path(__file__).parent.parent.parent.absolute()
            config_path = os.path.join(root_dir, "config", "rtx4070_config.yaml")

        # 加载配置文件
        self.load_config(config_path)

        # 自动检测硬件并优化配置
        self.optimize_for_hardware()

    def load_config(self, config_path: str) -> None:
        """
        从YAML文件加载配置

        参数:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print(f"成功加载配置: {config_path}")
        except Exception as e:
            print(f"加载配置文件出错: {e}")
            # 使用默认配置
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "project_name": "aixunliancang",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "output_dir": "outputs",

            "hardware": {
                "gpu_name": "RTX 4070" if torch.cuda.is_available() else "CPU",
                "vram_gb": 12,
                "cuda_cores": 5888,
                "tensor_cores": 184,
            },

            "model": {
                "batch_size": 128,
                "mixed_precision": True,
                "gradient_checkpointing": True,
                "max_sequence_length": 8192,
            },

            "optimization": {
                "optimizer": "adamw",
                "learning_rate": 2e-4,
                "weight_decay": 0.01,
                "scheduler": "cosine_with_warmup",
                "warmup_steps": 100,
                "gradient_accumulation_steps": 1,
                "mixed_precision_dtype": "float16",
                "compiler_enabled": True,
            },

            "training": {
                "epochs": 100,
                "early_stopping_patience": 10,
                "checkpoint_interval": 1000,
                "eval_interval": 500,
                "log_interval": 50,
                "gradient_clip": 1.0,
            },

            "data": {
                "num_workers": 8,
                "prefetch_factor": 2,
                "pin_memory": True,
                "persistent_workers": True,
            }
        }

    def optimize_for_hardware(self) -> None:
        """根据检测到的硬件自动优化配置"""
        if not torch.cuda.is_available():
            # CPU模式
            self.config["device"] = "cpu"
            self.config["model"]["batch_size"] = 32
            self.config["model"]["mixed_precision"] = False
            self.config["optimization"]["compiler_enabled"] = False
            self.config["data"]["num_workers"] = min(8, os.cpu_count() or 1)
            return

        # 获取GPU信息
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)

        # 更新硬件信息
        self.config["hardware"]["gpu_name"] = gpu_name
        self.config["hardware"]["vram_gb"] = round(vram_gb, 1)

        # 针对不同显存大小优化批次
        if vram_gb < 8:
            self.config["model"]["batch_size"] = 32
            self.config["model"]["gradient_checkpointing"] = True
        elif vram_gb < 16:
            self.config["model"]["batch_size"] = 64
            self.config["model"]["gradient_checkpointing"] = True
        else:
            self.config["model"]["batch_size"] = 128

        # 根据CUDA版本调整编译器
        cuda_version = torch.version.cuda
        if cuda_version:
            major_version = int(cuda_version.split('.')[0])
            if major_version < 11:
                self.config["optimization"]["compiler_enabled"] = False

        # 优化工作线程数
        self.config["data"]["num_workers"] = min(8, os.cpu_count() or 1)

        print(f"配置已针对 {gpu_name} ({round(vram_gb, 1)}GB VRAM) 优化")

    def save_config(self, config_path: str) -> None:
        """
        保存配置到YAML文件

        参数:
            config_path: 目标配置文件路径
        """
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            print(f"配置已保存至: {config_path}")
        except Exception as e:
            print(f"保存配置文件出错: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        keys = key.split('.')
        config = self.config

        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def __getitem__(self, key: str) -> Any:
        """允许使用字典语法访问配置"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """允许使用字典语法设置配置"""
        self.set(key, value)

    def __str__(self) -> str:
        """返回可打印的配置表示"""
        return yaml.dump(self.config, default_flow_style=False, allow_unicode=True)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    加载配置的便捷函数

    参数:
        config_path: 配置文件路径

    返回:
        Config对象
    """
    return Config(config_path)