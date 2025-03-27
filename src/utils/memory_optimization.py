#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import torch
import numpy as np
import time
import functools
import inspect
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
import weakref
import contextlib
import threading
import math


class MemoryTracker:
    """
    内存和显存使用跟踪器
    监控、记录和分析训练过程中的内存和显存使用情况
    """

    def __init__(self, log_interval=10, enabled=True):
        """
        初始化内存跟踪器

        参数:
            log_interval: 日志记录间隔（步数）
            enabled: 是否启用
        """
        self.log_interval = log_interval
        self.enabled = enabled
        self.current_step = 0

        # 记录数据
        self.cpu_usage = []
        self.gpu_usage = []
        self.gpu_utilization = []
        self.timestamp = []

        # 峰值使用量
        self.peak_cpu = 0
        self.peak_gpu = 0

        # 模型大小
        self.model_size = 0

        # 记录线程
        self.monitoring_thread = None
        self.stop_event = threading.Event()

    def start(self, model=None):
        """
        开始跟踪

        参数:
            model: 要分析的模型
        """
        if not self.enabled:
            return

        # 重置状态
        self._reset()

        # 记录模型大小
        if model is not None:
            # 记录模型大小（参数+缓冲区）
            if isinstance(model, torch.nn.Module):
                self.model_size = sum(
                    p.numel() * p.element_size() for p in model.parameters()
                ) + sum(
                    b.numel() * b.element_size() for b in model.buffers()
                )
                print(f"模型大小: {self.model_size / (1024 ** 2):.2f} MB")

        # 启动监控线程
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitor_thread, daemon=True)
        self.monitoring_thread.start()

        print("内存跟踪器已启动")

    def stop(self):
        """停止跟踪"""
        if not self.enabled or self.monitoring_thread is None:
            return

        # 停止监控线程
        self.stop_event.set()
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)

        print("内存跟踪器已停止")

        # 打印摘要
        self._print_summary()

    def step(self):
        """记录一个步骤"""
        if not self.enabled:
            return

        self.current_step += 1

        # 定期记录内存使用情况
        if self.current_step % self.log_interval == 0:
            self._log_memory_usage()

    def _reset(self):
        """重置跟踪器状态"""
        self.current_step = 0
        self.cpu_usage = []
        self.gpu_usage = []
        self.gpu_utilization = []
        self.timestamp = []
        self.peak_cpu = 0
        self.peak_gpu = 0

    def _log_memory_usage(self):
        """记录当前内存使用情况"""
        # 获取CPU内存
        import psutil
        cpu_used = psutil.Process(os.getpid()).memory_info().rss

        # 获取GPU内存（如果可用）
        gpu_used = 0
        gpu_util = 0

        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_allocated()
            if hasattr(torch.cuda, 'utilization'):
                try:
                    gpu_util = torch.cuda.utilization()
                except:
                    # 如果utilization不可用，尝试使用pynvml
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = util.gpu
                    except:
                        pass

        # 更新峰值
        self.peak_cpu = max(self.peak_cpu, cpu_used)
        self.peak_gpu = max(self.peak_gpu, gpu_used)

        # 记录数据
        self.cpu_usage.append(cpu_used)
        self.gpu_usage.append(gpu_used)
        self.gpu_utilization.append(gpu_util)
        self.timestamp.append(time.time())

    def _monitor_thread(self):
        """监控线程函数"""
        # 每秒记录一次内存使用情况
        while not self.stop_event.is_set():
            self._log_memory_usage()
            time.sleep(1.0)

    def _print_summary(self):
        """打印使用摘要"""
        if not self.cpu_usage:
            print("没有收集到内存使用数据")
            return

        # 计算平均值
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage)

        print("\n" + "=" * 50)
        print("内存跟踪器摘要")
        print("=" * 50)

        # CPU内存
        print(f"CPU内存:")
        print(f"  平均使用: {avg_cpu / (1024 ** 2):.2f} MB")
        print(f"  峰值使用: {self.peak_cpu / (1024 ** 2):.2f} MB")

        # GPU内存
        if torch.cuda.is_available():
            avg_gpu = sum(self.gpu_usage) / len(self.gpu_usage)
            avg_util = sum(self.gpu_utilization) / len(self.gpu_utilization) if self.gpu_utilization else 0

            print(f"GPU内存 (设备: {torch.cuda.get_device_name(0)}):")
            print(f"  平均使用: {avg_gpu / (1024 ** 2):.2f} MB")
            print(f"  峰值使用: {self.peak_gpu / (1024 ** 2):.2f} MB")
            print(f"  总显存: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")

            if self.gpu_utilization:
                print(f"GPU利用率:")
                print(f"  平均利用率: {avg_util:.2f}%")

        # 效率分析
        if self.model_size > 0 and torch.cuda.is_available():
            memory_efficiency = self.model_size / self.peak_gpu
            print(f"内存效率:")
            print(f"  模型大小/峰值显存: {memory_efficiency:.4f} "
                  f"({'高' if memory_efficiency > 0.5 else '中' if memory_efficiency > 0.3 else '低'})")

        print("=" * 50)

    def plot(self, filename=None):
        """
        绘制内存使用情况图表

        参数:
            filename: 保存文件名（如果为None，则显示图表）
        """
        if not self.enabled or not self.cpu_usage:
            print("没有数据可绘制")
            return

        try:
            import matplotlib.pyplot as plt

            # 创建相对时间轴（从0开始）
            t0 = self.timestamp[0]
            rel_time = [(t - t0) for t in self.timestamp]

            # 转换为MB
            cpu_mb = [x / (1024 ** 2) for x in self.cpu_usage]
            gpu_mb = [x / (1024 ** 2) for x in self.gpu_usage]

            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # CPU内存
            ax1.plot(rel_time, cpu_mb, label='CPU内存')
            ax1.set_xlabel('时间 (秒)')
            ax1.set_ylabel('内存使用 (MB)')
            ax1.set_title('CPU内存使用')
            ax1.grid(True)

            # GPU内存和利用率
            if torch.cuda.is_available():
                ax2.plot(rel_time, gpu_mb, label='GPU内存')
                ax2.set_xlabel('时间 (秒)')
                ax2.set_ylabel('内存使用 (MB)')
                ax2.set_title(f'GPU内存使用 ({torch.cuda.get_device_name(0)})')
                ax2.grid(True)

                if self.gpu_utilization:
                    # 创建次坐标轴
                    ax3 = ax2.twinx()
                    ax3.plot(rel_time, self.gpu_utilization, 'r--', label='GPU利用率')
                    ax3.set_ylabel('利用率 (%)')

                    # 两个图例
                    lines1, labels1 = ax2.get_legend_handles_labels()
                    lines2, labels2 = ax3.get_legend_handles_labels()
                    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            plt.tight_layout()

            # 保存或显示
            if filename:
                plt.savefig(filename, dpi=300)
                print(f"图表已保存至 {filename}")
            else:
                plt.show()

        except ImportError:
            print("绘图需要matplotlib，请安装: pip install matplotlib")


def get_model_memory_profile(model, input_size, batch_size=1, device=None):
    """
    获取模型内存使用分析

    参数:
        model: 要分析的模型
        input_size: 输入大小（不包括批次维度）
        batch_size: 批次大小
        device: 运行设备

    返回:
        内存分析报告字典
    """
    if device is None:
        device = next(model.parameters()).device

    # 确保模型在评估模式
    model.eval()

    # 创建示例输入
    if isinstance(input_size, int):
        x = torch.rand(batch_size, input_size, device=device)
    elif isinstance(input_size, (list, tuple)):
        if len(input_size) == 1:
            x = torch.rand(batch_size, input_size[0], device=device)
        elif len(input_size) == 3:  # 假设为图像: C, H, W
            x = torch.rand(batch_size, *input_size, device=device)
        else:
            x = torch.rand(batch_size, *input_size, device=device)
    else:
        raise ValueError(f"不支持的input_size格式: {input_size}")

    # 清理CUDA缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 收集模型参数内存
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    # 前向传播内存分析
    if device.type == 'cuda':
        # 记录开始状态
        mem_before = torch.cuda.memory_allocated()

        # 运行前向传播
        with torch.no_grad():
            _ = model(x)

        # 强制完成所有CUDA操作
        torch.cuda.synchronize()

        # 记录结束状态
        mem_after = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()

        # 计算前向传播激活内存
        activation_memory = mem_after - mem_before

        # 假设每个参数的梯度与参数相同大小
        grad_memory = param_size

        # 优化器状态（假设Adam，每个参数2个状态）
        optimizer_memory = param_size * 2

        # 创建报告
        report = {
            'Parameter Memory (MB)': param_size / (1024 ** 2),
            'Buffer Memory (MB)': buffer_size / (1024 ** 2),
            'Activation Memory (MB)': activation_memory / (1024 ** 2),
            'Gradient Memory (MB)': grad_memory / (1024 ** 2),
            'Optimizer Memory (MB)': optimizer_memory / (1024 ** 2),
            'Total Memory (MB)': (param_size + buffer_size + activation_memory + grad_memory + optimizer_memory) / (
                        1024 ** 2),
            'Peak Memory (MB)': peak_mem / (1024 ** 2),
        }
    else:
        # CPU模式下只报告参数内存
        report = {
            'Parameter Memory (MB)': param_size / (1024 ** 2),
            'Buffer Memory (MB)': buffer_size / (1024 ** 2),
            'Total Parameter Memory (MB)': (param_size + buffer_size) / (1024 ** 2),
        }

    return report


@contextlib.contextmanager
def use_gradient_checkpointing(model, enabled=True):
    """
    临时启用/禁用梯度检查点

    参数:
        model: 模型
        enabled: 是否启用
    """
    # 保存原始状态
    modules = [m for m in model.modules() if hasattr(m, 'gradient_checkpointing')]
    original_states = [getattr(m, 'gradient_checkpointing') for m in modules]

    try:
        # 设置新状态
        for m in modules:
            if hasattr(m, 'gradient_checkpointing_enable') and callable(m.gradient_checkpointing_enable):
                if enabled:
                    m.gradient_checkpointing_enable()
                else:
                    setattr(m, 'gradient_checkpointing', False)
            else:
                setattr(m, 'gradient_checkpointing', enabled)

        yield
    finally:
        # 恢复原始状态
        for m, state in zip(modules, original_states):
            setattr(m, 'gradient_checkpointing', state)


class MemoryOptimizer:
    """
    内存优化器
    提供自动化的内存和显存优化策略
    """

    def __init__(self, model, max_vram_gb=12.0, config=None):
        """
        初始化内存优化器

        参数:
            model: 要优化的模型
            max_vram_gb: 最大可用显存（GB）
            config: 配置字典
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.max_vram_gb = max_vram_gb
        self.config = config or {}

        # 初始化优化标志
        self.applied_optimizations = {
            'gradient_checkpointing': False,
            'mixed_precision': False,
            'activation_checkpointing': False,
            'memory_efficient_attention': False,
            'cpu_offload': False,
            'weight_quantization': False,
        }

        # 检测GPU特性
        self.gpu_features = self._detect_gpu_features()

    def _detect_gpu_features(self):
        """检测GPU特性"""
        features = {
            'tensor_cores': False,
            'fp16_support': False,
            'bf16_support': False,
            'compute_capability': None,
        }

        if not torch.cuda.is_available():
            return features

        # 获取设备功能
        device_props = torch.cuda.get_device_properties(0)
        compute_major, compute_minor = device_props.major, device_props.minor
        features['compute_capability'] = (compute_major, compute_minor)

        # 检测特性
        if compute_major >= 7:  # Volta及更高架构
            features['tensor_cores'] = True
            features['fp16_support'] = True

        if compute_major >= 8:  # Ampere及更高架构
            features['bf16_support'] = True

        return features

    def optimize(self, input_size=None, batch_size=32, profile_mode='auto'):
        """
        优化模型内存使用

        参数:
            input_size: 输入大小
            batch_size: 批次大小
            profile_mode: 分析模式

        返回:
            优化报告
        """
        print("=" * 60)
        print("进行内存优化...")
        print("=" * 60)

        # 分析内存使用
        if input_size is not None and profile_mode in ('auto', 'profile'):
            print(f"分析模型内存使用 (批次大小: {batch_size})...")
            baseline_profile = get_model_memory_profile(self.model, input_size, batch_size, self.device)

            # 打印基准分析结果
            print("\n基准内存使用:")
            for k, v in baseline_profile.items():
                print(f"  {k}: {v:.2f} MB")

            # 预测训练时内存使用
            total_mem = baseline_profile.get('Total Memory (MB)', 0)
            peak_mem = baseline_profile.get('Peak Memory (MB)', 0)

            print(f"\n预测内存需求:")
            print(f"  单次前向+反向传播: {total_mem:.2f} MB")
            print(f"  峰值内存使用: {peak_mem:.2f} MB")

            # 检查显存限制
            vram_mb = self.max_vram_gb * 1024
            if peak_mem > vram_mb * 0.9:
                print(f"\n警告: 预测峰值内存 ({peak_mem:.2f} MB) 接近或超过可用显存 ({vram_mb:.2f} MB)")
                print("将应用内存优化...")

                # 确定需要的优化级别
                mem_ratio = peak_mem / vram_mb
                if mem_ratio > 1.5:
                    # 严重超出，应用所有优化
                    self._apply_all_optimizations()
                elif mem_ratio > 1.2:
                    # 中度超出，应用中度优化
                    self._apply_medium_optimizations()
                elif mem_ratio > 0.9:
                    # 轻度超出，应用轻度优化
                    self._apply_light_optimizations()
            else:
                print(f"\n内存使用在可接受范围内，仅应用性能优化")
                self._apply_performance_optimizations()
        else:
            # 根据配置应用优化
            if self.config.get('optimization_level', 'auto') == 'max':
                self._apply_all_optimizations()
            elif self.config.get('optimization_level', 'auto') == 'medium':
                self._apply_medium_optimizations()
            elif self.config.get('optimization_level', 'auto') == 'light':
                self._apply_light_optimizations()
            else:
                # 自动模式但没有分析，应用中等优化
                self._apply_medium_optimizations()

        # 创建优化报告
        report = {
            'applied_optimizations': self.applied_optimizations,
            'gpu_features': self.gpu_features,
            'recommended_batch_size': self._recommend_batch_size(batch_size),
        }

        # 打印报告
        print("\n应用的优化:")
        for opt, applied in self.applied_optimizations.items():
            print(f"  {opt}: {'已启用' if applied else '未启用'}")

        print(f"\n推荐批次大小: {report['recommended_batch_size']}")

        print("\n优化完成!")
        print("=" * 60)

        return report

    def _apply_light_optimizations(self):
        """应用轻度内存优化"""
        # 启用混合精度（如果支持）
        if self.gpu_features['fp16_support'] or self.gpu_features['bf16_support']:
            self._enable_mixed_precision()
            self.applied_optimizations['mixed_precision'] = True

        # 启用内存高效的注意力（如果是Transformer类模型）
        if self._is_transformer_model():
            self._enable_memory_efficient_attention()
            self.applied_optimizations['memory_efficient_attention'] = True

    def _apply_medium_optimizations(self):
        """应用中度内存优化"""
        # 应用轻度优化
        self._apply_light_optimizations()

        # 启用梯度检查点
        self._enable_gradient_checkpointing()
        self.applied_optimizations['gradient_checkpointing'] = True

        # 如果是大模型，启用激活检查点
        if self._is_large_model():
            self._enable_activation_checkpointing()
            self.applied_optimizations['activation_checkpointing'] = True

    def _apply_all_optimizations(self):
        """应用所有内存优化"""
        # 应用中度优化
        self._apply_medium_optimizations()

        # 启用CPU卸载
        self._enable_cpu_offload()
        self.applied_optimizations['cpu_offload'] = True

        # 尝试权重量化
        if self._can_quantize():
            self._enable_weight_quantization()
            self.applied_optimizations['weight_quantization'] = True

    def _apply_performance_optimizations(self):
        """应用性能优化，但不牺牲太多内存"""
        # 启用混合精度（如果支持）
        if self.gpu_features['fp16_support'] or self.gpu_features['bf16_support']:
            self._enable_mixed_precision()
            self.applied_optimizations['mixed_precision'] = True

        # 为Tensor Core优化参数尺寸
        if self.gpu_features['tensor_cores']:
            self._optimize_for_tensor_cores()

    def _enable_mixed_precision(self):
        """启用混合精度训练"""
        print("启用混合精度训练...")

        # 判断使用哪种精度
        if self.gpu_features['bf16_support'] and self.config.get('precision', 'auto') != 'fp16':
            # 如果支持BF16并且未指定使用FP16，则使用BF16
            print("  使用bfloat16精度")
            dtype = torch.bfloat16
        else:
            # 否则使用FP16
            print("  使用float16精度")
            dtype = torch.float16

        # 保存到配置
        if 'amp' not in self.config:
            self.config['amp'] = {}

        self.config['amp']['enabled'] = True
        self.config['amp']['dtype'] = str(dtype).split('.')[-1]

        # 这里只设置配置，实际的混合精度实现需要在训练循环中应用

    def _enable_gradient_checkpointing(self):
        """启用梯度检查点"""
        print("启用梯度检查点...")

        # 检查模型是否支持梯度检查点
        supports_checkpointing = False

        # 检查通用模型属性
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            supports_checkpointing = True
        elif hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
            supports_checkpointing = True

        # 检查各个子模块
        checkpointing_set = False
        for module in self.model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
                checkpointing_set = True
            elif hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
                checkpointing_set = True

        if checkpointing_set:
            supports_checkpointing = True

        if not supports_checkpointing:
            print("  警告: 模型可能不支持梯度检查点")

        # 保存到配置
        self.config['gradient_checkpointing'] = True

    def _enable_memory_efficient_attention(self):
        """启用内存高效的注意力机制"""
        print("启用内存高效的注意力机制...")

        # 检查是否安装了优化库
        try:
            import flash_attn
            has_flash_attention = True
        except ImportError:
            has_flash_attention = False

        if has_flash_attention:
            print("  检测到Flash Attention，将启用")
        else:
            print("  未检测到Flash Attention，将使用PyTorch内置的内存高效注意力")

        # 查找注意力模块
        attention_modules_found = 0
        for name, module in self.model.named_modules():
            # 检查常见的注意力模块命名模式
            if any(pattern in name.lower() for pattern in ['attention', 'attn']):
                if hasattr(module, 'use_flash_attention'):
                    module.use_flash_attention = has_flash_attention
                    attention_modules_found += 1
                elif hasattr(module, 'use_memory_efficient_attention'):
                    module.use_memory_efficient_attention = True
                    attention_modules_found += 1
                # 对于不支持直接配置的模块，将在配置中标记

        if attention_modules_found == 0:
            print("  警告: 未找到可配置的注意力模块")
        else:
            print(f"  已配置 {attention_modules_found} 个注意力模块")

        # 保存到配置
        self.config['memory_efficient_attention'] = True
        self.config['flash_attention'] = has_flash_attention

    def _enable_activation_checkpointing(self):
        """启用激活检查点"""
        print("启用激活检查点...")

        # 查找支持激活检查点的模块
        checkpointed_modules = 0
        for name, module in self.model.named_modules():
            # 检查是否是可以应用激活检查点的模块类型
            if any(pattern in module.__class__.__name__ for pattern in ['Block', 'Layer', 'Encoder', 'Decoder']):
                # 尝试应用torch.utils.checkpoint
                if hasattr(module, 'forward') and callable(module.forward):
                    # 保存原始前向传播函数
                    if not hasattr(module, '_original_forward'):
                        module._original_forward = module.forward

                    # 替换为检查点版本
                    module.forward = functools.partial(
                        torch.utils.checkpoint.checkpoint,
                        module._original_forward,
                        use_reentrant=False
                    )

                    checkpointed_modules += 1

        if checkpointed_modules == 0:
            print("  警告: 未找到适合激活检查点的模块")
        else:
            print(f"  已对 {checkpointed_modules} 个模块应用激活检查点")

        # 保存到配置
        self.config['activation_checkpointing'] = True

    def _enable_cpu_offload(self):
        """启用CPU卸载"""
        print("启用CPU卸载...")

        # CPU卸载通常需要在优化器中实现
        # 这里只设置配置，实际实现在创建优化器时应用

        # 保存到配置
        self.config['cpu_offload'] = True
        print("  注意: CPU卸载需要在创建优化器时应用")

    def _enable_weight_quantization(self):
        """启用权重量化"""
        print("启用权重量化...")

        # 权重量化通常需要专门的库和额外步骤
        # 这里只设置配置，实际实现通常需要在模型加载时应用

        # 检查是否安装了量化库
        try:
            import bitsandbytes as bnb
            has_bnb = True
        except ImportError:
            has_bnb = False

        if has_bnb:
            print("  检测到bitsandbytes，将使用8位量化")
            # 量化由专门的API完成，通常在模型创建时完成
        else:
            print("  未检测到bitsandbytes，将使用PyTorch原生量化")

            # 创建量化配置
            qconfig = torch.quantization.get_default_qconfig('fbgemm')
            self.model.qconfig = qconfig

            # 准备量化
            torch.quantization.prepare(self.model, inplace=True)
            # 注意：在实际应用中，需要用实际数据校准和完成量化

        # 保存到配置
        self.config['weight_quantization'] = True
        self.config['quantization_bits'] = 8  # 默认8位量化

    def _optimize_for_tensor_cores(self):
        """为Tensor Core优化模型"""
        print("优化模型以利用Tensor Cores...")

        # 启用TensorFloat32精度
        if hasattr(torch.cuda, 'matmul') and hasattr(torch.cuda.matmul, 'allow_tf32'):
            torch.cuda.matmul.allow_tf32 = True
            print("  已启用TensorFloat32用于矩阵乘法")

        if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
            print("  已启用TensorFloat32用于cuDNN")

        # 检查模型参数尺寸是否为8的倍数
        non_optimal_modules = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                # 检查输入和输出维度
                if isinstance(module, torch.nn.Linear):
                    in_features, out_features = module.in_features, module.out_features
                else:  # Conv2d
                    in_features, out_features = module.in_channels, module.out_channels

                # 对于Tensor Core，最佳性能需要维度是8的倍数
                if in_features % 8 != 0 or out_features % 8 != 0:
                    non_optimal_modules += 1

        if non_optimal_modules > 0:
            print(f"  警告: 发现 {non_optimal_modules} 个模块的维度不是8的倍数，可能无法充分利用Tensor Cores")
            print("  考虑调整模型架构，使维度为8的倍数以获得最佳性能")
        else:
            print("  所有模块维度已优化，可充分利用Tensor Cores")

        # 保存到配置
        self.config['tensor_core_optimized'] = True

    def _is_transformer_model(self):
        """检查是否是Transformer类型的模型"""
        # 检查模型类名
        if 'transformer' in self.model.__class__.__name__.lower():
            return True

        # 检查是否包含典型的Transformer组件
        for module in self.model.modules():
            if 'attention' in module.__class__.__name__.lower():
                return True

        return False

    def _is_large_model(self):
        """检查是否是大模型"""
        # 计算参数总量
        param_count = sum(p.numel() for p in self.model.parameters())

        # 超过1亿参数视为大模型
        return param_count > 100_000_000

    def _can_quantize(self):
        """检查是否可以量化"""
        # 检查是否安装了量化库
        try:
            import bitsandbytes
            return True
        except ImportError:
            pass

        # 检查PyTorch是否支持量化
        return hasattr(torch, 'quantization')

    def _recommend_batch_size(self, current_batch_size):
        """推荐批次大小"""
        # 根据优化情况调整推荐批次大小
        if self.applied_optimizations['weight_quantization'] and self.applied_optimizations['cpu_offload']:
            # 激进优化，批次大小可以保持不变
            return current_batch_size

        if self.applied_optimizations['gradient_checkpointing'] and self.applied_optimizations['mixed_precision']:
            # 中度优化，批次大小可以保持或略微减小
            return current_batch_size

        if self.applied_optimizations['mixed_precision']:
            # 轻度优化，可以尝试增加批次大小
            return current_batch_size * 2

        # 无优化或轻度优化情况
        # 估算最佳批次大小
        if self.device.type == 'cuda':
            vram_mb = self.max_vram_gb * 1024

            # 假设每个样本使用大约X MB内存
            # 这是一个粗略估计，实际使用需要根据具体模型调整
            sample_mb = self._estimate_sample_memory(current_batch_size)

            # 留出30%显存给系统和梯度
            usable_mb = vram_mb * 0.7

            # 计算可能的批次大小
            possible_batch = int(usable_mb / sample_mb) if sample_mb > 0 else current_batch_size

            # 确保批次大小是8的倍数（对Tensor Core友好）
            possible_batch = (possible_batch // 8) * 8

            # 不要推荐太小的批次
            recommended = max(8, min(possible_batch, current_batch_size * 2))

            return recommended

        # CPU训练，返回当前批次大小
        return current_batch_size

    def _estimate_sample_memory(self, batch_size):
        """估算每个样本的内存使用"""
        if self.device.type != 'cuda':
            return 0

        # 尝试使用之前的分析结果
        if hasattr(self, '_sample_memory'):
            return self._sample_memory

        # 尝试估算
        try:
            # 清空缓存
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 记录初始内存
            start_mem = torch.cuda.memory_allocated()

            # 运行模型（如果有input_size）
            if hasattr(self, '_input_size') and hasattr(self, '_batch_size'):
                with torch.no_grad():
                    _ = self.model(torch.rand(batch_size, *self._input_size, device=self.device))

                # 清空缓存
                torch.cuda.empty_cache()

                # 记录内存增加
                end_mem = torch.cuda.memory_allocated()
                mem_used = end_mem - start_mem

                # 估算每个样本的内存
                sample_mem = mem_used / batch_size

                # 保存结果
                self._sample_memory = sample_mem

                return sample_mem
        except Exception as e:
            print(f"内存估算出错: {e}")

        # 如果无法估算，使用默认值
        # 假设RTX 4070的12GB显存，默认每样本50MB
        return 50  # MB


@contextlib.contextmanager
def mixed_precision_context(enabled=True, dtype='float16'):
    """
    混合精度上下文管理器

    参数:
        enabled: 是否启用
        dtype: 精度类型 ('float16', 'bfloat16')
    """
    if not enabled:
        # 如果未启用，直接运行代码
        yield
        return

    # 确定数据类型
    if dtype == 'bfloat16' and torch.cuda.is_available() and hasattr(torch, 'bfloat16'):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    # 使用自动混合精度
    with torch.cuda.amp.autocast(enabled=enabled, dtype=amp_dtype):
        yield


def profile_memory_usage(func):
    """
    内存使用分析装饰器

    参数:
        func: 要分析的函数

    返回:
        包装后的函数
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 收集初始内存状态
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated()
            peak_start = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()

        # 执行函数
        result = func(*args, **kwargs)

        # 收集最终内存状态
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_mem = torch.cuda.memory_allocated()
            peak_end = torch.cuda.max_memory_allocated()

            # 打印内存统计信息
            print(f"\n--- 内存使用情况分析: {func.__name__} ---")
            print(f"  使用内存: {(end_mem - start_mem) / (1024 ** 2):.2f} MB")
            print(f"  峰值内存: {peak_end / (1024 ** 2):.2f} MB")
            print(f"  净增内存: {(end_mem - start_mem) / (1024 ** 2):.2f} MB")
            print(f"-----------------------------------")

        return result

    return wrapper


@contextlib.contextmanager
def temporary_parameter_copy(module, use_copy=False):
    """
    临时参数复制上下文，用于权重共享

    参数:
        module: 模块
        use_copy: 是否使用复制模式
    """
    if not use_copy:
        # 不使用复制，直接运行代码
        yield
        return

    # 保存原始参数
    original_params = {}
    for name, param in module.named_parameters():
        original_params[name] = param.data.clone()

    try:
        # 运行代码
        yield
    finally:
        # 恢复原始参数
        with torch.no_grad():
            for name, param in module.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])


class MemoryEfficientSwapLinear(torch.nn.Linear):
    """
    内存高效的线性层，支持权重交换
    当不需要时将权重卸载到CPU
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

        # 卸载标志
        self.weight_offloaded = False
        self.bias_offloaded = False

        # 权重副本
        self.weight_cpu = None
        self.bias_cpu = None

    def offload(self):
        """将权重卸载到CPU"""
        if self.weight_offloaded:
            return

        # 保存CPU副本
        self.weight_cpu = self.weight.data.cpu()

        if self.bias is not None:
            self.bias_cpu = self.bias.data.cpu()

        # 释放GPU内存
        self.weight.data = torch.empty(0, device=self.weight.device)

        if self.bias is not None:
            self.bias.data = torch.empty(0, device=self.bias.device)

        # 设置标志
        self.weight_offloaded = True
        self.bias_offloaded = True

    def reload(self):
        """将权重加载回GPU"""
        if not self.weight_offloaded:
            return

        # 恢复权重
        self.weight.data = self.weight_cpu.to(self.weight.device)

        if self.bias is not None and self.bias_offloaded:
            self.bias.data = self.bias_cpu.to(self.bias.device)

        # 清除CPU副本释放内存
        self.weight_cpu = None
        self.bias_cpu = None

        # 设置标志
        self.weight_offloaded = False
        self.bias_offloaded = False

    def forward(self, input):
        """前向传播"""
        # 如果权重已卸载，则自动加载
        if self.weight_offloaded:
            self.reload()

        # 调用原始前向传播
        result = super().forward(input)

        return result


if __name__ == "__main__":
    # 测试内存优化

    # 创建一个简单模型
    class TestModel(torch.nn.Module):
        def __init__(self, size=1000):
            super().__init__()
            # 故意创建一些大的参数
            self.fc1 = torch.nn.Linear(size, size)
            self.fc2 = torch.nn.Linear(size, size)
            self.fc3 = torch.nn.Linear(size, size)
            self.fc4 = torch.nn.Linear(size, size)
            self.fc5 = torch.nn.Linear(size, size)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = self.fc5(x)
            return x


    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"测试内存优化 (GPU: {torch.cuda.get_device_name(0)})")
        device = 'cuda'
    else:
        print("CUDA不可用，使用CPU进行测试")
        device = 'cpu'

    # 创建模型
    model = TestModel(1000).to(device)

    # 测试内存分析
    profile = get_model_memory_profile(model, 1000, batch_size=16, device=device)
    print("\n内存分析结果:")
    for k, v in profile.items():
        print(f"  {k}: {v:.2f} MB")

    # 测试内存优化器
    optimizer = MemoryOptimizer(model, max_vram_gb=12.0)
    optimizer.optimize(input_size=1000, batch_size=32)

    # 测试混合精度上下文
    x = torch.randn(16, 1000, device=device)

    with mixed_precision_context(enabled=True):
        y = model(x)
        print(f"混合精度输出数据类型: {y.dtype}")

    # 测试梯度检查点
    with use_gradient_checkpointing(model, enabled=True):
        y = model(x)
        print("已使用梯度检查点进行前向传播")