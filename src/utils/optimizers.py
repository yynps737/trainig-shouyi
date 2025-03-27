#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from typing import List, Optional, Callable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR

class AdamW8bit(Optimizer):
    """8位量化版AdamW优化器，节省显存"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                      weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        
        # 检查是否支持8位优化
        self.supports_8bit = hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'custom_fwd')
        if not self.supports_8bit:
            print("警告: 8位优化不可用，将使用标准AdamW")

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 参数更新
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW8bit不支持稀疏梯度')

                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    # 8位量化指数移动平均
                    if self.supports_8bit:
                        state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.uint8)
                        state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.uint8)
                        state['scale_exp_avg'] = torch.tensor(1.0, device=p.device)
                        state['scale_exp_avg_sq'] = torch.tensor(1.0, device=p.device)
                    else:
                        # 回退到32位
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                # 获取超参数
                beta1, beta2 = group['betas']
                state['step'] += 1

                # 执行带权重衰减的更新
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # 8位量化实现
                if self.supports_8bit:
                    # 反量化
                    exp_avg = state['exp_avg'].float() * state['scale_exp_avg']
                    exp_avg_sq = state['exp_avg_sq'].float() * state['scale_exp_avg_sq']
                    
                    # 动量更新
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    # 重新量化
                    max_val_avg = torch.max(torch.abs(exp_avg)).item()
                    max_val_sq = torch.max(torch.abs(exp_avg_sq)).item()
                    
                    state['scale_exp_avg'] = max_val_avg / 127.0
                    state['scale_exp_avg_sq'] = max_val_sq / 127.0
                    
                    state['exp_avg'] = torch.round(exp_avg / state['scale_exp_avg']).to(torch.int8)
                    state['exp_avg_sq'] = torch.round(exp_avg_sq / state['scale_exp_avg_sq']).to(torch.int8)
                    
                    # 计算偏差修正
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    # 应用更新
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                    
                    # 反量化用于更新
                    exp_avg = state['exp_avg'].float() * state['scale_exp_avg']
                    exp_avg_sq = state['exp_avg_sq'].float() * state['scale_exp_avg_sq']
                    
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    
                else:
                    # 标准AdamW实现
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    
                    # 更新动量
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    if group['amsgrad']:
                        torch.maximum(state['max_exp_avg_sq'], exp_avg_sq, out=state['max_exp_avg_sq'])
                        denom = state['max_exp_avg_sq'].sqrt().add_(group['eps'])
                    else:
                        denom = exp_avg_sq.sqrt().add_(group['eps'])
                    
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, 
    num_cycles: float = 0.5, last_epoch: int = -1
) -> LambdaLR:
    """
    创建余弦学习率调度器，包含预热期
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_gpu_optimized_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    use_8bit: bool = False
) -> Optimizer:
    """
    获取针对GPU优化的优化器
    
    参数:
        model: 模型
        lr: 学习率
        weight_decay: 权重衰减率
        use_8bit: 是否使用8位优化
    """
    # 区分没有权重衰减的参数
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # 选择优化器
    if use_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                optimizer_grouped_parameters,
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            print("使用8位AdamW优化器")
            return optimizer
        except ImportError:
            print("未找到bitsandbytes库，回退到自定义AdamW8bit")
            optimizer = AdamW8bit(
                optimizer_grouped_parameters,
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            return optimizer
    else:
        # 使用融合实现的AdamW
        if hasattr(torch.optim, 'AdamW') and hasattr(torch.optim.AdamW, 'fused') and torch.cuda.is_available():
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                fused=True  # 使用融合实现
            )
            print("使用融合AdamW优化器")
            return optimizer
        else:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            print("使用标准AdamW优化器")
            return optimizer

def get_optimizer_for_rtx4070(
    model: torch.nn.Module,
    config: dict
) -> tuple:
    """
    获取针对RTX 4070优化的优化器和调度器
    
    参数:
        model: 模型
        config: 配置字典
    
    返回:
        optimizer: 优化器
        scheduler: 学习率调度器
    """
    # 获取配置
    lr = config.get('learning_rate', 2e-4)
    weight_decay = config.get('weight_decay', 0.01)
    use_8bit = config.get('use_8bit', False)
    scheduler_type = config.get('scheduler', 'cosine_with_warmup')
    warmup_steps = config.get('warmup_steps', 100)
    total_steps = config.get('total_steps', 10000)
    
    # 创建优化器
    optimizer = get_gpu_optimized_optimizer(
        model, 
        lr=lr, 
        weight_decay=weight_decay, 
        use_8bit=use_8bit
    )
    
    # 创建学习率调度器
    if scheduler_type == 'cosine_with_warmup':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
    else:
        scheduler = None
    
    return optimizer, scheduler
