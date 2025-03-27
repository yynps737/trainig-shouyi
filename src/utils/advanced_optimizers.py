#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from typing import List, Dict, Optional, Union, Callable, Tuple, Any


class Lion(Optimizer):
    """
    实现Lion优化器 (Learning rate Instead Of Nestrov)
    参考论文: https://arxiv.org/abs/2302.06675

    这个优化器在一些情况下比Adam表现更好，尤其是在大模型训练中
    并且内存占用低于Adam(W)
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """
        初始化Lion优化器

        参数:
            params: 模型参数
            lr: 学习率
            betas: 动量系数 (默认: (0.9, 0.99))
            weight_decay: 权重衰减系数
        """
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # 梯度
                grad = p.grad

                # 权重衰减
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # 获取参数
                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                # 获取超参数
                lr = group['lr']
                beta1, beta2 = group['betas']

                # 更新动量
                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Lion更新规则
                update = exp_avg.sign()  # 仅使用梯度方向

                # 更新参数 (与SGD不同，Lion使用动量的符号，而不是直接使用梯度)
                p.add_(update, alpha=-lr)

        return loss


class AdaFactor(Optimizer):
    """
    实现AdaFactor优化器

    AdaFactor是一个内存高效的自适应优化器，特别适合大规模模型训练
    它减少了每个参数所需的内存，对大模型训练尤其有用
    """

    def __init__(self, params, lr=None, beta1=0.9, beta2=0.999, eps=1e-8,
                 clip_threshold=1.0, decay_rate=-0.8, scale_parameter=True,
                 relative_step=True, warmup_init=False, weight_decay=0.0):
        """
        初始化AdaFactor优化器

        参数:
            params: 模型参数
            lr: 学习率，如果None则使用相对步长
            beta1: 一阶动量系数
            beta2: 二阶动量系数
            eps: 数值稳定性参数
            clip_threshold: 梯度裁剪阈值
            decay_rate: 学习率调度衰减率
            scale_parameter: 是否缩放参数
            relative_step: 是否使用相对步长
            warmup_init: 是否使用预热初始化
            weight_decay: 权重衰减系数
        """
        if lr is not None and relative_step:
            raise ValueError("不能同时指定lr和relative_step=True")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
            weight_decay=weight_decay
        )
        super(AdaFactor, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # 处理初始化
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0

                    # 二阶矩参数
                    if p.ndim > 1:
                        # 对于矩阵，我们存储行和列的统计量
                        state['exp_avg_sq_row'] = torch.zeros(p.size(0), dtype=p.dtype, device=p.device)
                        state['exp_avg_sq_col'] = torch.zeros(p.size(1), dtype=p.dtype, device=p.device)
                    else:
                        # 对于向量，我们存储完整统计量
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    # 如果启用一阶动量
                    if group['beta1'] > 0.0:
                        state['exp_avg'] = torch.zeros_like(p)

                # 更新步数
                state['step'] += 1

                # 计算学习率
                if group['relative_step']:
                    # 使用相对步长
                    rel_step_sz = 1.0
                    if state['step'] < 10000:
                        rel_step_sz = math.sqrt(state['step'])
                    if group['warmup_init']:
                        rel_step_sz = min(1.0, rel_step_sz / math.sqrt(10000))
                    lr = rel_step_sz / math.sqrt(max(1, state['step']))
                else:
                    # 使用固定学习率
                    lr = group['lr']

                # 权重衰减
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # 梯度裁剪
                if group['clip_threshold'] > 0.0:
                    grad_norm = torch.norm(grad.pow(2).sum().sqrt())
                    clip_coef = group['clip_threshold'] / (grad_norm + group['eps'])
                    if clip_coef < 1.0:
                        grad.mul_(clip_coef)

                # 计算二阶矩
                beta2 = group['beta2']

                if p.ndim > 1:
                    # 对于矩阵，使用行列分解
                    grad_sq = grad.pow(2)

                    # 更新行统计量
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_row.mul_(beta2).add_(grad_sq.mean(dim=1), alpha=1 - beta2)

                    # 更新列统计量
                    exp_avg_sq_col = state['exp_avg_sq_col']
                    exp_avg_sq_col.mul_(beta2).add_(grad_sq.mean(dim=0), alpha=1 - beta2)

                    # 计算更新量
                    rms_row = exp_avg_sq_row.sqrt().add_(group['eps'])
                    rms_col = exp_avg_sq_col.sqrt().add_(group['eps'])

                    # 近似RMS
                    rms = torch.outer(rms_row, rms_col)
                    rms = rms / torch.sqrt(rms_row.mean() * rms_col.mean())
                else:
                    # 对于向量，使用标准方法
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    rms = exp_avg_sq.sqrt().add_(group['eps'])

                # 使用动量（如果启用）
                update = grad / rms
                if group['beta1'] > 0.0:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(update, alpha=1 - group['beta1'])
                    update = exp_avg

                # 缩放更新量
                if group['scale_parameter']:
                    update.mul_(lr)
                else:
                    update.mul_(lr * (rms.mean() / group['eps']))

                # 应用更新
                p.add_(-update)

        return loss


class SAM(Optimizer):
    """
    实现SAM优化器 (Sharpness-Aware Minimization)
    参考论文: https://arxiv.org/abs/2010.01412

    SAM寻找对抗性权重扰动，使得模型泛化能力更强
    SAM的工作原理是：
    1. 计算当前梯度
    2. 基于当前梯度进行小扰动
    3. 重新计算新位置的梯度
    4. 使用新梯度更新权重回到原位置

    这种方法可以帮助找到更平坦的最小值，通常泛化能力更好
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        初始化SAM优化器

        参数:
            params: 模型参数
            base_optimizer: 基础优化器类，如torch.optim.SGD
            rho: 扰动大小
            adaptive: 是否使用自适应扰动
            **kwargs: 传递给基础优化器的参数
        """
        assert rho >= 0.0, f"无效的rho, 应该是非负的: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        执行第一步：计算并应用扰动
        """
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # 保存原始参数副本
                self.state[p]["old_p"] = p.data.clone()

                # 计算扰动大小
                if group["adaptive"]:
                    # 自适应扰动：根据参数梯度的绝对值缩放扰动
                    e_w = p.grad * scale.to(p)
                else:
                    # 标准扰动：使用固定扰动大小
                    e_w = p.grad * scale

                # 应用扰动
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        执行第二步：重置参数并应用更新
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # 重置为原始参数
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]

        # 使用基础优化器应用更新
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行优化步骤

        注意：与标准优化器不同，SAM需要两次前向-后向传递，
        因此需要使用closure函数来执行第二次传递。

        SAM的典型用法是：
        1. 前向传播和反向传播计算梯度
        2. 调用optimizer.first_step(zero_grad=True)
        3. 使用closure再次执行前向传播和反向传播
        4. 调用optimizer.second_step()

        如果提供closure，我们在这一步中自动执行上述过程。
        """
        assert closure is not None, "SAM需要cloure函数执行第二次前向-后向传递"

        # 第一步：应用扰动
        self.first_step(zero_grad=True)

        # 第二次前向-后向传递
        with torch.enable_grad():
            closure()

        # 第二步：更新权重
        self.second_step()

    def _grad_norm(self):
        """计算梯度范数"""
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p.grad) if group["adaptive"] else p.grad).norm(p=2))
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        """加载状态字典"""
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class SophiaG(Optimizer):
    """
    实现SophiaG优化器
    参考: https://arxiv.org/abs/2305.14342

    Sophia是一种新型优化器，使用Hessian信息来改进收敛性能
    特别适合大型语言模型和Transformer模型
    """

    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-1, k=10, estimator="eig",
                 orth_steps=100, num_samples=2, device=None):
        """
        初始化SophiaG优化器

        参数:
            params: 模型参数
            lr: 学习率
            betas: 动量系数 (m_momentum, h_momentum)
            rho: 噪声大小
            weight_decay: 权重衰减系数
            k: 使用前k个特征值和特征向量
            estimator: Hessian估计器类型 ('eig', 'trace')
            orth_steps: 正交迭代步数
            num_samples: 采样次数
            device: 计算设备
        """
        defaults = dict(
            lr=lr, betas=betas, rho=rho, weight_decay=weight_decay,
            k=k, estimator=estimator, orth_steps=orth_steps,
            num_samples=num_samples, device=device
        )
        super(SophiaG, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行优化步骤"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # 获取梯度和参数
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("SophiaG不支持稀疏梯度")

                # 初始化状态
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)

                    # Hessian估计
                    if group['estimator'] == 'eig':
                        # 使用特征值分解估计
                        state['h_diag'] = torch.ones_like(p)
                    else:
                        # 使用迹估计
                        state['h_diag'] = torch.zeros_like(p)

                # 更新步数
                state['step'] += 1

                # 获取超参数
                beta1, beta2 = group['betas']
                lr = group['lr']
                weight_decay = group['weight_decay']
                rho = group['rho']

                # 权重衰减
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # 更新一阶动量
                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 估计Hessian对角线
                if state['step'] % group['k'] == 0:
                    h_diag = self._estimate_hessian_diagonal(p, grad, group)

                    # 更新Hessian估计
                    state['h_diag'].mul_(beta2).add_(h_diag, alpha=1 - beta2)

                # 计算更新量
                update = exp_avg / (state['h_diag'] + 1e-10)

                # 应用更新
                p.add_(update, alpha=-lr)

        return loss

    def _estimate_hessian_diagonal(self, p, grad, group, loss=None):
        """
        估计Hessian对角线
        """
        if group['estimator'] == 'eig':
            # 使用特征值分解估计
            with torch.enable_grad():
                # 生成随机噪声
                v = torch.randn_like(p)
                v = v / (v.norm() + 1e-10)

                # 计算Hessian-向量积
                p.requires_grad_(True)
                g = grad.detach()
                hvp = torch.autograd.grad(g, p, v, retain_graph=True)[0]
                p.requires_grad_(False)

                # 估计对角线
                h_diag = torch.abs(hvp / v)

            return h_diag
        else:
            # 使用迹估计
            h_diag = torch.zeros_like(p)

            # 多次采样
            for _ in range(group['num_samples']):
                with torch.enable_grad():
                    # 生成随机噪声
                    v = torch.randn_like(p)

                    # 添加小扰动
                    p_perturbed = p + group['rho'] * v

                    # 计算扰动后的梯度
                    p_perturbed.requires_grad_(True)
                    g_perturbed = torch.autograd.grad(loss, p_perturbed)[0]
                    p_perturbed.requires_grad_(False)

                    # 估计对角线
                    h_est = (g_perturbed - grad) / (group['rho'] * v)
                    h_diag.add_(h_est)

            # 取平均
            h_diag.div_(group['num_samples'])

            return h_diag


# 高级学习率调度器
def get_cosine_annealing_warmup_restarts(
        optimizer: Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 1e-3,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        gamma: float = 1.0
) -> LambdaLR:
    """
    创建余弦退火预热重启学习率调度器

    参数:
        optimizer: 优化器
        first_cycle_steps: 第一个周期的步数
        cycle_mult: 周期乘数
        max_lr: 最大学习率
        min_lr: 最小学习率
        warmup_steps: 预热步数
        gamma: 学习率衰减因子

    返回:
        学习率调度器
    """

    def lr_lambda(current_step):
        # 预热阶段
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        # 计算当前周期和步数
        cycle = 0
        cycle_steps = first_cycle_steps
        step_sum = 0

        while step_sum + cycle_steps <= current_step:
            step_sum += cycle_steps
            cycle += 1
            cycle_steps = int(cycle_steps * cycle_mult)

        # 计算当前周期中的步数
        cycle_step = current_step - step_sum

        # 计算缩放系数
        cycle_scale = gamma ** cycle

        # 计算余弦退火
        cos_value = math.cos(math.pi * cycle_step / cycle_steps) * 0.5 + 0.5
        lr_range = max_lr - min_lr

        return min_lr + lr_range * cos_value * cycle_scale

    return LambdaLR(optimizer, lr_lambda)


def get_advanced_optimizer(model: torch.nn.Module,
                           optimizer_type: str = 'adamw',
                           lr: float = 1e-4,
                           weight_decay: float = 0.01,
                           **kwargs) -> Optimizer:
    """
    获取高级优化器

    参数:
        model: 模型
        optimizer_type: 优化器类型 ('adamw', 'lion', 'adafactor', 'sam', 'sophia')
        lr: 学习率
        weight_decay: 权重衰减
        **kwargs: 其他参数

    返回:
        优化器实例
    """
    # 区分无权重衰减参数
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

    # 分组参数
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]

    # 创建优化器
    if optimizer_type.lower() == 'adamw':
        # 使用融合的AdamW实现（如果可用）
        if hasattr(torch.optim, 'AdamW') and hasattr(torch.optim.AdamW, 'fused') and torch.cuda.is_available():
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=lr,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8),
                fused=True  # 使用融合实现
            )
        else:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=lr,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
    elif optimizer_type.lower() == 'lion':
        optimizer = Lion(
            optimizer_grouped_parameters,
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.99)),
            weight_decay=0.0  # 权重衰减已经在参数分组中设置
        )
    elif optimizer_type.lower() == 'adafactor':
        optimizer = AdaFactor(
            optimizer_grouped_parameters,
            lr=lr,
            beta1=kwargs.get('beta1', 0.9),
            weight_decay=0.0,  # 权重衰减已经在参数分组中设置
            relative_step=kwargs.get('relative_step', False),
            scale_parameter=kwargs.get('scale_parameter', True),
            warmup_init=kwargs.get('warmup_init', False)
        )
    elif optimizer_type.lower() == 'sam':
        base_optimizer = torch.optim.SGD
        if 'base_optimizer' in kwargs:
            if kwargs['base_optimizer'].lower() == 'sgd':
                base_optimizer = torch.optim.SGD
            elif kwargs['base_optimizer'].lower() == 'adam':
                base_optimizer = torch.optim.Adam
            elif kwargs['base_optimizer'].lower() == 'adamw':
                base_optimizer = torch.optim.AdamW

        optimizer = SAM(
            optimizer_grouped_parameters,
            base_optimizer=base_optimizer,
            rho=kwargs.get('rho', 0.05),
            adaptive=kwargs.get('adaptive', False),
            lr=lr,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=0.0  # 权重衰减已经在参数分组中设置
        )
    elif optimizer_type.lower() == 'sophia':
        optimizer = SophiaG(
            optimizer_grouped_parameters,
            lr=lr,
            betas=kwargs.get('betas', (0.965, 0.99)),
            rho=kwargs.get('rho', 0.04),
            weight_decay=0.0,  # 权重衰减已经在参数分组中设置
            k=kwargs.get('k', 10),
            estimator=kwargs.get('estimator', 'eig')
        )
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

    return optimizer


def get_advanced_scheduler(optimizer: Optimizer,
                           scheduler_type: str = 'cosine_warmup',
                           num_warmup_steps: int = 100,
                           num_training_steps: int = 10000,
                           **kwargs) -> Any:
    """
    获取高级学习率调度器

    参数:
        optimizer: 优化器
        scheduler_type: 调度器类型 ('cosine_warmup', 'cosine_warmup_restarts', 'one_cycle')
        num_warmup_steps: 预热步数
        num_training_steps: 训练总步数
        **kwargs: 其他参数

    返回:
        学习率调度器
    """
    if scheduler_type == 'cosine_warmup':
        # 余弦预热
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == 'cosine_warmup_restarts':
        # 余弦预热重启
        return get_cosine_annealing_warmup_restarts(
            optimizer,
            first_cycle_steps=kwargs.get('first_cycle_steps', num_training_steps),
            cycle_mult=kwargs.get('cycle_mult', 1.0),
            max_lr=kwargs.get('max_lr', 1.0),
            min_lr=kwargs.get('min_lr', 0.0),
            warmup_steps=num_warmup_steps,
            gamma=kwargs.get('gamma', 1.0)
        )

    elif scheduler_type == 'one_cycle':
        # One Cycle Policy
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 1e-3),
            total_steps=num_training_steps,
            pct_start=float(num_warmup_steps) / float(num_training_steps),
            anneal_strategy=kwargs.get('anneal_strategy', 'cos'),
            div_factor=kwargs.get('div_factor', 25.0),
            final_div_factor=kwargs.get('final_div_factor', 1e4)
        )

    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")


# 量化辅助函数
def quantize_model(model: torch.nn.Module,
                   quantization_type: str = 'dynamic',
                   dtype: str = 'qint8',
                   calibrate: bool = False) -> torch.nn.Module:
    """
    量化模型

    参数:
        model: 要量化的模型
        quantization_type: 量化类型 ('dynamic', 'static', 'qat')
        dtype: 量化数据类型 ('qint8', 'float16')
        calibrate: 是否执行校准

    返回:
        量化后的模型
    """
    if dtype == 'float16' and torch.cuda.is_available():
        # 使用半精度（FP16）
        return model.half()

    # 如果模型没有量化支持，添加量化存根
    if not hasattr(model, 'qconfig'):
        model.qconfig = None

    if quantization_type == 'dynamic':
        # 动态量化
        model_quantized = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={torch.nn.Linear},
            dtype=getattr(torch, dtype.replace('q', ''))
        )

    elif quantization_type == 'static':
        # 静态量化
        model.eval()

        # 配置量化
        if torch.backends.quantized.engine == 'fbgemm':
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        else:
            model.qconfig = torch.quantization.get_default_qconfig('qnnpack')

        # 准备量化
        model_prepared = torch.quantization.prepare(model)

        if calibrate:
            # 校准（用户应提供校准数据）
            pass

        # 转换为量化模型
        model_quantized = torch.quantization.convert(model_prepared)

    elif quantization_type == 'qat':
        # 量化感知训练
        model.train()

        # 配置量化
        if torch.backends.quantized.engine == 'fbgemm':
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        else:
            model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')

        # 准备QAT
        model_prepared = torch.quantization.prepare_qat(model)

        # 转换为量化模型
        model_quantized = torch.quantization.convert(model_prepared)

    else:
        raise ValueError(f"不支持的量化类型: {quantization_type}")

    return model_quantized


if __name__ == "__main__":
    # 测试优化器和调度器
    import matplotlib.pyplot as plt

    # 创建一个简单模型
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # 测试不同优化器
    optimizers = {
        'AdamW': get_advanced_optimizer(model, 'adamw', lr=1e-3),
        'Lion': get_advanced_optimizer(model, 'lion', lr=1e-4),
        'AdaFactor': get_advanced_optimizer(model, 'adafactor', lr=1e-3),
    }

    # 测试不同调度器
    plt.figure(figsize=(10, 6))

    for name, optimizer in optimizers.items():
        # 重置优化器学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3

        # 创建余弦预热调度器
        scheduler = get_advanced_scheduler(
            optimizer,
            'cosine_warmup',
            num_warmup_steps=100,
            num_training_steps=1000
        )

        # 记录学习率
        lrs = []
        for _ in range(1000):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # 绘制学习率
        plt.plot(lrs, label=f"{name} + Cosine Warmup")

    # 绘制图表
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules')
    plt.legend()
    plt.grid(True)

    # 保存图表
    plt.savefig('optimizer_comparison.png')
    print("优化器对比图已保存至 optimizer_comparison.png")

    # 测试量化
    if torch.cuda.is_available():
        try:
            # 测试动态量化
            model_int8 = quantize_model(model, 'dynamic', 'qint8')
            print(
                f"动态量化模型大小: {sum(p.nelement() * p.element_size() for p in model_int8.parameters()) / 1024:.2f} KB")

            # 测试FP16
            model_fp16 = model.half().cuda()
            print(
                f"FP16模型大小: {sum(p.nelement() * p.element_size() for p in model_fp16.parameters()) / 1024:.2f} KB")

            # 对比原始FP32
            print(f"FP32模型大小: {sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024:.2f} KB")
        except Exception as e:
            print(f"量化测试失败: {e}")
    else:
        print("CUDA不可用，跳过量化测试")