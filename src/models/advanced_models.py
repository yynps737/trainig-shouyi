#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union, Any


class EfficientBlock(nn.Module):
    """优化的EfficientNet风格卷积块，针对RTX 4070优化"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 expansion_factor: float = 4.0,
                 reduction_ratio: int = 4,
                 dropout_rate: float = 0.2):
        super().__init__()

        # 扩展通道数
        expanded_channels = int(in_channels * expansion_factor)

        # 1x1卷积扩展通道
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)  # 使用SiLU激活函数提高性能
        )

        # 深度可分离卷积
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )

        # 注意力机制（squeeze-and-excitation）
        reduced_channels = max(1, expanded_channels // reduction_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, reduced_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, expanded_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 1x1卷积投影到输出通道
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 残差连接
        self.use_residual = (in_channels == out_channels and stride == 1)

        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        identity = x

        # 主路径
        x = self.expand(x)
        x = self.depthwise(x)

        # 应用SE注意力
        x = x * self.se(x)

        x = self.project(x)

        # 应用Dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # 残差连接
        if self.use_residual:
            x = x + identity

        return x


class RTXEfficientNet(nn.Module):
    """针对RTX 4070优化的EfficientNet实现"""

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 10,
                 width_multiplier: float = 1.0,
                 depth_multiplier: float = 1.0,
                 dropout_rate: float = 0.2,
                 enable_quantization: bool = False):
        super().__init__()

        # 网络配置
        config = [
            # t, c, n, s (扩展因子, 输出通道, 重复次数, 步长)
            [1, 16, 1, 1],  # 阶段1
            [6, 24, 2, 2],  # 阶段2
            [6, 40, 2, 2],  # 阶段3
            [6, 80, 3, 2],  # 阶段4
            [6, 112, 3, 1],  # 阶段5
            [6, 192, 4, 2],  # 阶段6
            [6, 320, 1, 1]  # 阶段7
        ]

        # 缩放通道
        def _round_channels(channels):
            return int(channels * width_multiplier)

        # 缩放深度
        def _round_repeats(repeats):
            return int(math.ceil(repeats * depth_multiplier))

        # 初始卷积层
        output_channels = _round_channels(32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True)
        )

        # 构建阶段
        input_channels = output_channels
        stages = []

        for t, c, n, s in config:
            output_channels = _round_channels(c)
            repeats = _round_repeats(n)

            # 第一个块有步长s，其余为1
            for i in range(repeats):
                stride = s if i == 0 else 1
                stages.append(
                    EfficientBlock(
                        in_channels=input_channels,
                        out_channels=output_channels,
                        expansion_factor=t,
                        stride=stride,
                        dropout_rate=dropout_rate
                    )
                )
                input_channels = output_channels

        self.stages = nn.Sequential(*stages)

        # 头部
        head_channels = _round_channels(1280)
        self.head = nn.Sequential(
            nn.Conv2d(input_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_channels, num_classes)
        )

        # 量化支持
        self.enable_quantization = enable_quantization
        if enable_quantization:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        # 量化
        if self.enable_quantization and not self.training:
            x = self.quant(x)

        # 前向传播
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        x = self.classifier(x)

        # 反量化
        if self.enable_quantization and not self.training:
            x = self.dequant(x)

        return x

    def _initialize_weights(self):
        # 使用RTX优化的初始化策略
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def fuse_model(self):
        """模型融合以加速推理"""
        if not self.enable_quantization:
            return

        # 融合卷积+BN层
        for m in self.modules():
            if isinstance(m, EfficientBlock):
                torch.quantization.fuse_modules(m.expand, ['0', '1'], inplace=True)
                torch.quantization.fuse_modules(m.depthwise, ['0', '1'], inplace=True)
                torch.quantization.fuse_modules(m.project, ['0', '1'], inplace=True)

        torch.quantization.fuse_modules(self.stem, ['0', '1'], inplace=True)

        for i, m in enumerate(self.head):
            if isinstance(m, nn.Conv2d) and i < len(self.head) - 1:
                if isinstance(self.head[i + 1], nn.BatchNorm2d):
                    torch.quantization.fuse_modules(self.head, [str(i), str(i + 1)], inplace=True)


class EnhancedFFN(nn.Module):
    """增强的前馈网络，针对RTX 4070 Tensor Core优化"""

    def __init__(self,
                 d_model: int,
                 dim_feedforward: int,
                 activation: str = "gelu",
                 dropout: float = 0.1):
        super().__init__()

        # 确保维度为Tensor Core优化的整数倍（32）
        dim_feedforward = ((dim_feedforward + 31) // 32) * 32

        # 标准前馈网络实现
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = self._get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 残差缩放因子
        self.scale = 1.0 / math.sqrt(2.0)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu" or activation == "swish":
            return nn.SiLU(inplace=True)
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

    def forward(self, x):
        # 残差连接
        residual = x

        # 前馈网络
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        # 残差缩放
        return residual + x * self.scale


class FlashAttention(nn.Module):
    """实现Flash Attention优化的注意力机制"""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 causal: bool = False):
        super().__init__()

        # 确保嵌入维度能被头数整除
        assert embed_dim % num_heads == 0, "嵌入维度必须是头数的整数倍"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal

        # 缩放因子
        self.scaling = self.head_dim ** -0.5

        # 投影矩阵
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 内存效率检查
        self.use_flash_attn = self._check_flash_attention_available()

    def _check_flash_attention_available(self):
        """检查是否可以使用Flash Attention"""
        try:
            # 尝试导入flash_attn库
            import flash_attn
            has_flash_attn = True
        except ImportError:
            has_flash_attn = False

        # 检查CUDA可用性
        cuda_available = torch.cuda.is_available()

        # 检查GPU架构是否支持（Ampere或更高）
        arch_supported = False
        if cuda_available:
            device_cap = torch.cuda.get_device_capability()
            arch_supported = device_cap[0] >= 8  # Ampere及更高架构

        return has_flash_attn and cuda_available and arch_supported

    def _flash_attention_forward(self, q, k, v, attn_mask=None):
        """使用Flash Attention实现的前向传播"""
        from flash_attn import flash_attn_func

        # 重塑为Flash Attention需要的格式 [batch, seqlen, num_heads, head_dim]
        batch_size, seq_len, _ = q.size()
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, k.size(1), self.num_heads, self.head_dim)
        v = v.view(batch_size, v.size(1), self.num_heads, self.head_dim)

        # 调用Flash Attention函数
        # 注意：flash_attn_func预期q, k, v的形状为 [batch, seqlen, nheads, headdim]
        dropout_p = self.dropout if self.training else 0.0

        # Flash Attention不直接支持注意力掩码，需特殊处理
        if attn_mask is not None:
            # 对于padding掩码，可以使用key_padding_mask参数
            if attn_mask.dim() == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                key_padding_mask = ~attn_mask.bool()
                out = flash_attn_func(
                    q, k, v,
                    dropout_p=dropout_p,
                    causal=self.causal,
                    key_padding_mask=key_padding_mask
                )
            else:
                # 对于更复杂的掩码，我们回退到标准注意力计算
                return self._standard_attention_forward(
                    q.view(batch_size, seq_len, -1),
                    k.view(batch_size, k.size(1), -1),
                    v.view(batch_size, v.size(1), -1),
                    attn_mask
                )
        else:
            out = flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                causal=self.causal
            )

        # 重塑输出 [batch, seqlen, embed_dim]
        return out.view(batch_size, seq_len, -1)

    def _standard_attention_forward(self, q, k, v, attn_mask=None):
        """标准注意力实现的前向传播"""
        # 重塑为多头形式 [batch, num_heads, seq_len, head_dim]
        batch_size, seq_len, _ = q.size()
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力
        q = q * self.scaling
        attn_weights = torch.matmul(q, k.transpose(-2, -1))

        # 应用掩码
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(~attn_mask.bool(), float("-inf"))

        # 对于因果掩码
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # 应用softmax和dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 注意力输出
        attn_output = torch.matmul(attn_weights, v)

        # 重塑回原始维度 [batch, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return attn_output

    def forward(self, q, k, v, attn_mask=None):
        """前向传播，根据可用性选择Flash Attention或标准注意力"""
        # 投影q, k, v
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 选择注意力实现
        if self.use_flash_attn and q.is_cuda:
            attn_output = self._flash_attention_forward(q, k, v, attn_mask)
        else:
            attn_output = self._standard_attention_forward(q, k, v, attn_mask)

        # 最终投影
        return self.out_proj(attn_output)


class RTXTransformerLayer(nn.Module):
    """针对RTX 4070优化的Transformer层"""

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 norm_first: bool = True,
                 use_flash_attention: bool = True,
                 causal: bool = False):
        super().__init__()

        # 优化维度为32的倍数（针对Tensor Core）
        d_model = ((d_model + 31) // 32) * 32
        dim_feedforward = ((dim_feedforward + 31) // 32) * 32

        # 自注意力层
        self.self_attn = FlashAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            causal=causal
        ) if use_flash_attention else nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # 前馈网络
        self.ffn = EnhancedFFN(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 是否使用预归一化（Pre-LN）
        self.norm_first = norm_first

        # 是否使用自定义FlashAttention
        self.use_flash_attention = use_flash_attention

    def forward(self, x, attn_mask=None):
        """前向传播"""
        # 预归一化或后归一化
        if self.norm_first:
            # 预归一化
            attn_output = self._attention_block(self.norm1(x), attn_mask)
            x = x + attn_output
            x = x + self._ff_block(self.norm2(x))
        else:
            # 后归一化
            attn_output = self._attention_block(x, attn_mask)
            x = self.norm1(x + attn_output)
            x = self.norm2(x + self._ff_block(x))

        return x

    def _attention_block(self, x, attn_mask):
        """注意力块"""
        if self.use_flash_attention:
            return self.self_attn(x, x, x, attn_mask)
        else:
            attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
            return self.dropout(attn_output)

    def _ff_block(self, x):
        """前馈块"""
        return self.ffn(x)


class FastRTXTransformer(nn.Module):
    """快速RTX Transformer模型，针对RTX 4070优化"""

    def __init__(self,
                 vocab_size: int = 30000,
                 max_seq_len: int = 512,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 norm_first: bool = True,
                 use_flash_attention: bool = True,
                 num_classes: int = 2,
                 share_embeddings: bool = True,
                 use_checkpoint: bool = False):
        super().__init__()

        # 优化维度为32的倍数（针对Tensor Core）
        d_model = ((d_model + 31) // 32) * 32
        dim_feedforward = ((dim_feedforward + 31) // 32) * 32

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_checkpoint = use_checkpoint

        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer编码器
        self.encoder_layers = nn.ModuleList([
            RTXTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                use_flash_attention=use_flash_attention
            ) for _ in range(num_layers)
        ])

        # 输出层
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            self._get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        # 如果共享嵌入
        if share_embeddings:
            # 使嵌入和分类器权重共享
            self.classifier[-1].weight = self.token_embedding.weight

        # 初始化参数
        self._init_parameters()

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu" or activation == "swish":
            return nn.SiLU(inplace=True)
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

    def _init_parameters(self):
        """参数初始化"""
        # Xavier初始化线性层
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 特殊初始化嵌入层
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)

    def forward(self, x, attention_mask=None):
        """前向传播"""
        # 获取序列长度
        batch_size, seq_len = x.size()
        device = x.device

        # 位置ID
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # 嵌入
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings

        # Dropout
        x = self.dropout(embeddings)

        # Transformer编码器层
        for layer in self.encoder_layers:
            if self.use_checkpoint and self.training:
                # 使用梯度检查点减少内存使用
                x = torch.utils.checkpoint.checkpoint(layer, x, attention_mask)
            else:
                x = layer(x, attention_mask)

        # 最终层归一化
        x = self.norm(x)

        # 分类（使用[CLS]位置的输出，即第一个token）
        x = x[:, 0]
        output = self.classifier(x)

        return output


# 辅助函数 - 创建量化模型
def create_quantized_model(model, quantization_backend='fbgemm'):
    """创建INT8量化模型"""
    # 为FP32模型准备量化
    model.eval()
    model.enable_quantization = True

    # 配置量化配置
    model.qconfig = torch.quantization.get_default_qconfig(quantization_backend)
    torch.quantization.prepare(model, inplace=True)

    # 校准（应在使用时执行）

    # 转换为量化模型
    torch.quantization.convert(model, inplace=True)

    return model


# 辅助函数 - 构建模型
def build_rtx_optimized_model(model_type, **kwargs):
    """构建针对RTX 4070优化的模型"""
    if model_type == 'efficient_net':
        return RTXEfficientNet(**kwargs)
    elif model_type == 'transformer':
        return FastRTXTransformer(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


if __name__ == "__main__":
    # 测试模型
    import time

    # 测试EfficientNet
    model_efficient = RTXEfficientNet(num_classes=10)
    x_img = torch.randn(8, 3, 224, 224)

    # 测试Transformer
    model_transformer = FastRTXTransformer(vocab_size=30000, num_classes=2)
    x_text = torch.randint(0, 30000, (8, 64))

    if torch.cuda.is_available():
        model_efficient = model_efficient.cuda()
        model_transformer = model_transformer.cuda()
        x_img = x_img.cuda()
        x_text = x_text.cuda()

    # 预热
    with torch.no_grad():
        _ = model_efficient(x_img)
        _ = model_transformer(x_text)

    # 测试EfficientNet性能
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            y_img = model_efficient(x_img)
    torch.cuda.synchronize()
    efficient_time = (time.time() - start) / 10

    # 测试Transformer性能
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            y_text = model_transformer(x_text)
    torch.cuda.synchronize()
    transformer_time = (time.time() - start) / 10

    # 测试编译后的模型性能（仅在PyTorch 2.0+和CUDA可用时）
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        print("使用torch.compile优化...")
        compiled_model = torch.compile(model_efficient, mode='max-autotune')

        # 预热
        _ = compiled_model(x_img)
        torch.cuda.synchronize()

        # 测量性能
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                y_img = compiled_model(x_img)
        torch.cuda.synchronize()
        compiled_time = (time.time() - start) / 10

        print(f"EfficientNet 编译前向传播时间: {efficient_time * 1000:.2f}ms")
        print(f"EfficientNet 编译后前向传播时间: {compiled_time * 1000:.2f}ms")
        print(f"编译加速比: {efficient_time / compiled_time:.2f}x")

    print(f"EfficientNet 前向传播时间: {efficient_time * 1000:.2f}ms")
    print(f"Transformer 前向传播时间: {transformer_time * 1000:.2f}ms")