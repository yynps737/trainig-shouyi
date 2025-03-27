#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedCNN(nn.Module):
    """针对RTX 4070优化的CNN模型"""
    
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        # 使用高效架构
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),  # 高效激活函数
            nn.MaxPool2d(2),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2),
            
            # 第四个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
        
        # 适应性池化，适应不同输入尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        # 使用正确的权重初始化
        self._initialize_weights()
    
    def forward(self, x):
        # 使用torch.compile加速
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            # 检查是否已编译
            if not hasattr(self, '_compiled_conv'):
                self._compiled_conv = torch.compile(self.conv_layers)
                self._compiled_fc = torch.compile(self.fc_layers)
            
            # 使用编译后的模块
            x = self._compiled_conv(x)
            x = self.adaptive_pool(x)
            x = torch.flatten(x, 1)
            x = self._compiled_fc(x)
        else:
            # 常规前向传播
            x = self.conv_layers(x)
            x = self.adaptive_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc_layers(x)
        
        return x
    
    def _initialize_weights(self):
        """使用优化的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

class FastTransformer(nn.Module):
    """针对RTX 4070优化的轻量级Transformer模型"""
    
    def __init__(self, vocab_size=30000, max_seq_len=512, embed_dim=256, 
                 num_heads=8, num_layers=6, num_classes=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # 使用nn.TransformerEncoder以利用CUDA优化
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # 预层归一化提高稳定性
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes)
        )
        
        # 初始化
        self._initialize_weights()
    
    def forward(self, x, attention_mask=None):
        # 创建位置ID
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x)
        
        # 嵌入
        token_embeddings = self.embedding(x)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings
        
        # 创建注意力掩码
        if attention_mask is not None:
            # 转换为PyTorch Transformer格式的掩码
            attention_mask = attention_mask.float()
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Transformer编码
        encoded = self.transformer_encoder(embeddings, src_key_padding_mask=attention_mask)
        
        # 分类 (使用[CLS]令牌的输出)
        output = self.classifier(encoded[:, 0, :])
        
        return output
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

if __name__ == "__main__":
    # 测试CNN模型
    model = OptimizedCNN()
    x = torch.randn(8, 3, 32, 32)
    
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
    
    # 测试前向传播
    y = model(x)
    print(f"CNN输出形状: {y.shape}")
    
    # 测试内存使用情况
    if torch.cuda.is_available():
        print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        y = model(x)
        torch.cuda.synchronize()
        print(f"前向传播峰值显存: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
