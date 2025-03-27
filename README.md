# AiXunLianCang Deep Learning Platform

<img src="https://img.shields.io/badge/CUDA-12.4-brightgreen" alt="CUDA Version"/> <img src="https://img.shields.io/badge/PyTorch-2.6.0-orange" alt="PyTorch Version"/> <img src="https://img.shields.io/badge/GPU-RTX%204070-blue" alt="GPU"/> <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>

## 项目概述

AiXunLianCang是一个针对NVIDIA RTX 4070 GPU架构优化的深度学习训练平台，结合WSL2和PyCharm提供Windows与Linux的无缝集成。本平台专为高性能模型训练设计，充分利用CUDA加速实现高达45倍的矩阵运算性能提升。

### 核心特性

- 💥 **性能优化**: 针对RTX 4070显卡的训练性能优化，实现高效GPU利用
- 🔄 **混合精度训练**: 支持FP16混合精度计算，提升2-3倍训练速度
- 📊 **实时监控**: GPU利用率、显存、温度和功耗的图形化实时监控
- 🛠️ **模块化设计**: 灵活可扩展的架构，支持多种模型和训练策略
- ⚙️ **自动配置**: 智能检测硬件资源并优化训练参数
- 📝 **详尽日志**: 训练过程的完整记录和可视化
- 🧩 **工作流管理**: 标准化的项目结构和开发流程

## 系统架构

### 全局架构