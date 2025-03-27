#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import platform
import time
import torch
import datetime
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional


def check_system():
    """检查系统环境"""
    print("=" * 60)
    print("系统环境检查")
    print("=" * 60)

    # 检查操作系统
    os_name = platform.system()
    os_version = platform.version()
    print(f"操作系统: {os_name} {os_version}")

    # 检查Python版本
    python_version = platform.python_version()
    print(f"Python版本: {python_version}")

    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")

    if cuda_available:
        # 检查CUDA版本
        cuda_version = torch.version.cuda
        print(f"CUDA版本: {cuda_version}")

        # 检查GPU设备信息
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            device_props = torch.cuda.get_device_properties(i)

            print(f"GPU {i}: {device_name}")
            print(f"  计算能力: {device_capability[0]}.{device_capability[1]}")
            print(f"  显存: {device_props.total_memory / (1024 ** 3):.2f} GB")
            print(f"  多处理器数量: {device_props.multi_processor_count}")

    # 检查PyTorch版本
    torch_version = torch.__version__
    print(f"PyTorch版本: {torch_version}")

    # 检查WSL
    is_wsl = "WSL" in os_version or "Microsoft" in os_version
    print(f"WSL环境: {is_wsl}")

    # 检查依赖项
    try:
        import torchvision
        print(f"torchvision版本: {torchvision.__version__}")
    except ImportError:
        print("警告: torchvision未安装")

    try:
        import matplotlib
        print(f"matplotlib版本: {matplotlib.__version__}")
    except ImportError:
        print("警告: matplotlib未安装")

    print("=" * 60)
    return cuda_available


def init_project(args):
    """初始化项目"""
    print("=" * 60)
    print("项目初始化")
    print("=" * 60)

    # 创建目录
    for dir_path in [
        "data/raw",
        "data/processed",
        "logs",
        "models",
        "visualizations",
        "outputs"
    ]:
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")

    # 创建或更新WSL配置
    if platform.system() == "Windows":
        try:
            home_dir = os.path.expanduser("~")
            wsl_config_path = os.path.join(home_dir, ".wslconfig")

            # 确定配置内容
            wsl_config = "[wsl2]\n"
            wsl_config += "memory=16GB\n"
            wsl_config += f"processors={os.cpu_count() // 2}\n"
            wsl_config += "swap=8GB\n"
            wsl_config += "localhostForwarding=true\n"

            # 写入配置
            with open(wsl_config_path, "w") as f:
                f.write(wsl_config)

            print(f"WSL配置已更新: {wsl_config_path}")
        except Exception as e:
            print(f"更新WSL配置失败: {e}")

    # 检查PyTorch安装
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"PyTorch CUDA支持: 已启用 (版本: {torch.version.cuda})")
        else:
            print("警告: PyTorch CUDA支持未启用")
    except ImportError:
        print("警告: PyTorch未安装")

    # 检查依赖并安装缺失的
    if args.install_deps:
        install_dependencies()

    # 运行CUDA基准测试
    if cuda_available and args.run_benchmark:
        print("\n执行CUDA基准测试...")
        subprocess.run([sys.executable, "cuda_benchmark.py"])

    # 创建环境摘要
    create_env_summary()

    print("\n项目初始化完成!")
    print("=" * 60)


def install_dependencies():
    """安装项目依赖"""
    print("\n安装项目依赖...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("依赖安装成功!")
    except subprocess.CalledProcessError as e:
        print(f"依赖安装失败: {e}")


def create_env_summary():
    """创建环境摘要文件"""
    summary = []

    # PyTorch信息
    import torch
    summary.append(f"PyTorch: {torch.__version__}")

    # CUDA信息
    if torch.cuda.is_available():
        summary.append(f"CUDA: {torch.version.cuda}")
        summary.append(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        summary.append("CUDA: 不可用")

    # 系统信息
    is_wsl = "Microsoft" in platform.version()
    if is_wsl:
        summary.append("系统: WSL2 Ubuntu")
    else:
        summary.append(f"系统: {platform.system()} {platform.release()}")

    # 其他信息
    summary.append(f"Python: {platform.python_version()}")

    # 写入文件
    with open("environment_summary.txt", "w") as f:
        f.write("\n".join(summary))

    print(f"环境摘要已保存至: environment_summary.txt")


def run_test(args):
    """运行测试"""
    print("=" * 60)
    print("运行测试")
    print("=" * 60)

    # 测试CUDA可用性
    print("测试CUDA可用性...")
    subprocess.run([sys.executable, "test_cuda.py"])

    # 测试模型
    if args.model:
        print(f"\n测试模型: {args.model}")
        subprocess.run([sys.executable, f"src/models/{args.model}.py"])

    # 运行GPU监控工具
    if args.monitor:
        print("\n启动GPU监控...")
        try:
            subprocess.run(["python", "gpu_monitor.py", "--interval", "0.5"])
        except KeyboardInterrupt:
            print("GPU监控已停止")

    print("=" * 60)


def optimize_wsl():
    """优化WSL配置"""
    if platform.system() != "Windows":
        print("此命令仅适用于Windows系统上的WSL")
        return

    print("=" * 60)
    print("优化WSL配置")
    print("=" * 60)

    # 创建WSL配置
    home_dir = os.path.expanduser("~")
    wsl_config_path = os.path.join(home_dir, ".wslconfig")

    # 获取系统信息
    import psutil
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB
    total_cpu = psutil.cpu_count(logical=True)

    # 生成配置
    wsl_config = "[wsl2]\n"

    # 内存配置 (最大16GB或系统内存的一半)
    allocated_memory = min(16, int(total_memory / 2))
    wsl_config += f"memory={allocated_memory}GB\n"

    # CPU配置 (最大16个或系统CPU的一半)
    allocated_cpu = min(16, int(total_cpu / 2))
    wsl_config += f"processors={allocated_cpu}\n"

    # 交换空间 (8GB)
    wsl_config += "swap=8GB\n"

    # 本地主机转发
    wsl_config += "localhostForwarding=true\n"

    # 写入配置
    with open(wsl_config_path, "w") as f:
        f.write(wsl_config)

    print(f"WSL配置已更新: {wsl_config_path}")
    print(f"配置详情:")
    print(f"  内存: {allocated_memory}GB")
    print(f"  处理器: {allocated_cpu}")
    print(f"  交换空间: 8GB")
    print(f"  本地主机转发: 已启用")

    print("\n重启WSL以应用更改:")
    print("  在Windows PowerShell中执行: wsl --shutdown")

    print("=" * 60)


def create_training_job(args):
    """创建训练作业"""
    print("=" * 60)
    print("创建训练作业")
    print("=" * 60)

    # 检查模型类型
    model_type = args.model_type
    if model_type not in ["cnn", "transformer"]:
        print(f"不支持的模型类型: {model_type}")
        print("支持的类型: cnn, transformer")
        return

    # 创建时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"{model_type}_{timestamp}"

    # 创建作业目录
    job_dir = os.path.join("outputs", job_name)
    os.makedirs(job_dir, exist_ok=True)

    # 复制配置文件
    config_src = os.path.join("config", "rtx4070_config.yaml")
    config_dst = os.path.join(job_dir, "config.yaml")
    shutil.copy2(config_src, config_dst)

    # 创建作业脚本
    script_path = os.path.join(job_dir, "train.sh")

    script_content = "#!/bin/bash\n\n"
    script_content += f"# 训练作业: {job_name}\n"
    script_content += f"# 创建时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    if model_type == "cnn":
        script_content += "# 运行CNN模型训练\n"
        script_content += f"python train.py --config {config_dst} --model cnn"
    else:
        script_content += "# 运行Transformer模型训练\n"
        script_content += f"python train.py --config {config_dst} --model transformer"

    with open(script_path, "w") as f:
        f.write(script_content)

    # 设置执行权限
    os.chmod(script_path, 0o755)

    print(f"训练作业已创建: {job_dir}")
    print(f"配置文件: {config_dst}")
    print(f"启动脚本: {script_path}")

    print("\n运行作业:")
    print(f"  cd {job_dir}")
    print(f"  ./train.sh")

    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AiXunLianCang深度学习训练平台工作流管理工具")

    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # info命令
    info_parser = subparsers.add_parser("info", help="显示系统信息")

    # init命令
    init_parser = subparsers.add_parser("init", help="初始化项目")
    init_parser.add_argument("--install-deps", action="store_true", help="安装依赖")
    init_parser.add_argument("--run-benchmark", action="store_true", help="运行CUDA基准测试")

    # test命令
    test_parser = subparsers.add_parser("test", help="运行测试")
    test_parser.add_argument("--model", type=str, help="测试特定模型")
    test_parser.add_argument("--monitor", action="store_true", help="启动GPU监控")

    # optimize命令
    optimize_parser = subparsers.add_parser("optimize", help="优化系统配置")

    # train命令
    train_parser = subparsers.add_parser("train", help="创建训练作业")
    train_parser.add_argument("--model-type", type=str, required=True, help="模型类型 (cnn, transformer)")

    # 解析参数
    args = parser.parse_args()

    # 处理命令
    if args.command == "info":
        check_system()
    elif args.command == "init":
        init_project(args)
    elif args.command == "test":
        run_test(args)
    elif args.command == "optimize":
        optimize_wsl()
    elif args.command == "train":
        create_training_job(args)
    else:
        # 默认显示帮助
        parser.print_help()


if __name__ == "__main__":
    main()