import torch
import time
import numpy as np

def benchmark_cuda():
    """运行简单的CUDA性能测试"""
    print("执行CUDA性能测试...")
    
    # 测试参数
    sizes = [1000, 2000, 4000, 8000]
    repeats = 10
    
    results = []
    
    for size in sizes:
        # 创建大矩阵
        cpu_times = []
        gpu_times = []
        
        # CPU测试
        for _ in range(repeats):
            a = torch.randn(size, size)
            b = torch.randn(size, size)
            
            start = time.time()
            c = torch.matmul(a, b)
            cpu_time = time.time() - start
            cpu_times.append(cpu_time)
        
        # GPU测试
        if torch.cuda.is_available():
            # 预热GPU
            a = torch.randn(1000, 1000, device='cuda')
            b = torch.randn(1000, 1000, device='cuda')
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            for _ in range(repeats):
                a = torch.randn(size, size, device='cuda')
                b = torch.randn(size, size, device='cuda')
                
                start = time.time()
                c = torch.matmul(a, b)
                torch.cuda.synchronize()  # 确保GPU操作完成
                gpu_time = time.time() - start
                gpu_times.append(gpu_time)
        
        # 计算平均时间
        avg_cpu = np.mean(cpu_times)
        avg_gpu = np.mean(gpu_times) if gpu_times else float('inf')
        
        # 计算加速比
        speedup = avg_cpu / avg_gpu if avg_gpu > 0 else 0
        
        results.append({
            'size': size,
            'cpu_time': avg_cpu,
            'gpu_time': avg_gpu,
            'speedup': speedup
        })
        
        print(f"矩阵大小: {size}x{size}")
        print(f"CPU时间: {avg_cpu:.4f}秒")
        print(f"GPU时间: {avg_gpu:.4f}秒")
        print(f"加速比: {speedup:.2f}x")
        print("-" * 40)
    
    return results

if __name__ == "__main__":
    # 显示系统和CUDA信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print("-" * 40)
        
        # 运行基准测试
        benchmark_cuda()
    else:
        print("CUDA不可用，无法执行GPU基准测试")
