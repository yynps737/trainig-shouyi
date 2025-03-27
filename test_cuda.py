import torch

def main():
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        
        # 测试CUDA张量操作
        x = torch.rand(5, 5).cuda()
        y = torch.rand(5, 5).cuda()
        z = x @ y
        print(f"矩阵乘法结果形状: {z.shape}")
        print(f"在设备: {z.device}")

if __name__ == "__main__":
    main()
