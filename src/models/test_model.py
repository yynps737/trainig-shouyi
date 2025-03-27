import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    # 测试模型
    model = SimpleNet()
    x = torch.randn(1, 10)
    y = model(x)
    print(f"模型输出形状: {y.shape}")
    
    # 测试CUDA
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        y = model(x)
        print(f"CUDA模型输出形状: {y.shape}")
        print(f"使用的GPU: {torch.cuda.get_device_name(0)}")
