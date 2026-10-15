import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml

# ---------- 1. 初始化 DirectML 设备 ----------
dml = torch_directml.device()
print(f"正在使用设备: {torch_directml.device_name(0)}")

# ---------- 2. 准备数据（移动到 GPU） ----------
# 生成 1000 个样本，每个样本 10 个特征
X = torch.randn(1000, 10).to(dml)
y = torch.randint(0, 2, (1000, 1)).float().to(dml)  # 二分类标签

# ---------- 3. 定义模型（移动到 GPU） ----------
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = SimpleNet().to(dml)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------- 4. 训练循环 ----------
print("\n开始训练...")
for epoch in range(5):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

print("训练完成！你的 Intel Arc 显卡正在加速工作。")