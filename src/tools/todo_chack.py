import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import os


# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


model = SimpleModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 定义指数衰减学习率调度器
scheduler = ExponentialLR(optimizer, gamma=0.9)

# 检查点文件路径
checkpoint_path = 'checkpoint.pth'

# 加载检查点（如果存在）
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded, starting from epoch {start_epoch}")

# 训练循环参数
num_epochs = 100
learning_rates = []

for epoch in range(start_epoch, num_epochs):
    model.train()
    # 模拟一个训练过程
    optimizer.zero_grad()
    output = model(torch.randn(1, 10))
    loss = criterion(output, torch.randn(1, 1))
    loss.backward()
    optimizer.step()

    # 每个 epoch 结束时更新学习率
    scheduler.step()

    # 记录当前学习率
    learning_rates.append(scheduler.get_last_lr()[0])

    # 保存检查点
    if (epoch + 1) % 10 == 0:  # 每10个epoch保存一次
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

# 打印最后的学习率
print(f'Final Learning Rate: {learning_rates[-1]}')

# 绘制学习率
plt.plot(range(start_epoch, num_epochs), learning_rates[start_epoch:], marker='o', linestyle='-', color='b')
plt.title('Learning Rate over Epochs using ExponentialLR')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.show()
