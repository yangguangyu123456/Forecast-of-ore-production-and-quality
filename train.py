import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_data
from model import LSTMPredictor
import numpy as np

# 参数设置
input_size = 10  # 输入特征数
hidden_size = 50  # LSTM隐藏层单元数
num_layers = 2  # LSTM层数
output_size = 1  # 输出特征数
num_epochs = 100  # 训练轮数
learning_rate = 0.001  # 学习率
batch_size = 32  # 批次大小

# 加载数据
train_data = load_data('data/train.csv')
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

# 数据预处理
X_train = np.expand_dims(X_train, axis=2)
y_train = np.expand_dims(y_train, axis=1)

# 创建数据加载器
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
model = LSTMPredictor(input_size, hidden_size, num_layers, output_size).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(model.device), targets.to(model.device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'model.pth')