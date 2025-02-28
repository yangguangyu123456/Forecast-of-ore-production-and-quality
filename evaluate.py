import torch
from data_loader import load_data
from model import LSTMPredictor
import numpy as np

# 加载模型
input_size = 10
hidden_size = 50
num_layers = 2
output_size = 1

model = LSTMPredictor(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 加载测试数据
test_data = load_data('data/test.csv')
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# 数据预处理
X_test = np.expand_dims(X_test, axis=2)
y_test = np.expand_dims(y_test, axis=1)

# 转换为Tensor
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 预测
with torch.no_grad():
    predictions = model(X_test)

# 评估
mse = torch.nn.functional.mse_loss(predictions, y_test)
print(f'Mean Squared Error: {mse.item():.4f}')