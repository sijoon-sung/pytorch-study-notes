# 💻 CHAP_03 실습 구현 (Code)

## 1. 사인 함수 예측 (3차 다항식 근사)

### ① 랜덤 가중치 초기화
```python
import math
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-math.pi, math.pi, 1000)
y = torch.sin(x)

# 가중치 초기화
a, b, c, d = torch.randn(()), torch.randn(()), torch.randn(()), torch.randn(())
y_random = a * x**3 + b * x**2 + c * x + d
```

### ② 학습 루프
```python
learning_rate = 1e-6 

for epoch in range(2000):
    y_pred = a * x**3 + b * x**2 + c * x + d
    loss = (y_pred - y).pow(2).sum().item()
    
    # 역전파(수동 미분)
    grad_y_pred = 2.0 * (y_pred - y) 
    grad_a = (grad_y_pred * x ** 3).sum()
    grad_b = (grad_y_pred * x ** 2).sum()
    grad_c = (grad_y_pred * x).sum()
    grad_d = grad_y_pred.sum()
    
    # 가중치 업데이트
    a -= learning_rate * grad_a 
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
```

## 2. 캘리포니아 집값 예측 (회귀)

### ① 데이터 준비 (Pandas)
```python
import pandas as pd
from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()
dataFrame = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
dataFrame["target"] = dataset["target"]
```

### ② 모델 학습
```python
import torch.nn as nn
from torch.optim.adam import Adam

model = nn.Sequential(
   nn.Linear(8, 100), nn.ReLU(), nn.Linear(100, 1)
)

X = dataFrame.iloc[:, :-1].values
Y = dataFrame["target"].values
optim = Adam(model.parameters(), lr=0.001)

for epoch in range(200):
   for i in range(len(X)//100):
       start, end = i*100, (i+1)*100
       x = torch.FloatTensor(X[start:end])
       y = torch.FloatTensor(Y[start:end]).view(-1, 1)

       optim.zero_grad()
       preds = model(x)
       loss = nn.MSELoss()(preds, y)
       loss.backward(); optim.step()
```

## 3. MNIST 손글씨 분류 (다중 분류)

### ① 데이터 로더 정의
```python
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader

training_data = MNIST(root="./", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
```

### ② 모델 학습 및 저장
```python
model = nn.Sequential(
    nn.Linear(784, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 10)
).to(device)

optim = Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    for data, label in train_loader:
        optim.zero_grad()
        data = torch.reshape(data, (-1, 784)).to(device)
        preds = model(data)
        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward(); optim.step()

torch.save(model.state_dict(), "MNIST.pth")
```
