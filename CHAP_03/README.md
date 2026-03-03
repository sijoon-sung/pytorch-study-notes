# 📘 CHAP_03: 실습 첫걸음 (회귀와 다중분류)

이번 장에서는 딥러닝의 기초인 **회귀(Regression)**와 **다중분류(Classification)**를 사인 함수, 집값 데이터, MNIST 손글씨 데이터를 통해 직접 구현하며 학습합니다.

---

## 🟢 3.1 사인 함수 예측하기 (3차 다항식 근사)

### 💡 핵심 개념: 모델과 가중치
- **모델(Model):** 인공 신경망(ANN)의 또 다른 이름.
- **가중치(Weight):** 딥러닝이 스스로 학습하며 조절하는 미지수. GPT-3는 무려 1,750억 개의 가중치를 가집니다.
- **실습 목표:** 사인 곡선을 3차 다항식($y = ax^3 + bx^2 + cx + d$)으로 근사하며, 4개의 계수($a, b, c, d$)를 가중치로 학습시킵니다.

### ✅ STEP 1: 랜덤 가중치로 곡선 그리기 (학습 전)
학습 전에는 가중치가 랜덤하게 설정되어 실제 사인 곡선과 완전히 다른 모양이 나옵니다.

```python
# 필요한 라이브러리 불러오기
import math # 수학 패키지
import torch # 파이토치 모듈
import matplotlib.pyplot as plt # 시각화 라이브러리

# 1. -pi부터 pi 사이에서 일정한 간격의 점 1,000개 추출 (입력 데이터 x)
x = torch.linspace(-math.pi, math.pi, 1000)

# 2. 실제 사인곡선에서 추출한 값으로 정답(y true) 만들기
y = torch.sin(x)

# 3. 예측 사인곡선에 사용할 임의의 가중치(계수) 4개 뽑기
a = torch.randn(())
b = torch.randn(())
c = torch.randn(())
d = torch.randn(())

# 사인 함수를 근사할 3차 다항식 정의 (초기 예측값, y pred)
y_random = a * x**3 + b * x**2 + c * x + d

# 4. 실제 사인곡선 그리기 (위쪽)
plt.subplot(2, 1, 1) # 2행 1열의 첫 번째 영역
plt.title("y true")
plt.plot(x, y)

# 5. 예측 사인곡선 그리기 (아래쪽)
plt.subplot(2, 1, 2) # 2행 1열의 두 번째 영역
plt.title("y pred")
plt.plot(x, y_random)

# 6. 실제와 예측 사인곡선 화면에 출력하기
plt.show()
```

### ✅ STEP 2: 가중치 학습 루프 (Training Loop)
4단계(예측 → 손실 → 미분 → 업데이트)를 통해 실제 사인 곡선과 유사해지도록 학습합니다.

```python
# 학습률 정의 (보폭 지정)
learning_rate = 1e-6 

# 총 2,000번의 반복 학습 진행
for epoch in range(2000):
    # 1. 순전파: 현재 가중치로 예측값 계산
    y_pred = a * x**3 + b * x**2 + c * x + d
    
    # 2. 손실 계산: 예측값과 실제 정답(y)의 오차(제곱오차) 계산
    loss = (y_pred - y).pow(2).sum().item() 
    
    # 100번 에포크마다 현재 손실값 출력
    if epoch % 100 == 0:
        print(f"epoch {epoch+1} loss: {loss}")
        
    # 3. 역전파(수동): 손실값을 미분하여 각 가중치의 기울기(Gradient) 계산
    grad_y_pred = 2.0 * (y_pred - y) 
    grad_a = (grad_y_pred * x ** 3).sum()
    grad_b = (grad_y_pred * x ** 2).sum()
    grad_c = (grad_y_pred * x).sum()
    grad_d = grad_y_pred.sum()
    
    # 4. 가중치 업데이트: 기울기의 반대 방향으로 학습률만큼 이동
    a -= learning_rate * grad_a 
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
```

---

## 🟢 3.2 캘리포니아 집값 예측하기 (회귀 분석)

### 💡 핵심 개념: MLP(Multi-Layer Perceptron)
- **특징(Feature):** 결과 예측에 사용되는 데이터 요소(방 개수, 인구수, 소득 수준 등).
- **완전연결층(FC Layer):** 이전 층의 모든 노드와 다음 층의 모든 노드가 연결된 구조.
- **학습 단위:** 
  - **배치(Batch):** 가중치 업데이트 시 사용하는 데이터 묶음 단위.
  - **에포크(Epoch):** 전체 데이터를 한 번씩 다 학습하는 단위.
  - **이터레이션(Iteration):** 1에포크를 완성하는 데 필요한 배치의 반복 횟수.

### ✅ STEP 1: 데이터 불러오기 및 살펴보기 (Pandas)
기존 보스턴 데이터셋의 인종 차별적 요소를 배제하기 위해 최신 환경에서는 **캘리포니아 집값 데이터**를 사용합니다.

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing

# 1. 데이터셋 불러오기
dataset = fetch_california_housing()

# 2. 판다스 데이터프레임으로 변환 및 정답(target) 열 추가
dataFrame = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
dataFrame["target"] = dataset["target"]

# 3. 상위 5개 행 출력
print(dataFrame.head())
```

### ✅ STEP 2: 모델 정의 및 학습 (PyTorch)
`nn.Sequential`을 통해 층을 쌓고 `Adam` 최적화기를 활용합니다.

```python
import torch
import torch.nn as nn
from torch.optim.adam import Adam

# ❶ 모델 정의 (다층 신경망, MLP)
model = nn.Sequential(
    nn.Linear(8, 100), # 입력층(특징 8개) -> 은닉층(노드 100개)
    nn.ReLU(),         # 활성화 함수
    nn.Linear(100, 1)  # 은닉층 -> 출력층(결과값 1개)
)

X = dataFrame.iloc[:, :-1].values # 특징
Y = dataFrame["target"].values    # 정답

# 최적화 기법 정의 (Adam)
learning_rate = 0.001
optim = Adam(model.parameters(), lr=learning_rate)

# 학습 루프 (200 에포크)
for epoch in range(200):
    for i in range(len(X) // 100): # 배치 크기 100
        start, end = i * 100, (i + 1) * 100
        x = torch.FloatTensor(X[start:end])
        y = torch.FloatTensor(Y[start:end]).view(-1, 1) # 정답 모양 맞추기

        optim.zero_grad()                 # 기울기 초기화
        preds = model(x)                  # 순전파
        loss = nn.MSELoss()(preds, y)     # 손실 계산 (MSE)
        loss.backward()                   # 역전파
        optim.step()                      # 가중치 업데이트

    if epoch % 20 == 0:
        print(f"epoch {epoch} loss:{loss.item()}")
```

---

## 🟢 3.3 MNIST 손글씨 분류하기 (다중 분류)

### 💡 핵심 개념: 분류와 소프트맥스
- **분류(Classification):** 출력을 확률 분포로 바꾸어(Softmax) 어떤 카테고리에 속할지 결정.
- **Flatten (평탄화):** 28x28의 2차원 이미지를 784개의 1차원 데이터로 변환.
- **CrossEntropyLoss:** 분류 문제에서 정답 확률과 예측 확률의 차이를 줄이는 손실 함수.

### ✅ STEP 1: MNIST 데이터셋 및 로더 정의
`torchvision`을 사용해 데이터를 내려받고 `ToTensor()`로 텐서 변환 및 정규화를 수행합니다.

```python
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader

# 데이터셋 다운로드
training_data = MNIST(root="./", train=True, download=True, transform=ToTensor())
test_data = MNIST(root="./", train=False, download=True, transform=ToTensor())

# 데이터 로더 정의 (배치 32)
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
```

### ✅ STEP 2: GPU 기반 모델 학습 및 저장
이미지 데이터는 연산량이 많으므로 `cuda` 장치를 활용합니다.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 정의
model = nn.Sequential(
    nn.Linear(784, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 10) # 0~9 숫자 분류
).to(device)

optim = Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    for data, label in train_loader:
        optim.zero_grad()
        data = torch.reshape(data, (-1, 784)).to(device) # Flatten
        preds = model(data)
        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

# 가중치 저장
torch.save(model.state_dict(), "MNIST.pth")
```

### ✅ STEP 3: 모델 성능 평가
`with torch.no_grad()`로 기울기 계산을 제외하여 메모리를 절약하고 정확도를 측정합니다.

```python
model.load_state_dict(torch.load("MNIST.pth", map_location=device))
num_corr = 0

with torch.no_grad():
    for data, label in test_loader:
        data = torch.reshape(data, (-1, 784)).to(device)
        output = model(data)
        preds = output.data.max(1)[1] # 가장 높은 확률의 클래스(숫자) 선택
        num_corr += preds.eq(label.to(device).data).sum().item()

print(f"Accuracy: {num_corr / len(test_data)}") # 목표: 92% 이상 (실제 약 97%)
```

---
*본 문서는 [텐초의 파이토치] 3장 실습 내용을 바탕으로 정리되었습니다.*
