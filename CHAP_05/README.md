# 📘 CHAP_05: ResNet과 배치 정규화

이번 장에서는 스킵 커넥션을 사용하는 CNN 모델인 **ResNet**을 알아보고, 직접 구현하여 CIFAR-10 데이터를 학습해 봅니다.

---

## 🟢 5.1 이해하기: ResNet

### 💡 1. ResNet의 등장 배경: "깊을수록 좋은 게 아니었어?"
- **기울기 소실(Gradient Vanishing):** 층이 깊어질수록 입력층에 가까운 가중치의 기울기가 0에 수렴하여 학습이 멈추는 문제.
- **정보 압축의 한계:** 층을 거칠 때마다 픽셀이 압축되어 원본 이미지의 디테일한 정보가 뭉개지는 현상.

### 💡 2. 스킵 커넥션(Skip Connection): 왜 압도적으로 좋은가?
![[Pasted image 20260303112649.png]]
- **수학적인 생명줄:** 입력값 $x$를 미분하면 1이 되므로, 아무리 깊어져도 최소 1이라는 기울기가 보장됩니다.
- **잔차 학습(Residual Learning):** 신경망이 정답 $H(x)$ 대신 $x$와의 차이(잔차)인 $F(x) = H(x) - x$만 학습하도록 하여 목표를 뚜렷하게 만듭니다.

### 💡 3. 비유: 요리로 이해하는 스킵 커넥션
- **기존 모델:** 원재료로 새로운 요리를 처음부터 끝까지 창조 (부담 큼).
- **ResNet 모델:** 밀키트(원재료 $x$)를 한쪽에 두고, 부족한 간(잔차 $F(x)$)만 맞춤 (학습이 쉬움).

---

## 🟢 5.2 이해하기: 배치 정규화 (Batch Normalization)

### 💡 필요성: 데이터의 불균형 해결
배치마다 데이터의 분포(Scale)가 다르면 모델이 학습 시 혼란을 겪습니다. (예: 방의 개수 1~5 vs 대지 면적 30~3,000)

### 💡 효과
- **안정성:** 배치마다 들어오는 값들을 평균 0, 분산 1 근처로 강제로 모아줍니다.
- **속도 향상:** 학습 속도가 빨라지고 초기 가중치 설정에 덜 민감해집니다.

![[Pasted image 20260226172126.png]]

---

## 🟢 5.4 ResNet 모델 정의하기

### 💡 1. ResNet 기본 블록 (Basic Block)
`nn.Sequential`의 한계를 극복하기 위해 `nn.Module`을 상속받아 복잡한 데이터 흐름(스킵 커넥션)을 제어합니다.

#### ✅ 1) 다운샘플 (Downsample)
스킵 커넥션에서 채널 수가 다른 입력값과 합성곱 결과를 더해주기 위해 $1 	imes 1$ 합성곱을 사용하여 규격을 맞춰줍니다.

#### ✅ 2) 기본 블록 구현 (BasicBlock)

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BasicBlock, self).__init__()
        
        # 1. 메인 합성곱층 (특징 추출용)
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        
        # 2. 다운샘플층 (스킵 커넥션 채널 맞춤용)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # 3. 배치 정규화층
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x_ = x # 스킵 커넥션을 위한 백업
        
        # 메인 경로
        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)
        
        # 지름길 경로 (채널 맞춤)
        x_ = self.downsample(x_)
        
        # 합산 및 최종 활성화
        x += x_
        x = self.relu(x)
        
        return x
```

### 💡 2. 전체 ResNet 모델 조립
특징 추출(기본 블록 3회) + 평탄화(Flatten) + 분류(Classifier)로 구성됩니다.

#### ✅ 전체 모델 구현

```python
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        
        # 특징 추출기 (채널 수를 점진적으로 확장)
        self.b1 = BasicBlock(in_channels=3, out_channels=64)
        self.b2 = BasicBlock(in_channels=64, out_channels=128)
        self.b3 = BasicBlock(in_channels=128, out_channels=256)
        
        # 평균 풀링 (정보 손실 감소)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 최종 분류기
        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # 특징 추출 사이클
        x = self.b1(x); x = self.pool(x)
        x = self.b2(x); x = self.pool(x)
        x = self.b3(x); x = self.pool(x)
        
        # 평탄화 및 분류
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x); x = self.relu(x)
        x = self.fc2(x); x = self.relu(x)
        x = self.fc3(x)
        
        return x
```

---

## 🟢 5.5 모델 학습하기

### 💡 1. 데이터 증강 및 로드
`RandomCrop`, `RandomHorizontalFlip`을 통해 모델의 실전 강인함을 높입니다.

```python
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader

transforms = Compose([
    RandomCrop((32, 32), padding=4),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

train_loader = DataLoader(CIFAR10(root="./", train=True, download=True, transform=transforms), batch_size=32, shuffle=True)
```

### 💡 2. 학습 루프
30 에포크 동안 Adam 최적화기를 사용하여 학습하고 결과를 저장합니다.

```python
import tqdm
from torch.optim.adam import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet(num_classes=10).to(device)
optim = Adam(model.parameters(), lr=1e-4)

for epoch in range(30):
    for data, label in tqdm.tqdm(train_loader):
        optim.zero_grad()
        preds = model(data.to(device))
        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

torch.save(model.state_dict(), "ResNet.pth")
```

---

## 🟢 5.6 모델 성능 평가하기

### 💡 1. 실전 평가 및 정확도
`torch.no_grad()`를 사용하여 기울기 계산을 끄고 CIFAR-10 테스트 데이터셋에 대한 정확도를 측정합니다.

```python
model.load_state_dict(torch.load("ResNet.pth", map_location=device))
num_corr = 0

with torch.no_grad():
    for data, label in test_loader:
        output = model(data.to(device))
        preds = output.data.max(1)[1]
        num_corr += preds.eq(label.to(device).data).sum().item()

print(f"Accuracy: {num_corr / len(test_data)}") # 목표 약 88.2%
```

### 💡 2. 딥러닝 핵심 교훈
1. **뼈대(구조)의 힘:** 기본 CNN(100회 학습, 83%)보다 ResNet(30회 학습, 88%)의 성능이 우월합니다.
2. **거인의 어깨 (전이 학습):** 직접 설계한 ResNet보다 사전 학습된 VGG(92.7%)의 성능이 좋은 경우가 많습니다. 전이 학습은 가장 가성비 좋은 필살기입니다.

---
*본 문서는 [텐초의 파이토치] 5장 실습 내용을 바탕으로 정리되었습니다.*
