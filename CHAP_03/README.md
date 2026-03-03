# 📘 3장. 실습 첫걸음: 회귀와 다중분류

이번 장에서는 딥러닝의 가장 기본이 되는 **회귀(Regression)**와 **다중분류(Classification)** 문제를 직접 해결하며 파이토치의 기초를 다집니다.

---

# 3.1 사인 함수 예측하기

# 딥러닝 모델의 가중치(Weight)와 실습 첫걸음

> **💡 용어 사전: 모델 (Model)**
> 
> 딥러닝에서 '모델'은 **인공 신경망(Artificial Neural Network)** 과 같은 의미로 사용됩니다. 일반적으로 딥러닝 신경망을 줄여서 '딥러닝 모델'이라고 부릅니다.

## 1. 딥러닝 모델은 가중치를 몇 개나 가질까?

우리가 흔히 접하는 딥러닝 모델 안에는 상상을 초월하는 개수의 가중치(파라미터)가 들어 있습니다.

- **이미지 인식 모델:** 보통 **천만 개에서 1억 개**가 넘는 가중치를 가집니다.
- **거대 언어 모델 (예: OpenAI GPT-3):** 무려 **1,750억 개**의 가중치를 가집니다.

이 어마어마한 가중치의 개수는, 직관적으로 비유하자면 **'1,750억 차 방정식을 풀어야 한다'** 는 뜻과 같습니다. 컴퓨터가 이 수많은 미지수(가중치)를 스스로 조율하며 정답을 찾아가는 과정이 바로 딥러닝의 학습입니다.

## 2. 실습 첫걸음: 사인(Sine) 곡선 근사하기

걷지도 못하는 아이가 뛸 수는 없듯, 처음부터 1,750억 개의 파라미터를 가진 복잡한 모델을 만들 수는 없습니다. 따라서 기초 원리를 탄탄하게 다지기 위해 **아주 단순한 모델**부터 차근차근 만들어 봅니다.

- **실습 목표:** 친숙한 삼각함수인 **사인(Sine) 곡선을 3차 다항식으로 근사(Approximation)** 하기.
- **사용할 모델:** 3차 다항식 ($y = ax^3 + bx^2 + cx + d$)
- **가중치의 개수:** 단 **4개** ($a, b, c, d$라는 4개의 계수)

> **🎯 핵심 포인트**
> 
> 여기서 3차 다항식의 **계수(Coefficient)가 곧 인공 신경망의 가중치(Weight)** 역할을 합니다. 파라미터 4개짜리 아주 귀여운(?) 모델을 통해, 신경망이 어떻게 최적의 계수(가중치)를 찾아 곡선을 완벽하게 따라가는지 그 기본 원리를 직접 확인하게 될 것입니다.

# 🎯 실습 예제 소개: 사인 함수 예측

|**항목**|**상세 내용**|
|---|---|
|**문제 정의**|사인 함수를 3차 다항식의 계수를 이용해 예측합니다.|
|**난이도**|★☆☆☆☆|
|**이름**|사인 함수 예측|
|**알고리즘**|MLP|
|**데이터셋 소개**|별도의 데이터셋을 사용하지 않습니다. `linspace()` 함수를 이용해 사인 함수를 만드는 데 필요한 값을 직접 생성하겠습니다.|
|**문제 유형**|회귀 (Regression)|
|**평가지표**|평균 제곱 오차 (MSE, Mean Squared Error)|
|**주요 패키지**|`torch`, `torch.nn`|
|**예제 코드 노트**|• 위치: [colab 링크](https://colab.research.google.com/drive/1cu7iaM0Q6I3hlI6zcWtX7qAoT5cvlKw5)<br>• 단축 URL: [http://t2m.kr/F2hYp](http://t2m.kr/F2hYp)<br>• 파일: `ex3_1.ipynb`|

딥러닝을 구현하기 위해서는 크게 두 가지가 필요합니다. 
첫째는 우리가 학습시키고자 하는 **'모델(Model)'** 이고, 둘째는 모델이 어떻게 정답을 찾아갈지 정해주는 **'학습 루프(Training Loop)'** 입니다.

## 1. 딥러닝 모델의 기본 학습 루프 (Training Loop)

모델은 한 번에 정답을 맞히지 못합니다. 데이터를 여러 번 반복해서 보며 스스로 오차를 줄여나가는 과정을 거치는데, 이를 학습 루프라고 합니다.

![[Pasted image 20260302111136.png]]

1. **모델 정의 및 데이터 준비:** 딥러닝 모델을 정의하고, 데이터를 불러온 뒤 원하는 만큼 반복 학습을 시작합니다.
2. **순전파 (Forward Propagation):** 불러온 데이터를 이용해 모델의 예측값을 계산합니다. 데이터가 입력층에서 출력층 방향으로 흘러가며 계산되기 때문에 '순전파'라고 부릅니다.
3. **오차(Loss) 계산:** 모델이 내놓은 예측값과 실제 정답을 손실 함수(Loss Function)를 이용해 비교하고 오차를 계산합니다.
4. **역전파 (Backpropagation) 및 가중치 수정:** 오차를 뒤로(출력층에서 입력층 방향으로) 전달하며, 오차를 줄이기 위해 모델의 가중치(Weight)를 수정합니다.
5. **학습 종료:** 지정한 횟수만큼 반복(Epoch)했다면 학습을 마칩니다.

---

## 2. 실습: 랜덤하게 가중치를 적용해 사인곡선 그리기 (학습 전)

본격적인 학습 루프를 돌리기 전, 모델이 가장 처음 가지게 되는 '초기 상태(랜덤 가중치)'를 눈으로 확인해 보는 실습입니다.

- **목표:** $y = \sin(x)$ 곡선을 모델이 예측하도록 만들기.
- **사용할 모델:** 사인 함수를 근사하기 위한 **3차 다항식** ($y = ax^3 + bx^2 + cx + d$)
- **가중치(Weight):** 3차항의 계수부터 상수항까지 총 4개의 계수($a, b, c, d$)가 모델의 가중치가 됩니다. 처음에는 이 4개의 값을 랜덤하게 뽑아서 사용합니다.

### 💻 구현 코드

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

![[Pasted image 20260302112718.png]]

> _그래프 결과: 아직 가중치를 학습하기 전이라, 예측 그래프(`y pred`)는 실제 사인 함수(`y true`)와 완전히 다른 모양을 띱니다. 랜덤한 값이 들어갔기 때문에 코드를 실행할 때마다 다른 모양이 나옵니다._

---

## 3. 📚 코드 상세 분석 및 새로 등장한 함수 정리

### 1) 텐서(Tensor) 생성 및 수학 함수 (`torch`)

| **함수 원형**           | **설명**                                                                                                |
| ------------------- | ----------------------------------------------------------------------------------------------------- |
| `linspace(A, B, C)` | 시작점 A부터 종료점 B까지 **간격이 동일한** 데이터 C개를 반환합니다. (예: `-math.pi`부터 `math.pi`까지 1000개의 점 추출) |
| `sin(A)`            | 입력 A에 대한 사인(Sine) 함수의 결과값을 반환합니다.                                                                     |
| `randn()`           | **정규분포**를 따르는 랜덤한 값을 반환합니다. 괄호를 비워 `()`를 넣으면 스칼라값 1개를 뽑습니다.                                           |

### 2) 시각화 함수 (`matplotlib.pyplot`)

|**함수 원형**|**설명**|
|---|---|
|`subplot(pos)`|그림의 위치(`pos`)에 그래프를 지정해 줍니다. 여러 개의 그래프를 그릴 때 사용합니다.|
|`title(str)`|그림의 제목을 지정합니다.|
|`plot(x, y)`|x(입력값)와 y(함숫값)를 이용해 선 그래프를 그려줍니다.|
|`show()`|메모리에 그려진 그림을 최종적으로 화면에 출력해 줍니다.|

> **💡 [TIP] `subplot()`이 그래프를 배치하는 원리**
> 
> `plt.subplot(행의 개수, 열의 개수, 그릴 위치)` 형태로 인수를 전달합니다.
> 
> 예를 들어 `plt.subplot(2, 3, 5)`라고 작성하면, 전체 도화지를 2행 3열(총 6칸)로 나누고, 그중 5번째 위치에 그래프를 그리겠다는 뜻입니다.

---

# 3.1.2 가중치를 학습시켜서 사인곡선 그리기

앞서 정의한 4개의 랜덤 가중치($a, b, c, d$)를 업데이트하여, 3차 다항식 모델이 실제 사인 곡선과 유사해지도록 만드는 전체 학습 루프(Training Loop)입니다.

## 1. 학습 루프 및 시각화 전체 코드

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

# ----------------- 학습 결과 시각화 -----------------

# 1) 실제 사인 곡선 (y true)
plt.subplot(3, 1, 1)
plt.title("y true")
plt.plot(x, y)

# 2) 학습을 완료한 가중치의 예측 사인 곡선 (y pred)
plt.subplot(3, 1, 2)
plt.title("y pred")
plt.plot(x, y_pred)

# 3) 처음 랜덤하게 뽑았던 가중치의 사인 곡선 (y random)
plt.subplot(3, 1, 3)
plt.title("y random")
plt.plot(y_random)

# 그래프 화면에 출력
plt.show()
```

---

## 2. 🧠 핵심 코드 원리 파헤치기

코드가 이전보다 조금 복잡해졌지만, 딥러닝 학습의 4단계(예측 → 손실 → 미분 → 업데이트)를 그대로 따르고 있습니다.

### ① 손실(Loss) 정의하기

```python
loss = (y_pred - y).pow(2).sum().item()
```

오차를 계산하는 **제곱오차(Squared Error)** 방식입니다.
- **`pow(2)`:** 오차가 음수가 나오는 것을 막기 위해 제곱을 해줍니다.
- **`sum()`:** 모든 데이터 포인트(1000개)에서 발생한 오차를 하나로 다 합칩니다.
- **`item()`:** 파이토치 텐서(Tensor) 안에 들어있는 스칼라값을 일반적인 파이썬 실수(float) 값으로 꺼내옵니다.

### ② 수동으로 기울기(Gradient) 계산하기

```python
grad_y_pred = 2.0 * (y_pred - y) 
```

가중치를 업데이트하기 위해 손실값을 미분한 결과입니다.

### ③ 가중치 업데이트 (기울기의 반대 방향)

```python
a -= learning_rate * grad_a
```

- **왜 빼기(`-=`)를 할까?** 가중치는 오차가 줄어드는 '기울기의 반대 방향'으로 움직여야 합니다.
- **`learning_rate` (학습률):** 한 번에 얼마나 가중치를 수정할지 보폭을 결정합니다.

---

## 3. 결과 해석 및 한계점

위 코드를 실행하면 총 3개의 그래프가 나옵니다.
1. **y true:** 우리가 목표로 하는 완벽한 진짜 사인 곡선입니다.
2. **y pred:** 2000번의 학습을 거쳐 수정한 가중치($a, b, c, d$)로 그린 곡선입니다. 진짜 곡선과 거의 똑같이 맞춰진 것을 볼 수 있습니다.
3. **y random:** 학습 전, 랜덤 값이었을 때의 엉망진창 곡선입니다.

> **🚨 수동 계산의 한계와 다음 스텝**
> 
> 이번 실습에서는 4개의 가중치($a, b, c, d$)를 사용했기 때문에 사람이 직접 손으로 미분 공식을 쓰고 코드로 일일이 업데이트할 수 있었습니다. 하지만 앞서 배운 GPT-3 같은 모델은 **1,750억 개**의 가중치를 가집니다. 수백만 개의 가중치를 이렇게 수동으로 미분하고 업데이트하는 것은 불가능합니다.
> 
> 그래서 파이토치(PyTorch)에는 이 복잡한 미분과 가중치 계산을 알아서 다 해주는 **'자동 미분(Autograd)'** 기능이 존재합니다.

---

# 3.2 보스턴 집값 예측하기 : 회귀 분석 

이번에는 파이토치가 제공하는 신경망을 사용해 더 어려운 문제를 해결하겠습니다. 결과 예측에 사용되는 데이터 요소를 **‘특징 feature’** 이라고 부릅니다. 특징에 모델의 가중치를 반영해 결과를 도출합니다. 이번에는 집값만을 예측하기 때문에 출력은 하나만 나옵니다.

![[Pasted image 20260302135156.png]]

> **💡 [Gemini의 캔더(Candor) 팁] 왜 이름이 다를까요?**
> 
> 기존 머신러닝 교재들은 주로 '보스턴 집값 데이터'(`load_boston()`)를 사용했습니다. 하지만 이 데이터셋 내에 인종 차별적인 요소가 포함되어 있어, 최근 Scikit-learn 라이브러리에서는 이를 대체하기 위해 **캘리포니아 집값 데이터(`fetch_california_housing()`)** 를 사용하도록 업데이트되었답니다!

# 3.2.1 데이터 살펴보기

Python

```python
from sklearn.datasets import fetch_california_housing
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 캘리포니아 데이터셋 불러오기
dataset = fetch_california_housing()

# 데이터셋이 가지고 있는 키(key) 출력
print(dataset.keys())
# 결과: dict_keys(['data', 'target', 'frame', 'target_names', 'feature_names', 'DESCR'])
```

### 🔑 데이터셋의 Key 구성 요소

| **키 이름 (Key)**      | **설명**                                | **상세 내용 (캘리포니아 데이터 기준)**                  |
| ------------------- | ------------------------------------- | ----------------------------------------- |
| **`data`**          | 우리가 모델 학습에 사용할 **입력 특징 데이터** | 20,640개의 샘플 × 8개의 특성(Feature)             |
| **`target`**        | 모델이 맞혀야 할 **예측 정답값**                  | 집값 (단위: 100,000 달러)                       |
| **`feature_names`** | `data`에 들어있는 각 특성의 이름 리스트             | `['MedInc', 'HouseAge', 'AveRooms', ...]` |
| **`target_names`**  | 타겟 값의 이름 (보통 회귀에서는 잘 안 씀)             | 집값 척도 이름                                  |
| **`DESCR`**         | 데이터셋에 대한 전반적인 설명 (Description)        | 데이터 수집 배경 등 설명 문자열                        |

---

## 💡 [Box] 데이터를 다루는 방법 : 판다스(Pandas)와 PIL

- **PIL (Python Imaging Library):** 이미지 데이터 처리에 특화.
- **Pandas (판다스):** 엑셀 파일처럼 **테이블 형식의 데이터**를 다루는 데 특화. 데이터를 행렬 구조인 **데이터프레임(DataFrame)** 형태로 저장합니다.

---

# 3.2.2 데이터 불러오기 (판다스 활용)

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing

# 1. 데이터셋 불러오기
dataset = fetch_california_housing()

# 2. 특징(data) 데이터만 가져와서 판다스 데이터프레임으로 변환
dataFrame = pd.DataFrame(dataset["data"])

# 3. 데이터프레임의 열(Column) 이름을 특징 이름(feature_names)으로 덮어쓰기
dataFrame.columns = dataset["feature_names"]

# 4. 데이터프레임의 맨 끝에 "target"이라는 이름으로 정답(집값) 열 추가
dataFrame["target"] = dataset["target"]

# 5. 데이터프레임이 잘 만들어졌는지 상위 5개 행만 요약해서 출력
print(dataFrame.head())
```

![[Pasted image 20260302140225.png]]

- **`dataset["data"]` & `pd.DataFrame()`:** 데이터를 표 형태의 데이터프레임으로 변환합니다.
- **`dataFrame.columns`:** 각 열에 특징 이름표를 붙여줍니다.
- **`dataFrame.head()`:** 일부 데이터(기본 5개)만 요약 출력합니다.

---

# 3.2.3 모델 정의 및 학습하기

## 1. 선형 회귀(Linear Regression)와 평균 제곱 오차(MSE)

- **선형 회귀:** 데이터를 $y$와 $x$의 관계를 나타내는 '직선'으로 나타내어 예측하는 방법.
- **학습 방식:** 예측값과 실제 정답을 비교해 오차를 줄여나가는 방식.
- **MSE (Mean Squared Error):** 오차에 제곱을 취한 뒤 평균을 낸 값. 회귀 문제에서 유용하게 쓰입니다.

![[Pasted image 20260302140702.png]]

---

## 2. 파이토치를 이용한 다층 신경망(MLP) 정의

- **`torch.nn.Sequential()`:** 모듈(층)을 순서대로 집어넣는 컨테이너.
- **`nn.Linear`:** 선형 회귀 모델을 구현할 때 사용하는 파이토치 모듈.
- **MLP (다층 신경망):** 각 층의 뉴런이 다음 층과 빠짐없이 연결된 **완전연결층(FC)** 구조.

![[Pasted image 20260302141657.png]]

---

## 3. 딥러닝 필수 학습 단위: 배치, 에포크, 이터레이션

|**용어**|**상세 설명**|
|---|---|
|**배치 (Batch)**|가중치 업데이트 시 사용하는 **데이터의 묶음 단위**. (예: 100개씩 처리)|
|**에포크 (Epoch)**|배치 크기 단위로 쪼개어, **전체 데이터를 모두 한 번씩 다 학습**하는 단위.|
|**이터레이션 (Iteration)**|1에포크를 완성하는 데 필요한 **배치의 반복 횟수**.|

---

## 4. 실습 코드 (모델 정의 및 학습)

```python
import torch
import torch.nn as nn
from torch.optim.adam import Adam

# ❶ 모델 정의 (다층 신경망)
model = nn.Sequential(
   nn.Linear(8, 100),
   nn.ReLU(),
   nn.Linear(100, 1)
)

X = dataFrame.iloc[:, :-1].values # ❷ 정답을 제외한 특징
Y = dataFrame["target"].values    # 정답 값 추출

batch_size = 100
learning_rate = 0.001

# ❸ 최적화 정의 (Adam)
optim = Adam(model.parameters(), lr=learning_rate)

# 에포크 반복
for epoch in range(200):
   # 배치 반복
   for i in range(len(X)//batch_size):
       start = i*batch_size 
       end = start + batch_size

       # 텐서 변환
       x = torch.FloatTensor(X[start:end])
       y = torch.FloatTensor(Y[start:end]).view(-1, 1) # 정답 모양 맞추기 (수정됨)

       optim.zero_grad()             # ❺ 기울기 초기화
       preds = model(x)              # ❻ 모델의 예측값 계산
       loss = nn.MSELoss()(preds, y) # ❼ MSE 손실 계산
       loss.backward()               # 오차 역전파
       optim.step()                  # 가중치 업데이트

   if epoch % 20 == 0:
       print(f"epoch {epoch} loss:{loss.item()}")
```

---

# 3.2.4 모델 성능 평가하기

무작위로 하나의 행을 추출하여 실제 정답과 모델의 예측값을 비교해 봅니다.

```python
import torch

# 1. 첫 번째 데이터의 특징 추출 후 텐서로 변환 (캘리포니아 기준 특징 8개)
prediction = model(torch.FloatTensor(X[0, :8])) 

# 2. 실제 정답
real = Y[0]

# 3. 결과 출력
print(f"prediction: {prediction.item()} / real: {real}")
```

> **🚨 [Trouble Shooting]** 
> 구버전 교재 코드 `X[0, :13]`은 보스턴 데이터 기준입니다. 현재 캘리포니아 데이터는 특징이 **8개**이므로 `X[0, :8]`로 수정해야 에러가 발생하지 않습니다.

---

# 3.3 손글씨 분류하기 : 다중분류

하나의 데이터로 여러 가지를 동시에 예측하는 모델을 설계합니다. 인공 신경망의 출력 노드는 원하는 만큼 늘릴 수 있습니다.

![[Pasted image 20260302145020.png]]

## 💡 회귀(Regression) vs 분류(Classification)

|**구분**|**회귀 모델**|**분류 모델**|
|---|---|---|
|**출력 방식**|출력값을 **그대로 사용**|출력을 **확률 분포로 바꿔서 사용**|
|**최종 처리**|별도 변환 없음|**소프트맥스(Softmax) 함수** 통과|

> **🎯 소프트맥스(Softmax)를 쓰는 이유:** 모든 출력값의 범위를 0과 1 사이로 제한하고 합을 1로 만들어, "각 범주에 속할 확률"로 해석하기 위해 사용합니다.

![[Pasted image 20260302145154.png]]

---

# 3.3.1 데이터 살펴보기 : MNIST 데이터셋

MNIST는 0~9까지 10가지 손글씨 숫자 이미지 데이터셋입니다.

```python
import matplotlib.pyplot as plt
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

# 1. 학습용 및 평가용 데이터 다운로드
training_data = MNIST(root="./", train=True, download=True, transform=ToTensor())
test_data = MNIST(root="./", train=False, download=True, transform=ToTensor())

# 2. 시각화
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(training_data.data[i], cmap='gray')
plt.show()
```

- **`transform=ToTensor()`:** PIL 이미지나 넘파이 배열을 **파이토치 텐서**료 변환하고 픽셀값을 0~1 사이로 정규화합니다.

---

# 3.3.2 데이터 로더(DataLoader) 정의하기

```python
from torch.utils.data.dataloader import DataLoader

# 1. 학습용 데이터 로더: 배치 32, 섞기(True)
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)

# 2. 평가용 데이터 로더: 배치 32, 안 섞음(False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
```

- **학습용 섞는 이유:** 특정 순서에 편향되게 학습하는 것을 방지하기 위함입니다.

---

# 3.3.3 모델 정의 및 학습하기 (손글씨 다중분류)

## 🚨 [핵심] 2차원 이미지를 1차원으로 펴기 (Flatten)
MNIST 흑백 이미지는 `28×28` (784픽셀)의 2차원 데이터입니다. MLP는 1차원 배열(벡터)만 입력받을 수 있으므로 `784` 길이로 쭉 펴주는 작업이 필수입니다.

![[Pasted image 20260302150322.png]]

```python
import torch
import torch.nn as nn
from torch.optim.adam import Adam

# 1. GPU/CPU 지정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 다중분류 모델 정의 (출력 노드 10개)
model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
model.to(device)

# 3. 최적화 설정
optim = Adam(model.parameters(), lr=1e-3)

# 4. 학습 루프 (20 에포크)
for epoch in range(20):
    for data, label in train_loader:
        optim.zero_grad()
        
        # Flatten: (-1, 784)로 모양 변경
        data = torch.reshape(data, (-1, 784)).to(device)
        preds = model(data)
        
        # 다중분류용 손실 함수 (CrossEntropyLoss)
        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        
        loss.backward()
        optim.step()
        
    print(f"epoch {epoch+1} loss:{loss.item()}")

# 5. 가중치 저장
torch.save(model.state_dict(), "MNIST.pth")
```

---

# 3.3.4 모델 성능 평가하기

```python
# 1. 가중치 불러오기
model.load_state_dict(torch.load("MNIST.pth", map_location=device))

num_corr = 0

# 2. 평가 모드 (기울기 계산 끔)
with torch.no_grad(): 
    for data, label in test_loader:
        data = torch.reshape(data, (-1, 784)).to(device)
        output = model(data.to(device))
        
        # 3. 가장 큰 확률의 인덱스 추출
        preds = output.data.max(1)[1] 
        
        # 4. 정답 일치 개수 합산
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr

# 최종 정확도
print(f"Accuracy: {num_corr / len(test_data)}")
```

### 🧠 중요 개념
- **`with torch.no_grad():`** 평가 시 기울기 트래킹을 멈춰 메모리와 속도를 개선합니다.
- **`max(1)[1]`:** 확률 분포 중 가장 높은 값의 인덱스(예측 숫자)를 선택합니다.
- **`Accuracy` 기준:** 92% 이상이면 훌륭한 모델, 80% 미만이면 개선이 필요한 수준입니다.

---
https://colab.research.google.com/drive/1d-JXJH_VEHUGedh_O8eWKwdR6_-d3P5M?usp=sharing
