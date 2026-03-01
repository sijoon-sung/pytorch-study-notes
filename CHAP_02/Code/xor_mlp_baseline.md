# [Code] XOR MLP Baseline Implementation

## 🔬 Implementation Goals
- `nn.Module`을 상속받은 `XORNet` 클래스 설계.
- 2.3절의 가중치($w$)와 편향($b$)을 직접 주입하여 순전파 결과 검증 (`manual_weight_init`).
- `nn.Sequential`을 이용한 은닉층 구성 및 `ReLU`, `Sigmoid` 적용 모듈화.

---

## 💻 Baseline Code Template

```python
import torch
import torch.nn as nn

class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        # TODO: Define layers based on Chapter 2.3 specifications
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass

# Verification
if __name__ == "__main__":
    # TODO: Add manual weight initialization and testing logic
    print("XOR MLP Baseline created.")
```

---

## 🔗 Related Concepts
- [[2장_개념]] - 퍼셉트론 및 다층 신경망 이론
