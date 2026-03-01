# [Code] Baseline Models for Regression and Classification

## 🔬 Implementation Goals
- Implement a 3rd-degree polynomial regressor for Sine wave approximation.
- Build a Multi-Layer Perceptron (MLP) for California Housing price prediction.
- Develop a handwritten digit classifier using the MNIST dataset.

---

## 💻 Baseline Templates

### 1. Polynomial Regressor (Sine Wave)
```python
import torch
import torch.nn as nn

# y = ax^3 + bx^2 + cx + d
x = torch.linspace(-math.pi, math.pi, 1000)
y = torch.sin(x)

# Manual implementation of backprop logic
# grad_a = (2.0 * (y_pred - y) * x**3).sum()
```

### 2. Housing Price MLP
```python
model = nn.Sequential(
    nn.Linear(8, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### 3. MNIST Digit Classifier
```python
model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
# Use CrossEntropyLoss for multi-class classification
criterion = nn.CrossEntropyLoss()
```

---

## 🔗 Related Concepts
- [[3장_개념]] - PyTorch architecture and training loops.
