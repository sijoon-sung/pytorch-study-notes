# [Experiment] Optimization and Activation Analysis

## 🧪 Experiment Goals
- **Learning Rate($\alpha$) 민감도 테스트**: 학습률이 너무 클 때(발산)와 작을 때(지연)의 손실 곡선 비교.
- **Activation Function 비교**: 깊은 층(Deep)에서 Sigmoid 사용 시 실제 기울기 값이 얼마나 빨리 0에 수렴하는지 측정 (Vanishing Gradient 시각화).
- **Architecture Depth vs Width**: 실무 TIP에서 언급된 OOM 대응을 위해 층의 깊이와 너비가 메모리 사용량에 미치는 영향 분석.

---

## 🔬 Experimental Setup
- Dataset: Synthetic XOR Data or MNIST for gradient flow analysis.
- Model: Deep MLP with variable activation functions.
- Metric: Gradient norm per layer, Training Loss, Memory Footprint.

---

## 📊 Expected Results
- Sigmoid activation will show significant gradient vanishing as depth increases (>10 layers).
- ReLU will maintain stable gradient flow even in deeper networks.
- Memory usage will increase linearly with layer depth due to intermediate activation storage.

---

## 🔗 Related Concepts
- [[2장_개념]] - 경사 하강법 및 기울기 소실 이론
