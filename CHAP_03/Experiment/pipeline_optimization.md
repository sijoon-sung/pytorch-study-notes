# [Experiment] Pipeline Optimization and Generalization

## 🧪 Experiment Goals
- **Hyperparameter Sensitivity**: Analyze the impact of `learning_rate` and `batch_size` on convergence speed.
- **Normalization Analysis**: Compare model performance with and without feature scaling (Standardization/Min-Max) for Housing data.
- **Activation Function Efficiency**: Benchmark `ReLU` vs `Sigmoid` in the MNIST classifier regarding gradient vanishing.

---

## 🔬 Experimental Setup
- **Dataset**: MNIST, fetch_california_housing.
- **Metric**: Validation Accuracy, Training Loss Curve, Inference Latency.
- **Hardware**: CUDA-enabled GPU vs CPU performance comparison.

---

## 📊 Expected Results
- Adam optimizer will likely outperform SGD in terms of convergence speed due to adaptive moment estimation.
- Feature scaling will significantly reduce the initial loss and stabilize training for regression tasks.
- GPU acceleration (`.to(device)`) will show exponential speedup as the model width/depth increases.

---

## 🔗 Related Concepts
- [[3장_개념]] - Optimization algorithms and GPU handling.
