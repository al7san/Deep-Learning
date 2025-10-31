# 🧠 Week 6 — Hardware and Software for Deep Learning

> Based on *Justin Johnson’s Lecture 9 (2019): Hardware and Software for Deep Learning*  
> Expanded with detailed hardware comparisons, framework analysis, and interactive review questions.

---

## 1. Introduction — Why Hardware Matters

Deep learning requires enormous computational power for training large neural networks.  
The choice of hardware — **CPU**, **GPU**, or **TPU** — directly affects training speed, energy cost, and scalability.

💡 **Key idea:**  
Neural network training is dominated by *matrix multiplications*, which are highly parallelizable.  
Thus, hardware that supports parallel computation performs dramatically faster.

---

## 2. CPU vs GPU — Parallel vs Sequential Computation

| Feature | CPU | GPU |
|----------|------|------|
| **Cores** | 4–16 | 1000+ (smaller cores) |
| **Clock Speed** | 3–4 GHz | 1–1.5 GHz |
| **Parallelism** | Low (few threads) | Massive (thousands of threads) |
| **Memory Bandwidth** | Limited | High |
| **Task Type** | Sequential logic | Parallel numerical computation |
| **Ideal For** | Inference, lightweight tasks | Training large deep models |

📘 **In short:**  
CPUs are versatile and good at general tasks,  
while GPUs are optimized for running many operations at once — perfect for matrix-heavy deep learning.

---

## 3. Inside a GPU — SMs, Tensor Cores, and Parallelism

A **GPU (Graphics Processing Unit)** consists of many **Streaming Multiprocessors (SMs)**,  
each with hundreds of small cores capable of running thousands of threads in parallel.  

Modern GPUs (e.g., *NVIDIA Titan RTX*) have:
- 72 SMs  
- 4608 CUDA cores  
- 576 Tensor Cores  
- Memory Bandwidth ≈ 672 GB/s  

Tensor Cores accelerate **matrix multiplications** and **mixed-precision** arithmetic.

<details>
<summary>🧩 Example: Checking GPU in PyTorch</summary>

```python
import torch
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0))
````

</details>

---

## 4. TPUs — Google’s Specialized Accelerators

**TPUs (Tensor Processing Units)** are custom ASICs designed by Google
specifically for accelerating TensorFlow computations.

| Version    | Performance | Memory     | Cloud Cost |
| ---------- | ----------- | ---------- | ---------- |
| **TPU v2** | 45 TFLOPs   | 64 GB HBM  | ~$4.50/hr  |
| **TPU v3** | 90 TFLOPs   | 128 GB HBM | ~$8.00/hr  |

📘 **Use case:**
When working with Google Cloud or massive production-scale models,
TPUs can train models faster and more efficiently than GPUs.

---

## 5. Deep Learning Frameworks

Frameworks simplify model building, training, and deployment.
They handle **automatic differentiation**, **GPU acceleration**, and **I/O pipelines**.

### 🧩 Major Frameworks Overview

| Framework      | Language     | Notable Features                    |
| -------------- | ------------ | ----------------------------------- |
| **PyTorch**    | Python       | Dynamic graphs, research-friendly   |
| **TensorFlow** | Python / C++ | Graph execution, production-ready   |
| **MXNet**      | Python / C++ | Lightweight, efficient for edge use |
| **Caffe**      | C++          | Fast, used for classic CNNs         |
| **CNTK**       | C++ / .NET   | Microsoft’s early framework         |

---

### ⚖️ PyTorch vs TensorFlow Comparison

| Feature               | **PyTorch**                 | **TensorFlow**                         |
| --------------------- | --------------------------- | -------------------------------------- |
| **Ease of Use**       | ⭐⭐⭐⭐⭐ Dynamic, Pythonic     | ⭐⭐⭐ More setup required                |
| **Community Support** | ⭐⭐⭐⭐                        | ⭐⭐⭐⭐⭐                                  |
| **Deployment**        | Moderate (TorchServe, ONNX) | Excellent (TensorFlow Serving, TFLite) |
| **Best For**          | Research, experimentation   | Production, cloud scaling              |
| **Autograd Engine**   | Dynamic                     | Static (with optional eager mode)      |

---

## 6. Mixed Precision and Memory Optimization

Training with 32-bit floating-point (FP32) is accurate but memory-intensive.
**Mixed Precision Training** uses both **FP16 (half precision)** and **FP32**
to accelerate computations and reduce memory usage.

💡 **Benefits:**

* Faster computation on GPUs with Tensor Cores.
* Less GPU memory usage → larger batch sizes.

<details>
<summary>🧩 Example: Enable Mixed Precision in PyTorch</summary>

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, labels in data_loader:
    optimizer.zero_grad()
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

</details>

---

## 7. Deployment & Cost Trade-offs

Choosing the right hardware depends on **budget, speed, and scalability**.

| Scenario                 | Best Hardware | Reason                            |
| ------------------------ | ------------- | --------------------------------- |
| Research & Prototyping   | GPU (local)   | Flexible and fast experimentation |
| Large-Scale Training     | TPU (cloud)   | Faster throughput, scalable       |
| Edge or Mobile Inference | CPU           | Cheaper, low power consumption    |

💡 *Rule of thumb:*
Train on **GPU/TPU**, infer on **CPU**.

---

## 8. Comparison Table: CPU vs GPU vs TPU

| Feature          | **CPU**                 | **GPU**       | **TPU**              |
| ---------------- | ----------------------- | ------------- | -------------------- |
| Cores            | 4–16                    | 1000+         | 64–128               |
| Clock Speed      | High                    | Medium        | Medium               |
| Parallelism      | Low                     | High          | Very High            |
| Memory Bandwidth | Low                     | High          | Very High            |
| Power Efficiency | Moderate                | High          | High                 |
| Cost             | $                       | $$            | $$$                  |
| Flexibility      | High                    | Medium        | Low                  |
| Ideal For        | Inference, small models | Training CNNs | Large-scale training |

---

## 🧩 Interactive Quiz – Week 6 Review

### 💬 Question 1

Why are GPUs preferred for deep learning training instead of CPUs?

<details>
<summary>✅ Show Answer</summary>
Because GPUs contain thousands of small cores optimized for parallel matrix operations,  
making them ideal for handling the massive computations in neural networks.
</details>

---

### 💬 Question 2

What is the main advantage of TPUs over GPUs?

<details>
<summary>✅ Show Answer</summary>
TPUs are specialized hardware for TensorFlow workloads.  
They achieve higher throughput and power efficiency for large-scale training in the cloud.
</details>

---

### 💬 Question 3

What is the difference between PyTorch and TensorFlow in terms of computation graphs?

<details>
<summary>✅ Show Answer</summary>
PyTorch uses **dynamic computation graphs** (define-by-run),  
while TensorFlow originally used **static graphs** (define-and-run).  
Dynamic graphs are more intuitive for research.
</details>

---

### 💬 Question 4

Why is mixed precision training useful?

<details>
<summary>✅ Show Answer</summary>
It uses both FP16 and FP32 to reduce memory use and speed up training  
without significant accuracy loss — especially on GPUs with Tensor Cores.
</details>

---

### 💬 Question 5

In what situation would a CPU still be the best choice?

<details>
<summary>✅ Show Answer</summary>
When performing inference on small models or deploying to devices  
where energy efficiency and cost are more important than raw speed.
</details>

---

## 🧭 Navigation Panel

> ⬅️ [Previous Week (Week 5 — Advanced CNNs)](Week5_CNN_Advanced.md)
> 🧩 [Go to Week 6 Scenarios](/Scenarios/Week6_Scenarios.md)
> ➡️ [Next Week (Week 7 — Optimization in Practice)](Week7_Optimization.md)


