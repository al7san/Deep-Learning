# Week 7 — Training Neural Networks (Part 1)

> Based on *Justin Johnson’s Lecture 10 (2019): Training Neural Networks (Part 1)*  
> Expanded with best practices for initialization, activation, and regularization in modern deep learning.

---

## 1. Introduction

Before building large-scale neural networks, we must establish good **training foundations**.  
Performance depends not only on model architecture but also on **activation choice**, **data normalization**, **weight initialization**, and **regularization techniques**.

---

## 2. Activation Functions

Activation functions introduce **non-linearity**, enabling networks to learn complex mappings.  
Choosing the right function affects gradient flow, convergence, and accuracy.

### Common Activation Functions

| Function | Formula / Behavior | Advantages | Drawbacks | Typical Use |
|-----------|--------------------|-------------|-------------|--------------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1 + e^{-x}}$ | Smooth output, bounded | Vanishing gradients | Old models, binary output |
| **Tanh** | $\tanh(x) = 2\sigma(2x) - 1$ | Centered around 0 | Still saturates | RNNs, legacy models |
| **ReLU** | $f(x) = \max(0, x)$ | Simple, efficient | Dying ReLU problem | CNNs, most modern models |
| **Leaky ReLU** | $f(x) = \max(0.01x, x)$ | Avoids dead neurons | Slight bias | CNNs, deep nets |
| **ELU** | Exponential linear unit | Faster convergence | Higher computation | Deep architectures |
| **SELU** | Scaled ELU | Self-normalizing | Requires specific init | Self-Norm networks |

<details>
<summary>Example: Using LeakyReLU in PyTorch</summary>

```python
import torch.nn as nn
activation = nn.LeakyReLU(0.1)
````

</details>

---

## 3. Data Preprocessing

Data preprocessing ensures stable and efficient training by keeping all input features on similar scales.

### 3.1 Normalization

Rescales each feature to have zero mean and unit variance:

$$
x' = \frac{x - \mu}{\sigma}
$$

### 3.2 PCA and Whitening

* **PCA (Principal Component Analysis):** decorrelates features.
* **Whitening:** scales principal components to have unit variance.

These steps help gradient-based optimization converge faster.

---

## 4. Weight Initialization

Bad initialization can cause **exploding** or **vanishing gradients**.
Good initialization ensures that activations and gradients have controlled variance across layers.

| Method                  | Formula                                      | Best for        | Notes                          |
| ----------------------- | -------------------------------------------- | --------------- | ------------------------------ |
| **Xavier / Glorot**     | $\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$ | Sigmoid / Tanh  | Balances input/output variance |
| **Kaiming / He (MSRA)** | $\text{Var}(w) = \frac{2}{n_{in}}$           | ReLU, LeakyReLU | Prevents dying neurons         |
| **Orthogonal**          | $W^T W = I$                                  | RNNs            | Preserves gradient norm        |

<details>
<summary>Example: Kaiming Initialization in PyTorch</summary>

```python
import torch.nn as nn
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```

</details>

---

## 5. Regularization

Regularization prevents overfitting and improves generalization.
It acts as a form of "controlled noise" that helps the model learn more robust patterns.

### 5.1 L1 and L2 Penalties

* **L1 (Lasso):** encourages sparsity (many weights → 0).
* **L2 (Ridge):** penalizes large weights, leading to smoother models.

$$
L_{reg} = \lambda_1 ||w||_1 + \lambda_2 ||w||_2^2
$$

### 5.2 Dropout

Randomly disables a portion of neurons during training.

$$
p = 0.5 \text{ (commonly used)}
$$

This forces neurons to learn redundant representations.

### 5.3 Batch Normalization

Normalizes activations inside the network, reducing internal covariate shift.
Allows higher learning rates and faster convergence.

### 5.4 Data Augmentation

Artificially increases training data diversity via:

* Flipping, rotation, translation
* Brightness/contrast changes
* Random crops or noise injection

### 5.5 Mixup

Blends two training examples:

$$
\tilde{x} = \lambda x_i + (1 - \lambda)x_j
$$
$$
\tilde{y} = \lambda y_i + (1 - \lambda)y_j
$$

Helps generalization by smoothing decision boundaries.

---

## 6. Training Dynamics (Preview)

Lecture 10 Part 1 introduces setup techniques; Part 2 will explore **learning rates**, **momentum**, and **schedulers** in depth.

---

## 7. After Training — Model Ensembles and Transfer Learning

* **Ensemble methods:** average predictions from multiple trained models → improves stability.
* **Transfer learning:** fine-tune a pretrained model on a related dataset → saves compute and improves accuracy.

---

### 8. Review

 Q1. Why is ReLU generally preferred over Sigmoid or Tanh in deep CNNs?

<details><summary>Show Answer</summary>
Because ReLU does not saturate for positive values and avoids vanishing gradients,  
making training faster and more stable.
</details>



Q2. What problem does Batch Normalization solve?

<details><summary>Show Answer</summary>
It reduces internal covariate shift by keeping activations normalized within layers,  
allowing higher learning rates and smoother convergence.
</details>



Q3. What type of initialization is recommended for ReLU-based networks?

<details><summary>Show Answer</summary>
**Kaiming (He) initialization**, as it maintains activation variance in ReLU layers.
</details>


Q4. How does Dropout improve generalization?

<details><summary>Show Answer</summary>
By randomly deactivating neurons during training, it prevents co-adaptation  
and encourages redundancy in learned representations.
</details>



Q5. What is the main idea of Mixup augmentation?

<details><summary>Show Answer</summary>
Mixup linearly combines two input samples and their labels,  
forcing the model to learn smoother decision boundaries.
</details>

---


> ⬅️ [Back to Week 6 — Hardware and Software](Week6_HardwareandSoftware.md)
> 🧩 [Go to Week 7 Scenarios](../Scenarios/Week7_Scenarios.md)
> ➡️ [Next Week (Week 8 — Optimization in Practice)](Week8_Optimization.md)

