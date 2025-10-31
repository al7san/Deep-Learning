# 🧠 Week 4 — Convolutional Neural Networks (CNNs)

> Based on *Justin Johnson’s Lecture 7 (2019): Convolutional Networks*  
> Expanded with real-world analogies, exam-style insights, and practical notes from our study sessions.

---

## 1. Introduction — How CNNs See the World
Deep learning models can “see” images through Convolutional Neural Networks (CNNs).  
Instead of looking at all pixels at once, a CNN focuses on **small local regions** (patches) and learns patterns such as edges, corners, or textures.  
This makes CNNs excellent for computer-vision tasks like image classification, object detection, and medical imaging.

---

## 2. Convolution Layer — The Eyes of the Network
A **convolution layer** slides small filters (also called kernels) over the image and performs dot-products.  
Each filter learns to detect a specific pattern: vertical edges, color gradients, or object parts.  

### 🧮 What happens mathematically
If your input image is size `(H × W × C)` and your filter is `(K × K × C)`,  
the convolution operation computes a new feature map by multiplying overlapping values and summing them.

- **Input:** image or previous layer activations  
- **Weights:** small learnable filters  
- **Output:** feature maps (representations of learned patterns)

<details>
<summary>🧩 Example: simple convolution in PyTorch (click to view)</summary>

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
x = torch.randn(1, 3, 32, 32)   # (batch, channels, height, width)
y = conv(x)
print(y.shape)  # -> (1, 8, 32, 32)
````

</details>

---

## 3. Padding and Stride — Controlling Image Size

Without padding, each convolution shrinks the image (edges are lost).
To preserve size, we add **padding** (extra zeros around the border).
The **stride** defines how far the filter moves at each step.

| Parameter       | Meaning                  | Effect                     |
| --------------- | ------------------------ | -------------------------- |
| **Padding (P)** | Adds a border of zeros   | Preserves edge information |
| **Stride (S)**  | Step size for the filter | Controls downsampling      |

**Rule of thumb:**
Use `padding="same"` and `stride=1` to keep feature maps the same size.

---

## 4. Pooling Layers — Learning What Really Matters

Pooling layers **downsample** feature maps to make the network focus on the most important information.
They reduce the spatial size while keeping key features.

* **Max Pooling:** takes the maximum value in each region (captures strongest signal).
* **Average Pooling:** takes the average (smooths out details).

| Hyperparameter  | Description                          |
| --------------- | ------------------------------------ |
| Kernel size (K) | Region over which pooling is applied |
| Stride (S)      | Step size of pooling window          |

📘 Typical choices:
`MaxPool, K=2, S=2` (used in LeNet and AlexNet).

<details>
<summary>💡 Example: Max Pooling in PyTorch</summary>

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = torch.randn(1, 8, 32, 32)
y = pool(x)
print(y.shape)  # -> (1, 8, 16, 16)
```

</details>

---

## 5. Activation — ReLU and Beyond

After each convolution, we apply an **activation function** to introduce *non-linearity*.
Without it, the network would behave like a linear model regardless of depth.

* **ReLU (Rectified Linear Unit):** `f(x) = max(0, x)`

  * Fast, simple, and avoids saturation.
  * May cause the *dying ReLU* problem (neurons stuck at zero).
* Alternatives: **LeakyReLU**, **ELU**, **GELU**, etc.

<details>
<summary>⚙️ Example: ReLU usage</summary>

```python
relu = nn.ReLU()
x = torch.tensor([[-1.0, 0.5, 2.0]])
print(relu(x))  # -> tensor([[0.0, 0.5, 2.0]])
```

</details>

---

## 6. LeNet-5 — The Classic CNN

One of the earliest CNNs by *LeCun et al., 1998*, used for handwritten-digit recognition (MNIST).

| Layer   | Output Size | Filter Details |
| ------- | ----------- | -------------- |
| Input   | 1×28×28     | —              |
| Conv1   | 20×28×28    | K=5, P=2, S=1  |
| ReLU    | 20×28×28    | —              |
| MaxPool | 20×14×14    | K=2, S=2       |
| Conv2   | 50×14×14    | K=5, P=2, S=1  |
| ReLU    | 50×14×14    | —              |
| MaxPool | 50×7×7      | K=2, S=2       |
| Flatten | 2450        | —              |
| Linear  | 500         | —              |
| ReLU    | 500         | —              |
| Linear  | 10          | Output classes |

---

## 7. Normalization Layers — Keeping Values in Check

Training deep CNNs can become unstable as activations vary across layers.
**Normalization** layers fix this by re-centering and re-scaling activations.

### 7.1 Batch Normalization

* Normalizes activations within each mini-batch
* Keeps learning stable and allows higher learning rates
* Formula:
  [
  y = \gamma \frac{(x - \mu)}{\sigma} + \beta
  ]
  where ( \mu, \sigma ) are batch mean and std, and ( \gamma, \beta ) are learnable.

### 7.2 Layer / Instance / Group Normalization

| Type             | Normalizes over      | Best for       |
| ---------------- | -------------------- | -------------- |
| **BatchNorm**    | Across batch samples | Large batches  |
| **LayerNorm**    | Within each sample   | Transformers   |
| **InstanceNorm** | Per image/channel    | Style transfer |
| **GroupNorm**    | Groups of channels   | Small batches  |

<details>
<summary>⚖️ Example: Batch Normalization</summary>

```python
bn = nn.BatchNorm2d(num_features=8)
x = torch.randn(4, 8, 32, 32)
y = bn(x)
print(y.mean().item(), y.std().item())  # ≈ 0, 1
```

</details>

---

## 8. Modern Practices — Strided Convolutions vs Pooling

In newer CNNs (like ResNet, EfficientNet), pooling is often replaced by **convolutions with stride > 1**.
This allows the model to *learn* how to downsample intelligently rather than using a fixed rule.

| Technique       | Learns? | Keeps Details? | Common Use                    |
| --------------- | ------- | -------------- | ----------------------------- |
| Max Pooling     | ❌       | ❌              | Simple/fast models            |
| Average Pooling | ❌       | ✅              | Medical or texture analysis   |
| Strided Conv    | ✅       | ✅              | Modern CNNs, object detection |

---

## 9. Fine-Tuning Deep Models

When reusing a pretrained network, we **freeze the early layers** and retrain the later ones.

* Early layers learn *general features* (edges, colors).
* Later layers learn *task-specific features* (bird species, car types).
  Freezing early layers provides a **baseline** for new tasks and saves computation.

<details>
<summary>🔁 Example: Freezing Layers in PyTorch</summary>

```python
model = torchvision.models.resnet18(pretrained=True)
for param in model.layer1.parameters():
    param.requires_grad = False  # freeze early layers
```

</details>

---

## 10. Summary Cheat Sheet

| Concept           | Key Idea                  | Purpose                       |
| ----------------- | ------------------------- | ----------------------------- |
| **Convolution**   | Learn spatial features    | Extract patterns              |
| **Padding**       | Keep image size           | Preserve borders              |
| **Stride**        | Move filters by steps     | Control downsampling          |
| **Pooling**       | Reduce dimensions         | Focus on key info             |
| **ReLU**          | Non-linear activation     | Enable complex learning       |
| **Normalization** | Stabilize training        | Faster, smoother convergence  |
| **Fine-Tuning**   | Reuse pretrained features | Save time & data              |
| **Strided Conv**  | Learn to downsample       | Modern alternative to pooling |


