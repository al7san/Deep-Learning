#  Week 5 — Advanced CNNs, Transfer Learning & Optimization

> Based on *Justin Johnson’s Lecture 8 (2019): Convolutional Networks – Part 2*  
> Expanded with real-world analogies, interactive questions, and PyTorch code examples.

---

## 1. Introduction — From Seeing to Understanding

In [Week 4](../Week4_CNN.md), we learned how CNNs **see** images by detecting patterns such as edges and shapes.  
In Week 5, we explore how they **understand and generalize** — connecting those features to make decisions.

After convolution and pooling layers extract spatial features,  
the **fully connected layers** (FC layers) combine them into a meaningful prediction —  
like classifying whether an image is a cat, a flower, or a road sign.

---

## 2. Dropout Regularization — Fighting Overfitting

Overfitting happens when a model memorizes training data instead of learning patterns.  
**Dropout** randomly deactivates a portion of neurons during training,  
forcing the network to learn multiple paths to the correct output.

📘 **Effect:** Improves generalization by reducing co-dependence among neurons.

<details>
<summary>🧩 Example: Dropout in PyTorch</summary>

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, 10)
)
print(model)
````

</details>

---

## 3. Data Augmentation — Expanding the Dataset

When we have limited or homogeneous data, **Data Augmentation** helps us simulate diversity.
It generates new samples by modifying existing images — rotations, flips, brightness changes, and crops.

🎯 **Goal:** Teach the model to recognize objects under various orientations and lighting conditions.

<details>
<summary>🧩 Example: Using torchvision transforms</summary>

```python
from torchvision import transforms

augment = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2)
])
```

</details>

---

## 4. Transfer Learning — Learning from Pretrained Knowledge

Instead of training a CNN from scratch, we can reuse a model already trained on a large dataset
(e.g., **ImageNet**) and adapt it to our own task.

📘 **Key idea:** Early layers of pretrained models already understand generic patterns like edges and textures.

Steps:

1. Load a pretrained model (e.g., ResNet50).
2. Replace the final classification layer.
3. Train only the new layers.

<details>
<summary>🧩 Example: Transfer Learning in PyTorch</summary>

```python
import torchvision.models as models
import torch.nn as nn

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # freeze all layers

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)  # e.g., 5 flower classes
```

</details>

---

## 5. Fine-Tuning — Adjusting Pretrained Models

Fine-tuning means retraining some (not all) layers of a pretrained model to better match your dataset.
We usually **unfreeze** a few top layers so they can adapt to new patterns.

| Dataset similarity   | Strategy                                  |
| -------------------- | ----------------------------------------- |
| Very similar         | Freeze most layers, train only classifier |
| Moderately different | Unfreeze last few layers                  |
| Very different       | Retrain most of the network               |

<details>
<summary>🧩 Example: Partial Fine-Tuning</summary>

```python
for param in model.layer4.parameters():
    param.requires_grad = True  # unfreeze last block only
```

</details>

---

## 6. Optimization & Learning Rate Scheduling

Training deep networks involves choosing an **optimizer** —
a rule for updating weights to minimize the loss function.
Each optimizer has strengths and weaknesses.

### ⚙️ Common Optimizers

| Optimizer    | Concept                                            | Strength                          | Weakness                           | Best Use Case               |
| ------------ | -------------------------------------------------- | --------------------------------- | ---------------------------------- | --------------------------- |
| **SGD**      | Fixed learning rate, simple gradient update        | Stable, general                   | Slow convergence                   | Large datasets              |
| **Momentum** | Adds velocity term to SGD                          | Speeds up convergence             | May overshoot minima               | Deep linear models          |
| **RMSProp**  | Scales learning rate by recent gradient magnitudes | Handles non-stationary objectives | Sensitive to hyperparameters       | Recurrent / online learning |
| **Adam**     | Combines Momentum + RMSProp                        | Fastest, adaptive                 | Sometimes unstable for fine-tuning | Most CNN and NLP tasks      |

<details>
<summary>🧩 Example: Adam with scheduler</summary>

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
```

</details>

---

## 7. CNN Architectures — The Evolution of Vision Models

### **LeNet-5 (1998)**

* First successful CNN for handwritten digit recognition (MNIST).
* Structure: Conv → Pool → Conv → Pool → FC → Softmax.
* Activation: `tanh`, `sigmoid`.

---

### **AlexNet (2012)**

* The breakthrough model that won the ImageNet challenge.
* Introduced **ReLU**, **Dropout**, and **GPU training**.
* 8 layers: 5 Conv + 3 FC.
* Huge jump in accuracy — CNNs became mainstream.

---

### **VGGNet (2014)**

* Simple but deep (16–19 layers).
* Used uniform 3×3 filters.
* Easy to understand, but computationally expensive.

---

### **ResNet (2015)**

* Introduced **Residual (Skip) Connections** to fix vanishing gradients.
* Enabled training of 50–152 layer networks.
* Became the new baseline for computer vision.

---

### **Inception (GoogLeNet, 2015)**

* Used multiple filter sizes (1×1, 3×3, 5×5) in parallel.
* Captures both small and large features.
* More efficient than VGG.

---

### **EfficientNet (2019)**

* Balanced scaling of depth, width, and resolution.
* Achieved high accuracy with fewer parameters.
* Excellent for mobile and real-time applications.

---

## 8. Summary Cheat Sheet

| Concept                      | Description                               | Benefit                       |
| ---------------------------- | ----------------------------------------- | ----------------------------- |
| **Dropout**                  | Randomly disables neurons during training | Reduces overfitting           |
| **Data Augmentation**        | Creates new samples by modifying images   | Improves generalization       |
| **Transfer Learning**        | Reuses pretrained model weights           | Saves time and resources      |
| **Fine-Tuning**              | Retrains specific layers                  | Adapts to new domain          |
| **Optimization**             | Weight update algorithms                  | Faster, stable training       |
| **Learning Rate Scheduling** | Gradual LR reduction                      | Smoother convergence          |
| **Architectures**            | Evolved CNN models                        | Accuracy–Efficiency trade-off |

---

## 🧩 Review

 1- Why is dropout applied only during training and not during testing?

<details>
<summary>✅ Show Answer</summary>
Because dropout introduces randomness to prevent overfitting during training.  
At test time, we use all neurons to get deterministic predictions.
</details>



2- What is the main difference between Data Augmentation and Transfer Learning?

<details>
<summary>✅ Show Answer</summary>
Augmentation increases *data diversity* by creating new samples.  
Transfer Learning increases *model knowledge* by reusing pretrained features.
</details>


3- When should Fine-Tuning be preferred over simple Transfer Learning?

<details>
<summary>✅ Show Answer</summary>
When your dataset differs significantly from the original dataset (e.g., medical vs. natural images).  
Retraining some layers helps the model adapt better to new patterns.
</details>



4- Which optimizer adapts learning rates for each parameter automatically?

<details>
<summary>✅ Show Answer</summary>
**Adam** — combines ideas from Momentum and RMSProp for adaptive, efficient learning.
</details>



5- Why did ResNet outperform VGG despite having more layers?

<details>
<summary>✅ Show Answer</summary>
Because ResNet’s residual (skip) connections prevent vanishing gradients,  
allowing much deeper networks to train effectively.
</details>

---



> ⬅️ [Previous Week (Week 4 CNNs)](../Week4_CNNs/Summary/Week4_CNNs.md)
> 🧩 [Go to Week 5 Scenarios](Week5_Scenarios.md)
> ➡️ [Next Week (Week 6 RNNs)](../Week6_RNNs/Week6_RNNs.md)

