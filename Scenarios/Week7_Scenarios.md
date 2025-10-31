# Week 7 — Applied Scenarios & Exam-Style Questions

> Complementary to *Week 7 — Training Neural Networks (Part 1)*  
> Each scenario explores practical challenges in activation, initialization, and regularization.

---

## Scenario 1 — Dying ReLU Problem

**Context:**  
A student trains a deep CNN using ReLU activations.  
After several epochs, many neurons start outputting only zeros, and the model stops improving.

**Question:**  
What caused this, and how can it be fixed?

<details>
<summary>Show Answer</summary>
This is the **Dying ReLU problem** — neurons receive negative inputs and never reactivate.  
Possible fixes:
- Use **LeakyReLU** or **ELU** to allow small negative gradients.  
- Lower the learning rate to avoid large negative weight updates.
</details>

---

## Scenario 2 — Normalization in Medical Imaging

**Context:**  
A medical AI team notices unstable training when using unnormalized X-ray pixel values (range 0–255).  

**Question:**  
What preprocessing step can stabilize training?

<details>
<summary>Show Answer</summary>
Apply **Normalization** to scale pixel values (e.g., divide by 255 or standardize to zero mean and unit variance).  
This keeps gradients well-conditioned and prevents exploding activations.
</details>

---

## Scenario 3 — Choosing Initialization Strategy

**Context:**  
A researcher uses ReLU activations but initializes weights using Xavier initialization.  
The network fails to converge.

**Question:**  
Why did this happen, and what is the proper fix?

<details>
<summary>Show Answer</summary>
Xavier initialization assumes symmetric activations like Sigmoid/Tanh.  
ReLU networks require **Kaiming (He) initialization**, which maintains proper activation variance.  
Switching to Kaiming ensures stable forward/backward propagation.
</details>

---

## Scenario 4 — Batch Normalization Benefits

**Context:**  
A deep CNN converges slowly even with a good optimizer.  
After adding Batch Normalization, training accelerates significantly.

**Question:**  
Why does BatchNorm improve training speed?

<details>
<summary>Show Answer</summary>
Batch Normalization normalizes activations within each mini-batch,  
reducing internal covariate shift and allowing higher learning rates.  
It also acts as a mild regularizer.
</details>

---

## Scenario 5 — Overfitting on Small Dataset

**Context:**  
A company trains a network with 2,000 samples and observes perfect training accuracy but poor validation accuracy.

**Question:**  
Which regularization methods could improve performance?

<details>
<summary>Show Answer</summary>
Use **Dropout**, **L2 regularization**, or **Data Augmentation**.  
They prevent overfitting by introducing noise, penalizing large weights, or increasing data diversity.
</details>

---

## Scenario 6 — Mixup and Generalization

**Context:**  
A data scientist tests Mixup augmentation and finds smoother decision boundaries and fewer misclassifications.

**Question:**  
Why does Mixup help with generalization?

<details>
<summary>Show Answer</summary>
Mixup interpolates between samples ($\tilde{x} = \lambda x_i + (1 - \lambda)x_j$),  
forcing the model to learn linear transitions between classes rather than memorizing sharp boundaries.
</details>

---

## Scenario 7 — Gradient Explosion in RNN

**Context:**  
An RNN trained on text data shows rapidly increasing loss after a few epochs.

**Question:**  
Which initialization or technique can help control this?

<details>
<summary>Show Answer</summary>
Use **Orthogonal initialization** or apply **Gradient Clipping**.  
Orthogonal matrices preserve gradient norms, preventing exponential growth through timesteps.
</details>

---

## Scenario 8 — Combining Techniques

**Context:**  
A team builds a CNN using ReLU activations, Kaiming initialization, BatchNorm, and Dropout.  
Training is stable and validation accuracy is high.

**Question:**  
Why does combining multiple techniques improve performance?

<details>
<summary>Show Answer</summary>
Each component targets a different challenge:
- **ReLU:** efficient non-linearity  
- **Kaiming init:** stabilizes activations  
- **BatchNorm:** smooths optimization  
- **Dropout:** reduces overfitting  
Together, they ensure balanced and reliable training.
</details>

---

## Quick Summary

| Problem | Technique | Benefit |
|----------|------------|----------|
| Dying ReLU | LeakyReLU / ELU | Keeps gradients alive |
| Unstable input | Normalization | Prevents exploding activations |
| Poor convergence (ReLU) | Kaiming Initialization | Stable gradients |
| Slow training | BatchNorm | Faster convergence |
| Overfitting | Dropout / L2 / Augmentation | Better generalization |
| Sharp decision boundaries | Mixup | Smoother class separation |
| Gradient explosion (RNN) | Orthogonal Init / Clipping | Stable sequence training |

---

## True / False — Quick Review

1️⃣ Sigmoid activations are ideal for deep networks because they prevent vanishing gradients.  
<details><summary>Show Answer</summary>❌ False — Sigmoid often *causes* vanishing gradients in deep networks.</details>

2️⃣ Batch Normalization can be used even during inference.  
<details><summary>Show Answer</summary>✅ True — During inference, it uses running statistics collected during training.</details>

3️⃣ Kaiming initialization is specifically designed for ReLU-based activations.  
<details><summary>Show Answer</summary>✅ True — It maintains proper variance for ReLU layers.</details>

4️⃣ Dropout is applied during both training and inference.  
<details><summary>Show Answer</summary>❌ False — It is only active during training to add stochastic regularization.</details>

5️⃣ Mixup combines input samples but not their labels.  
<details><summary>Show Answer</summary>❌ False — Mixup linearly combines both images and labels.</details>


6️⃣ Orthogonal initialization helps maintain gradient magnitude in RNNs.  
<details><summary>Show Answer</summary>✅ True — It preserves gradient norm across time steps.</details>

---


> ⬅️ [Back to Summary](../summary/Week7_TrainingNeuralNetworks.md)  
> ➡️ [Next Week (Week 8 Scenarios)](Week8_Optimization.md)
