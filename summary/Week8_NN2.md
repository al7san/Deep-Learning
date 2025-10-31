# Week 8 — Training Neural Networks (Part 2)

> Based on *Justin Johnson’s Lecture 11 (2019): Training Neural Networks (Part 2)*  
> Expanded with practical learning-rate schedules, hyperparameter optimization, and post-training techniques.

---

## 1. Introduction — Revisiting Training Dynamics

Training deep neural networks involves more than architecture design.  
Optimizing *how* we train — learning rates, stopping criteria, and fine-tuning strategies — often has a greater effect than structural tweaks.

A well-tuned training process improves:
- Convergence speed  
- Stability and reproducibility  
- Generalization performance  

---

## 2. Learning Rate Scheduling

The **learning rate ($\eta$)** controls the step size in gradient descent.  
Choosing it wisely balances between convergence speed and stability.

### Common Scheduling Strategies

| Schedule Type | Formula | Description | Pros | Cons | Typical Use |
|----------------|----------|--------------|------|------|--------------|
| **Constant** | $\eta_t = \eta_0$ | Fixed learning rate throughout training | Simple | Risk of oscillation or stagnation | Small datasets, quick tests |
| **Step Decay** | $\eta_t = \eta_0 \times \gamma^{\lfloor t / s \rfloor}$ | Reduces rate by $\gamma$ every $s$ epochs | Easy to tune | Abrupt jumps | Standard CNNs |
| **Exponential Decay** | $\eta_t = \eta_0 e^{-kt}$ | Smooth exponential decrease | Continuous change | Sensitive to $k$ | Long training schedules |
| **Linear Decay** | $\eta_t = \eta_0 (1 - t / T)$ | Decreases linearly to zero | Predictable end | Too aggressive near finish | Transformer training |
| **Cosine Annealing** | $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\pi t / T))$ | Follows a cosine curve | Smooth convergence | Needs full-epoch cycles | ImageNet-scale tasks |
| **Inverse Square Root** | $\eta_t = \eta_0 / \sqrt{t}$ | Common in NLP training | Stabilizes large-batch training | Slow decay | Transformers, RNNs |

<details>
<summary>Example: StepLR in PyTorch</summary>

```python
from torch.optim.lr_scheduler import StepLR
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
for epoch in range(30):
    train(...)
    scheduler.step()
````

</details>

---

## 3. Early Stopping

**Early stopping** halts training when validation loss stops improving.
It prevents overfitting and saves compute time.

### Implementation Logic

1. Track validation loss each epoch.
2. If no improvement for $p$ epochs → stop.
3. Optionally, restore the best model checkpoint.

<details>
<summary>Example: Early Stopping Concept</summary>

```python
if val_loss > best_loss:
    epochs_no_improve += 1
    if epochs_no_improve >= patience:
        print("Stopping early...")
        break
else:
    best_loss = val_loss
    epochs_no_improve = 0
```

</details>

---

## 4. Hyperparameter Search

Selecting hyperparameters — learning rate, batch size, weight decay —
is critical to achieving good results.

### 4.1 Grid Search

* Exhaustively tests all combinations.
* Reliable but computationally expensive.

### 4.2 Random Search

* Samples hyperparameters randomly.
* More efficient for high-dimensional spaces.
* Works surprisingly well in practice (Bergstra & Bengio, 2012).

| Method                    | Pros                  | Cons                   | Best Use                |
| ------------------------- | --------------------- | ---------------------- | ----------------------- |
| **Grid Search**           | Systematic            | Exponential cost       | Small spaces            |
| **Random Search**         | Efficient exploration | Less coverage          | High-dimensional tuning |
| **Bayesian Optimization** | Informed search       | Complex implementation | Large-scale tuning      |

---

## 5. After Training — Advanced Techniques

### 5.1 Model Ensembles

Train multiple models with different initializations and average their predictions.

$$
p(y|x) = \frac{1}{N} \sum_{i=1}^{N} p_i(y|x)
$$

Advantages:

* Reduces variance and overfitting
* Improves robustness to noise

### 5.2 Transfer Learning

Reuses features learned on large datasets.
Freeze early layers, fine-tune later ones for domain-specific adaptation.

### 5.3 Distributed / Large-Batch Training

* Parallelize training across GPUs or machines.
* Large batches require **learning rate scaling**:
  $$
  \eta_{new} = \eta_{base} \times \frac{B_{new}}{B_{base}}
  $$
* Synchronization and communication overhead must be managed carefully.

---

## 6. Practical Tips

### 6.1 Ratio of Update to Weight

Monitor the average ratio:
$$
r = \frac{||\Delta w||}{||w||}
$$
If $r > 10^{-3}$, the learning rate may be too high;
if $r < 10^{-6}$, it may be too low.

### 6.2 Polyak Averaging

Maintain an exponential moving average (EMA) of model parameters:
$$
\theta_{avg} = \alpha \theta_{avg} + (1 - \alpha)\theta_{t}
$$
It stabilizes inference and often improves final test accuracy.

### 6.3 Fine-Tuning Best Practices

* Start with pretrained weights.
* Lower learning rate (e.g., $0.1 \times$ of base rate).
* Optionally freeze early convolutional layers.

---

## 7.Review

Q1. Why use learning-rate schedules instead of a constant rate?

<details><summary>Show Answer</summary>
Because a decaying learning rate allows large updates early for fast convergence  
and smaller steps later for fine-tuning near minima.
</details>

 Q2. What problem does early stopping solve?

<details><summary>Show Answer</summary>
It prevents overfitting by halting training once validation loss stops improving.
</details>

 Q3. How does Random Search improve over Grid Search?

<details><summary>Show Answer</summary>
Random Search explores a wider range of values more efficiently in high-dimensional parameter spaces.
</details>

 Q4. Why do large-batch trainings require learning-rate scaling?

<details><summary>Show Answer</summary>
Because more samples per update mean fewer updates overall;  
scaling the learning rate keeps gradient magnitudes balanced.
</details>


 Q5. What is Polyak averaging used for?

<details><summary>Show Answer</summary>
To smooth parameter updates by keeping an exponential moving average of weights,  
leading to more stable evaluation performance.
</details>

---

## Navigation Panel

> ⬅️ [Back to Week 7 — Training Neural Networks (Part 1)](Week7_TrainingNN.md)
> 🧩 [Go to Week 8 Scenarios](../Scenarios/Week8_Scenarios.md)

