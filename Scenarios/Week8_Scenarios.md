# Week 8 — Applied Scenarios 

> Complementary to *Week 8 — Training Neural Networks (Part 2)*  
> These scenarios focus on learning rate dynamics, early stopping, and optimization tuning.

---

## Scenario 1 — Step Decay vs Cosine Annealing

**Context:**  
A student trains a ResNet model with **StepLR** and notices accuracy jumps every few epochs.  
Switching to **Cosine Annealing** produces a smoother learning curve.

**Question:**  
Why does the Cosine schedule produce smoother convergence?

<details>
<summary>Show Answer</summary>
Step Decay reduces the learning rate abruptly at fixed intervals,  
causing sudden loss spikes.  
Cosine Annealing decreases the learning rate gradually following a cosine curve,  
resulting in smooth and stable convergence.
</details>

---

## Scenario 2 — Early Stopping on Overfitting Model

**Context:**  
A small CNN starts to overfit after 20 epochs while validation loss stops improving.  
The researcher continues training until epoch 60.

**Question:**  
What was the mistake, and what should have been done instead?

<details>
<summary>Show Answer</summary>
The training should have been stopped earlier using **Early Stopping** based on validation loss.  
Continuing training after convergence causes overfitting and wasted compute time.
</details>

---

## Scenario 3 — Choosing Between Grid and Random Search

**Context:**  
A data scientist wants to tune learning rate, batch size, and dropout for a CNN.  
Running every possible combination would take days.

**Question:**  
Which search method should be used and why?

<details>
<summary>Show Answer</summary>
Use **Random Search** — it samples hyperparameter combinations randomly,  
covering a wider space efficiently and often finds near-optimal results faster  
than exhaustive Grid Search.
</details>

---

## Scenario 4 — Large-Batch Training Instability

**Context:**  
A model trained with batch size 1024 shows unstable loss and poor generalization.  
The same architecture with batch size 128 trains well.

**Question:**  
What adjustment could improve stability for large batches?

<details>
<summary>Show Answer</summary>
Increase the learning rate proportionally to batch size  
($\eta_{new} = \eta_{base} \times \frac{B_{new}}{B_{base}}$)  
and apply **Learning Rate Warmup** or **Gradient Clipping**  
to maintain stability during early epochs.
</details>

---

## Scenario 5 — Model Ensemble for Robustness

**Context:**  
A Kaggle team combines predictions from five independently trained CNNs.  
Validation accuracy improves by 1.5%.

**Question:**  
Why does ensembling improve results?

<details>
<summary>Show Answer</summary>
Ensembling averages predictions from multiple models,  
reducing variance and mitigating the effect of any single model’s noise.  
It increases robustness and generalization.
</details>

---

## Scenario 6 — Monitoring Update-to-Weight Ratio

**Context:**  
A researcher monitors the ratio $r = \frac{||\Delta w||}{||w||}$  
and finds $r = 1e^{-2}$ in early epochs.

**Question:**  
What does this indicate about the learning rate?

<details>
<summary>Show Answer</summary>
An $r$ value around $10^{-2}$ suggests the learning rate is too high.  
Gradients cause large updates relative to weight magnitude,  
risking instability.  
The learning rate should be reduced.
</details>

---

## Scenario 7 — Polyak Averaging for Stability

**Context:**  
During fine-tuning, validation accuracy fluctuates slightly after each epoch.  
Applying Polyak (EMA) averaging produces a smoother accuracy curve.

**Question:**  
Why does EMA help stabilize evaluation?

<details>
<summary>Show Answer</summary>
Polyak Averaging maintains an exponential moving average of weights:  
$\theta_{avg} = \alpha \theta_{avg} + (1 - \alpha)\theta_t$.  
It smooths short-term fluctuations, improving evaluation stability.
</details>

---

## Scenario 8 — Fine-Tuning from a Pretrained Model

**Context:**  
A team uses a pretrained VGG model for a new dataset of medical images.  
They retrain all layers with a high learning rate and lose the pretrained benefits.

**Question:**  
What went wrong, and how should fine-tuning be done?

<details>
<summary>Show Answer</summary>
The learning rate was too high — it destroyed pretrained weights.  
Fine-tuning requires:
- Lower learning rate (≈ 0.1× base rate)  
- Freezing early layers  
- Training only later layers or the classifier first
</details>

---

## Quick Summary

| Challenge | Technique | Purpose |
|------------|------------|----------|
| Learning rate instability | Cosine / Step Schedules | Smooth convergence |
| Overfitting | Early Stopping | Stop before validation loss rises |
| Costly tuning | Random Search | Efficient hyperparameter exploration |
| Large-batch instability | LR scaling + Warmup | Balanced updates |
| Model variance | Ensemble | Reduce prediction noise |
| Oversized updates | Monitor $r = ||\Delta w|| / ||w||$ | Detect unstable learning |
| Accuracy fluctuation | Polyak Averaging | Smooth evaluation |
| Losing pretrained info | Low LR + Partial Freeze | Safe fine-tuning |

---

## True / False — Quick Review

1️⃣ Cosine Annealing lowers the learning rate in abrupt jumps every few epochs.  
<details><summary>Show Answer</summary>❌ False — Cosine Annealing gradually reduces the rate in a smooth wave pattern.</details>

2️⃣ Early stopping can reduce overfitting by halting training when validation loss plateaus.  
<details><summary>Show Answer</summary>✅ True — It prevents unnecessary extra epochs once generalization stops improving.</details>

3️⃣ Grid Search is more efficient than Random Search for high-dimensional hyperparameter spaces.  
<details><summary>Show Answer</summary>❌ False — Random Search explores the space more effectively with fewer trials.</details>

4️⃣ Large-batch training always improves generalization.  
<details><summary>Show Answer</summary>❌ False — It can harm generalization unless the learning rate and warmup are tuned properly.</details>

5️⃣ Polyak averaging smooths weight updates and stabilizes model evaluation.  
<details><summary>Show Answer</summary>✅ True — It maintains a running average of parameters for smoother performance.</details>

6️⃣ Fine-tuning with a high learning rate helps the model adapt faster to new data.  
<details><summary>Show Answer</summary>❌ False — High LR overwrites pretrained knowledge; a smaller rate is safer.</details>

---



> ⬅️ [Back to Summary](../summary/Week8_TrainingNN2.md)  

