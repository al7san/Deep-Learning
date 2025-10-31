#  Week 6 — Applied Scenarios 

> Complementary to *Week 6 — Hardware and Software for Deep Learning*  
> Each scenario reflects real-world system design decisions for deep learning acceleration.

---

##  Scenario 1 — Choosing Between CPU and GPU

**Context:**  
A research team trains a CNN for image classification on a small dataset using a local workstation.  
Training on CPU takes several hours, while GPU training completes in minutes.

**Question:**  
Why does the GPU perform so much faster, and when would CPU training still be appropriate?

<details>
<summary> Show Answer</summary>
GPUs are optimized for parallel operations — they execute thousands of matrix multiplications simultaneously.  
CPUs, while faster per core, handle fewer operations at once.  
CPU training remains suitable for small models or inference tasks where parallelism is not critical.
</details>

---

##  Scenario 2 — TPU vs GPU for Cloud Training

**Context:**  
A company trains a large Transformer-based language model in Google Cloud.  
They must choose between NVIDIA GPUs and Google TPUs.

**Question:**  
Which hardware is more suitable and why?

<details>
<summary> Show Answer</summary>
**TPUs** are purpose-built for large-scale tensor operations and TensorFlow workloads.  
They deliver higher throughput and energy efficiency for distributed cloud training.  
However, **GPUs** offer more flexibility, especially for PyTorch or mixed-framework setups.
</details>

---

##  Scenario 3 — Selecting a Deep Learning Framework

**Context:**  
A startup needs to prototype models quickly for computer vision experiments,  
but later wants to deploy them on mobile devices.

**Question:**  
Should they choose PyTorch or TensorFlow?

<details>
<summary> Show Answer</summary>
**PyTorch** is ideal for fast research and experimentation due to its dynamic computation graph.  
For deployment on mobile or cloud environments, **TensorFlow** excels through TensorFlow Lite and TensorFlow Serving.  
A hybrid workflow (PyTorch for research, TensorFlow for production) is also common.
</details>

---

##  Scenario 4 — Memory Optimization with Mixed Precision

**Context:**  
A GPU-based model runs out of memory when increasing the batch size from 16 to 64.  

**Question:**  
How can the team solve this issue without buying a new GPU?

<details>
<summary> Show Answer</summary>
Enable **Mixed Precision Training** (FP16 + FP32).  
This reduces memory usage and speeds up training on Tensor Core GPUs,  
allowing larger batch sizes without extra hardware.
</details>

---

##  Scenario 5 — Cost-Efficient Training Strategy

**Context:**  
A university runs multiple student experiments on deep learning models with limited budget.  
They can’t afford cloud GPUs 24/7.

**Question:**  
What is a cost-efficient approach to manage experiments?

<details>
<summary> Show Answer</summary>
Use **local GPUs** or **shared servers** for prototyping and debugging small models.  
For final large-scale training, rent **cloud GPUs/TPUs** only as needed.  
This hybrid approach minimizes cloud costs while keeping flexibility.
</details>

---

##  Scenario 6 — Framework Performance Differences

**Context:**  
Two identical CNN models are trained — one in PyTorch, one in TensorFlow —  
but TensorFlow finishes 15% faster.

**Question:**  
What could explain this performance difference?

<details>
<summary> Show Answer</summary>
TensorFlow often uses graph-level optimizations (e.g., XLA compiler)  
that fuse operations and reduce kernel launches.  
PyTorch is slightly slower in graph optimization but easier to debug interactively.
</details>

---

##  Scenario 7 — When CPU Outperforms GPU

**Context:**  
A developer runs a small linear regression model on a GPU-enabled laptop.  
Surprisingly, GPU performance is slower than CPU.

**Question:**  
Why does this happen?

<details>
<summary> Show Answer</summary>
For small workloads, the **GPU overhead** (data transfer between CPU and GPU memory)  
outweighs its parallel advantage.  
CPUs are faster when the computation is minimal or highly sequential.
</details>

---

##  Scenario 8 — Framework Compatibility in Cloud Deployment

**Context:**  
A production team trained models in PyTorch but needs to deploy them on TensorFlow Serving.

**Question:**  
What’s the best way to make this transition?

<details>
<summary> Show Answer</summary>
Export the trained PyTorch models to **ONNX (Open Neural Network Exchange)** format,  
then convert to TensorFlow if necessary.  
ONNX ensures cross-framework compatibility and smoother deployment.
</details>

---

##  Quick Summary

| Challenge | Solution | Key Benefit |
|------------|-----------|--------------|
| Slow training | Use GPUs | Parallel processing |
| Large-scale TensorFlow tasks | TPUs | Cloud efficiency |
| Prototyping vs deployment | PyTorch → TensorFlow | Flexibility |
| GPU memory limits | Mixed Precision | Lower usage, faster speed |
| Budget constraints | Local + Cloud hybrid | Cost control |
| Framework speed | TensorFlow XLA | Graph optimization |
| Small models | CPU | Low overhead |
| Cross-framework deployment | ONNX | Compatibility |

---

## 🧩 True / False — Quick Review

1️⃣ GPUs outperform CPUs because they have higher clock speeds.  
<details><summary> Show Answer</summary>❌ False — GPUs are slower per core but excel via massive parallelism.</details>

2️⃣ TPUs are general-purpose processors that work efficiently with any framework.  
<details><summary> Show Answer</summary>❌ False — TPUs are specialized for TensorFlow and large tensor workloads.</details>


3️⃣ PyTorch uses dynamic graphs, making it easier for experimentation.  
<details><summary> Show Answer</summary>✅ True — Define-by-run execution simplifies debugging and iteration.</details>


4️⃣ Mixed precision training decreases both memory use and training speed.  
<details><summary> Show Answer</summary>❌ False — It decreases memory usage but *increases* speed on Tensor Core GPUs.</details>

5️⃣ CPU training is still useful for lightweight models and inference.  
<details><summary> Show Answer</summary>✅ True — CPUs handle small or sequential workloads efficiently.</details>

6️⃣  ONNX allows model interoperability between PyTorch and TensorFlow.  
<details><summary> Show Answer</summary>✅ True — ONNX is the common format for converting models across frameworks.</details>

---


> ⬅️ [Back to Summary](../summary/Week6_HardwareandSoftware.md)  
> ➡️ [Next Week (Week 7 Scenarios)](Week7_Scenarios.md)
