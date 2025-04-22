# Paper-to-System Mapping

## 1. A Survey on Mixture of Experts in LLMs (Completed)

**Core Concepts Applied:**
- **Sparse Expert Activation:** Simulated by splitting matrix rows into discrete chunks, each processed by an independent thread. This mimics sparse MoE behavior where only a subset of experts (threads) are active per input.
- **Task Routing:** Implemented via `Threads.@threads` loop, assigning each data slice to a distinct compute path, aligned with top-k gating logic in MoE systems.
- **Expert Isolation:** Each thread operates on its own data subset with no overlap, ensuring thread-safe parallelism—an analogue to expert independence in MoE models.
- **Load Balancing Awareness:** Fixed chunk sizing ensures balanced work distribution across threads; the system is prepared for dynamic balancing in later stages.
- **Computational Efficiency & Benchmarking:** The system logs memory, runtime, and allocation statistics for every matrix decomposition, capturing the performance benefits of selective execution (MoE principle of compute sparsity).

**Implemented Artifacts:**
- `task_based_svd()` function to perform multithreaded matrix decomposition
- Benchmark logging system via `log_benchmark(...)` to track scalability and resource usage
- Output CSV tracks: timestamp, matrix size, chunk count, thread count, runtime, memory, and allocations

**Outcome:**  
A reproducible, measurable prototype that operationalizes MoE routing and sparsity concepts in a controlled HPC context.

(10000, 100)

Running task-based SVD...
  1.668 s (152 allocations: 214.12 MiB)

## 2. Sparse Backpropagation for MoE Training (SparseMixer) (In Progress)

**Core Concepts:**

- **SparseMixer** is a backpropagation technique designed to improve the training efficiency of Mixture of Experts (MoE) models.
- In typical MoE training, only the activated experts (e.g., Top-1 or Top-2) receive gradients, which leads to underutilization of the rest.
- SparseMixer introduces a **gradient approximation method** using a **second-order ODE solver** (midpoint method), enabling **non-activated experts** to receive **partial updates**.
- This results in **better expert utilization**, **balanced learning**, and **faster convergence**, achieving **dense learning with sparse compute**.

**Implementation Considerations for Our System:**

- Our current prototype mimics sparse expert activation using fixed chunk-thread assignment via `Threads.@threads`.
- A next step could involve:
  - Logging which chunks (experts) were “inactive”
  - Simulating gradient propagation by tagging “inactive” paths with approximated metrics
  - Designing a gating + feedback loop to mimic SparseMixer’s learning update approximation

**Planned Enhancements:**

- Track inactive threads or "unused expert slots" per input
- Extend `log_benchmark()` to include placeholder metrics for simulated gradient distribution
- Eventually, model partial update logic for inactive expert paths to reflect SparseMixer-inspired behavior

### TechRxiv Paper — Optimizing Expert Routing for Sparse MoE

- **Focus:** Enhancing routing quality without increasing compute.
- **Key Idea:** Penalize bad routing using a new Routing Optimization Loss (ROL).
- **Top-K Mask Decay:** Smooth transitions in expert activation (like soft switches).
- **Outcome:** Improved convergence, better generalization, and more even expert usage.
- **Relevance to Our Project:** Reinforces why routing metrics (like our gradient sim) matter — ties directly into task-based expert load tracking.

### ArXiv Paper — GShard: Scaling Giant Models with Conditional Computation

- **Source:** Google Brain, core architecture behind models like Switch Transformer.
- **Key Idea:** Sparse expert activation + parallelism to scale to 600B+ params.
- **Mechanism:** Gating network with top-k routing + load balancing loss.
- **Conditional Computation:** Makes giant models computationally feasible by activating only relevant experts.
- **Outcome:** Trained faster, used less memory, and maintained high performance.
- **Relevance to Our Project:** Informs how we simulate partial activation and track per-expert compute impact.
