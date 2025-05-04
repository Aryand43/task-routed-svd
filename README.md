# Task-Routed SVD

This project explores **task-based parallelism** in matrix decomposition using Julia, drawing inspiration from **Mixture-of-Experts (MoE)** architectures used in large-scale language models.

The system dynamically selects which chunks ("experts") of a matrix to activate during SVD computation, optimizing compute usage while simulating ML-like routing logic and gradient feedback.

---

## Features

- Multithreaded SVD decomposition via `Threads.@threads`
- Softmax-based simulated routing to prioritize important chunks
- Dynamic load balancing through a thread-safe expert queue
- Simulated gradient logging (active vs. inactive experts)
- Performance + memory benchmarking with CSV logging
- Modular and extensible structure for future ML integration

---

## Motivation

Inspired by cutting-edge techniques in scalable ML and HPC:

- Sparse Mixture-of-Experts (MoE) architectures
- Sparse backpropagation techniques (e.g., SparseMixer)
- Google’s GShard and HuggingFace engineering posts
- Efficient SVD decomposition in parallel compute environments

---

## Requirements

- Julia ≥ 1.9
- Packages:
  - `BenchmarkTools`
  - `LinearAlgebra`
  - `CSV`
  - `DataFrames`
  - `ThreadsX`

---

## How to Run

```bash
julia --threads auto src/task_routed_svd.jl
