# Task-Routed SVD in Julia

This project explores task-based parallelism in matrix decomposition using Julia. The core idea is inspired by Mixture-of-Experts (MoE) routing, where only specific "experts" (or matrix chunks) are activated per computation — leading to efficiency gains.

## Features
- Threaded matrix SVD decomposition using `Threads.@threads`
- Simulates task routing via chunked matrix ops
- Benchmarking and timing utilities

## Inspired By
- [Mixture of Experts in LLMs – Survey](link-to-paper)
- HuggingFace Sparse MoE engineering posts

## Requirements
- Julia ≥ 1.9
- `BenchmarkTools`, `LinearAlgebra`, `Threads`

## Run Example
```bash
julia --threads auto src/task_routed_svd.jl
