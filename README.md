# Task-Routed SVD

This project explores task-based parallelism in matrix decomposition using Julia.  
Inspired by Mixture-of-Experts (MoE) architectures in large language models, the idea is to selectively activate chunks ("experts") of a matrix during SVD computations to gain efficiency improvements.

## Features
- Threaded matrix SVD decomposition using `Threads.@threads`
- Simulated task routing across matrix partitions
- Benchmarking scripts and timing utilities
- Modular, extensible code structure for future experiments

## Motivation
Inspired by:
- Mixture of Experts architectures (Sparse MoE models)
- HuggingFace engineering posts on scalable sparse compute
- Parallelized linear algebra needs in HPC and AI scaling

## Requirements
- Julia ≥ 1.9
- Packages: `BenchmarkTools`, `LinearAlgebra`, `Threads`

## How to Run
```bash
julia --threads auto src/task_routed_svd.jl
