using LinearAlgebra
using BenchmarkTools
using CSV, DataFrames
using Dates
using ThreadsX
using Base.Threads: @threads, nthreads, lock, unlock, ReentrantLock, Atomic, atomic_add!, atomic_get

function softmax(x::Vector{Float64})
    exps = exp.(x .- maximum(x))
    return exps ./ sum(exps)
end

function task_based_svd(A::Matrix{Float64}, chunks::Int)
    n = size(A, 1)
    chunk_size = div(n, chunks)
    svd_results = Vector{Any}(undef, chunks)
    raw_scores = randn(chunks) 
    softmax_scores = softmax(raw_scores)
    sorted_chunks = sortperm(softmax_scores, rev=true)
    activated_experts = Set{Int}()
    lock_obj = ReentrantLock()
    work_queue = [Threads.Atomic{Int}(i) for i in sorted_chunks]
    queue_index = Threads.Atomic{Int}(1)
    @threads for thread_id in 1:nthreads()
        while true
            i = queue_index[]
            if i > length(work_queue)
                break
            end
            chunk_id = work_queue[i].value
            queue_index[] += 1
            idx_start = (chunk_id - 1) * chunk_size + 1
            idx_end = chunk_id == chunks ? n : chunk_id * chunk_size
    
            svd_results[chunk_id] = svd(A[idx_start:idx_end, :])
    
            lock(lock_obj) do
                push!(activated_experts, chunk_id)
            end
        end
    end    
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    logfile = "logs/expert_gradients.csv"
    mkpath(dirname(logfile))  # Ensure logs/ exists
    log_entries = Vector{NamedTuple}()

    for i in 1:chunks
        if i in activated_experts
            gradient_val = rand(0.05:0.01:0.10)
            println("Expert $i activated — simulated_gradient_impact = ", gradient_val)
            push!(log_entries, (
                Timestamp = timestamp,
                Expert = i,
                Activated = true,
                GradientValue = gradient_val,
                Rows = size(A, 1),
                Cols = size(A, 2),
                Chunks = chunks
            ))
        else
            gradient_val = rand(0.001:0.001:0.015)
            println("Expert $i: Partial update — approximated_gradient = ", gradient_val)
            push!(log_entries, (
                Timestamp = timestamp,
                Expert = i,
                Activated = false,
                GradientValue = gradient_val,
                Rows = size(A, 1),
                Cols = size(A, 2),
                Chunks = chunks
            ))
        end
    end

    log_gradient_to_csv(logfile, log_entries)
    return svd_results
end

function run_benchmark()
    A = randn(4000, 1000)
    println("Running task-based SVD...")
    @btime task_based_svd($A, 4)
end

function log_benchmark(matrix_rows, matrix_cols, chunks; logfile="benchmarks/performance_log.csv")
    A = randn(matrix_rows, matrix_cols)
    bench = @benchmark task_based_svd($A, $chunks) setup=(GC.gc())
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    thread_count = nthreads()
    time_ms = minimum(bench).time / 1e6
    memory_mb = minimum(bench).memory / 1024^2
    allocations = minimum(bench).allocs

    row = DataFrame(
        Timestamp = [timestamp],
        Rows = [matrix_rows],
        Cols = [matrix_cols],
        Chunks = [chunks],
        Threads = [thread_count],
        Time_ms = [time_ms],
        Allocations = [allocations],
        Memory_MB = [memory_mb]
    )

    mkpath(dirname(logfile))
    if isfile(logfile)
        CSV.write(logfile, row; append=true)
    else
        CSV.write(logfile, row)
    end
end

function log_gradient_to_csv(logfile::String, entries::Vector{NamedTuple})
    df = DataFrame(entries)
    mkpath(dirname(logfile))
    if isfile(logfile)
        CSV.write(logfile, df; append=true)
    else
        CSV.write(logfile, df)
    end
end

# === RUN ===
log_benchmark(10000, 1000, 4)
run_benchmark()
