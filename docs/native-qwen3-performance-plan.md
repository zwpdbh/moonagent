# Native Qwen3 performance plan

## Purpose and current measurement

The native CUDA implementation is correct but slow. A release-mode chapter 2
basic generation run produced 100 tokens in 821.45 seconds, or 0.122 tokens/sec.
This is a correctness-first implementation; the number is a baseline, not an
indication that MoonBit or CUDA is inherently slow.

The plan is to measure first, then remove host work, synchronization, allocation,
and redundant computation in that order. Every optimization must preserve the
existing CPU/CUDA and basic/cached generation tests.

Related implementation plan: `docs/native-qwen3-moonxi-plan.md`.

## Initial profile results

Opt-in profiling is implemented in
`moonxi-net-gpu/tensor/llm_profile.mbt`. It records operation counts, host time,
CUDA-event time, index volume, and allocation volume. CUDA-event mode
synchronizes each measured operation, so these measurements describe the
instrumented run and should not be treated as uninstrumented kernel benchmarks.

Run a short profile with:

```bash
MOONXI_LLM_PROFILE=1 MOONXI_MAX_TOKENS=4 \
  moon run playground/cmd/main --target native --release -- \
  ex02 "Explain large language models."
```

Set `MOONXI_DEVICE=cpu` for the native CPU total. The default device remains
automatic CUDA selection when available.

### Four-token totals

| Backend and mode | Seconds | Tokens/sec |
| --- | ---: | ---: |
| Native CUDA basic | 4.957 | 0.807 |
| Native CUDA cached | 5.020 | 0.797 |
| Native CPU basic | 20.834 | 0.192 |
| Native CPU cached | 20.778 | 0.193 |

These are short, non-warmed runs used to identify overhead, not stable
performance benchmarks. Four tokens are too few to show the expected basic/cache
algorithmic difference.

### Instrumented CUDA cached breakdown

The latest four-token cached profile used 5.020 seconds in total. Important host
and CUDA entries were:

| Entry | Time | Share of total |
| --- | ---: | ---: |
| Gather host total | 4380.7 ms | 87.3% |
| Gather index construction | 3746.6 ms | 74.6% |
| Gather kernel CUDA events | 272.3 ms | 5.4% |
| Linear CUDA events | 192.7 ms | 3.8% |
| RMSNorm CUDA events | 125.7 ms | 2.5% |
| Batched matmul CUDA events | 55.3 ms | 1.1% |
| Tensor allocation host time | 78.2 ms | 1.6% |
| Gather index H2D host time | 30.5 ms | 0.6% |
| Gather index free host time | 5.3 ms | 0.1% |

Host categories overlap and CUDA-event categories synchronize work, so do not sum
these rows into a separate total. They identify where to optimize first.

The same run recorded:

- 1,684 generic gather calls.
- 28,161,536 generated gather indices.
- 112,646,144 bytes of gather-index H2D traffic.
- 1,684 index-buffer allocations and frees.
- 5,060 tensor allocations, representing 445,740,048 bytes allocated over the
  run (cumulative allocation volume, not peak memory).

### Conclusions that replace earlier hypotheses

1. **Host index construction is the dominant measured cost.** Direct layout
   kernels or cached index maps should come before numerical micro-optimization.
2. **Index H2D transfer and immediate frees are small in this profile.** They are
   worth removing as part of direct kernels, but the current data does not show
   them as the primary bottleneck.
3. **Output allocation is visible but secondary.** Workspace reuse remains useful
   after the dominant host indexing work is removed.
4. **Linear and RMSNorm kernels are the next largest arithmetic contributors.**
   Their CUDA-event timings justify later kernel work, but not before layout
   overhead is addressed.
5. **Four-token basic and cached totals are nearly equal.** Longer fixed-token
   runs are required to evaluate cache benefits.

## Direct layout kernel result

Phase C's direct layout implementation is now working. The shared helper is
`moonxi-net-gpu/tensor/llm.mbt:98`; the coordinate-computing CUDA kernel is
`moonxi-net-gpu/cuda/cuda_kernels.cu:2700`.

The direct kernel handles:

- `permute` (`llm.mbt:140`).
- `narrow` (`llm.mbt:179`).
- `expand` (`llm.mbt:258`).
- `repeat_interleave` (`llm.mbt:271`).

Each call transfers only a fixed 64-byte metadata record containing the
operation, shapes, axes, and dimensions. Source offsets are computed by GPU
threads. The previous per-element host index array is not constructed. The old
gather helper remains public as a reference implementation.

### Measured four-token improvement

| Mode | Generic gather | Direct layout | Speedup |
| --- | ---: | ---: | ---: |
| CUDA basic | 4.957 s | 0.305 s | 16.2× |
| CUDA cached | 5.020 s | 0.414 s | 12.1× |

The direct cached profile still processes the same 28,161,536 output elements,
but reports 1,684 direct-layout calls and no generic gather calls. Its
direct-layout CUDA-event total was 19.9 ms, versus 272.3 ms for gather CUDA events
in the earlier instrumented profile. The metadata volume is 107,776 bytes over
1,684 calls, replacing 112.6 MB of per-element index traffic.

These short runs demonstrate that the host-indexing bottleneck has been removed.
They do not provide a new result for the original 100-token workload; that should
be rerun after further allocation/cache work. The next priorities are longer
basic/cache comparisons, safe workspace allocation, and the remaining RMSNorm,
linear, and attention-kernel costs.

Correctness after the change: `moonagent` passes 219 release tests and the fork
passes 123 tests, with clean `moon check` results in both workspaces.

## Hundred-token comparison with Rust/Candle

The original 100-token workload has now been rerun after direct layout
optimization. All implementations used release mode, the same prompt, and the
same requested token count. Model loading was excluded from each generation
timer, and the programs ran sequentially on the RTX 3090.

| Implementation | Basic seconds | Basic tokens/sec | Cached seconds | Cached tokens/sec |
| --- | ---: | ---: | ---: | ---: |
| Native MoonBit, F32 | 40.636 | 2.461 | 30.882 | 3.238 |
| Rust/Candle, BF16 | 2.470 | about 40 | 2.170 | about 46 |

Rust/Candle is therefore about 16.5× faster on basic generation and 14.2× faster
on cached generation in this comparison. The native basic result is 20.2× faster
than its original 821-second baseline, but it remains far from the Rust
implementation.

The generated text is not identical across all implementations. The current
comparison measures runtime for the same workload, not numerical equivalence
between F32 native execution and BF16 Candle execution.

### Instrumented native cached breakdown

The same 100-token native cached run took 78.503 seconds with CUDA-event timing
enabled, compared with 30.882 seconds without it. This difference demonstrates
that per-operation event synchronization materially changes execution. The
following values describe the instrumented run, not an uninstrumented timeline:

| Entry | Instrumented time | Share of 78.503 s |
| --- | ---: | ---: |
| Linear CUDA events | 55.907 s | 71.2% |
| Tensor allocation host time | 5.892 s | 7.5% |
| RMSNorm CUDA events | 3.116 s | 4.0% |
| Softmax CUDA events | 1.518 s | 1.9% |
| Batched matmul CUDA events | 1.447 s | 1.8% |
| Direct layout CUDA events | 1.303 s | 1.7% |
| Elementwise multiplication CUDA events | 0.567 s | 0.7% |

The run recorded 19,700 linear calls, 126,500 tensor allocations, and
48,242,611,600 bytes of cumulative tensor allocation. Direct layout no longer
builds host index maps; its remaining cost is small relative to linear arithmetic
and allocation in this instrumented profile.

### Updated priorities

1. Profile linear and attention arithmetic without adding synchronization to
   every operation; use an application-level CUDA event pair or an external
   profiler for a less distorted timeline.
2. Introduce safe workspace reuse to reduce the 126,500 allocations and their
   host cost.
3. Improve RMSNorm/softmax and batched attention after arithmetic timing is
   confirmed.
4. Continue using longer fixed-token runs for basic/cache comparisons.
5. Keep the fair wall-clock comparison separate from the instrumented breakdown.

## Current execution paths

### Basic generation recomputes the sequence

`generate_native_text_basic_stream` calls the full model on every token with
`cache=None` and then appends the selected ID to the growing input array:

- `ch02/ch02.mbt:272` — basic generator.
- `ch02/ch02.mbt:292` — full forward pass.
- `ch02/ch02.mbt:297` — append the generated ID.

The chapter's `ex01` runner selects this basic path at `ch02/ch02.mbt:63`.
Consequently, work grows as the generated sequence grows. This is expected
algorithmic overhead, but it does not by itself explain the very low token rate.

### Cached generation still performs a forward pass per token

`generate_native_text_basic_stream_cache` prefills the cache and then forwards
only the latest token:

- `ch02/ch02.mbt:310` — cached generator.
- `ch02/ch02.mbt:338` — forward pass with the cache.
- `ch02/ch02.mbt:351` — schedule the next one-token input.

This avoids recomputing old activations, but still executes every layer and many
of the same allocation, layout, and synchronization operations for each token.

### Qwen3 calls the expensive layout paths repeatedly

The native model uses layout operations throughout RoPE and attention:

- `qwen3/native_model.mbt:157` — RoPE.
- `qwen3/native_model.mbt:175` — narrow/scale/concat used to rotate halves.
- `qwen3/native_model.mbt:197` — attention.
- `qwen3/native_model.mbt:242-243` — concatenate new K/V with cached K/V.
- `qwen3/native_model.mbt:255-256` — materialize repeated KV heads.
- `qwen3/native_model.mbt:257` — permute K and compute attention scores.
- `qwen3/native_model.mbt:266` — attention-value matmul and output layout.

These operations run across 28 layers. A small fixed overhead is therefore
multiplied by layer count, token count, and—on the basic path—sequence length.

## Problem primer: what is host index construction?

**Host** means the CPU side of the program. **Device** means the GPU. An index is
just the position of a value in a tensor's flat storage; it is not a token ID.

A layout operation often only changes which existing value belongs at each output
position. For example, transposing this row-major matrix:

```text
input:  [1, 2, 3]
        [4, 5, 6]
```

has flat input storage `[1, 2, 3, 4, 5, 6]`. Its transposed output is:

```text
output: [1, 4]
        [2, 5]
        [3, 6]
```

One way to compute the output on a GPU is to provide a source position for every
output element:

```text
source indices: [0, 3, 1, 4, 2, 5]
```

The gather kernel then performs `output[i] = input[source_indices[i]]`.

Our current implementation builds that entire index array in MoonBit on the CPU,
copies it to the GPU, and only then runs the gather. This is **host index
construction**. The actual GPU copy is cheap; building millions of coordinates
and indices with CPU loops is the dominant measured cost.

A direct layout kernel can instead compute the source position from the output
position, shape, and strides inside the GPU kernel. That avoids constructing,
storing, and transferring a separate index array. This is why direct layout
kernels are the first planned optimization.

## Specific implementation costs

### 1. Host index generation and H2D transfer

The common GPU gather helper is `gpu_llm_gather` at
`moonxi-net-gpu/tensor/llm.mbt:55`.

For every output element it currently:

1. Computes coordinates and a source index on the host (`llm.mbt:63-65`).
2. Allocates an index buffer on the GPU (`llm.mbt:66`).
3. Copies the index array from host to device (`llm.mbt:67`).
4. Allocates the output tensor (`llm.mbt:68`).
5. Launches the gather kernel (`llm.mbt:69-72`).
6. Frees the temporary index buffer (`llm.mbt:73`).

The gather kernel itself is simple and appropriate:
`moonxi-net-gpu/cuda/cuda_kernels.cu:2548`. The concern is the surrounding host
work and transfer, not gather arithmetic.

This helper is used by:

- `permute` — `llm.mbt:78`.
- `narrow` — `llm.mbt:123`.
- `expand` — `llm.mbt:203`.
- `repeat_interleave` — `llm.mbt:224`.

Embedding also creates and transfers an index buffer at `llm.mbt:335`, although
that occurs only when new input tokens are embedded rather than for every
attention operation.

### 2. Immediate synchronous frees

`gpu_llm_gather` frees every temporary index buffer immediately after launching
the gather (`llm.mbt:73`). The free reaches `cudaFree` through
`cuda_free_buffer` at `cuda_kernels.cu:2687`.

CUDA deallocation can synchronize or serialize execution. Even where it does not
force a full device wait in every context, immediate alloc/copy/free cycles make
it difficult to maintain an efficient asynchronous pipeline. Exact time spent
here must be measured rather than inferred.

Do not fix this by leaking index buffers. Replace immediate frees with reusable
owned buffers, direct kernels, or a correctly scoped workspace.

### 3. Fresh output allocation for nearly every operation

`GpuTensor::alloc_st` at `moonxi-net-gpu/tensor/gpu_tensor.mbt:273` allocates
through `cuda.alloc` unless pooling is enabled (`gpu_tensor.mbt:276-283`).

The native Qwen3 CLI does not call `gpu_enable_pool`. Linear projection,
attention matmul, normalization, masking, layout operations, and residual
arithmetic therefore allocate fresh outputs repeatedly. The current simple
ownership model also makes it difficult to know which intermediate buffers can
be safely reused.

A workspace reset must never invalidate model weights, RoPE tables, or live KV
cache entries. Those objects need permanent allocations or lifetimes separate
from temporary forward-pass buffers.

### 4. Serial row scans in normalization kernels

The current softmax kernel assigns one GPU thread to each row and serially scans
that row for maximum, exponential sum, and normalization:

- `cuda_kernels.cu:2571-2584`.

RMSNorm similarly uses one thread per row and serial loops for mean square and
scaling:

- `cuda_kernels.cu:2603-2616`.

These implementations are simple and correct but underuse parallelism within
rows. They may matter less than host/index overhead; profile before rewriting.

### 5. Batched matmul uses a host loop over batches

`GpuTensor::batched_matmul` calls `cuda.llm_batched_matmul` at `llm.mbt:277-316`.
The CUDA wrapper at `cuda_kernels.cu:2554` loops over batches and invokes the
existing cuBLAS matmul for each one. Using cuBLAS is good, but many small calls
can add launch overhead. A strided-batched cuBLAS call is a later optimization.

### 6. Materialized KV expansion and growing cache copies

The model repeats KV heads at `qwen3/native_model.mbt:255-256`, allocating and
copying expanded tensors. It also concatenates new K/V onto existing cached
tensors at `native_model.mbt:242-243`, so cache storage grows through repeated
copying.

These are useful correctness implementations, but later versions can index the
original KV heads directly during attention and use preallocated cache capacity.

### 7. F32 memory traffic

The native loader converts BF16 checkpoint values to F32 in
`qwen3/native_model.mbt:124`. F32 doubles weight storage relative to BF16 and
increases memory traffic. This is deliberate for the first port. BF16 support is
not an early optimization until allocation and scheduling issues are addressed.

## Phase A — Establish repeatable measurements

### Work

1. Add a benchmark-only runner that accepts backend, cache mode, prompt, token
   count, and warmup count. Do not change the existing educational CLI behavior.
2. Measure release builds with the same prompt and generated-token counts.
3. Record native CPU basic/cached, native CUDA basic/cached, and PyTorch
   basic/cached results. Rust/Candle can be an additional reference.
4. Measure model loading separately from generation.
5. Run at least one warmup and three measured runs; report median and variance.
6. Add counters for:
   - Forward calls and generated tokens.
   - Gather calls and generated index elements.
   - H2D index bytes.
   - Tensor and index allocations.
   - Index-buffer frees and output allocations.
7. Use host monotonic timing around the total operation and CUDA events around
   GPU regions. The fork already has event/profiling support in
   `moonxi-net-gpu/event_tensor/event_tensor.mbt`.
8. Optionally use Nsight Systems for an independent timeline if available.

### Acceptance

- A documented table reproduces the 0.122-token/sec baseline or explains the
  changed environment.
- Results distinguish host time, GPU execution, synchronization, and loading.
- The profile identifies the largest contributors instead of assuming them.

## Phase B — Add fair generation benchmarks

### Work

1. Add timing to cached generation, equivalent to the existing basic runner.
2. Compare basic and cached CUDA generation over identical token counts.
3. Keep EOS disabled in benchmark mode so every run performs the requested work.

### Acceptance

- Cached generation is faster than basic generation on the same workload, or the
  profile explains why fixed overhead hides the difference.
- Basic/cached token IDs remain equal on the fixed correctness prompts.

## Phase C — Remove temporary-index alloc/copy/free cycles

### Work

Prefer direct kernels for the hot operations:

1. `permute`: compute output coordinates and source offsets inside the CUDA
   kernel from axes, shapes, and strides.
2. `narrow`: compute the shifted source offset directly.
3. `expand` and `repeat_interleave`: compute source coordinates directly.
4. Keep the generic gather implementation as a fallback and correctness oracle.
5. If a temporary buffer is still required, reuse an owned buffer with sufficient
   capacity and no immediate `cudaFree` in the operation loop.

An intermediate, lower-effort step may cache shape/index maps that are truly
constant across calls. Direct kernels are the cleaner final implementation.

### Acceptance

- No host index-array construction for the optimized hot operations.
- No per-call index H2D transfer or index-buffer free on those paths.
- Existing CUDA LLM tests pass; add multi-dimensional shape tests.

## Phase D — Introduce safe workspace allocation

### Work

1. Classify allocations as permanent (weights, RoPE tables, KV cache) or temporary
   (layer intermediates, index scratch).
2. Use separate ownership/allocation scopes for these classes.
3. Reuse temporary buffers only when their sizes and lifetimes are known.
4. Add pool/workspace statistics and peak-memory reporting.
5. Never reset a workspace while any returned tensor or cache entry still refers
   to its memory.

### Acceptance

- Peak allocation count and allocation time decrease in the profile.
- Weights and caches remain valid across multiple forward calls and generations.
- Repeated cached generation does not corrupt output or grow temporary memory
  without bound.

## Phase E — Reduce attention and cache copying

### Work

1. Preallocate KV-cache capacity and write new K/V at the current position instead
   of concatenating on every token.
2. Avoid materializing repeated KV heads; map each query head to its source KV
   head during the attention operation.
3. Consider strided-batched cuBLAS for attention matmul after measuring the
   existing host loop.

### Acceptance

- Cached generation performs no growing K/V concatenation.
- Repeated-head tensors are not allocated on the hot path.
- Cache/full-forward numerical checks still pass.

## Phase F — Improve numerical kernels

### Work

1. Implement warp/block reductions for RMSNorm and softmax.
2. Fuse obvious operation chains where profiling shows memory-bound overhead,
   especially RoPE and SwiGLU.
3. Evaluate BF16 storage/computation only after the F32 execution pipeline is
   efficient and numerically stable.

### Acceptance

- Report kernel-level speedups independently of end-to-end changes.
- Preserve numerical tolerances and fixed generated-token sequences.

## Optimization order and decision rule

Implement phases in order A → B → C → D → E → F. Do not start with BF16 or complex
kernel fusion while host index generation, synchronization, and allocation remain
unmeasured.

For every phase, record:

- Before/after tokens per second.
- Host and GPU time where available.
- Allocation and transfer counters.
- Peak memory.
- Correctness-test result.

If an optimization does not improve a repeatable measurement, simplify or revert
it rather than retaining unnecessary complexity.

## Completion target

The performance work is successful when native cached CUDA generation no longer
constructs/frees host-generated index maps in its hot loop, temporary allocation
lifetimes are explicit and reusable, KV caches avoid growing concatenation, and
repeatable profiling demonstrates a substantial improvement over the 0.122
tokens/sec correctness baseline without changing expected model behavior.
