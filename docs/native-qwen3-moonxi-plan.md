# Native Qwen3 with moonxi-net

## Goal

Run the chapter 2 Qwen3 example with native MoonBit tensor implementations from
`chnlkw/moonxi-net`, using the standard MoonBit compiler. CUDA kernels must be
built externally with NVIDIA `nvcc`; the project must not require the
CUDA-patched MoonBit compiler fork.

The existing `zw/torch` PyTorch bridge remains available as a reference and test
oracle during migration. Native inference must not initialize CPython or require
`PYTHONPATH`.

## Current baseline

- The Qwen3 tokenizer and generation algorithms already run in MoonBit.
- Active model execution uses PyTorch tensors through `zw/torch`.
- `educational/safetensors` reads the real BF16 checkpoint and converts tensor
  values to F32, but is not part of active inference.
- The local `moonxi-net` fork has been migrated to current `moon.mod` manifests.
- Its CPU module passes its existing tests with the standard compiler.
- `qwen3/native_ops.mbt` contains a tested rank-2 `NpArray` linear-projection
  prototype. This is only an integration checkpoint, not a native Qwen3 model.
- The GPU package still declares `cuda_kernels.cu` as a MoonBit native stub, so
  the standard compiler incorrectly sends it to the C compiler.
- `moonxi-net` currently provides F32 tensors and rank-2 matmul/transpose. It
  lacks several tensor operations required by Qwen3.

## Design boundaries

Generic tensor mathematics belongs in `moonxi-net`:

- Shape, layout, slicing, broadcasting, and concatenation operations.
- Batched matmul, linear projection, embedding, reductions, softmax, SiLU,
  RMSNorm, masking, and argmax.
- CPU and CUDA backend implementations of the same operations.
- Autograd support only when a later training task requires it.

Qwen-specific behavior belongs in `moonagent`:

- Configuration and checkpoint loading into named native tensors.
- Qwen3 layer composition, RoPE convention, grouped-query attention, and SwiGLU.
- KV-cache ownership and updates.
- Tokenization, greedy generation, EOS handling, streaming, CLI, and statistics.
- Integration tests against known Qwen3 outputs.

Do not add Qwen names, tokenizer behavior, or generation policy to
`moonxi-net`.

## Phase 1 — Build CUDA with the standard MoonBit compiler

Goal: prove that the compiler fork is unnecessary before relying on the GPU
backend for Qwen3.

**Status: completed.** `scripts/build_cuda_kernels.sh` compiles the CUDA source
with `nvcc` and caches a static library under `_build/cuda`. Standard MoonBit
compiles the C stub; `moonxi-net-gpu/build_config.js` propagates native linking
settings to dependents. A
clean build of the GPU linear example ran successfully and learned approximately
`w=3`, `b=1`; the CPU tests also pass. Explicit `cuInit(0)` was needed to make
cuBLAS/cuDNN initialization reliable in this environment. No compiler fork is
used.

### Changes in `moonxi-net`

1. Add a reproducible CUDA build script for
   `moonxi-net-gpu/cuda/cuda_kernels.cu`.
   - Discover `nvcc` through `NVCC`, then `CUDA_HOME/bin/nvcc`, then
     `/usr/local/cuda/bin/nvcc`.
   - Compile position-independent CUDA objects and archive them as a static
     library such as `libmoonxi_cuda_kernels.a`.
   - Store generated binaries under `_build`; never commit them.
   - Preserve the existing GPU architecture flags, with an override for local
     development.
   - Rebuild when CUDA sources or flags change.
2. Change `moonxi-net-gpu/cuda/moon.pkg`:
   - Keep `cuda_stub.c` as the only C native stub.
   - Remove `cuda_kernels.cu` from `native-stub` and remove `nvcc-flags`.
   - Link the prebuilt kernel library plus `cuda`, `cudart`, `cublas`, `cudnn`,
     and `stdc++`.
   - Verify how package-relative native library search paths are resolved; do
     not hard-code a user-specific build path.
3. Keep FFI signatures and tensor APIs unchanged during this build-system
   migration. Add explicit CUDA driver initialization if required by the host.
4. Document the two-step build: run the CUDA script, then use ordinary `moon`
   commands. CPU-only users must not need `nvcc`, CUDA, or cuDNN.

### Acceptance

- `moon version` shows the standard compiler.
- `moon test moonxi-net --target native` passes without CUDA tools.
- The existing GPU linear example builds and learns the expected values after
  running the external CUDA build script.
- No CUDA-patched compiler is installed or invoked.
- The build works from a clean `_build` directory.

### Environment note

The host has CUDA 13.0 and cuDNN 9.27 runtime libraries, but cuDNN development
headers were not found during investigation. Install the documented development
package before validating the existing full GPU package. Qwen3 itself does not
need cuDNN; separating LLM/CUDA support from optional CNN/cuDNN support can be a
later cleanup.

## Phase 2 — Implement generic native CPU tensor operations

Goal: support a complete Qwen3 forward pass using `NpArray` and F32, without
CPython.

**Status: completed for Qwen3 inference.** `moonxi-net/nparray/llm.mbt` now
provides copying implementations of permutation, slicing, selection,
concatenation, broadcasting, KV repetition, batched matmul, linear projection,
stable softmax, integer-ID embedding, SiLU, RMSNorm, masking, and argmax. Generic
unit tests cover multi-batch arithmetic and cache-style indexing. The native
model uses these operations directly; no autograd wrapper is involved.

### Changes in `moonxi-net`

Implement operations with explicit rank/shape validation and generic tests:

1. Layout and indexing:
   - Arbitrary-rank `reshape`/`view`, `permute`, `narrow`, `select`, and
     `unsqueeze`.
   - `concat`, `expand`, and the KV expansion operation needed by grouped-query
     attention.
   - Define copy versus view behavior explicitly. A simple copying CPU
     implementation is acceptable initially.
2. Arithmetic and reductions:
   - Batched matmul for Qwen's rank-3/rank-4 attention tensors.
   - Linear projection over arbitrary leading dimensions.
   - Axis-based sum/mean and stable last-dimension softmax.
   - Elementwise exponential/division or other primitives needed by the public
     operations.
3. Neural-network operations:
   - Embedding lookup using MoonBit integer token IDs at the API boundary.
   - SiLU, RMSNorm, masked fill, and last-dimension argmax.
4. Generalize the rank-2 `native_linear_projection` prototype into a reusable
   library operation, then remove the temporary Qwen-local wrapper.
5. Do not add backward closures yet. Inference uses raw tensors rather than
   `Grad[T]`.

### Acceptance

- Each operation has small, hand-computable tests plus shape-error tests.
- Batched operations cover more than one batch/head, not only rank-2 examples.
- Equivalent `@torch` operations provide reference values where practical.
- `moon test moonxi-net --target native` passes.

## Phase 3 — Native CPU Qwen3 inference

Goal: generate text using only MoonBit, native tensor operations, and the local
model cache.

**Status: completed.** `NativeQwen3Model` loads the BF16 safetensors checkpoint
as F32 `NpArray` weights and implements the complete forward pass, RoPE, grouped
attention, MLP, and KV cache. `ch02` provides native basic and cached lazy token
streams. Real-checkpoint tests verify the first token and that both generation
paths produce `"<think>\nOkay,"`; no Python runtime is used.

### Changes in `moonagent`

1. Promote or adapt the educational safetensors reader into an active native
   loading path.
   - Preserve lazy per-tensor reads and validate byte counts against shapes.
   - Convert BF16 checkpoint values to F32 initially.
   - Construct `NpArray` weights without passing through Python.
2. Add a temporary native Qwen3 model path beside the PyTorch reference:
   - Parse the existing `Config` and validate expected tensor shapes.
   - Implement embeddings, 28 transformer blocks, Q/K/V projections, Q/K norm,
     RoPE, grouped-query attention, SwiGLU, residual connections, final norm,
     and tied output logits.
   - Reuse generic operations from `moonxi-net`; keep RoPE and model composition
     in `moonagent`.
3. Implement native KV-cache prefill and token-by-token updates.
4. Port both existing generation paths to the native model:
   - `generate_text_basic_stream`.
   - `generate_text_basic_stream_cache`.
   - Preserve lazy EOS exhaustion, maximum-token limits, and streaming output.
5. Do not remove the PyTorch implementation yet. Use it to diagnose numerical
   or architectural differences.

### Acceptance

- The real checkpoint loads without `PYTHONPATH` or CPython initialization.
- Fixed intermediate tensors match the F32 reference within documented
  tolerances.
- Cached and non-cached paths generate the same token IDs for fixed prompts.
- The existing four-token real-model prefix remains `"<think>\nOkay,"` unless a
  demonstrated F32/BF16 numerical difference requires a reviewed expectation
  change.
- `cache-check` confirms that cached and full-forward final logits agree within
  F32 tolerances.
- CPU performance is measured but is not an acceptance criterion.

## Phase 4 — Native CUDA Qwen3 inference

Goal: run the same native model on `GpuTensor` using the standard MoonBit
compiler and externally built CUDA kernels.

**Status: completed for correctness.** CUDA kernels and FFI wrappers now cover
Qwen3 layout, arithmetic, normalization, masking, embedding, and argmax. The
`NativeTensor` trait shares the model and generation logic between `NpArray` and
`GpuTensor`. Real-checkpoint tests pass for both basic and cached CUDA streams.

The initial layout implementation transfers host-generated integer index maps
and frees them synchronously; it is not optimized. A complete 100-token native
CUDA basic run took 821 seconds (0.12 tokens/sec). Replacing index maps with
specialized kernels, reducing synchronization, and improving allocation/cache
behavior are necessary performance follow-ups.

### Changes in `moonxi-net`

1. Implement CUDA equivalents of the generic operations introduced in phase 2.
2. Support batched matmul and attention layouts without host-side per-element
   loops or unnecessary device-to-host transfers.
3. Keep F32 as the first native GPU dtype. Do not mix BF16 kernel development
   into the initial correctness port.
4. Avoid host transfers for ordinary forward operations; token selection may
   transfer the final scalar ID as required by the generation loop.

### Changes in `moonagent`

1. Load native checkpoint values and transfer model weights to the GPU.
2. Share model and generation logic with the CPU path where the tensor traits
   make that clear; otherwise use thin backend-specific operations rather than
   duplicating Qwen architecture.
3. Keep tokenization and text output native and unchanged.

### Acceptance

- The program runs under the standard MoonBit compiler after the external CUDA
  build step.
- Native CUDA and native CPU paths produce equivalent logits within tolerance
  and the same reviewed token sequences.
- Basic and cached generation both work on the real checkpoint.
- Report tokens/second for both paths and identify synchronization or allocation
  bottlenecks before optimizing.

## Phase 5 — Make native inference the default

**Status: completed.** The chapter CLI selects native CUDA when a device is
available and native CPU otherwise. Native stdout replaced the Python-based
streaming print helper, and no Python path setup is used. The roadmap and tensor
documentation describe the native architecture. Both workspaces pass clean
checks; `moonagent` passes 219 release tests and the fork passes 123 tests.

1. Switch chapter 2 CLI examples to the native backend.
2. Remove Python-path setup, `without_grad`, and Python device detection from the
   native execution path.
3. Keep `zw/torch` for PyTorch examples and as an optional reference test
   backend; removing that package is not required by this plan.
4. Update `docs/roadmap/ch02.md`, `docs/tensor-design.md`, `torch/README.md`, and
   applicable `AGENTS.md` files so no documentation claims that active Qwen3
   inference still requires PyTorch.
5. Clean deprecation warnings in touched `moonxi-net` packages in separate
   mechanical commits or patches, not mixed with numerical changes.
6. Run `moon info && moon fmt`, the complete workspace tests, the native CPU
   example, and the native CUDA example.

## Verification commands

Use package selection to avoid accidentally building CUDA for CPU-only tests:

```bash
# moonxi-net CPU baseline, from its workspace root
moon test moonxi-net --target native

# moonagent tests while the PyTorch reference remains active
PYTHONPATH=/home/zw/code/moonbit_programming/moonagent/.venv/lib/python3.12/site-packages \
  moon test --target native

# Native chapter 2 example; must not need PYTHONPATH after phase 3
moon run playground/cmd/main --target native -- \
  ex01 "Explain large language models."
```

The exact CUDA script and GPU example commands will be added to the fork's
README during phase 1.

## Risks and deliberate limitations

- Converting the BF16 checkpoint to F32 approximately doubles weight storage.
  Correctness comes before memory optimization.
- F32 native results can differ numerically from BF16 CUDA/PyTorch results.
  Compare like-for-like references and do not require bitwise equality.
- The current cache concatenates tensors and may copy increasing amounts of
  data. Preallocated caches are an optimization after correctness.
- Native CPU matmul and GPU F32 kernels may be slower than PyTorch BF16. This is
  acceptable for the initial port.
- The fork's deprecation warnings have been cleaned up; both workspaces
  currently pass `moon check --target native` without warnings.

## Completion criteria

This migration is complete when both chapter 2 streaming examples run on the
real Qwen3 checkpoint using native MoonBit tensor operations, the CUDA version
uses standard MoonBit plus externally invoked `nvcc`, no Python runtime is
needed by native inference, cached and non-cached outputs agree, and all project
and fork tests pass.
