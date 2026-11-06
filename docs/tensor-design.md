# Tensor design principles

## Native Qwen3 backend

Chapter 2 now uses `NativeTensor` to run the same Qwen3 model and generation
logic on `NpArray` (CPU) and `GpuTensor` (CUDA). Generic tensor operations live in
`moonxi-net`; Qwen3 architecture, checkpoint loading, KV caching, tokenization,
and generation remain in `moonagent`. Native execution uses F32 and does not
initialize CPython. The PyTorch backend described below remains available as a
reference implementation.

## Torch purpose

The `zw/torch` package provides MoonBit access to real PyTorch tensors and
operations through `python.mbt` and the CPython C API. Qwen-specific architecture
and inference orchestration remain in `qwen3`; educational model-training code
lives in `torch_example`.

The former native `zw/tensor` package has been removed. Numerical kernels,
broadcasting, strides, device placement, and autograd are delegated to PyTorch
rather than reimplemented in MoonBit.

## Torch tensors are Python objects

`Tensor` wraps a `PyObject` that refers to an actual `torch.Tensor`.
Operations such as matrix multiplication, softmax, RMSNorm, and attention execute
inside PyTorch. `TorchModule`, `TorchOptimizer`, and `TorchDataLoader` similarly
wrap their Python counterparts.

MoonBit code provides typed function signatures and labeled arguments, but values
arriving from Python still require runtime type checks. Boundary failures use
`TorchError` with distinct import, lookup, invocation, shape, and type cases.

## Data conversion and ownership

Floating-point tensor construction currently copies `Array[Double]` through a
Python list, producing PyTorch's default `float32` tensor. Integer labels use
`int_tensor` with `Array[Int64]`. `to_array()` flattens a tensor through
`reshape([-1]).tolist()` and copies the values back to `Array[Double]`.

There is currently no zero-copy buffer interface. Keep computations as
coarse-grained PyTorch operations where practical instead of repeatedly crossing
the boundary for individual values.

`select`, `reshape`, `view`, and `transpose` preserve PyTorch's own view and
stride semantics. MoonBit code does not expose a separate contiguous-only
`TensorView` abstraction.

## Runtime and devices

The bridge is native-only and requires CPython 3.12 development libraries plus a
Python environment containing PyTorch. The embedded interpreter must be able to
find that environment's `site-packages`, currently configured with `PYTHONPATH`.
MoonBit dependency declarations do not install Python packages.

Tensors and modules can be transferred with `.to(device="cpu")` or
`.to(device="cuda")`. Operands must follow PyTorch's normal device rules.
Verified environments use PyTorch 2.13.0 with CUDA 12.6 on an RTX 3090.

## Autograd and training

PyTorch owns computation graphs and gradients. Tensor constructors reshape first
and then apply `requires_grad_(true)` so trainable values remain leaf tensors.
Training code uses the same `zero_grad()`, `backward()`, and `step()` sequence as
PyTorch.

`torch_example` uses `torch.nn.Sequential`, `TensorDataset`, `DataLoader`,
cross-entropy, and SGD. DataLoader epochs are currently materialized into
MoonBit batch arrays; this is simple for toy examples but is not intended for
large streaming datasets.

## Incremental feature growth

Add wrappers when concrete MoonBit examples require them, rather than attempting
to duplicate the entire PyTorch API. Preserve PyTorch behavior and validate
results against real PyTorch execution. Priorities for deeper integration are
zero-copy buffers, explicit Python object lifetime management, checkpointing,
and scalable iteration.
