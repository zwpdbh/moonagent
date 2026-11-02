# Using PyTorch from MoonBit

This package explores machine learning with real PyTorch while writing the
surrounding code in MoonBit. It uses `python.mbt` to call CPython directly; it
does not reimplement PyTorch operations in MoonBit.

The current proof of concept covers tensor creation, shapes, data types,
reshaping, views, transposition, matrix multiplication, and conversion back to
MoonBit arrays. The examples and tests follow
[PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs](https://sebastianraschka.com/teaching/pytorch-1h/).

## Architecture

```text
MoonBit code
    |
    v
zw/torch wrapper
    |
    v
Kaida-Amethyst/python (python.mbt)
    |
    v
CPython C API
    |
    v
Python's torch package
```

`TorchTensor` wraps a Python `torch.Tensor` object. Operations such as
`matmul`, `reshape`, and `transpose` execute inside PyTorch. MoonBit arrays and
Python lists are currently converted by copying.

The existing native `zw/tensor` package is separate from this experiment.

## Repository layout

This guide assumes these existing paths:

```text
/home/zw/code/moonbit_programming/moonagent
/home/zw/code/moonbit_programming/forked/python.mbt
```

The workspace file `moonagent/moon.work` includes both the wrapper and the
local `python.mbt` fork:

```toml
members = [
  "./playground",
  "./utils",
  "./qwen3",
  "./tensor",
  "./torch",
  "../forked/python.mbt",
]
```

The fork provides package `Kaida-Amethyst/python@0.1.6`. The `torch/moon.mod`
module imports that version, and `torch/moon.pkg` imports the package for the
native target.

## 1. Install prerequisites

The integration requires:

- A recent MoonBit toolchain with native-target support.
- Python 3.12 and its development files (`Python.h`, `python3-config`, and
  `libpython3.12`).
- A C compiler.
- [`uv`](https://docs.astral.sh/uv/) for creating an isolated Python
  environment.

Python 3.12 is used here because the bridge build links against the system
Python 3.12 library. Using a virtual environment with a different Python
version can make packages unavailable to, or incompatible with, the embedded
interpreter.

## 2. Create the Python environment

Run these commands from the workspace root:

```bash
cd /home/zw/code/moonbit_programming/moonagent
uv venv --python /usr/bin/python3.12 .venv
uv pip install --python .venv/bin/python torch numpy
```

This installs PyTorch and NumPy only in `.venv`, leaving the system Python
unchanged. `.venv/` is excluded by the repository's `.gitignore`.

Verify the installation without GPU initialization:

```bash
.venv/bin/python -c 'import torch; print(torch.__version__)'
```

## 3. Make the environment visible to embedded CPython

`python.mbt` embeds CPython in the MoonBit native executable. Merely creating or
activating `.venv` does not reliably make its packages visible to that embedded
interpreter, so set `PYTHONPATH` explicitly when running tests or programs:

```bash
export PYTHONPATH=/home/zw/code/moonbit_programming/moonagent/.venv/lib/python3.12/site-packages
```

Then run the tests from `torch/`:

```bash
cd /home/zw/code/moonbit_programming/moonagent/torch
moon test --target native
```

Without activating the environment in the shell, the equivalent one-command
form is:

```bash
cd /home/zw/code/moonbit_programming/moonagent/torch
PYTHONPATH=/home/zw/code/moonbit_programming/moonagent/.venv/lib/python3.12/site-packages \
  moon test --target native
```

## 4. Follow PyTorch code in MoonBit

The article's common tensor-operation example looks like this in Python:

```python
tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
transposed = tensor2d.T
result = tensor2d.matmul(transposed)
```

The corresponding MoonBit test code is:

```moonbit
let tensor2d = @torch.tensor(
  data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
  shape=[2, 3],
)
let transposed = tensor2d.t()
let result = @torch.matmul(left=tensor2d, right=transposed)

assert_eq(result.shape(), [2, 2])
assert_eq(result.to_array(), [14.0, 32.0, 32.0, 77.0])
```

The wrapper currently uses flat input data plus an explicit shape instead of
nested Python-style lists:

| PyTorch | MoonBit wrapper |
| --- | --- |
| `torch.tensor(data)` | `@torch.tensor(data=values, shape=dimensions)` |
| `x.reshape(3, 2)` | `x.reshape([3, 2])` |
| `x.view(3, 2)` | `x.view([3, 2])` |
| `x.transpose(0, 1)` | `x.transpose(dim0=0, dim1=1)` |
| `x.T` | `x.t()` |
| `x.matmul(y)` | `@torch.matmul(left=x, right=y)` |
| `x.shape` | `x.shape()` |
| `x.dtype` | `x.dtype()` |
| `x.tolist()` | `x.to_array()` for a flat `Array[Double]` |

`reshape` and `view` delegate shape validation to PyTorch, so PyTorch features
such as a single inferred `-1` dimension remain available.

## 5. How the binding works

The implementation in `torch/torch.mbt` performs these steps:

1. Import and cache Python's `torch` module with `@python.pyimport("torch")`.
2. Convert MoonBit numeric arrays into Python lists.
3. Resolve module functions and tensor methods with `get_attr`.
4. Build Python positional arguments with `PyTuple`.
5. Invoke the resolved `PyCallable`.
6. Check that tensor results contain Python objects and wrap them as
   `TorchTensor`.
7. Convert `Tensor.tolist()` results back into `Array[Double]` when requested.

Errors at this dynamic boundary are represented by `TorchError`. Python import,
attribute lookup, invocation, result-type, and constructor-shape failures are
kept distinct.

## Tests

`torch/torch_test.mbt` currently checks examples from sections 2.1–2.3 of the
article:

- Creating scalar, vector, matrix, and 3D tensors.
- Reading shapes and the default `torch.float32` dtype.
- Reshaping and viewing a `[2, 3]` tensor as `[3, 2]`.
- Transposing a matrix.
- Verifying `tensor2d.matmul(tensor2d.T)` returns
  `[[14, 32], [32, 77]]`.

These tests exercise real calls through MoonBit, `python.mbt`, CPython, and
PyTorch rather than comparing against a mock implementation.

## Current limitations and next steps

- The wrapper is native-only and requires CPython plus an installed PyTorch
  package.
- Tensor construction currently accepts `Array[Double]`, which PyTorch receives
  as Python floats and converts to its default `float32` tensors.
- Array/list conversion copies data. There is no zero-copy buffer integration.
- Autograd, `torch.nn`, data loaders, training loops, model serialization, and
  device transfer are not wrapped yet.
- The article-based tests currently exercise CPU tensors.

A useful next milestone is the article's automatic-differentiation example:
support `requires_grad`, arithmetic operations, `sigmoid`, binary cross
entropy, `backward`, and access to `.grad`.

## CUDA status

The host detects an NVIDIA GeForce RTX 3090. However, with the currently
installed `torch 2.15.0+cu132` build, calling `torch.cuda.is_available()` causes
a segmentation fault even in plain Python. CUDA execution is therefore not yet
verified, and this issue is outside the MoonBit wrapper: the article-based CPU
tests pass through the complete MoonBit-to-PyTorch path.
