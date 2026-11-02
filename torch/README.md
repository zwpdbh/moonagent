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

## Using the local `python.mbt` fork as a dependency

The fork is an independent Git repository hosted at
[https://github.com/zwpdbh/python.mbt.git](https://github.com/zwpdbh/python.mbt.git). If it has not been cloned yet,
create it beside the `moonagent` workspace:

```bash
cd /home/zw/code/moonbit_programming
git clone https://github.com/zwpdbh/python.mbt.git forked/python.mbt
```

The resulting layout is:

```text
/home/zw/code/moonbit_programming/moonagent
/home/zw/code/moonbit_programming/forked/python.mbt
```

This project does not import the fork through its Git URL. Instead, MoonBit
resolves it as a local workspace member. The workspace file
`moonagent/moon.work` contains:

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

Member paths are relative to `moon.work`, so `../forked/python.mbt` resolves to
`/home/zw/code/moonbit_programming/forked/python.mbt`. Local changes in the fork
are therefore used by subsequent MoonBit builds without publishing or
reinstalling a package.

The fork's own `moon.mod` declares the module name
`Kaida-Amethyst/python` and version `0.1.6`. Consumers use that module name,
not the Git repository URL. Two declarations are needed in `zw/torch`:

1. `torch/moon.mod` makes the module and its version available to the
   `zw/torch` module:

   ```toml
   import {
     "Kaida-Amethyst/python@0.1.6",
   }
   ```
2. `torch/moon.pkg` imports the package into the current package's namespace:

   ```toml
   import {
     "Kaida-Amethyst/python",
   }

   supported_targets = "native"
   ```

This allows `torch/torch.mbt` to use APIs such as `@python.pyimport`,
`PyObject`, `PyCallable`, `PyTuple`, and `PyList`. The native restriction comes
from the fork's CPython C-API integration.

Because the fork is a workspace member, no `moon add` command is needed for
this local setup. Running `moon add Kaida-Amethyst/python` would instead be the
starting point for using the published package, not for selecting this local
Git fork.

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
uv venv --python /usr/bin/python3.12 --clear .venv
uv pip install --python .venv/bin/python \
  --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu126 \
  torch==2.13.0 numpy
```

This installs the verified PyTorch 2.13.0 + CUDA 12.6 stack and NumPy only in
`.venv`, leaving the system Python unchanged. The SJTU mirror can substantially
speed up downloads in China. To use the official PyTorch repository instead,
replace the mirror URL with `https://download.pytorch.org/whl/cu126`. `.venv/`
is excluded by the repository's `.gitignore`.

Verify the installation and CUDA detection:

```bash
.venv/bin/python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

The expected output is `2.13.0+cu126` followed by `True` on a compatible NVIDIA
system.

## 3. Make the environment visible to embedded CPython

Running `moon test` without additional configuration can fail with:

```text
ModuleNotFoundError: No module named 'torch'
zw/torch.TorchError.ImportFailed
```

This can happen even when `.venv/bin/python -c 'import torch'` succeeds.
`moon test` starts a native MoonBit executable that embeds the system CPython
3.12 library; it does not launch `.venv/bin/python`. Consequently, the embedded
interpreter does not automatically search the virtual environment's
`site-packages`. The `Kaida-Amethyst/python` MoonBit dependency provides only
the bridge, not the PyTorch package itself.

Set `PYTHONPATH` explicitly when running tests or programs:

```bash
export PYTHONPATH=/home/zw/code/moonbit_programming/moonagent/.venv/lib/python3.12/site-packages
```

Then run the tests from `torch/`:

```bash
cd /home/zw/code/moonbit_programming/moonagent/torch
moon test --target native
```

The recommended project-specific, one-command form is:

```bash
cd /home/zw/code/moonbit_programming/moonagent/torch
PYTHONPATH=/home/zw/code/moonbit_programming/moonagent/.venv/lib/python3.12/site-packages \
  moon test --target native
```

### Optional: persist the setting in your shell

For a personal setup where you want plain `moon test` to work in every new Bash
terminal, add this line to `~/.bashrc` and then run `source ~/.bashrc`:

```bash
export PYTHONPATH="/home/zw/code/moonbit_programming/moonagent/.venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
```

For Zsh, add the same line to `~/.zshrc` and run `source ~/.zshrc`.

A shell-wide `PYTHONPATH` affects every Python program started from that shell,
including other projects and virtual environments. The project-specific command
above is therefore preferable when working with multiple Python environments.
Avoid installing PyTorch into the system Python merely to bypass this search-
path issue.

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

| PyTorch                | MoonBit wrapper                                  |
| ---------------------- | ------------------------------------------------ |
| `torch.tensor(data)` | `@torch.tensor(data=values, shape=dimensions)` |
| `x.reshape(3, 2)`    | `x.reshape([3, 2])`                            |
| `x.view(3, 2)`       | `x.view([3, 2])`                               |
| `x.transpose(0, 1)`  | `x.transpose(dim0=0, dim1=1)`                  |
| `x.T`                | `x.t()`                                        |
| `x.matmul(y)`        | `@torch.matmul(left=x, right=y)`               |
| `x.shape`            | `x.shape()`                                    |
| `x.dtype`            | `x.dtype()`                                    |
| `x.tolist()`         | `x.to_array()` for a flat `Array[Double]`    |

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

`torch/torch_test.mbt` checks examples from sections 2.1–2.3 of the article:

- Creating scalar, vector, matrix, and 3D tensors.
- Reading shapes and the default `torch.float32` dtype.
- Reshaping and viewing a `[2, 3]` tensor as `[3, 2]`.
- Transposing a matrix.
- Verifying `tensor2d.matmul(tensor2d.T)` returns
  `[[14, 32], [32, 77]]`.

These tests exercise real calls through MoonBit, `python.mbt`, CPython, and
PyTorch rather than comparing against a mock implementation.

## Training example

`torch_example/training.mbt` ports the article's neural-network and training
sections. It builds the same `Linear → ReLU → Linear → ReLU → Linear`
architecture, uses the five-example toy dataset, and trains with mini-batches,
cross-entropy loss, and SGD. It also computes train/test accuracy and supports
CPU or CUDA devices.

The example uses PyTorch's `TensorDataset`, which is equivalent to the article's
custom in-memory `ToyDataset` for this use case. `TorchDataLoader.batches()`
materializes one epoch as MoonBit batches so the training loop can use ordinary
MoonBit `for` loops.

Run the example from the workspace root after setting up Python as described
above:

```bash
cd /home/zw/code/moonbit_programming/moonagent/torch_example
PYTHONPATH=/home/zw/code/moonbit_programming/moonagent/.venv/lib/python3.12/site-packages \
  moon run cmd/main --target native
```

`torch_example/training_test.mbt` adds further article-based checks:

- The logistic-regression autograd example and its weight/bias gradients.
- Three training epochs, decreasing loss, and 100% train/test accuracy.
- Tensor addition on CUDA when a GPU is available.
- A CUDA training epoch when a GPU is available.

## Current limitations and next steps

- The wrapper is native-only and requires CPython plus an installed PyTorch
  package.
- Floating-point tensor construction accepts `Array[Double]`; integer labels
  use `int_tensor` with `Array[Int64]`.
- Array/list conversion copies data. There is no zero-copy buffer integration.
- Models use wrapped `torch.nn.Sequential` modules rather than MoonBit-defined
  subclasses of `torch.nn.Module`.
- Model serialization and distributed/multi-GPU training are not wrapped yet.

Useful next milestones are model checkpointing, more dataset/iterator
operations, and broader `torch.nn` coverage.

## CUDA status

PyTorch `2.13.0+cu126` is verified on the host's NVIDIA GeForce RTX 3090.
`torch.cuda.is_available()` returns `True`, and an actual CUDA matrix
multiplication produced the expected `[[14, 32], [32, 77]]` result. The
MoonBit integration tests also verify CUDA tensor addition and a CUDA training
epoch through `zw/torch`.

The previously tested `torch 2.15.0+cu132` build segfaulted during CUDA
detection. Rebuilding `.venv` cleanly with the CUDA 12.6 stack resolved that
issue.
