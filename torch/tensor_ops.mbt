///|
fn binary_tensor_operation(
  name : String,
  left : Tensor,
  right : Tensor,
) -> Tensor raise TorchError {
  let args = PyTuple::new(2)
  args.set(0, left.obj)
  args.set(1, right.obj)
  expect_tensor(invoke(get_torch_function(name), args, name), name)
}

///|
/// Add two tensors element by element. PyTorch broadcasting rules apply, so the
/// shapes may differ when compatible.
/// Example: `[1, 2] + [3, 4]` produces `[4, 6]`; adding shape `[2]` to `[2, 2]`
/// broadcasts the smaller tensor across rows.
/// Call: `@torch.add(left=first, right=second)`.
pub fn add(left~ : Tensor, right~ : Tensor) -> Tensor raise TorchError {
  binary_tensor_operation("add", left, right)
}

///|
/// Multiply two tensors element by element, with PyTorch broadcasting. This is
/// not matrix multiplication; use `matmul` for that operation.
/// Example: `[1, 2, 3] * [4, 5, 6]` produces `[4, 10, 18]`.
/// Call: `@torch.mul(left=probabilities, right=values)`.
pub fn mul(left~ : Tensor, right~ : Tensor) -> Tensor raise TorchError {
  binary_tensor_operation("mul", left, right)
}

///|
/// Apply `1 / (1 + exp(-x))` to every input element, producing values between
/// zero and one while preserving the input shape.
/// Example: `sigmoid([0.0])` produces `[0.5]`.
/// Call: `@torch.sigmoid(input=logits)`.
pub fn sigmoid(input~ : Tensor) -> Tensor raise TorchError {
  let args = PyTuple::new(1)
  args.set(0, input.obj)
  expect_tensor(
    invoke(get_torch_function("sigmoid"), args, "torch.sigmoid"),
    "torch.sigmoid",
  )
}

///|
/// Convert scores into non-negative weights that sum to 1 along `dim`.
/// The output has the same shape as the input. Use `dim=-1` to normalize along
/// the last dimension, as in attention and vocabulary logits.
/// Example: scores `[1.0, 1.0]` produce weights `[0.5, 0.5]`.
/// Call: `@torch.softmax(input=scores, dim=-1)`.
pub fn softmax(input~ : Tensor, dim~ : Int32) -> Tensor raise TorchError {
  let args = PyTuple::new(2)
  args.set(0, input.obj)
  args.set(1, @python.PyInteger::from(dim.to_int64()))
  expect_tensor(
    invoke(get_torch_function("softmax"), args, "torch.softmax"),
    "torch.softmax",
  )
}

///|
/// Return the indices of the maximum values along `dim` as an integer tensor.
/// The selected dimension is removed because this operation does not retain a
/// singleton dimension. Use `dim=-1` for the last dimension.
/// Example: `argmax([2.0, 7.0, 1.0], dim=0)` returns index `1`.
/// For input shape `[batch, vocabulary]`, `dim=-1` returns shape `[batch]`.
/// Call `reshape([batch, 1])` when the next model input needs that extra axis.
pub fn argmax(input~ : Tensor, dim~ : Int32) -> Tensor raise TorchError {
  let args = PyTuple::new(2)
  args.set(0, input.obj)
  args.set(1, @python.PyInteger::from(dim.to_int64()))
  expect_tensor(
    invoke(get_torch_function("argmax"), args, "torch.argmax"),
    "torch.argmax",
  )
}

///|
/// Return a tensor on the requested device, for example `device="cpu"`,
/// `device="cuda"`, or `device="cuda:0"`. Values and shape are preserved.
/// Example: `let gpu_weight = weight.to(device="cuda")`.
/// This is commonly used for both model weights and their input tensors.
pub fn Tensor::to(self : Tensor, device~ : String) -> Tensor raise TorchError {
  let args = PyTuple::new(1)
  args.set(0, @python.PyString::from(device))
  expect_tensor(self.call("to", args), "Tensor.to")
}

///|
/// Return the tensor's current device as text, such as `cpu` or `cuda:0`.
/// Example: a newly created CPU tensor returns `"cpu"` from `tensor.device()`.
/// The device string can help diagnose operations whose inputs are on different
/// devices.
pub fn Tensor::device(self : Tensor) -> String raise TorchError {
  guard self.obj.get_attr("device") is Some(PyClass(device)) else {
    raise UnexpectedType("Tensor.device is not a torch.device")
  }
  device.to_string()
}

///|
/// Extract the only element of a one-element tensor as `Double`. Use this for a
/// single loss, score, or token ID; calling it on a multi-element tensor fails.
/// Example: a tensor containing only `5` returns `5.0` from `.item()`.
/// Convert the result explicitly when an integer ID is required:
/// `let token_id = next_token.item().to_int64()`.
pub fn Tensor::item(self : Tensor) -> Double raise TorchError {
  match self.call("item", PyTuple::new(0)) {
    @python.PyFloat(value) => value.to_double()
    @python.PyInteger(value) => value.to_double()
    _ => raise UnexpectedType("Tensor.item returned a non-numeric value")
  }
}

///|
/// Run backpropagation from this tensor and accumulate gradients in tensors
/// created with `requires_grad=true`. In the common training case, call this on
/// a scalar loss after the forward pass and before the optimizer step.
/// Example: `loss.backward()`.
pub fn Tensor::backward(self : Tensor) -> Unit raise TorchError {
  invoke_unit(
    get_method(self.obj, "backward"),
    PyTuple::new(0),
    "Tensor.backward",
  )
}

///|
/// Return this tensor's accumulated gradient after `backward()`. The gradient
/// has the same shape as the tensor. A tensor without an accumulated gradient
/// does not produce a valid result.
/// Example: after `loss.backward()`, use `let d_weight = weight.grad()`.
pub fn Tensor::grad(self : Tensor) -> Tensor raise TorchError {
  guard self.obj.get_attr("grad") is Some(PyClass(gradient)) else {
    raise UnexpectedType("Tensor.grad is not a tensor")
  }
  Tensor::{ obj: gradient, }
}

///|
/// Check whether this PyTorch installation can use a compatible CUDA device.
/// Return `true` when CUDA is available, not merely when CUDA support was built
/// into PyTorch. Use the result to choose `device="cuda"` or `device="cpu"`.
/// Example: `let use_cuda = @torch.cuda_is_available()`.
pub fn cuda_is_available() -> Bool raise TorchError {
  guard get_torch_module().get_attr("cuda") is Some(PyModule(cuda)) else {
    raise AttributeMissing("torch.cuda")
  }
  guard cuda.get_attr("is_available") is Some(PyCallable(is_available)) else {
    raise AttributeMissing("torch.cuda.is_available")
  }
  guard invoke(is_available, PyTuple::new(0), "torch.cuda.is_available")
    is @python.PyBool(available) else {
    raise UnexpectedType("torch.cuda.is_available did not return a bool")
  }
  available.to_bool()
}

///|
/// Select one index along `dim` and remove that dimension from the result.
/// Example: selecting row 1 from `[[1, 2, 3], [4, 5, 6]]` with `dim=0, index=1`
/// returns `[4, 5, 6]` with shape `[3]`.
/// This corresponds to PyTorch's `input.select(dim, index)` or `input[index]`.
/// Use `narrow` instead when you want to retain the selected dimension.
pub fn select(
  input~ : Tensor,
  dim~ : Int32,
  index~ : Int32,
) -> Tensor raise TorchError {
  let args = PyTuple::new(2)
  args.set(0, @python.PyInteger::from(dim.to_int64()))
  args.set(1, @python.PyInteger::from(index.to_int64()))
  expect_tensor(input.call("select", args), "Tensor.select")
}

///|
/// Concatenate tensors along an existing dimension. All other dimensions must
/// match. The output size along `dim` is the sum of the input sizes.
/// Example: concatenating `[1, 2]` and `[3, 4]` with `dim=0` returns
/// `[1, 2, 3, 4]`. Two `[2, 4]` matrices with `dim=0` return shape `[4, 4]`.
/// KV-cache updates use this operation to append keys and values along the
/// sequence dimension.
pub fn concat(inputs : Array[Tensor], dim~ : Int32) -> Tensor raise TorchError {
  let values = PyList::new()
  for input in inputs {
    values.append(input.obj)
  }
  let args = PyTuple::new(2)
  args.set(0, values)
  args.set(1, @python.PyInteger::from(dim.to_int64()))
  expect_tensor(
    invoke(get_torch_function("cat"), args, "torch.cat"),
    "torch.cat",
  )
}

///|
/// Create a CPU `float32` tensor filled with pseudo-random values uniformly
/// distributed in `[0, 1)`. The shape's product determines the element count.
/// Example: `let matrix = @torch.rand([3, 3])` creates nine values in a `3 × 3`
/// matrix. Use `Tensor.to(device="cuda")` afterward for GPU execution.
pub fn rand(shape : Array[Int32]) -> Tensor raise TorchError {
  ignore(shape_size(shape))
  let args = PyTuple::new(1)
  args.set(0, shape_arguments(shape))
  expect_tensor(
    invoke(get_torch_function("rand"), args, "torch.rand"),
    "torch.rand",
  )
}

///|
/// Multiply every input element by the same scalar while preserving shape.
/// Example: `[1.0, 2.0]` scaled by `0.5` produces `[0.5, 1.0]`.
/// Call: `@torch.scale(input=value, scalar=attention_weight)`.
/// This is scalar multiplication, not tensor-tensor element-wise `mul`.
pub fn scale(input~ : Tensor, scalar~ : Double) -> Tensor raise TorchError {
  let args = PyTuple::new(1)
  args.set(0, @python.PyFloat::from(scalar))
  expect_tensor(input.call("mul", args), "Tensor.mul")
}

///|
/// Compute a dot product. For two vectors, the result is a one-element scalar
/// tensor; call `.item()` to extract its numeric value.
/// Example: `[1, 2, 3]` dot `[4, 5, 6]` is `1×4 + 2×5 + 3×6 = 32`.
/// PyTorch also permits vector/matrix forms, with shape rules determined by
/// operand rank. Use `matmul` for general matrix multiplication.
pub fn dot(left~ : Tensor, right~ : Tensor) -> Tensor raise TorchError {
  binary_tensor_operation("dot", left, right)
}

///|
/// Expose the underlying Python tensor without copying for custom CPython
/// integrations. Most callers should use the tensor operations instead; this is
/// an escape hatch for PyTorch APIs not wrapped yet.
/// Example: `let python_object = tensor.as_python()`.
pub fn Tensor::as_python(self : Tensor) -> PyObject {
  self.obj
}

///|
/// Wrap an existing Python object as a tensor without copying. The caller must
/// guarantee that `obj` is a `torch.Tensor`; the wrapper cannot validate that
/// contract from the MoonBit type alone. Use this with `as_python` when building
/// a thin wrapper around another PyTorch operation.
/// Example: `let tensor = Tensor::from_python(python_object)`.
pub fn Tensor::from_python(obj : PyObject) -> Tensor {
  { obj, }
}

///|
/// Convert values to a named PyTorch dtype while preserving shape. Typical names
/// include `float32`, `float16`, `bfloat16`, and `int64`.
/// Example: `let bf16 = tensor.to_dtype(dtype="bfloat16")` converts values such
/// as `1.0` and `2.0` to BF16 while retaining the same tensor shape.
pub fn Tensor::to_dtype(
  self : Tensor,
  dtype~ : String,
) -> Tensor raise TorchError {
  guard get_torch_module().get_attr(dtype) is Some(PyClass(dtype_object)) else {
    raise AttributeMissing("torch.\{dtype}")
  }
  let args = PyTuple::new(1)
  args.set(0, dtype_object)
  expect_tensor(self.call("to", args), "Tensor.to(dtype)")
}

///|
/// Select a consecutive range of `length` elements beginning at `start` along
/// `dim`. Unlike `select`, this retains the dimension with size `length`.
/// Example: `[1, 2, 3, 4, 5, 6]` with `dim=0, start=2, length=3` returns
/// `[3, 4, 5]` with shape `[3]`. For shape `[4, 6]`, narrowing with `dim=1,
/// start=1, length=3` returns shape `[4, 3]`.
/// RoPE uses this operation to split the first and second halves of head vectors.
pub fn narrow(
  input~ : Tensor,
  dim~ : Int32,
  start~ : Int32,
  length~ : Int32,
) -> Tensor raise TorchError {
  let args = PyTuple::new(3)
  args.set(0, @python.PyInteger::from(dim.to_int64()))
  args.set(1, @python.PyInteger::from(start.to_int64()))
  args.set(2, @python.PyInteger::from(length.to_int64()))
  expect_tensor(input.call("narrow", args), "Tensor.narrow")
}

///|
/// Insert a new dimension of size 1 at `dim`, increasing rank by one.
/// Example: a `[2, 3]` matrix becomes `[1, 2, 3]` with `dim=0`; its values remain
/// unchanged. This is commonly used to make shapes compatible for broadcasting.
pub fn unsqueeze(input~ : Tensor, dim~ : Int32) -> Tensor raise TorchError {
  let args = PyTuple::new(1)
  args.set(0, @python.PyInteger::from(dim.to_int64()))
  expect_tensor(input.call("unsqueeze", args), "Tensor.unsqueeze")
}

///|
/// Repeat each element along `dim` `repeats` times before moving to the next
/// element. Qwen3 uses this to share each key/value head with a group of query
/// heads. Example: `[10, 20]` with `repeats=2, dim=0` becomes `[10, 10, 20, 20]`.
pub fn repeat_interleave(
  input~ : Tensor,
  repeats~ : Int32,
  dim~ : Int32,
) -> Tensor raise TorchError {
  let args = PyTuple::new(2)
  args.set(0, @python.PyInteger::from(repeats.to_int64()))
  args.set(1, @python.PyInteger::from(dim.to_int64()))
  expect_tensor(
    input.call("repeat_interleave", args),
    "Tensor.repeat_interleave",
  )
}

///|
/// Create a boolean tensor whose elements are all `true`, directly on `device`.
/// Despite its general name, this wrapper always creates `torch.bool` values.
/// Example: `@torch.ones([2, 2], device="cpu")` produces
/// `[[true, true], [true, true]]`. It is useful for constructing causal masks
/// without transferring a mask from CPU to GPU afterward.
pub fn ones(shape : Array[Int32], device~ : String) -> Tensor raise TorchError {
  ignore(shape_size(shape))
  let args = PyTuple::new(1)
  args.set(0, shape_arguments(shape))
  guard get_torch_module().get_attr("bool") is Some(PyClass(dtype)) else {
    raise AttributeMissing("torch.bool")
  }
  let kwargs = @python.PyDict::new()
  kwargs.set("dtype", dtype)
  kwargs.set("device", @python.PyString::from(device))
  expect_tensor(
    invoke_with_kwargs(get_torch_function("ones"), args, kwargs, "torch.ones"),
    "torch.ones",
  )
}

///|
/// Return the upper-triangular part of a matrix. Entries below the selected
/// diagonal become zero. With `diagonal=1`, the main diagonal is also excluded.
/// Example: `[[1, 2], [3, 4]]` becomes `[[0, 2], [0, 0]]` with `diagonal=1`.
/// This is the first step in constructing Qwen3's causal attention mask.
pub fn triu(input~ : Tensor, diagonal~ : Int32) -> Tensor raise TorchError {
  let args = PyTuple::new(2)
  args.set(0, input.obj)
  args.set(1, @python.PyInteger::from(diagonal.to_int64()))
  expect_tensor(
    invoke(get_torch_function("triu"), args, "torch.triu"),
    "torch.triu",
  )
}

///|
/// Replace elements with `value` wherever the boolean `mask` is true. The mask
/// may broadcast across input dimensions. Example: scores `[1.0, 2.0]` with mask
/// `[false, true]` and `value=0.0` produce `[1.0, 0.0]`. Attention uses negative
/// infinity so masked positions receive zero weight after softmax.
/// Call: `@torch.masked_fill(input=scores, mask=causal_mask, value=-1.0e30)`.
pub fn masked_fill(
  input~ : Tensor,
  mask~ : Tensor,
  value~ : Double,
) -> Tensor raise TorchError {
  let args = PyTuple::new(2)
  args.set(0, mask.obj)
  args.set(1, @python.PyFloat::from(value))
  expect_tensor(input.call("masked_fill", args), "Tensor.masked_fill")
}

///|
/// Eagerly load every tensor in a safetensors file into a name-to-tensor map.
/// Checkpoint names and shapes are preserved. The returned tensors initially use
/// the loader's default device; call `.to(device=...)` on each tensor when GPU
/// placement is required. The Python `safetensors` package must be installed.
///
/// Example: `let weights = @torch.load_safetensors("model.safetensors")`.
pub fn load_safetensors(path : String) -> Map[String, Tensor] raise TorchError {
  guard @python.pyimport("safetensors.torch", print_err=true)
    is Some(python_module) else {
    raise ImportFailed("safetensors.torch")
  }
  let args = PyTuple::new(1)
  args.set(0, @python.PyString::from(path))
  guard invoke(
      get_module_function(python_module, "load_file"),
      args,
      "load_file",
    )
    is PyDict(state_dict) else {
    raise UnexpectedType("safetensors.load_file did not return a dictionary")
  }
  let tensors = Map([])
  let items = state_dict.items()
  for index in 0..<items.len() {
    guard items.get(index) is Some(PyTuple(item)) else {
      raise UnexpectedType("state dictionary item is not a tuple")
    }
    guard item.get(0) is Some(@python.PyString(name)) else {
      raise UnexpectedType("state dictionary key is not a string")
    }
    guard item.get(1) is Some(PyClass(value)) else {
      raise UnexpectedType("state dictionary value is not a tensor")
    }
    tensors.set(name.to_string(), Tensor::from_python(value))
  }
  tensors
}
