///|
/// Core bridge between MoonBit and real PyTorch. `Tensor` holds a CPython
/// `torch.Tensor`, while functions in this package delegate numerical work to
/// PyTorch rather than reimplementing kernels in MoonBit.
using @python {
  type PyCallable,
  type PyList,
  type PyModule,
  type PyObject,
  type PyObjectEnum,
  type PyTuple,
}

///|
/// Errors raised at the dynamic CPython boundary.
/// - ImportFailed: Python or safetensors cannot be imported.
/// - AttributeMissing: a required module function, tensor method, or attribute
///   does not exist.
/// - CallFailed: invoking a Python callable raised an exception.
/// - UnexpectedType: a result has an unexpected Python type or value.
/// - InvalidShape: MoonBit data and requested dimensions are inconsistent.
pub(all) suberror TorchError {
  ImportFailed(String)
  AttributeMissing(String)
  CallFailed(String)
  UnexpectedType(String)
  InvalidShape(String)
}

///|
/// Format an error with both its category and operation-specific details.
pub impl Show for TorchError with fn output(self, logger) {
  let (kind, message) = match self {
    ImportFailed(message) => ("import failed", message)
    AttributeMissing(message) => ("attribute missing", message)
    CallFailed(message) => ("call failed", message)
    UnexpectedType(message) => ("unexpected type", message)
    InvalidShape(message) => ("invalid shape", message)
  }
  logger.write_string("\{kind}: \{message}")
}

///|
pub extend TorchError with Show::{to_string, output}

///|
/// A lightweight owner of a Python `torch.Tensor` object.
/// Tensor operations usually reuse the underlying Python object; conversion to
/// a MoonBit array is a separate copying operation performed by `to_array`.
pub struct Tensor {
  priv obj : PyObject
}

///|
/// Create a lazy import thunk. Delaying this import until a tensor operation is
/// used lets applications configure `sys.path` before importing PyTorch.
fn import_torch_module() -> () -> PyModule? {
  fn() { @python.pyimport("torch", print_err=true) }
}

///|
let torch_module : () -> PyModule? = import_torch_module()

///|
/// Return Python's `torch` module, or raise ImportFailed if it is unavailable.
/// CPython's own import cache reuses an already imported module on later calls.
fn get_torch_module() -> PyModule raise TorchError {
  guard torch_module() is Some(torch_mod) else { raise ImportFailed("torch") }
  torch_mod
}

///|
/// Resolve a function such as `matmul` from the `torch` module.
fn get_torch_function(name : String) -> PyCallable raise TorchError {
  guard get_torch_module().get_attr(name) is Some(PyCallable(function)) else {
    raise AttributeMissing("torch.\{name}")
  }
  function
}

///|
/// Resolve a callable attribute from any Python object, such as `reshape` on a
/// tensor. This lookup happens dynamically for every operation.
fn get_method(object : PyObject, name : String) -> PyCallable raise TorchError {
  guard object.get_attr(name) is Some(PyCallable(callable)) else {
    raise AttributeMissing(name)
  }
  callable
}

///|
/// Invoke a Python callable with positional and keyword arguments. This variant
/// allows `None` as a legitimate Python result for methods such as `eval`.
fn invoke_optional(
  callable : PyCallable,
  args : PyTuple,
  kwargs : @python.PyDict,
  name : String,
) -> PyObjectEnum? raise TorchError {
  callable.invoke(args~, kwargs~, print_err=true) catch {
    _ => raise CallFailed(name)
  }
}

///|
/// Invoke with only positional arguments and require a non-`None` result.
/// Tensor-producing operations use this wrapper.
fn invoke(
  callable : PyCallable,
  args : PyTuple,
  name : String,
) -> PyObjectEnum raise TorchError {
  guard invoke_optional(callable, args, @python.PyDict::new(), name)
    is Some(value) else {
    raise UnexpectedType("\{name} returned None")
  }
  value
}

///|
/// Invoke with both positional arguments and a keyword dictionary. For example,
/// RMSNorm passes `weight` and `eps` by name.
fn invoke_with_kwargs(
  callable : PyCallable,
  args : PyTuple,
  kwargs : @python.PyDict,
  name : String,
) -> PyObjectEnum raise TorchError {
  guard invoke_optional(callable, args, kwargs, name) is Some(value) else {
    raise UnexpectedType("\{name} returned None")
  }
  value
}

///|
/// Invoke a method for its side effect and discard the returned Python value.
/// Used for operations such as `backward`, `eval`, and optimizer steps.
fn invoke_unit(
  callable : PyCallable,
  args : PyTuple,
  name : String,
) -> Unit raise TorchError {
  ignore(invoke_optional(callable, args, @python.PyDict::new(), name))
}

///|
/// Wrap a Python object as a `Tensor`. `operation` is included in the error so
/// callers can tell which PyTorch call returned an unexpected type.
fn expect_tensor(
  value : PyObjectEnum,
  operation : String,
) -> Tensor raise TorchError {
  guard value is PyClass(object) else {
    raise UnexpectedType("\{operation} did not return a torch.Tensor")
  }
  Tensor::{ obj: object, }
}

///|
/// Compute the element count of a shape. For example, `[2, 3]` requires six
/// values, while the scalar shape `[]` requires one. Negative dimensions are
/// rejected before calling PyTorch.
fn shape_size(shape : Array[Int32]) -> Int32 raise TorchError {
  let mut size = 1
  for dimension in shape {
    if dimension < 0 {
      raise InvalidShape("tensor dimensions must not be negative")
    }
    size *= dimension
  }
  size
}

///|
/// Convert MoonBit dimensions into a Python tuple used by `reshape` and `view`.
fn shape_arguments(shape : Array[Int32]) -> PyTuple {
  let args = PyTuple::new(shape.length().to_uint64())
  for index, dimension in shape {
    args.set(index, @python.PyInteger::from(dimension.to_int64()))
  }
  args
}

///|
/// Call a method on the underlying Python tensor with positional arguments.
/// This is the low-level helper behind operations such as `reshape` and `view`.
fn Tensor::call(
  self : Tensor,
  name : String,
  args : PyTuple,
) -> PyObjectEnum raise TorchError {
  invoke(get_method(self.obj, name), args, name)
}

///|
/// Construct a floating-point PyTorch tensor from flat values and a shape.
/// Values are copied into a Python list and reshaped, so `[1, 2, 3, 4, 5, 6]`
/// with shape `[2, 3]` becomes a 2-by-3 matrix. By default PyTorch uses
/// `torch.float32`. Set `requires_grad=true` for trainable parameters.
pub fn tensor(
  data~ : Array[Double],
  shape~ : Array[Int32],
  requires_grad~ : Bool = false,
) -> Tensor raise TorchError {
  if data.length() != shape_size(shape) {
    raise InvalidShape("data length does not match tensor shape")
  }
  let values = PyList::from(data.map(@python.PyFloat::from))
  let args = PyTuple::new(1)
  args.set(0, values)
  let result = expect_tensor(
    invoke(get_torch_function("tensor"), args, "torch.tensor"),
    "torch.tensor",
  ).reshape(shape)
  if requires_grad {
    let grad_args = PyTuple::new(1)
    grad_args.set(0, @python.PyBool::from(true))
    expect_tensor(
      result.call("requires_grad_", grad_args),
      "Tensor.requires_grad_",
    )
  } else {
    result
  }
}

///|
/// Construct an integer tensor from flat values and a shape. This is used for
/// token IDs and class labels. PyTorch typically infers an integer dtype such
/// as `torch.int64` from these values.
pub fn int_tensor(
  data~ : Array[Int64],
  shape~ : Array[Int32],
) -> Tensor raise TorchError {
  if data.length() != shape_size(shape) {
    raise InvalidShape("data length does not match tensor shape")
  }
  let values = PyList::from(data.map(@python.PyInteger::from))
  let args = PyTuple::new(1)
  args.set(0, values)
  expect_tensor(
    invoke(get_torch_function("tensor"), args, "torch.tensor"),
    "torch.tensor",
  ).reshape(shape)
}

///|
/// Package the dimensions as one tuple argument. PyTorch also permits one `-1`
/// entry, which infers that dimension from the element count.
fn reshape_arguments(shape : Array[Int32]) -> PyTuple {
  let args = PyTuple::new(1)
  args.set(0, shape_arguments(shape))
  args
}

///|
/// Change a tensor's shape without changing its element count. For example,
/// `[2, 3]` and `[3, 2]` both contain six values. PyTorch may return a view or a
/// copy depending on the original layout.
pub fn Tensor::reshape(
  self : Tensor,
  shape : Array[Int32],
) -> Tensor raise TorchError {
  expect_tensor(
    self.call("reshape", reshape_arguments(shape)),
    "Tensor.reshape",
  )
}

///|
/// Reinterpret the same underlying data with a new shape. Unlike `reshape`,
/// `view` requires a compatible memory layout and never makes a copy merely to
/// make the shape work. This is useful for zero-copy slices of contiguous data.
pub fn Tensor::view(
  self : Tensor,
  shape : Array[Int32],
) -> Tensor raise TorchError {
  expect_tensor(self.call("view", reshape_arguments(shape)), "Tensor.view")
}

///|
/// Swap two dimensions while preserving the other dimensions. For a matrix,
/// `transpose(dim0=0, dim1=1)` changes shape `[rows, columns]` to
/// `[columns, rows]`.
pub fn Tensor::transpose(
  self : Tensor,
  dim0~ : Int32,
  dim1~ : Int32,
) -> Tensor raise TorchError {
  let args = PyTuple::new(2)
  args.set(0, @python.PyInteger::from(dim0.to_int64()))
  args.set(1, @python.PyInteger::from(dim1.to_int64()))
  expect_tensor(self.call("transpose", args), "Tensor.transpose")
}

///|
/// Access PyTorch's `T` attribute, the common shorthand for transposing a
/// two-dimensional matrix. For example, a `[2, 3]` tensor's `t()` has shape
/// `[3, 2]`.
pub fn Tensor::t(self : Tensor) -> Tensor raise TorchError {
  guard self.obj.get_attr("T") is Some(attribute) else {
    raise AttributeMissing("Tensor.T")
  }
  expect_tensor(attribute, "Tensor.T")
}

///|
/// Return the PyTorch dtype name, such as `torch.float32`, `torch.bfloat16`, or
/// `torch.int64`. Use `to_dtype` when you need to change the dtype instead.
pub fn Tensor::dtype(self : Tensor) -> String raise TorchError {
  guard self.obj.get_attr("dtype") is Some(PyClass(dtype)) else {
    raise UnexpectedType("Tensor.dtype is not a torch.dtype")
  }
  dtype.to_string()
}

///|
/// Perform PyTorch matrix multiplication. With two matrices, the left shape is
/// `[m, k]` and the right shape is `[k, n]`, producing `[m, n]`. For example,
/// multiplying shapes `[2, 3]` and `[3, 2]` produces shape `[2, 2]`.
pub fn matmul(left~ : Tensor, right~ : Tensor) -> Tensor raise TorchError {
  let args = PyTuple::new(2)
  args.set(0, left.obj)
  args.set(1, right.obj)
  expect_tensor(
    invoke(get_torch_function("matmul"), args, "torch.matmul"),
    "torch.matmul",
  )
}

///|
/// Return the tensor's dimensions as a MoonBit array. The array length is the
/// tensor rank. For example, a matrix with two rows and three columns returns
/// `[2, 3]`.
pub fn Tensor::shape(self : Tensor) -> Array[Int32] raise TorchError {
  guard self.obj.get_attr("shape") is Some(PyTuple(shape)) else {
    raise UnexpectedType("Tensor.shape is not a tuple")
  }
  Array::makei(shape.len().to_int(), index => {
    guard shape.get(index) is Some(@python.PyInteger(dimension)) else {
      raise UnexpectedType("Tensor.shape contains a non-integer")
    }
    dimension.to_int64().to_int()
  })
}

///|
/// Copy every tensor value into a flat `Array[Double]` in row-major order.
/// Shape information is discarded; call `shape()` before flattening if you need
/// to retain it. PyTorch `tolist` also converts values to ordinary Python
/// numbers, so this is a host-side copy rather than a zero-copy view.
pub fn Tensor::to_array(self : Tensor) -> Array[Double] raise TorchError {
  let flat = self.reshape([-1])
  guard flat.call("tolist", PyTuple::new(0)) is PyList(values) else {
    raise UnexpectedType("Tensor.tolist did not return a flat list")
  }
  Array::makei(values.len(), index => {
    match values.get(index) {
      Some(@python.PyFloat(value)) => value.to_double()
      Some(@python.PyInteger(value)) => value.to_double()
      _ => raise UnexpectedType("Tensor.tolist returned a non-numeric value")
    }
  })
}
