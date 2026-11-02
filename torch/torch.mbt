///|
using @python {
  type PyCallable,
  type PyList,
  type PyModule,
  type PyObject,
  type PyObjectEnum,
  type PyTuple,
}

///|
pub(all) suberror TorchError {
  ImportFailed(String)
  AttributeMissing(String)
  CallFailed(String)
  UnexpectedType(String)
  InvalidShape(String)
}

///|
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
pub struct Tensor {
  priv obj : PyObject
}

///|
fn import_torch_module() -> () -> PyModule? {
  fn() { @python.pyimport("torch", print_err=true) }
}

///|
let torch_module : () -> PyModule? = import_torch_module()

///|
fn get_torch_module() -> PyModule raise TorchError {
  guard torch_module() is Some(torch_mod) else { raise ImportFailed("torch") }
  torch_mod
}

///|
fn get_torch_function(name : String) -> PyCallable raise TorchError {
  guard get_torch_module().get_attr(name) is Some(PyCallable(function)) else {
    raise AttributeMissing("torch.\{name}")
  }
  function
}

///|
fn get_method(object : PyObject, name : String) -> PyCallable raise TorchError {
  guard object.get_attr(name) is Some(PyCallable(callable)) else {
    raise AttributeMissing(name)
  }
  callable
}

///|
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
fn invoke_unit(
  callable : PyCallable,
  args : PyTuple,
  name : String,
) -> Unit raise TorchError {
  ignore(invoke_optional(callable, args, @python.PyDict::new(), name))
}

///|
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
fn shape_arguments(shape : Array[Int32]) -> PyTuple {
  let args = PyTuple::new(shape.length().to_uint64())
  for index, dimension in shape {
    args.set(index, @python.PyInteger::from(dimension.to_int64()))
  }
  args
}

///|
fn Tensor::call(
  self : Tensor,
  name : String,
  args : PyTuple,
) -> PyObjectEnum raise TorchError {
  invoke(get_method(self.obj, name), args, name)
}

///|
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
fn reshape_arguments(shape : Array[Int32]) -> PyTuple {
  let args = PyTuple::new(1)
  args.set(0, shape_arguments(shape))
  args
}

///|
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
pub fn Tensor::view(
  self : Tensor,
  shape : Array[Int32],
) -> Tensor raise TorchError {
  expect_tensor(self.call("view", reshape_arguments(shape)), "Tensor.view")
}

///|
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
pub fn Tensor::t(self : Tensor) -> Tensor raise TorchError {
  guard self.obj.get_attr("T") is Some(attribute) else {
    raise AttributeMissing("Tensor.T")
  }
  expect_tensor(attribute, "Tensor.T")
}

///|
pub fn Tensor::dtype(self : Tensor) -> String raise TorchError {
  guard self.obj.get_attr("dtype") is Some(PyClass(dtype)) else {
    raise UnexpectedType("Tensor.dtype is not a torch.dtype")
  }
  dtype.to_string()
}

///|
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
