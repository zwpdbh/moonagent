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
pub fn add(left~ : Tensor, right~ : Tensor) -> Tensor raise TorchError {
  binary_tensor_operation("add", left, right)
}

///|
pub fn mul(left~ : Tensor, right~ : Tensor) -> Tensor raise TorchError {
  binary_tensor_operation("mul", left, right)
}

///|
pub fn sigmoid(input~ : Tensor) -> Tensor raise TorchError {
  let args = PyTuple::new(1)
  args.set(0, input.obj)
  expect_tensor(
    invoke(get_torch_function("sigmoid"), args, "torch.sigmoid"),
    "torch.sigmoid",
  )
}

///|
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
pub fn Tensor::to(self : Tensor, device~ : String) -> Tensor raise TorchError {
  let args = PyTuple::new(1)
  args.set(0, @python.PyString::from(device))
  expect_tensor(self.call("to", args), "Tensor.to")
}

///|
pub fn Tensor::device(self : Tensor) -> String raise TorchError {
  guard self.obj.get_attr("device") is Some(PyClass(device)) else {
    raise UnexpectedType("Tensor.device is not a torch.device")
  }
  device.to_string()
}

///|
pub fn Tensor::item(self : Tensor) -> Double raise TorchError {
  match self.call("item", PyTuple::new(0)) {
    @python.PyFloat(value) => value.to_double()
    @python.PyInteger(value) => value.to_double()
    _ => raise UnexpectedType("Tensor.item returned a non-numeric value")
  }
}

///|
pub fn Tensor::backward(self : Tensor) -> Unit raise TorchError {
  invoke_unit(
    get_method(self.obj, "backward"),
    PyTuple::new(0),
    "Tensor.backward",
  )
}

///|
pub fn Tensor::grad(self : Tensor) -> Tensor raise TorchError {
  guard self.obj.get_attr("grad") is Some(PyClass(gradient)) else {
    raise UnexpectedType("Tensor.grad is not a tensor")
  }
  Tensor::{ obj: gradient, }
}

///|
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
pub fn scale(input~ : Tensor, scalar~ : Double) -> Tensor raise TorchError {
  let args = PyTuple::new(1)
  args.set(0, @python.PyFloat::from(scalar))
  expect_tensor(input.call("mul", args), "Tensor.mul")
}

///|
pub fn dot(left~ : Tensor, right~ : Tensor) -> Tensor raise TorchError {
  binary_tensor_operation("dot", left, right)
}
