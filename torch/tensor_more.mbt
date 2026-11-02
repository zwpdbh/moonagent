///|
pub fn select(
  input~ : TorchTensor,
  dim~ : Int32,
  index~ : Int32,
) -> TorchTensor raise TorchError {
  let args = PyTuple::new(2)
  args.set(0, @python.PyInteger::from(dim.to_int64()))
  args.set(1, @python.PyInteger::from(index.to_int64()))
  expect_tensor(input.call("select", args), "Tensor.select")
}

///|
pub fn concat(
  inputs : Array[TorchTensor],
  dim~ : Int32,
) -> TorchTensor raise TorchError {
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
pub fn rand(shape : Array[Int32]) -> TorchTensor raise TorchError {
  ignore(shape_size(shape))
  let args = PyTuple::new(1)
  args.set(0, shape_arguments(shape))
  expect_tensor(
    invoke(get_torch_function("rand"), args, "torch.rand"),
    "torch.rand",
  )
}

///|
pub fn scale(
  input~ : TorchTensor,
  scalar~ : Double,
) -> TorchTensor raise TorchError {
  let args = PyTuple::new(1)
  args.set(0, @python.PyFloat::from(scalar))
  expect_tensor(input.call("mul", args), "Tensor.mul")
}

///|
pub fn dot(
  left~ : TorchTensor,
  right~ : TorchTensor,
) -> TorchTensor raise TorchError {
  binary_tensor_operation("dot", left, right)
}
