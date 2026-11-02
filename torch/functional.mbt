///|
pub fn silu(input~ : Tensor) -> Tensor raise TorchError {
  let functional = get_functional_module()
  let args = PyTuple::new(1)
  args.set(0, input.obj)
  expect_tensor(
    invoke(get_module_function(functional, "silu"), args, "silu"),
    "silu",
  )
}

///|
pub fn rms_norm(
  input~ : Tensor,
  weight~ : Tensor,
  eps~ : Double,
) -> Tensor raise TorchError {
  let shape = input.shape()
  guard shape.length() > 0 else {
    raise InvalidShape("RMSNorm input must have at least one dimension")
  }
  let functional = get_functional_module()
  let args = PyTuple::new(2)
  args.set(0, input.obj)
  args.set(1, shape_arguments([shape[shape.length() - 1]]))
  let kwargs = @python.PyDict::new()
  kwargs.set("weight", weight.obj)
  kwargs.set("eps", @python.PyFloat::from(eps))
  expect_tensor(
    invoke_with_kwargs(
      get_module_function(functional, "rms_norm"),
      args,
      kwargs,
      "rms_norm",
    ),
    "rms_norm",
  )
}
