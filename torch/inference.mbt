///|
fn set_grad_enabled(enabled : Bool) -> Unit raise TorchError {
  let args = PyTuple::new(1)
  args.set(0, @python.PyBool::from(enabled))
  invoke_unit(
    get_torch_function("set_grad_enabled"),
    args,
    "torch.set_grad_enabled",
  )
}

///|
/// Run a computation with gradient recording disabled.
pub fn[T] without_grad(body : () -> T raise TorchError) -> T raise TorchError {
  set_grad_enabled(false)
  defer set_grad_enabled(true)
  body()
}
