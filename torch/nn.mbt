///|
pub struct TorchModule {
  priv obj : PyObject
}

///|
pub struct TorchParameters {
  priv obj : PyObject
}

///|
pub struct TorchOptimizer {
  priv obj : PyObject
}

///|
fn expect_object(
  value : PyObjectEnum,
  operation : String,
) -> PyObject raise TorchError {
  match value {
    PyClass(object) => object
    PyCallable(callable) => callable.obj()
    _ => raise UnexpectedType("\{operation} returned an unexpected Python type")
  }
}

///|
fn expect_module(
  value : PyObjectEnum,
  operation : String,
) -> TorchModule raise TorchError {
  TorchModule::{ obj: expect_object(value, operation), }
}

///|
fn get_submodule(parent : PyModule, name : String) -> PyModule raise TorchError {
  guard parent.get_attr(name) is Some(PyModule(child)) else {
    raise AttributeMissing("\{parent.get_name()}.\{name}")
  }
  child
}

///|
fn get_module_function(
  parent : PyModule,
  name : String,
) -> @python.PyCallable raise TorchError {
  guard parent.get_attr(name) is Some(PyCallable(function)) else {
    raise AttributeMissing("\{parent.get_name()}.\{name}")
  }
  function
}

///|
fn get_nn_module() -> PyModule raise TorchError {
  get_submodule(get_torch_module(), "nn")
}

///|
fn get_functional_module() -> PyModule raise TorchError {
  get_submodule(get_nn_module(), "functional")
}

///|
fn module_constructor(
  parent : PyModule,
  name : String,
  args : PyTuple,
) -> TorchModule raise TorchError {
  expect_module(invoke(get_module_function(parent, name), args, name), name)
}

///|
pub fn linear(
  in_features~ : Int32,
  out_features~ : Int32,
) -> TorchModule raise TorchError {
  let args = PyTuple::new(2)
  args.set(0, @python.PyInteger::from(in_features.to_int64()))
  args.set(1, @python.PyInteger::from(out_features.to_int64()))
  module_constructor(get_nn_module(), "Linear", args)
}

///|
pub fn relu() -> TorchModule raise TorchError {
  module_constructor(get_nn_module(), "ReLU", PyTuple::new(0))
}

///|
pub fn sequential(layers : Array[TorchModule]) -> TorchModule raise TorchError {
  let args = PyTuple::new(layers.length().to_uint64())
  for index, layer in layers {
    args.set(index, layer.obj)
  }
  module_constructor(get_nn_module(), "Sequential", args)
}

///|
pub fn TorchModule::forward(
  self : TorchModule,
  input~ : TorchTensor,
) -> TorchTensor raise TorchError {
  let args = PyTuple::new(1)
  args.set(0, input.obj)
  expect_tensor(
    invoke(get_method(self.obj, "forward"), args, "Module.forward"),
    "Module.forward",
  )
}

///|
pub fn TorchModule::train(self : TorchModule) -> Unit raise TorchError {
  invoke_unit(get_method(self.obj, "train"), PyTuple::new(0), "Module.train")
}

///|
pub fn TorchModule::eval(self : TorchModule) -> Unit raise TorchError {
  invoke_unit(get_method(self.obj, "eval"), PyTuple::new(0), "Module.eval")
}

///|
pub fn TorchModule::to(
  self : TorchModule,
  device~ : String,
) -> Unit raise TorchError {
  let args = PyTuple::new(1)
  args.set(0, @python.PyString::from(device))
  invoke_unit(get_method(self.obj, "to"), args, "Module.to")
}

///|
pub fn TorchModule::parameters(
  self : TorchModule,
) -> TorchParameters raise TorchError {
  let result = invoke(
    get_method(self.obj, "parameters"),
    PyTuple::new(0),
    "Module.parameters",
  )
  TorchParameters::{ obj: expect_object(result, "Module.parameters"), }
}

///|
pub fn sgd(
  parameters~ : TorchParameters,
  learning_rate~ : Double,
) -> TorchOptimizer raise TorchError {
  let optim = get_submodule(get_torch_module(), "optim")
  let args = PyTuple::new(1)
  args.set(0, parameters.obj)
  let kwargs = @python.PyDict::new()
  kwargs.set("lr", @python.PyFloat::from(learning_rate))
  let result = invoke_with_kwargs(
    get_module_function(optim, "SGD"),
    args,
    kwargs,
    "torch.optim.SGD",
  )
  TorchOptimizer::{ obj: expect_object(result, "torch.optim.SGD"), }
}

///|
pub fn TorchOptimizer::zero_grad(
  self : TorchOptimizer,
) -> Unit raise TorchError {
  invoke_unit(
    get_method(self.obj, "zero_grad"),
    PyTuple::new(0),
    "Optimizer.zero_grad",
  )
}

///|
pub fn TorchOptimizer::step(self : TorchOptimizer) -> Unit raise TorchError {
  invoke_unit(get_method(self.obj, "step"), PyTuple::new(0), "Optimizer.step")
}

///|
fn functional_loss(
  name : String,
  input : TorchTensor,
  target : TorchTensor,
) -> TorchTensor raise TorchError {
  let args = PyTuple::new(2)
  args.set(0, input.obj)
  args.set(1, target.obj)
  let functional = get_functional_module()
  expect_tensor(invoke(get_module_function(functional, name), args, name), name)
}

///|
pub fn cross_entropy(
  input~ : TorchTensor,
  target~ : TorchTensor,
) -> TorchTensor raise TorchError {
  functional_loss("cross_entropy", input, target)
}

///|
pub fn binary_cross_entropy(
  input~ : TorchTensor,
  target~ : TorchTensor,
) -> TorchTensor raise TorchError {
  functional_loss("binary_cross_entropy", input, target)
}

///|
pub fn manual_seed(seed~ : Int64) -> Unit raise TorchError {
  let args = PyTuple::new(1)
  args.set(0, @python.PyInteger::from(seed))
  ignore(invoke(get_torch_function("manual_seed"), args, "torch.manual_seed"))
}
