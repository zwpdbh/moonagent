///|
pub struct TorchDataset {
  priv obj : PyObject
}

///|
pub struct TorchDataLoader {
  priv obj : PyObject
}

///|
pub(all) struct TorchBatch {
  features : Tensor
  labels : Tensor
}

///|
fn get_data_module() -> PyModule raise TorchError {
  get_submodule(get_submodule(get_torch_module(), "utils"), "data")
}

///|
pub fn tensor_dataset(
  features~ : Tensor,
  labels~ : Tensor,
) -> TorchDataset raise TorchError {
  let data = get_data_module()
  let args = PyTuple::new(2)
  args.set(0, features.obj)
  args.set(1, labels.obj)
  let result = invoke(
    get_module_function(data, "TensorDataset"),
    args,
    "TensorDataset",
  )
  TorchDataset::{ obj: expect_object(result, "TensorDataset"), }
}

///|
pub fn data_loader(
  dataset~ : TorchDataset,
  batch_size~ : Int32,
  shuffle~ : Bool,
  num_workers~ : Int32,
  drop_last~ : Bool,
) -> TorchDataLoader raise TorchError {
  let data = get_data_module()
  let args = PyTuple::new(1)
  args.set(0, dataset.obj)
  let kwargs = @python.PyDict::new()
  kwargs.set("batch_size", @python.PyInteger::from(batch_size.to_int64()))
  kwargs.set("shuffle", @python.PyBool::from(shuffle))
  kwargs.set("num_workers", @python.PyInteger::from(num_workers.to_int64()))
  kwargs.set("drop_last", @python.PyBool::from(drop_last))
  let result = invoke_with_kwargs(
    get_module_function(data, "DataLoader"),
    args,
    kwargs,
    "DataLoader",
  )
  TorchDataLoader::{ obj: expect_object(result, "DataLoader"), }
}

///|
fn import_builtins_module() -> () -> PyModule? {
  let builtins = @python.pyimport("builtins", print_err=true)
  fn() { builtins }
}

///|
let builtins_module : () -> PyModule? = import_builtins_module()

///|
fn get_builtins_module() -> PyModule raise TorchError {
  guard builtins_module() is Some(builtins) else {
    raise ImportFailed("builtins")
  }
  builtins
}

///|
pub fn TorchDataLoader::batches(
  self : TorchDataLoader,
) -> Array[TorchBatch] raise TorchError {
  let args = PyTuple::new(1)
  args.set(0, self.obj)
  let result = invoke(
    get_module_function(get_builtins_module(), "list"),
    args,
    "list(DataLoader)",
  )
  guard result is PyList(batch_list) else {
    raise UnexpectedType("list(DataLoader) did not return a list")
  }
  Array::makei(batch_list.len(), index => {
    guard batch_list.get(index) is Some(batch) else {
      raise UnexpectedType("DataLoader returned an invalid batch")
    }
    let features = match batch {
      PyTuple(values) => values.get(0)
      PyList(values) => values.get(0)
      _ => raise UnexpectedType("DataLoader batch is not a tuple or list")
    }
    let labels = match batch {
      PyTuple(values) => values.get(1)
      PyList(values) => values.get(1)
      _ => raise UnexpectedType("DataLoader batch is not a tuple or list")
    }
    guard features is Some(features) else {
      raise UnexpectedType("DataLoader batch has no features")
    }
    guard labels is Some(labels) else {
      raise UnexpectedType("DataLoader batch has no labels")
    }
    TorchBatch::{
      features: expect_tensor(features, "DataLoader features"),
      labels: expect_tensor(labels, "DataLoader labels"),
    }
  })
}
