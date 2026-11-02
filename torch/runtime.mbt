///|
fn import_python_module(name : String) -> PyModule raise TorchError {
  guard @python.pyimport(name, print_err=true) is Some(python_module) else {
    raise ImportFailed(name)
  }
  python_module
}

///|
/// Add a directory to CPython's module search path and refresh importer caches.
pub fn add_python_path(path : String) -> Unit raise TorchError {
  let sys_module = import_python_module("sys")
  guard sys_module.get_attr("path") is Some(PyList(paths)) else {
    raise AttributeMissing("sys.path")
  }
  let args = PyTuple::new(2)
  args.set(0, @python.PyInteger::from(0))
  args.set(1, @python.PyString::from(path))
  invoke_unit(get_method(paths.obj(), "insert"), args, "sys.path.insert")
  invoke_unit(
    get_module_function(import_python_module("importlib"), "invalidate_caches"),
    PyTuple::new(0),
    "importlib.invalidate_caches",
  )
}
