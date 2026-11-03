///|
using @python {type PyTuple}

///|
pub(all) suberror PrintError {
  ImportFailed(String)
  CallFailed(String)
}

///|
/// Print a value without appending a newline. This uses the embedded Python
/// `print` with `end=""`, and flushes immediately so streaming output appears
/// as soon as it is produced.
pub fn[T : Show] print_x(value : T) -> Unit raise PrintError {
  guard @python.pyimport("builtins", print_err=true) is Some(python_module) else {
    raise ImportFailed("builtins")
  }
  guard python_module.get_attr("print") is Some(PyCallable(print_fn)) else {
    raise ImportFailed("builtins.print")
  }
  let args = PyTuple::new(1)
  args.set(0, @python.PyString::from(value.to_string()))
  let kwargs = @python.PyDict::new()
  kwargs.set("end", @python.PyString::from(""))
  kwargs.set("flush", @python.PyBool::from(true))
  ignore(
    print_fn.invoke(args~, kwargs~, print_err=true) catch {
      _ => raise CallFailed("print_x")
    },
  )
}
