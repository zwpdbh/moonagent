///|
pub(all) suberror PrintError {
  WriteFailed(String)
}

///|
pub impl Show for PrintError with fn output(self, logger) {
  match self {
    WriteFailed(message) => logger.write_string(message)
  }
}

///|
pub extend PrintError with Show::{to_string, output}

///|
/// Print a value without appending a newline and write it to native stdout.
pub async fn[T : Show] print_x(value : T) -> Unit raise PrintError {
  let bytes = @encoding.encode(@encoding.UTF8, value.to_string())
  @stdio.stdout.write(bytes) catch {
    error => raise WriteFailed(error.to_string())
  }
}
