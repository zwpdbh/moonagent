///|
/// A pull-based lazy sequence. Each call to `next` runs the supplied function,
/// which returns `Some(value)` until the sequence is exhausted or `None`.
/// The function returns a Result so fallible generators can propagate errors
/// without losing the concrete error type.
pub struct LazyGenerator[T, E] {
  next_fn : () -> Result[T?, E]
}

///|
/// Create a lazy generator from a stateful function. The function is not called
/// until `next` is invoked, so work is performed on demand rather than eagerly.
pub fn[T, E] lazy_generator(next : () -> Result[T?, E]) -> LazyGenerator[T, E] {
  { next_fn: next, }
}

///|
/// Produce the next value, or `None` when the sequence is finished.
pub fn[T, E : Error] LazyGenerator::next(
  self : LazyGenerator[T, E],
) -> T? raise E {
  (self.next_fn)().unwrap_or_error()
}

///|
/// Consume the remaining sequence eagerly and collect all values in order.
pub fn[T, E : Error] LazyGenerator::collect(
  self : LazyGenerator[T, E],
) -> Array[T] raise E {
  let values : Array[T] = []
  while self.next() is Some(value) {
    values.push(value)
  }
  values
}
