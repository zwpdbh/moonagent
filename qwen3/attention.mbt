///|
/// query:           [head_dim]
/// keys:            [sequence_length, head_dim], flattened
/// values:          [sequence_length, head_dim], flattened
/// sequence_length: number of cached positions, including the current token
/// It means how many token positions have been processed in the current context
/// head_dim:        size of each query/key/value vector in this head
/// return:          [head_dim]
/// The problem it tries to solve: vanishing gradient problem.
/// Variance is exploding due to high dimension. and Variance Scaling Rule
/// tell us the when you multiple a variable by C, its variance is C squared.
fn scaled_dot_product_attention(
  query : ArrayView[Float],
  keys : ArrayView[Float],
  values : ArrayView[Float],
  sequence_length : Int32,
  head_dim : Int32,
) -> Array[Float] {
  abort("TODO")
}
