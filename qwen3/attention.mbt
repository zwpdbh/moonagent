///|
/// Calculate attention from query and key vectors
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
  query~ : ArrayView[Float],
  keys~ : ArrayView[Float],
  values~ : ArrayView[Float],
  sequence_length~ : Int32,
  head_dim~ : Int32,
) -> Array[Float] {
  if sequence_length <= 0 {
    abort("attention sequence length must be positive")
  }
  if head_dim <= 0 {
    abort("attention head dimension must be positive")
  }
  if query.length() != head_dim {
    abort("attention query length must equal head dimension")
  }
  let expected_length = sequence_length * head_dim
  if keys.length() != expected_length {
    abort("attention keys have an invalid length")
  }
  if values.length() != expected_length {
    abort("attention values have an invalid length")
  }

  let scale = Float::from_int(head_dim).sqrt()
  let scores = @tensor.mat_vec_mul(
    matrix=keys,
    shape=[sequence_length, head_dim],
    vector=query,
  ).map(score => score / scale)
  let attention_weights = softmax(scores)

  @tensor.vec_mat_mul(vector=attention_weights, matrix=values, shape=[
    sequence_length, head_dim,
  ])
}

///|
/// RoPE (Rotary Positional Encoding).
/// It is applied to 
/// - the query vector,
/// - each key vector before storing it in the KV cache
/// 
/// `input` is one head’s query or key vector of length head_dim
/// `position` is the token's absolute position in the sequence
/// `rope_theta` comes from `Config.rope_theta`— Qwen3 uses 1000000
fn apply_rope(
  input : ArrayView[Float],
  position : Int32,
  rope_theta : Float,
) -> Array[Float] {
  abort("TODO")
}
