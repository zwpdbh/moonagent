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
  query~ : TensorView,
  keys~ : TensorView,
  values~ : TensorView,
) -> Tensor {
  if query.rank() != 1 || keys.rank() != 2 || values.rank() != 2 {
    abort("attention expects a query vector and key/value matrices")
  }
  if keys.shape != values.shape {
    abort("attention keys and values must have the same shape")
  }
  let sequence_length = keys.dim(0)
  let head_dim = keys.dim(1)
  if sequence_length == 0 || head_dim == 0 {
    abort("attention sequence length and head dimension must be positive")
  }
  if query.dim(0) != head_dim {
    abort("attention query length must equal head dimension")
  }

  let scale = Float::from_int(head_dim).sqrt()
  let scores = @tensor.mat_vec_mul(matrix=keys, vector=query)
  let scaled_scores = Tensor::new(
    data=scores.data.map(score => score / scale),
    shape=scores.shape,
  )
  let attention_weights = softmax(input=scaled_scores.view())

  @tensor.vec_mat_mul(vector=attention_weights.view(), matrix=values)
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
/// Q and A ===
/// What is the token's absolute position? Token position is the position in the array of token IDs produced by the tokenizer
/// What is pair's index
/// How to find the head_dim? is it the width (columns) of the weight matrix?
///  
fn apply_rope(
  input~ : TensorView,
  position~ : Int32,
  rope_theta~ : Float,
) -> Tensor {
  if input.rank() != 1 {
    abort("RoPE input must be one head vector")
  }
  let head_dim = input.dim(0)
  if head_dim == 0 || head_dim % 2 != 0 {
    abort("RoPE head dimension must be positive and even")
  }
  if position < 0 {
    abort("RoPE position must not be negative")
  }
  if rope_theta <= 0.0 {
    abort("RoPE theta must be positive")
  }

  let half_dim = head_dim / 2
  let position_float = Float::from_int(position)
  let head_dim_float = Float::from_int(head_dim)

  let rotations = Array::makei(half_dim, pair_index => {
    let exponent = Float::from_int(2 * pair_index) / head_dim_float

    let frequency = 1.0F / @math.powf(rope_theta, exponent)
    let angle = position_float * frequency

    (@math.cosf(angle), @math.sinf(angle))
  })

  let data = Array::makei(head_dim, dimension => {
    let pair_index = if dimension < half_dim {
      dimension
    } else {
      dimension - half_dim
    }
    let (cosine, sine) = rotations[pair_index]
    let first = input.data[pair_index]
    let second = input.data[pair_index + half_dim]
    if dimension < half_dim {
      first * cosine - second * sine
    } else {
      first * sine + second * cosine
    }
  })
  Tensor::new(data~, shape=input.shape)
}

///|
/// Run grouped-query attention(GQA) for one token and concatenate all head outputs.
///
/// Expected shapes:
/// - queries: `[num_query_heads, head_dim]`
/// - keys:    `[num_key_value_heads, sequence_length, head_dim]`
/// - values:  `[num_key_value_heads, sequence_length, head_dim]`
///
/// Queries and keys must already have RoPE applied. Keys and values are
/// head-major, so each KV head's complete sequence cache is contiguous.
/// Each group of query heads shares one KV head. The output shape is
/// `[num_query_heads * head_dim]`.
/// 
/// Q and A === 
/// 1. What is tensor why it is used to represent queries, keys and values.
/// 2. What is queries, what is keys and values.
/// 3. Given a queries which is TensorView, what is its num_query_heads?
/// 4. Given keys and values, what is num_key_value head
/// 5. What is group size
fn grouped_query_attention(
  queries~ : TensorView,
  keys~ : TensorView,
  values~ : TensorView,
) -> Tensor {
  abort("TODO")
}
