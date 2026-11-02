///|
/// Calculate attention from query and key vectors using PyTorch.
/// query:  [head_dim]
/// keys:   [sequence_length, head_dim]
/// values: [sequence_length, head_dim]
/// return: [head_dim]
fn scaled_dot_product_attention(
  query~ : Tensor,
  keys~ : Tensor,
  values~ : Tensor,
) -> Tensor raise @torch.TorchError {
  let query_shape = query.shape()
  let keys_shape = keys.shape()
  let values_shape = values.shape()
  if query_shape.length() != 1 ||
    keys_shape.length() != 2 ||
    values_shape.length() != 2 {
    abort("attention expects a query vector and key/value matrices")
  }
  if keys_shape != values_shape {
    abort("attention keys and values must have the same shape")
  }
  let sequence_length = keys_shape[0]
  let head_dim = keys_shape[1]
  if sequence_length == 0 || head_dim == 0 {
    abort("attention sequence length and head dimension must be positive")
  }
  if query_shape[0] != head_dim {
    abort("attention query length must equal head dimension")
  }

  let scores = @torch.matmul(left=keys, right=query)
  let scaled_scores = @torch.scale(
    input=scores,
    scalar=1.0 / head_dim.to_double().sqrt(),
  )
  let attention_weights = @torch.softmax(input=scaled_scores, dim=0)
  @torch.matmul(left=attention_weights, right=values)
}

///|
/// Apply RoPE to one head's query or key vector.
pub fn apply_rope(
  input~ : Tensor,
  position~ : Int32,
  rope_theta~ : Float,
) -> Tensor raise @torch.TorchError {
  let input_shape = input.shape()
  if input_shape.length() != 1 {
    abort("RoPE input must be one head vector")
  }
  let head_dim = input_shape[0]
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
  let input_data = input.to_array()
  let data = Array::makei(head_dim, dimension => {
    let pair_index = if dimension < half_dim {
      dimension
    } else {
      dimension - half_dim
    }
    let (cosine, sine) = rotations[pair_index]
    let first = input_data[pair_index]
    let second = input_data[pair_index + half_dim]
    if dimension < half_dim {
      first * cosine.to_double() - second * sine.to_double()
    } else {
      first * sine.to_double() + second * cosine.to_double()
    }
  })
  @torch.tensor(data~, shape=input_shape)
}

///|
priv struct GqaDimensions {
  num_query_heads : Int32
  group_size : Int32
}

///|
/// Validate GQA inputs once and derive all dimensions needed by the computation.
fn validate_gqa_inputs(
  queries : Tensor,
  keys : Tensor,
  values : Tensor,
) -> GqaDimensions raise @torch.TorchError {
  let queries_shape = queries.shape()
  let keys_shape = keys.shape()
  let values_shape = values.shape()
  if queries_shape.length() != 2 ||
    keys_shape.length() != 3 ||
    values_shape.length() != 3 {
    abort("GQA expects rank-2 queries and rank-3 keys/values")
  }
  if keys_shape != values_shape {
    abort("GQA keys and values must have identical shapes")
  }

  let num_query_heads = queries_shape[0]
  let head_dim = queries_shape[1]
  let num_key_value_heads = keys_shape[0]
  let sequence_length = keys_shape[1]
  if num_query_heads <= 0 ||
    num_key_value_heads <= 0 ||
    sequence_length <= 0 ||
    head_dim <= 0 {
    abort("GQA head counts, sequence length, and head_dim must be positive")
  }
  if keys_shape[2] != head_dim {
    abort("GQA queries and KV caches must have matching head_dim")
  }
  if num_query_heads % num_key_value_heads != 0 {
    abort("GQA query heads must be divisible by KV heads")
  }
  let group_size = num_query_heads / num_key_value_heads
  { num_query_heads, group_size, }
}

///|
/// Run grouped-query attention for one token and concatenate all head outputs.
/// queries: [num_query_heads, head_dim]
/// keys:    [num_key_value_heads, sequence_length, head_dim]
/// values:  [num_key_value_heads, sequence_length, head_dim]
/// output:  [num_query_heads * head_dim]
pub fn grouped_query_attention(
  queries~ : Tensor,
  keys~ : Tensor,
  values~ : Tensor,
) -> Tensor raise @torch.TorchError {
  let dims = validate_gqa_inputs(queries, keys, values)
  let head_outputs = Array::makei(dims.num_query_heads, query_head_index => {
    let kv_head_index = query_head_index / dims.group_size
    let query = @torch.select(input=queries, dim=0, index=query_head_index)
    let head_keys = @torch.select(input=keys, dim=0, index=kv_head_index)
    let head_values = @torch.select(input=values, dim=0, index=kv_head_index)
    scaled_dot_product_attention(query~, keys=head_keys, values=head_values)
  })
  @torch.concat(head_outputs, dim=0)
}

///|
test "understanding query, key, value in transformers and LLMs" {
  let token_dog = @torch.tensor(data=[1.0, -0.5, 0.2], shape=[3])
  let token_plays = @torch.tensor(data=[0.4, 0.1, 0.4], shape=[3])
  let token_fetch = @torch.tensor(data=[0.6, -0.8, 0.7], shape=[3])
  let w_q = @torch.rand([3, 3])
  let w_k = @torch.rand([3, 3])
  let w_v = @torch.rand([3, 3])

  let query_plays = @torch.matmul(left=token_plays, right=w_q)
  let key_dog = @torch.matmul(left=token_dog, right=w_k)
  let value_dog = @torch.matmul(left=token_dog, right=w_v)
  let key_fetch = @torch.matmul(left=token_fetch, right=w_k)
  let value_fetch = @torch.matmul(left=token_fetch, right=w_v)

  let similarity_plays_dog = @torch.dot(left=query_plays, right=key_dog).item()
  let similarity_plays_fetch = @torch.dot(left=query_plays, right=key_fetch).item()
  let similarity_scores = @torch.tensor(
    data=[similarity_plays_dog, similarity_plays_fetch],
    shape=[2],
  )
  let attention_weights = @torch.softmax(input=similarity_scores, dim=0).to_array()
  let weighted_value_dog = @torch.scale(
    input=value_dog,
    scalar=attention_weights[0],
  )
  let weighted_value_fetch = @torch.scale(
    input=value_fetch,
    scalar=attention_weights[1],
  )
  ignore(@torch.add(left=weighted_value_dog, right=weighted_value_fetch))
}
