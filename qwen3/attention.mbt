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
  let attention_weights = @tensor.softmax(input=scaled_scores.view())

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
/// Each token generates query, key and value vector.
/// Query: What is the token is seeking (what information it needs)
/// Key: The information the token holds (how relevant it is to other tokens)
/// Value: The actual content the token shares if its key is found relevant
/// 
/// These vectors are derived from the token's embedding through learned 
/// transformations using weight matrices: 
/// Wq(for queries)
/// Wk(for keys)
/// Wv(for values)
/// 
/// Each token has its query, key, and value vectors, the transformer follows a
/// search-match-update process to determine which tokens provide the most relevant context:
/// Use dot product between a token's query with other token's key to measure relevance.
/// The similarity scores are passed through a softmax function to normalize them into
/// attention weights. The transformer then compute a weighted sum of the value vectors 
/// based on these attention weights.
fn grouped_query_attention(
  queries~ : TensorView,
  keys~ : TensorView,
  values~ : TensorView,
) -> Tensor {
  // 1. Validate the ranks: queries = 2, keys = 3, values = 3.
  //    Require keys.shape == values.shape, nonzero head counts/dimensions,
  //    and matching head_dim between queries and the KV caches.
  if queries.rank() != 2 {
    abort("queries rank need to be 2")
  }

  if keys.rank() != 3 {
    abort("keys' rank need to be 3")
  }

  if values.rank() != 3 {
    abort("values' rank need to be 3")
  }

  if keys.shape != values.shape {
    abort("keys.shape should eq with values.shape")
  }

  // 2. Derive all dimensions from the tensor shapes:
  //    num_query_heads     = queries.dim(0)
  //    head_dim            = queries.dim(1)
  //    num_key_value_heads = keys.dim(0)
  //    sequence_length     = keys.dim(1)
  //    Also check keys.dim(2) == head_dim.
  let num_query_heads = queries.dim(0)
  let head_dim = queries.dim(1)

  let num_key_value_heads = keys.dim(0)
  let sequence_length = keys.dim(1)

  if keys.dim(2) != head_dim {
    abort("Need matching head_dim between queries and the KV caches")
  }

  // 3. Require num_query_heads % num_key_value_heads == 0, then calculate:
  //    group_size = num_query_heads / num_key_value_heads.
  //    Qwen3 has group_size = 16 / 8 = 2.

  // 4. Create one output per query head with Array::makei(num_query_heads, ...).
  //    Inside that function, calculate the shared KV head:
  //    kv_head_index = query_head_index / group_size.

  // 5. Select the current head's inputs without copying:
  //    query       = queries.slice_first_axis(query_head_index) // [head_dim]
  //    head_keys   = keys.slice_first_axis(kv_head_index)       // [sequence_length, head_dim]
  //    head_values = values.slice_first_axis(kv_head_index)     // [sequence_length, head_dim]

  // 6. Call scaled_dot_product_attention(query=query, keys=head_keys,
  //    values=head_values). Collect each returned Tensor in head_outputs.
  //    Every output has shape [head_dim].

  // 7. Concatenate head_outputs into one flat Array[Float] of length
  //    num_query_heads * head_dim. For each flat output index, you can use:
  //    head_index = flat_index / head_dim
  //    dimension  = flat_index % head_dim
  //    value      = head_outputs[head_index].data[dimension]

  // 8. Return Tensor::new(data=concatenated_data,
  //    shape=[num_query_heads * head_dim]).
  abort("TODO")
}

///|
test "understanding query, key, value in transformers and LLMs" {
  // Create the vector representations (embeddings) for each token
  // These embeddings, as simplified vectors, represent how the tokens are encoded in the transformer model, 
  // capturing their “meaning” for further computations:
  let token_dog = Tensor::new(data=[1.0, -0.5, 0.2], shape=[3])
  let token_plays = Tensor::new(data=[0.4, 0.1, 0.4], shape=[3])
  let token_fetch = Tensor::new(data=[0.6, -0.8, 0.7], shape=[3])

  // Next, we create randomized weight matrices for the query, key, and value transformations.
  // In a real transformer model, these weights are learned during training (and with way more layers).
  let w_q = Tensor::rand([3, 3])
  let w_k = Tensor::rand([3, 3])
  let w_v = Tensor::rand([3, 3])

  // Now we compute the query, key and value vectors for tokens.
  // We will focus on "plays" as the querying token, seeking information
  // from "dog" and "fetch".
  // The query vector tells us what "plays" is seeking to understand.
  // The key vectors for "dog" and "fetch" represent the information those tokens hold,
  // and the value vectors represent the actal content those tokens will contribute if found relevant.

  // "plays" is the querying token
  let query_plays = @tensor.vec_mat_mul(
    vector=token_plays.view(),
    matrix=w_q.view(),
  )

  // compute key and value for "dog"
  let key_dog = @tensor.vec_mat_mul(vector=token_dog.view(), matrix=w_k.view())
  let value_dot = @tensor.vec_mat_mul(
    vector=token_dog.view(),
    matrix=w_v.view(),
  )

  // compute key and value for "fetch"
  let key_fetch = @tensor.vec_mat_mul(
    vector=token_fetch.view(),
    matrix=w_k.view(),
  )
  let value_fetch = @tensor.vec_mat_mul(
    vector=token_fetch.view(),
    matrix=w_v.view(),
  )

  // To determine how relevant "dog" and "fetch" are to "plays", we calculate the similarity scores
  // by taking the dot product between the query from "plays" and the keys of the other tokens.
  // cals sim score for "plays" query to "dog" key 
  let similarity_plays_dog = @tensor.dot(
    left=query_plays.data,
    right=key_dog.data,
  )
  // cals sim score for "plays" query to "fetch" key
  let similarity_plays_fetch = @tensor.dot(
    left=query_plays.data,
    right=key_fetch.data,
  )

  // next use the softmax function to covert these similarity scores into attension weights.
  // this weights determine how much attention "plays" should pay to "dog" and "fetch"
  let similarity_score = Tensor::new(
    data=[similarity_plays_dog, similarity_plays_fetch],
    shape=[2],
  )
  let attention_weights = @tensor.softmax(input=similarity_score.view())

  // using these attention weights, we compute the weighted sum of the value vector from "dog" and "fetch"
  // to update the context for "plays"
  let weighted_value_dog = attention_weights[0] * value_dog
  let weighted_value_fetch = attention_weights[1] * value_fetch
  let updated_context_plays = weighted_value_dog + weighted_value_fetch
}
