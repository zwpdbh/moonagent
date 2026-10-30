///|
/// This file contains basic tensor operations
/// embedding lookup
/// matrix multiplication / linear projection
/// RMS normalization
/// SiLU activation
/// element-wise arithmetic

///|
/// Given embedding shape which is [vocab_size, hidden_size], in Qwen3 it is [151936, 1024] and token ID
/// the embedding is one row of the matrix:
/// start = token_id * hidden_size
/// end = start + hidden_size
/// "Hidden Size" means the number of Float values used to represent one token insize the model.
/// Some token id -> 1024 Float values
fn embedding_lookup(
  weights~ : @tensor.TensorView,
  token_id~ : Int32,
) -> @tensor.TensorView {
  if weights.rank() != 2 {
    abort("embedding weights must be a rank-2 tensor")
  }
  weights.slice_first_axis(token_id)
}

///|
/// RMSNorm (root-mean-square normalization)
/// Its purpose is helping a deep neural network remain numerically stable.
/// It usually applied when:
/// before attention
/// before MLP
/// after the final decoder layer.
/// The 'eps' is a small positive number added for numerical stability.
/// In short: epsilon protects the division without significantly changing normal RMSNorm behavior.
fn rms_norm(
  input~ : @tensor.TensorView,
  weight~ : @tensor.TensorView,
  eps~ : Float,
) -> @tensor.Tensor {
  if input.rank() != 1 || weight.rank() != 1 || input.shape != weight.shape {
    abort("RMSNorm input and weight must be vectors with the same shape")
  }
  if input.data.length() == 0 {
    abort("RMSNorm input must not be empty")
  }
  if eps < 0.0 {
    abort("RMSNorm epsilon must not be negative")
  }
  let sum_of_squares = input.data.fold(init=0.0F, (sum, value) => {
    sum + value * value
  })
  let mean_square = sum_of_squares / Float::from_int(input.data.length())
  let inverse_rms = 1.0F / (mean_square + eps).sqrt()
  let data = input.data.mapi((i, value) => value * inverse_rms * weight.data[i])
  @tensor.Tensor::new(data~, shape=input.shape)
}

///|
/// Multiply an input vector by a learned weight matrix.
/// Weight shape is `[out_features, in_features]`; output shape is
/// `[out_features]`.
fn linear(
  input~ : @tensor.TensorView,
  weight~ : @tensor.TensorView,
) -> @tensor.Tensor {
  @tensor.mat_vec_mul(matrix=weight, vector=input)
}

///|
/// SiLU activation (Sigmoid Linear Unit)
/// It is used in every MLP block
fn silu(input~ : @tensor.TensorView) -> @tensor.Tensor {
  let data = input.data.map(value => value / (1.0F + @math.expf(-value)))
  @tensor.Tensor::new(data~, shape=input.shape)
}

///|
/// Element-wise multiplication
fn elementwise_mul(
  left~ : @tensor.TensorView,
  right~ : @tensor.TensorView,
) -> @tensor.Tensor {
  if left.shape != right.shape {
    abort("element-wise multiplication requires equal shapes")
  }
  let data = left.data.mapi((index, value) => value * right.data[index])
  @tensor.Tensor::new(data~, shape=left.shape)
}

///|
/// Qwen3 gated MLP: down_proj(SiLU(gate_proj(input)) * up_proj(input)).
fn mlp(
  input~ : @tensor.TensorView,
  gate_weight~ : @tensor.TensorView,
  up_weight~ : @tensor.TensorView,
  down_weight~ : @tensor.TensorView,
) -> @tensor.Tensor {
  let gate_projection = linear(input~, weight=gate_weight)
  let activated_gate = silu(input=gate_projection.view())
  let up_projection = linear(input~, weight=up_weight)
  let gated = elementwise_mul(
    left=activated_gate.view(),
    right=up_projection.view(),
  )
  linear(input=gated.view(), weight=down_weight)
}

///|
/// Convert a vector of scores to weights that sum to 1.
fn softmax(input~ : @tensor.TensorView) -> @tensor.Tensor {
  if input.rank() != 1 || input.data.length() == 0 {
    abort("softmax input must be a non-empty vector")
  }

  let maximum = input.data.fold(init=input.data[0], (current_max, value) => {
    if value > current_max {
      value
    } else {
      current_max
    }
  })
  let exponentials = input.data.map(value => @math.expf(value - maximum))
  let sum = exponentials.fold(init=0.0F, (acc, value) => acc + value)
  let data = exponentials.map(value => value / sum)
  @tensor.Tensor::new(data~, shape=input.shape)
}
