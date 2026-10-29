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
  weights : Array[Float],
  shape : Array[Int32],
  token_id : Int32,
) -> ArrayView[Float] {
  let vocab_size = shape[0]
  let hidden_size = shape[1]

  let start = token_id * hidden_size
  let end = start + hidden_size

  weights[start:end]
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
  input : ArrayView[Float],
  weight : ArrayView[Float],
  eps : Float,
) -> Array[Float] {
  if input.length() != weight.length() {
    abort("RMSNorm input and weight must have the same length")
  }
  if input.length() == 0 {
    abort("RMSNorm input must not be empty")
  }
  if eps < 0.0 {
    abort("RMSNorm epsilon must not be negative")
  }
  // use "F" to make it is 32-bit float, by default it is 64-bit double
  let sum_of_squares = input.fold(init=0.0F, (sum, value) => sum + value * value)
  let mean_square = sum_of_squares / Float::from_int(input.length())
  let inverse_rms = 1.0F / (mean_square + eps).sqrt()

  input.mapi((i, value) => value * inverse_rms * weight[i])
}
