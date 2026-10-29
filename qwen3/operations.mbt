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

///|
/// Multiplying an input vector by a learned weight matrix.
/// shape  = [out_features, in_features]
/// input  = in_features values
/// weight = out_features × in_features values, flattened
/// output = out_features values 
/// Weight:  m × n matrix, Input:   n × 1 vector, Output:  m × 1 vector
/// Each output value is the dot product of one weight-matrix row with the input:
fn linear(
  input : ArrayView[Float],
  weight : ArrayView[Float],
  shape : Array[Int32],
) -> Array[Float] {
  let out_features = shape[0]
  let in_features = shape[1]

  if input.length() != in_features {
    abort(
      "matrix multiplication rank not correct, expect: \{in_features}, got: \{input.length()}",
    )
  }

  Array::makei(out_features, out_index => {
    let row_start = out_index * in_features
    let row = weight[row_start:row_start + in_features]
    row.foldi(init=0.0F, (in_index, sum, value) => sum + value * input[in_index])
  })
}
