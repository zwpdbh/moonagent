///|
/// Basic tensor operations for Qwen3, backed by PyTorch.
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
  weights~ : Tensor,
  token_id~ : Int32,
) -> Tensor raise @torch.TorchError {
  if weights.shape().length() != 2 {
    abort("embedding weights must be a rank-2 tensor")
  }
  @torch.select(input=weights, dim=0, index=token_id)
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
  input~ : Tensor,
  weight~ : Tensor,
  eps~ : Float,
) -> Tensor raise @torch.TorchError {
  if input.shape().length() != 1 || input.shape() != weight.shape() {
    abort("RMSNorm input and weight must be vectors with the same shape")
  }
  if input.shape()[0] == 0 {
    abort("RMSNorm input must not be empty")
  }
  if eps < 0.0 {
    abort("RMSNorm epsilon must not be negative")
  }
  @torch.rms_norm(input~, weight~, eps=eps.to_double())
}

///|
/// Multiply an input vector by a learned weight matrix.
/// Weight shape is `[out_features, in_features]`; output shape is
/// `[out_features]`.
fn linear(input~ : Tensor, weight~ : Tensor) -> Tensor raise @torch.TorchError {
  @torch.matmul(left=weight, right=input)
}

///|
fn softmax(input~ : Tensor, dim~ : Int32) -> Tensor raise @torch.TorchError {
  let shape = input.shape()
  if shape.length() != 1 || shape[0] == 0 {
    abort("softmax input must be a non-empty vector")
  }
  @torch.softmax(input~, dim~)
}

///|
/// SiLU activation (Sigmoid Linear Unit)
/// It is used in every MLP block
fn silu(input~ : Tensor) -> Tensor raise @torch.TorchError {
  @torch.silu(input~)
}

///|
/// Element-wise multiplication
fn elementwise_mul(
  left~ : Tensor,
  right~ : Tensor,
) -> Tensor raise @torch.TorchError {
  if left.shape() != right.shape() {
    abort("element-wise multiplication requires equal shapes")
  }
  @torch.mul(left~, right~)
}

///|
/// Qwen3 gated MLP: down_proj(SiLU(gate_proj(input)) * up_proj(input)).
fn mlp(
  input~ : Tensor,
  gate_weight~ : Tensor,
  up_weight~ : Tensor,
  down_weight~ : Tensor,
) -> Tensor raise @torch.TorchError {
  let gate_projection = linear(input~, weight=gate_weight)
  let activated_gate = silu(input=gate_projection)
  let up_projection = linear(input~, weight=up_weight)
  let gated = elementwise_mul(left=activated_gate, right=up_projection)
  linear(input=gated, weight=down_weight)
}
