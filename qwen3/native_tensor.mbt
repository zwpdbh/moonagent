///|
using @nparray {type NpArray}

///|
/// Tensor operations required by the native Qwen3 inference implementation.
pub(open) trait NativeTensor: Add {
  fn from_host(FixedArray[Float], Array[Int32]) -> Self
  fn shape(Self) -> Array[Int32]
  fn view(Self, Array[Int32]) -> Self
  fn permute(Self, Array[Int32]) -> Self
  fn unsqueeze(Self, dim~ : Int32) -> Self
  fn narrow(Self, dim~ : Int32, start~ : Int32, length~ : Int32) -> Self
  fn select(Self, dim~ : Int32, index~ : Int32) -> Self
  fn concat(Array[Self], dim~ : Int32) -> Self
  fn repeat_interleave(Self, repeats~ : Int32, dim~ : Int32) -> Self
  fn batched_matmul(Self, Self) -> Self
  fn linear(Self, Self) -> Self
  fn softmax_last(Self) -> Self
  fn embedding(Self, Array[Int32]) -> Self
  fn silu(Self) -> Self
  fn rms_norm(Self, Self, eps~ : Float) -> Self
  fn masked_fill(Self, Self, value~ : Float) -> Self
  fn argmax_last(Self) -> Self
  fn item_int(Self) -> Int32
  fn mul_elem(Self, Self) -> Self
  fn scale(Self, Float) -> Self
}

///|
pub impl NativeTensor for NpArray with fn from_host(data, shape) {
  NpArray::from_host(data, shape)
}

///|
pub impl NativeTensor for NpArray with fn shape(value) {
  value.shape.shape.iter().collect()
}

///|
pub impl NativeTensor for NpArray with fn view(value, shape) {
  value.view(FixedArray::from_array(shape))
}

///|
pub impl NativeTensor for NpArray with fn permute(value, axes) {
  value.permute(axes)
}

///|
pub impl NativeTensor for NpArray with fn unsqueeze(value, dim~) {
  value.unsqueeze(dim~)
}

///|
pub impl NativeTensor for NpArray with fn narrow(value, dim~, start~, length~) {
  value.narrow(dim~, start~, length~)
}

///|
pub impl NativeTensor for NpArray with fn select(value, dim~, index~) {
  value.select(dim~, index~)
}

///|
pub impl NativeTensor for NpArray with fn concat(values, dim~) {
  NpArray::concat(values, dim~)
}

///|
pub impl NativeTensor for NpArray with fn repeat_interleave(
  value,
  repeats~,
  dim~,
) {
  value.repeat_interleave(repeats~, dim~)
}

///|
pub impl NativeTensor for NpArray with fn batched_matmul(left, right) {
  left.batched_matmul(right)
}

///|
pub impl NativeTensor for NpArray with fn linear(input, weight) {
  input.linear(weight)
}

///|
pub impl NativeTensor for NpArray with fn softmax_last(input) {
  input.softmax_last()
}

///|
pub impl NativeTensor for NpArray with fn embedding(weight, token_ids) {
  weight.embedding(token_ids)
}

///|
pub impl NativeTensor for NpArray with fn silu(input) {
  input.silu()
}

///|
pub impl NativeTensor for NpArray with fn rms_norm(input, weight, eps~) {
  input.rms_norm(weight, eps~)
}

///|
pub impl NativeTensor for NpArray with fn masked_fill(input, mask, value~) {
  input.masked_fill(mask, value~)
}

///|
pub impl NativeTensor for NpArray with fn argmax_last(input) {
  input.argmax_last()
}

///|
pub impl NativeTensor for NpArray with fn item_int(input) {
  input.data[0].to_int()
}

///|
pub impl NativeTensor for NpArray with fn mul_elem(left, right) {
  left.mul_elem(right)
}

///|
pub impl NativeTensor for NpArray with fn scale(input, scalar) {
  input.scale(scalar)
}
