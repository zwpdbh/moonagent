///|
pub impl NativeTensor for @gt.GpuTensor with fn from_host(data, shape) {
  @gt.GpuTensor::from_host(data, shape)
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn shape(value) {
  value.shape.shape.iter().collect()
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn view(value, shape) {
  value.view(shape)
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn permute(value, axes) {
  value.permute(axes)
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn unsqueeze(value, dim~) {
  value.unsqueeze(dim~)
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn narrow(
  value,
  dim~,
  start~,
  length~,
) {
  value.narrow(dim~, start~, length~)
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn select(value, dim~, index~) {
  value.select(dim~, index~)
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn concat(values, dim~) {
  @gt.GpuTensor::concat(values, dim~)
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn repeat_interleave(
  value,
  repeats~,
  dim~,
) {
  value.repeat_interleave(repeats~, dim~)
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn batched_matmul(left, right) {
  left.batched_matmul(right)
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn linear(input, weight) {
  input.linear(weight)
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn softmax_last(input) {
  input.softmax_last()
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn embedding(weight, token_ids) {
  weight.embedding(token_ids)
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn silu(input) {
  input.silu()
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn rms_norm(input, weight, eps~) {
  input.rms_norm(weight, eps~)
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn masked_fill(input, mask, value~) {
  input.masked_fill(mask, value~)
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn argmax_last(input) {
  input.argmax_last()
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn item_int(input) {
  input.item().to_int()
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn mul_elem(left, right) {
  left.mul_elem(right)
}

///|
pub impl NativeTensor for @gt.GpuTensor with fn scale(input, scalar) {
  input.scale(scalar)
}
