///|
test "article creates scalar, vector, matrix, and 3D tensors" {
  let tensor0d = @torch.tensor(data=[1.0], shape=[])
  let tensor1d = @torch.tensor(data=[1.0, 2.0, 3.0], shape=[3])
  let tensor2d = @torch.tensor(data=[1.0, 2.0, 3.0, 4.0], shape=[2, 2])
  let tensor3d = @torch.tensor(data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], shape=[
    2, 2, 2,
  ])

  assert_eq(tensor0d.shape(), [])
  assert_eq(tensor1d.shape(), [3])
  assert_eq(tensor2d.shape(), [2, 2])
  assert_eq(tensor3d.shape(), [2, 2, 2])
  assert_eq(tensor1d.dtype(), "torch.float32")
}

///|
test "article performs common PyTorch tensor operations" {
  let tensor2d = @torch.tensor(data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])

  let reshaped = tensor2d.reshape([3, 2])
  assert_eq(reshaped.shape(), [3, 2])
  assert_eq(reshaped.to_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

  let viewed = tensor2d.view([3, 2])
  assert_eq(viewed.shape(), [3, 2])
  assert_eq(viewed.to_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

  let transposed = tensor2d.t()
  assert_eq(transposed.shape(), [3, 2])
  assert_eq(transposed.to_array(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

  let result = @torch.matmul(left=tensor2d, right=transposed)
  assert_eq(result.shape(), [2, 2])
  assert_eq(result.to_array(), [14.0, 32.0, 32.0, 77.0])
}
