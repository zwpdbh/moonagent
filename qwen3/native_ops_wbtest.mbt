///|
test "native linear projection uses moonxi-net CPU tensors" {
  let input = @nparray.NpArray::from2d([[1.0, 2.0], [3.0, 4.0]])
  let weight = @nparray.NpArray::from2d([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
  let result = native_linear_projection(input~, weight~)

  assert_eq(result.shape.shape.length(), 2)
  assert_eq(result.shape.shape[0], 2)
  assert_eq(result.shape.shape[1], 3)
  assert_eq(result.data.length(), 6)
  let expected = [1.0F, 2.0F, 3.0F, 3.0F, 4.0F, 7.0F]
  for i in 0..<expected.length() {
    assert_eq(result.data[i], expected[i])
  }
}
