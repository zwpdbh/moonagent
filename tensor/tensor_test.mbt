///|
fn test_tensor(data : Array[Float], shape : Array[Int32]) -> @tensor.Tensor {
  @tensor.Tensor::new(data~, shape~)
}

///|
fn assert_tensor_eq(
  actual : @tensor.Tensor,
  data : Array[Float],
  shape : Array[Int32],
) -> Unit raise {
  assert_eq(actual.data, data)
  assert_eq(actual.shape, shape)
}

///|
test "dot computes the vector dot product" {
  assert_eq(@tensor.dot(left=[1.0, 2.0, 3.0], right=[4.0, 5.0, 6.0]), 32.0)
}

///|
test "panic dot rejects mismatched lengths" {
  ignore(@tensor.dot(left=[1.0], right=[1.0, 2.0]))
}

///|
test "mat_vec_mul multiplies a non-square matrix by a vector" {
  let result = @tensor.mat_vec_mul(
    matrix=test_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0], [2, 3]).view(),
    vector=test_tensor([1.0, 2.0, 3.0], [3]).view(),
  )
  assert_tensor_eq(result, [1.0, 5.0], [2])
}

///|
test "panic mat_vec_mul rejects a non-matrix shape" {
  ignore(
    @tensor.mat_vec_mul(
      matrix=test_tensor([1.0, 2.0], [2]).view(),
      vector=test_tensor([1.0, 2.0], [2]).view(),
    ),
  )
}

///|
test "panic mat_vec_mul rejects an invalid vector length" {
  ignore(
    @tensor.mat_vec_mul(
      matrix=test_tensor([1.0, 0.0, 0.0, 1.0], [2, 2]).view(),
      vector=test_tensor([1.0], [1]).view(),
    ),
  )
}

///|
test "vec_mat_mul multiplies a vector by a non-square matrix" {
  let result = @tensor.vec_mat_mul(
    vector=test_tensor([1.0, 2.0], [2]).view(),
    matrix=test_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0], [2, 3]).view(),
  )
  assert_tensor_eq(result, [1.0, 2.0, 2.0], [3])
}

///|
test "panic vec_mat_mul rejects an invalid vector length" {
  ignore(
    @tensor.vec_mat_mul(
      vector=test_tensor([1.0], [1]).view(),
      matrix=test_tensor([1.0, 0.0, 0.0, 1.0], [2, 2]).view(),
    ),
  )
}

///|
test "tensor validates shape and exposes dimensions" {
  let tensor = test_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let view = tensor.view()
  assert_eq(view.rank(), 2)
  assert_eq(view.dim(0), 2)
  assert_eq(view.dim(1), 3)
}

///|
test "tensor view slices the first axis without copying" {
  let tensor = test_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let row = tensor.view().slice_first_axis(1)
  assert_eq(row.shape, [3])
  assert_eq(row.data.to_owned(), [4.0, 5.0, 6.0])
}

///|
test "panic tensor rejects inconsistent data length" {
  ignore(@tensor.Tensor::new(data=[1.0, 2.0], shape=[3]))
}

///|
test "arithmetic operators apply element-wise operations" {
  let left = test_tensor([2.0, 4.0, 6.0, 8.0], [2, 2])
  let right = test_tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
  assert_tensor_eq(left + right, [3.0, 6.0, 9.0, 12.0], [2, 2])
  assert_tensor_eq(left - right, [1.0, 2.0, 3.0, 4.0], [2, 2])
  assert_tensor_eq(left * right, [2.0, 8.0, 18.0, 32.0], [2, 2])
  assert_tensor_eq(left / right, [2.0, 2.0, 2.0, 2.0], [2, 2])
}

///|
test "panic arithmetic operators reject mismatched shapes" {
  let left = test_tensor([1.0, 2.0], [2])
  let right = test_tensor([1.0, 2.0], [1, 2])
  ignore(left + right)
}
