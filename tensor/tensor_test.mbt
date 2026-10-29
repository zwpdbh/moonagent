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
    matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    shape=[2, 3],
    vector=[1.0, 2.0, 3.0],
  )
  assert_eq(result, [1.0, 5.0])
}

///|
test "panic mat_vec_mul rejects a non-matrix shape" {
  ignore(
    @tensor.mat_vec_mul(matrix=[1.0, 0.0, 0.0, 1.0], shape=[2], vector=[
      1.0, 2.0,
    ]),
  )
}

///|
test "panic mat_vec_mul rejects inconsistent matrix data" {
  ignore(
    @tensor.mat_vec_mul(matrix=[1.0, 0.0, 0.0], shape=[2, 2], vector=[1.0, 2.0]),
  )
}

///|
test "panic mat_vec_mul rejects an invalid vector length" {
  ignore(
    @tensor.mat_vec_mul(matrix=[1.0, 0.0, 0.0, 1.0], shape=[2, 2], vector=[1.0]),
  )
}

///|
test "vec_mat_mul multiplies a vector by a non-square matrix" {
  let result = @tensor.vec_mat_mul(
    vector=[1.0, 2.0],
    matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    shape=[2, 3],
  )
  assert_eq(result, [1.0, 2.0, 2.0])
}

///|
test "panic vec_mat_mul rejects an invalid vector length" {
  ignore(
    @tensor.vec_mat_mul(vector=[1.0], matrix=[1.0, 0.0, 0.0, 1.0], shape=[2, 2]),
  )
}
