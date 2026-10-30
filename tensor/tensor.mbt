///|
/// An owned, contiguous row-major tensor.
pub struct Tensor {
  data : Array[Float]
  shape : Array[Int32]
}

///|
/// A borrowed view of a contiguous row-major tensor.
pub struct TensorView {
  data : ArrayView[Float]
  shape : Array[Int32]
}

///|
fn checked_shape_size(shape : Array[Int32]) -> Int32 {
  let mut size = 1
  for dimension in shape {
    if dimension < 0 {
      abort("tensor dimensions must not be negative")
    }
    size *= dimension
  }
  size
}

///|
/// Construct an owned tensor after validating its data length and shape.
pub fn Tensor::new(data~ : Array[Float], shape~ : Array[Int32]) -> Tensor {
  if data.length() != checked_shape_size(shape) {
    abort("tensor data length does not match its shape")
  }
  { data, shape, }
}

///|
/// Construct a borrowed tensor view after validating its data length and shape.
pub fn TensorView::new(
  data~ : ArrayView[Float],
  shape~ : Array[Int32],
) -> TensorView {
  if data.length() != checked_shape_size(shape) {
    abort("tensor view data length does not match its shape")
  }
  { data, shape, }
}

///|
/// Borrow this owned tensor without copying its values.
pub fn Tensor::view(self : Tensor) -> TensorView {
  { data: self.data[:], shape: self.shape, }
}

///|
/// Return the number of dimensions.
pub fn TensorView::rank(self : TensorView) -> Int32 {
  self.shape.length()
}

///|
/// Return the size of one dimension.
pub fn TensorView::dim(self : TensorView, axis : Int32) -> Int32 {
  if axis < 0 || axis >= self.rank() {
    abort("tensor axis is out of bounds")
  }
  self.shape[axis]
}

///|
/// Select one index from axis 0, the leftmost dimension in `shape`, without
/// copying data. The selected dimension is removed, so the returned rank is one
/// less than the input rank. Axis 0 may represent a matrix row, an attention
/// head, or another outermost dimension depending on the tensor.
///
/// The returned view is contiguous because it selects one complete block from
/// the original row-major storage. Its starting offset is
/// `index * product(shape[1:])`.
///
/// For example, a `[3, 2, 2]` tensor can represent three KV heads, two sequence
/// positions, and two feature dimensions:
///
/// ```text
/// [
///   [[1, 2], [3, 4]],      // KV head 0
///   [[5, 6], [7, 8]],      // KV head 1
///   [[9, 10], [11, 12]],   // KV head 2
/// ]
/// Flat data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
/// ```
///
/// `slice_first_axis(1)` selects KV head 1 and returns a view with flat data
/// `[5, 6, 7, 8]` and shape `[2, 2]`. This is equivalent to PyTorch's
/// `tensor.select(dim=0, index=1)` or `tensor[1]`.
pub fn TensorView::slice_first_axis(
  self : TensorView,
  index : Int32,
) -> TensorView {
  if self.rank() == 0 {
    abort("cannot slice the first axis of a scalar tensor")
  }
  if index < 0 || index >= self.shape[0] {
    abort("tensor slice index is out of bounds")
  }
  let shape = self.shape[1:].to_owned()
  let block_size = checked_shape_size(shape)
  let start = index * block_size
  { data: self.data[start:start + block_size], shape, }
}

///|
/// Compute the dot product of two equal-length vectors.
///
/// Mathematically, this calculates `sum(left[i] * right[i])` and returns one
/// scalar. Both inputs are vectors even though MoonBit represents them with the
/// same `ArrayView[Float]` type. Labeled arguments make their roles explicit at
/// the call site and prevent accidentally relying on parameter position.
pub fn dot(left~ : ArrayView[Float], right~ : ArrayView[Float]) -> Float {
  if left.length() != right.length() {
    abort("dot product inputs must have equal lengths")
  }
  left.foldi(init=0.0F, (index, sum, value) => sum + value * right[index])
}

///|
/// Validate that a tensor is a matrix and return `(rows, columns)`.
fn matrix_dimensions(matrix : TensorView) -> (Int32, Int32) {
  if matrix.rank() != 2 {
    abort("expected a rank-2 matrix")
  }
  (matrix.shape[0], matrix.shape[1])
}

///|
/// Multiply a matrix by a column vector: `matrix[rows, columns] * vector[columns]`.
///
/// The result is an owned rank-1 tensor with shape `[rows]`. Each output value
/// is the dot product of one matrix row with `vector`.
pub fn mat_vec_mul(matrix~ : TensorView, vector~ : TensorView) -> Tensor {
  let (rows, columns) = matrix_dimensions(matrix)
  if vector.rank() != 1 || vector.shape[0] != columns {
    abort("matrix columns must equal the vector length")
  }
  let data = Array::makei(rows, row_index => {
    let row_start = row_index * columns
    let row = matrix.data[row_start:row_start + columns]
    dot(left=row, right=vector.data)
  })
  Tensor::new(data~, shape=[rows])
}

///|
/// Multiply a row vector by a matrix: `vector[rows] * matrix[rows, columns]`.
///
/// The result is an owned rank-1 tensor with shape `[columns]`. Each output
/// value combines one matrix column with the vector.
pub fn vec_mat_mul(vector~ : TensorView, matrix~ : TensorView) -> Tensor {
  let (rows, columns) = matrix_dimensions(matrix)
  if vector.rank() != 1 || vector.shape[0] != rows {
    abort("vector length must equal the matrix rows")
  }
  let data = Array::makei(columns, column => {
    vector.data.foldi(init=0.0F, (row, sum, value) => {
      sum + value * matrix.data[row * columns + column]
    })
  })
  Tensor::new(data~, shape=[columns])
}

///|
/// Combine two tensors with equal shapes element by element.
fn zip_with(
  left : TensorView,
  right : TensorView,
  operation : (Float, Float) -> Float,
) -> Tensor {
  if left.shape != right.shape {
    abort("element-wise operations require equal tensor shapes")
  }
  let data = left.data.mapi((index, value) => {
    operation(value, right.data[index])
  })
  Tensor::new(data~, shape=left.shape)
}

///|
/// Add corresponding elements without broadcasting.
pub fn elementwise_add(left~ : TensorView, right~ : TensorView) -> Tensor {
  zip_with(left, right, (a, b) => a + b)
}

///|
/// Subtract corresponding elements without broadcasting.
pub fn elementwise_sub(left~ : TensorView, right~ : TensorView) -> Tensor {
  zip_with(left, right, (a, b) => a - b)
}

///|
/// Multiply corresponding elements without broadcasting.
/// This is not matrix multiplication.
pub fn elementwise_mul(left~ : TensorView, right~ : TensorView) -> Tensor {
  zip_with(left, right, (a, b) => a * b)
}

///|
/// Divide corresponding elements without broadcasting.
pub fn elementwise_div(left~ : TensorView, right~ : TensorView) -> Tensor {
  zip_with(left, right, (a, b) => a / b)
}

///|
/// `+` is element-wise addition for owned tensors.
pub impl Add for Tensor with fn add(self, other) {
  elementwise_add(left=self.view(), right=other.view())
}

///|
/// `-` is element-wise subtraction for owned tensors.
pub impl Sub for Tensor with fn sub(self, other) {
  elementwise_sub(left=self.view(), right=other.view())
}

///|
/// `*` is element-wise multiplication, not matrix multiplication.
pub impl Mul for Tensor with fn mul(self, other) {
  elementwise_mul(left=self.view(), right=other.view())
}

///|
/// `/` is element-wise division for owned tensors.
pub impl Div for Tensor with fn div(self, other) {
  elementwise_div(left=self.view(), right=other.view())
}

///|
pub extend Tensor with Add::{add}

///|
pub extend Tensor with Sub::{sub}

///|
pub extend Tensor with Mul::{mul}

///|
pub extend Tensor with Div::{div}
