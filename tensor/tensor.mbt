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
/// Validate matrix metadata and return `(rows, columns)`.
///
/// A matrix is stored as one flat row-major array: all columns of row 0 first,
/// then all columns of row 1, and so on. `shape` supplies the two-dimensional
/// interpretation that the flat `ArrayView[Float]` cannot carry by itself.
fn matrix_dimensions(
  matrix : ArrayView[Float],
  shape : Array[Int32],
) -> (Int32, Int32) {
  if shape.length() != 2 {
    abort("matrix shape must contain exactly two dimensions")
  }
  let rows = shape[0]
  let columns = shape[1]
  if rows < 0 || columns < 0 {
    abort("matrix dimensions must not be negative")
  }
  if matrix.length() != rows * columns {
    abort("matrix data length does not match its shape")
  }
  (rows, columns)
}

///|
/// Multiply a matrix by a column vector: `matrix[rows, columns] * vector[columns]`.
///
/// The result contains `rows` values. Each output value is the dot product of
/// one matrix row with `vector`:
/// `output[row] = dot(matrix[row, :], vector)`.
///
/// `matrix` uses flat row-major storage and `shape` must be `[rows, columns]`.
/// The labeled arguments distinguish the matrix from the vector even though
/// both have type `ArrayView[Float]`.
pub fn mat_vec_mul(
  matrix~ : ArrayView[Float],
  shape~ : Array[Int32],
  vector~ : ArrayView[Float],
) -> Array[Float] {
  let (rows, columns) = matrix_dimensions(matrix, shape)
  if vector.length() != columns {
    abort("matrix columns must equal the vector length")
  }
  Array::makei(rows, row_index => {
    let row_start = row_index * columns
    let row = matrix[row_start:row_start + columns]
    dot(left=row, right=vector)
  })
}

///|
/// Multiply a row vector by a matrix: `vector[rows] * matrix[rows, columns]`.
///
/// The result contains `columns` values. Each output value combines one matrix
/// column with the vector:
/// `output[column] = sum(vector[row] * matrix[row, column])` over every row.
///
/// This is the orientation used for `attention_weights * values` in attention.
/// `matrix` uses flat row-major storage and `shape` must be `[rows, columns]`.
/// Labeled arguments make the multiplication direction clear at the call site.
pub fn vec_mat_mul(
  vector~ : ArrayView[Float],
  matrix~ : ArrayView[Float],
  shape~ : Array[Int32],
) -> Array[Float] {
  let (rows, columns) = matrix_dimensions(matrix, shape)
  if vector.length() != rows {
    abort("vector length must equal the matrix rows")
  }
  Array::makei(columns, column => {
    vector.foldi(init=0.0F, (row, sum, value) => {
      sum + value * matrix[row * columns + column]
    })
  })
}
