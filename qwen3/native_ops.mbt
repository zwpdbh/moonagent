///|
/// Compute a rank-2 linear projection using the native moonxi-net CPU backend.
pub fn native_linear_projection(
  input~ : @nparray.NpArray,
  weight~ : @nparray.NpArray,
) -> @nparray.NpArray {
  input.matmul(weight.transpose())
}
