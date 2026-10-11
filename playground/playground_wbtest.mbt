// Whitebox tests run within the package scope.
// Use them to validate internal helpers or invariants.
// Keep them focused; public behavior belongs in `_test.mbt`.

///|
test "whitebox: can call private shout()" {
  assert_eq(shout("inside"), "INSIDE")
}
