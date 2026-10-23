// Main package file for your library code.
// Put public APIs and shared helpers here.
// Keep blocks small and focused for easy refactors.

///|
/// Greets someone by name. (Demo of library-package code called from cmd/main.)
pub fn greet(name : String) -> String {
  "Hello, \{name}! (from the playground library package)"
}

///|
/// Helper not marked pub — invisible outside this package, but testable here.
fn shout(s : String) -> String {
  s.to_upper()
}

///|
/// Inline test (Rust: #[test] in the same file). Lives next to the code,
/// can see private items like `shout`.
test "shout uppercases" {
  assert_eq(shout("moon"), "MOON")
}
