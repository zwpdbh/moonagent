// Blackbox tests run against the package's public API. Use `@playground` to import.
// Prefer high-level behaviors over implementation details.
// Run `moon test` to execute and `moon test --update` to refresh snapshots.

///|
test "blackbox: greet via public API only" {
  // Only pub items are visible here — `shout` would NOT compile in this file.
  assert_eq(@playground.greet("test"), "Hello, test! (from the playground library package)")
}

///|
test "snapshot style with inspect" {
  // inspect() records the printed representation as a snapshot;
  // run `moon test --update` to write/refresh the expected value.
  inspect(@playground.greet("snap"), content="Hello, snap! (from the playground library package)")
}
