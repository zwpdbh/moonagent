// Whitebox tests run within the package scope.
// Use them to validate internal helpers or invariants.
// Keep them focused; public behavior belongs in `_test.mbt`.

///|
test "split_with_pre_tokenizer uses string.Regex for supported patterns" {
  assert_eq(split_with_pre_tokenizer("a1b22c", "[[:digit:]]+"), [
    "a", "1", "b", "22", "c",
  ])
}

///|
test "split_with_pre_tokenizer uses regexp for Unicode property patterns" {
  assert_eq(split_with_pre_tokenizer("123abc 中文😀", "\\p{L}+"), [
    "123", "abc", " ", "中文", "😀",
  ])
}

///|
test "split_with_pre_tokenizer uses the Qwen manual fallback" {
  assert_true(
    try_split_with_string_regex("hello", qwen_manual_fallback_pattern) is None,
  )
  assert_true(
    try_split_with_regexp("hello", qwen_manual_fallback_pattern) is None,
  )
  assert_eq(split_with_pre_tokenizer("  hello", qwen_manual_fallback_pattern), [
    " ", " hello",
  ])
}
