///|
pub(all) suberror TokenizerError {
  InvalidFormat(String)
  UnknownToken(Int)
}

///|
#warnings("-struct_never_constructed-unused_field")
struct Tokenizer {
  // for encoding
  vocab : Map[String, Int]
  // for decoding
  ids : Map[Int, String]
}

///|
/// Build a Tokenizer from the *contents* of tokenizer.json. Pure: no I/O.
#warnings("-unused_error_type")
pub fn Tokenizer::parse(json : String) -> Tokenizer raise TokenizerError {
  let _parsed = @json.parse(json) catch {
    e => raise InvalidFormat(e.to_string())
  }
  ignore(json)
  abort("TODO: parse model.vocab into vocab map (+ reverse index for decode)")
}

///|
/// Load tokenizer.json from disk, then parse.
#warnings("-unused_async-unused_error_type")
pub async fn Tokenizer::load(path : String) -> Tokenizer raise TokenizerError {
  ignore(path)
  abort("TODO: read file contents, delegate to parse")
}

///|
/// Text -> token ids.
pub fn Tokenizer::encode(self : Tokenizer, text : String) -> Array[Int] {
  ignore(self)
  ignore(text)
  abort("TODO: pre-tokenize + BPE merges")
}

///|
/// Token ids -> text. Raises UnknownToken for ids outside the vocab.
#warnings("-unused_error_type")
pub fn Tokenizer::decode(
  self : Tokenizer,
  ids : Array[Int],
) -> String raise TokenizerError {
  ignore(self)
  ignore(ids)
  abort("TODO: byte-alphabet mapping + utf8 decode")
}

///|
test "parse builds vocab map from model.vocab" {
  let snippet =
    #|{"model": {"vocab": {"a": 0, "b": 1, "!": 2}}}
  let tok = Tokenizer::parse(snippet)
  assert_eq(tok.vocab.length(), 3)
  assert_eq(tok.vocab.get("b"), Some(1))
  assert_eq(tok.vocab.get("!"), Some(2))
}

///|
test "parse builds reverse index (id -> token) for decode" {
  // NOTE: requires adding a second field to `struct Tokenizer`
  // (spec README step 4), e.g. `ids : Array[String]`.
  let snippet =
    #|{"model": {"vocab": {"a": 0, "b": 1, "!": 2}}}
  let tok = Tokenizer::parse(snippet)
  assert_eq(tok.ids.length(), 3)
  assert_eq(tok.ids[0], "a")
  assert_eq(tok.ids[1], "b")
  assert_eq(tok.ids[2], "!")
}

///|
test "parse rejects invalid JSON" {
  let raised = try {
    Tokenizer::parse("not json") |> ignore
    false
  } catch {
    InvalidFormat(_) => true
  }
  assert_true(raised)
}

///|
test "parse rejects json without model.vocab" {
  let raised = try {
    Tokenizer::parse("{}") |> ignore
    false
  } catch {
    InvalidFormat(_) => true
  }
  assert_true(raised)
}
