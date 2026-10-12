///|
pub(all) suberror TokenizerError {
  InvalidFormat(String)
  UnknownToken(Int)
  InvalidModel(String)
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
#warnings("-unused_field")
priv struct SpecialControlToken {
  id : Int
  content : String
  single_word : Bool
  lstrip : Bool
  rstrip : Bool
  normalized : Bool
  special : Bool
} derive(FromJson)

///|
priv struct QwenTokenizerModel {
  vocab : Map[String, Int]
  model_type : String // JSON key "type" — renamed because `type` is a keyword
}

///|
/// Manual decoder (instead of derive) so the JSON key "type" can map to the
/// field `model_type`. Mirrors what derive(FromJson) generates.
impl FromJson for QwenTokenizerModel with fn from_json(json, path) {
  guard json is Json::Object(obj) else {
    raise JsonDecodeError((path, "QwenTokenizerModel: expected object"))
  }
  let vocab : Map[String, Int] = match obj.get("vocab") {
    Some(v) => @json.from_json(v, path=path.add_key("vocab"))
    None =>
      raise JsonDecodeError((path.add_key("vocab"), "missing \"vocab\" field"))
  }
  let model_type : String = match obj.get("type") {
    Some(t) => @json.from_json(t, path=path.add_key("type"))
    None =>
      raise JsonDecodeError((path.add_key("type"), "missing \"type\" field"))
  }
  { vocab, model_type, }
}

///|
#warnings("-unused_field")
priv struct QwenTokenizerFile {
  added_tokens : Array[SpecialControlToken]
  model : QwenTokenizerModel
} derive(FromJson)

///|
/// Build a Tokenizer from the *contents* of tokenizer.json. Pure: no I/O.
pub fn Tokenizer::parse(json : String) -> Tokenizer raise TokenizerError {
  let parsed = @json.parse(json) catch {
    e => raise InvalidFormat(e.to_string())
  }
  let file : QwenTokenizerFile = @json.from_json(parsed) catch {
    e => raise InvalidFormat(e.to_string())
  }

  guard file.model.model_type == "BPE" else {
    raise InvalidModel("expected BPE tokenzier, got \{file.model.model_type} }")
  }

  let vocab = file.model.vocab

  // merge those special tokens
  for each in file.added_tokens {
    vocab[each.content] = each.id
  }

  // build revers one
  let ids : Map[Int, String] = Map([])
  for token, id in vocab {
    if ids.contains(id) {
      raise InvalidFormat("duplicated id \{id} for token: \{token}")
    }
    ids[id] = token
  }

  { vocab, ids, }
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
    #|{"model": {"vocab": {"a": 0, "b": 1, "!": 2}, "type": "BPE"}, "added_tokens": []}
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
    #|{"model": {"vocab": {"a": 0, "b": 1, "!": 2}, "type": "BPE"}, "added_tokens": []}
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
    e => raise e
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
    e => raise e
  }
  assert_true(raised)
}

///|
test "manual FromJson maps JSON key \"type\" to field model_type" {
  let j = @json.parse(
    (
      #|{"model": {"vocab": {"a": 0}, "type": "BPE"}, "added_tokens": []}
    ),
  )
  let file : QwenTokenizerFile = @json.from_json(j)
  assert_eq(file.model.model_type, "BPE")
  assert_eq(file.model.vocab.get("a"), Some(0))
}

///|
test "parse folds added_tokens into both maps" {
  let snippet =
    #|{"model": {"vocab": {"a": 0, "b": 1}, "type": "BPE"},
    #| "added_tokens": [
    #|   {"id": 2, "content": "<|endoftext|>", "single_word": false,
    #|    "lstrip": false, "rstrip": false, "normalized": false, "special": true}
    #| ]}
  let tok = Tokenizer::parse(snippet)
  assert_eq(tok.vocab.get("<|endoftext|>"), Some(2))
  assert_eq(tok.ids[2], "<|endoftext|>")
}
