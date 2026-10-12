///|
pub(all) suberror TokenizerError {
  InvalidFormat(String)
  UnknownToken(Int)
  InvalidModel(String)
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
    raise @json.JsonDecodeError((path, "QwenTokenizerModel: expected object"))
  }
  let vocab : Map[String, Int] = match obj.get("vocab") {
    Some(v) => @json.from_json(v, path=path.add_key("vocab"))
    None =>
      raise @json.JsonDecodeError(
        (path.add_key("vocab"), "missing \"vocab\" field"),
      )
  }
  let model_type : String = match obj.get("type") {
    Some(t) => @json.from_json(t, path=path.add_key("type"))
    None =>
      raise @json.JsonDecodeError(
        (path.add_key("type"), "missing \"type\" field"),
      )
  }
  { vocab, model_type, }
}

///|
/// Mirror of the "pre_tokenizer" section — an externally-tagged enum:
/// the JSON "type" field picks the variant.
priv enum PreTokenizer {
  Sequence(Array[PreTokenizer])
  Split(pattern~ : String, behavior~ : String, invert~ : Bool)
  ByteLevel(add_prefix_space~ : Bool, trim_offsets~ : Bool, use_regex~ : Bool)
}

///|
/// Helper: extract required field `key` and decode it as `T`,
/// keeping the JSON path for error messages.
fn[T : FromJson] field(
  obj : Map[String, Json],
  key : String,
  path : @json.JsonPath,
) -> T raise @json.JsonDecodeError {
  match obj.get(key) {
    Some(v) => @json.from_json(v, path=path.add_key(key))
    None =>
      raise @json.JsonDecodeError(
        (path.add_key(key), "missing \"\{key}\" field"),
      )
  }
}

///|
impl FromJson for PreTokenizer with fn from_json(json, path) {
  guard json is Json::Object(obj) else {
    raise @json.JsonDecodeError((path, "PreTokenizer: expected object"))
  }
  guard obj.get("type") is Some(Json::String(kind)) else {
    raise @json.JsonDecodeError(
      (path, "PreTokenizer: missing \"type\" discriminator"),
    )
  }
  match kind {
    "Sequence" => Sequence(field(obj, "pretokenizers", path))
    "Split" => {
      // pattern is nested: {"Regex": "..."}
      let pattern_map : Map[String, String] = field(obj, "pattern", path)
      guard pattern_map.get("Regex") is Some(regex) else {
        raise @json.JsonDecodeError(
          (path.add_key("pattern"), "Split: expected {\"Regex\": ...}"),
        )
      }
      Split(
        pattern=regex,
        behavior=field(obj, "behavior", path),
        invert=field(obj, "invert", path),
      )
    }
    "ByteLevel" =>
      ByteLevel(
        add_prefix_space=field(obj, "add_prefix_space", path),
        trim_offsets=field(obj, "trim_offsets", path),
        use_regex=field(obj, "use_regex", path),
      )
    _ =>
      raise @json.JsonDecodeError(
        (path, "PreTokenizer: unknown type \"\{kind}\""),
      )
  }
}

///|
#warnings("-unused_field")
priv struct QwenTokenizerDef {
  added_tokens : Array[SpecialControlToken]
  model : QwenTokenizerModel
  pre_tokenizer : PreTokenizer
} derive(FromJson)

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
pub fn Tokenizer::parse(json : String) -> Tokenizer raise TokenizerError {
  let parsed = @json.parse(json) catch {
    e => raise InvalidFormat(e.to_string())
  }
  let file : QwenTokenizerDef = @json.from_json(parsed) catch {
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
pub fn Tokenizer::load(path : String) -> Tokenizer raise TokenizerError {
  let content = @fs.read_file_to_string(path) catch {
    @fs.IOError(msg) =>
      raise InvalidFormat("cannot read file \{path}, error: \{msg}")
  }
  Tokenizer::parse(content)
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
    #|{"model": {"vocab": {"a": 0, "b": 1, "!": 2}, "type": "BPE"}, "added_tokens": [], "pre_tokenizer": {"type": "Sequence", "pretokenizers": []}}
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
    #|{"model": {"vocab": {"a": 0, "b": 1, "!": 2}, "type": "BPE"}, "added_tokens": [], "pre_tokenizer": {"type": "Sequence", "pretokenizers": []}}
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
      #|{"model": {"vocab": {"a": 0}, "type": "BPE"}, "added_tokens": [], "pre_tokenizer": {"type": "Sequence", "pretokenizers": []}}
    ),
  )
  let file : QwenTokenizerDef = @json.from_json(j)
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
    #| ],
    #| "pre_tokenizer": {"type": "Sequence", "pretokenizers": []}}
  let tok = Tokenizer::parse(snippet)
  assert_eq(tok.vocab.get("<|endoftext|>"), Some(2))
  assert_eq(tok.ids[2], "<|endoftext|>")
}

///|
test "load real tokenizer.json" {
  let tok = Tokenizer::load(@utils.qwen3_model_dir() + "/tokenizer.json")
  assert_eq(tok.vocab.length(), 151669) // 151643 vocab + 26 added
  assert_eq(tok.ids[151645], "<|im_end|>")
}

///|
test "pre_tokenizer decodes Sequence of Split + ByteLevel" {
  let j = @json.parse(
    (
      #|{"type": "Sequence",
      #| "pretokenizers": [
      #|   {"type": "Split",
      #|    "pattern": {"Regex": "(?i:'s|'t)"},
      #|    "behavior": "Isolated",
      #|    "invert": false},
      #|   {"type": "ByteLevel",
      #|    "add_prefix_space": false,
      #|    "trim_offsets": false,
      #|    "use_regex": false}
      #| ]}
    ),
  )
  let pt : PreTokenizer = @json.from_json(j)
  guard pt is Sequence(items) else { abort("expected Sequence") }
  assert_eq(items.length(), 2)
  guard items[0] is Split(pattern~, behavior~, invert~) else {
    abort("expected Split")
  }
  assert_eq(pattern, "(?i:'s|'t)")
  assert_eq(behavior, "Isolated")
  assert_eq(invert, false)

  guard items[1] is ByteLevel(add_prefix_space~, trim_offsets~, use_regex~) else {
    abort("expected ByteLevel")
  }
  assert_eq(add_prefix_space, false)
  assert_eq(use_regex, false)
  assert_eq(trim_offsets, false)
}
