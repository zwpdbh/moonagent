///|
pub(all) suberror TokenizerError {
  InvalidFormat(String)
  UnknownToken(Int32)
  InvalidModel(String)
}

///|
#warnings("-unused_field")
priv struct SpecialControlToken {
  id : Int32
  content : String
  single_word : Bool
  lstrip : Bool
  rstrip : Bool
  normalized : Bool
  special : Bool
} derive(FromJson)

///|
priv struct QwenTokenizerModel {
  vocab : Map[String, Int32]
  merges : Array[Array[String]]
  model_type : String // JSON key "type" — renamed because `type` is a keyword
}

///|
/// Manual decoder (instead of derive) so the JSON key "type" can map to the
/// field `model_type`. Mirrors what derive(FromJson) generates.
impl FromJson for QwenTokenizerModel with fn from_json(json, path) {
  guard json is Json::Object(obj) else {
    raise @json.JsonDecodeError((path, "QwenTokenizerModel: expected object"))
  }
  let vocab : Map[String, Int32] = field(obj, "vocab", path)
  let merges : Array[Array[String]] = field(obj, "merges", path)
  let model_type : String = match obj.get("type") {
    Some(t) => @json.from_json(t, path=path.add_key("type"))
    None =>
      raise @json.JsonDecodeError(
        (path.add_key("type"), "missing \"type\" field"),
      )
  }

  { vocab, merges, model_type, }
}

///|
/// Mirror of the "pre_tokenizer" section — an externally-tagged enum:
/// the JSON "type" field picks the variant.
priv enum PreTokenizer {
  // The key is when match on Sequence, it matches on Array of itself which 
  // then go further matching on Split and ByteLevel
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
/// Notice path grows as you descend during recurisve parse
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
  vocab : Map[String, Int32]
  // for decoding
  ids : Map[Int32, String]
  special_vocab : Map[String, Int32]
  special_ids : Map[Int32, String]
  // BPE merge table: pair -> rank (lower merges first)
  merges : Map[(String, String), Int32]
  // the pre-tokenizer regex pattern, kept for validation/reference.
  // NOTE: core's @string.Regex cannot parse this PCRE pattern (\\p{L}, lookahead),
  // so split_text hand-rolls its behavior instead of executing it.
  regex_pattern : String
}

///|
/// Extract the object at a JSON value, or raise with a clear message.
#warnings("-unused_value")
fn expect_object(
  j : Json,
  what~ : String,
) -> Map[String, Json] raise TokenizerError {
  guard j is Json::Object(o) else {
    raise InvalidFormat("\{what}: expected JSON object")
  }
  o
}

///|
/// Extract a required key from a JSON object, or raise with the key name.
#warnings("-unused_value")
fn expect_key(
  obj : Map[String, Json],
  key : String,
  what~ : String,
) -> Json raise TokenizerError {
  match obj.get(key) {
    Some(v) => v
    None => raise InvalidFormat("\{what}: missing \"{key}\"")
  }
}

///|
/// Build a Tokenizer from the *contents* of tokenizer.json. Pure: no I/O.
pub fn Tokenizer::new(
  tokenizer_json : String,
) -> Tokenizer raise TokenizerError {
  let parsed = @json.parse(tokenizer_json) catch {
    e => raise InvalidFormat(e.to_string())
  }
  let qwen_tokenizer_def : QwenTokenizerDef = @json.from_json(parsed) catch {
    e => raise InvalidFormat(e.to_string())
  }

  guard qwen_tokenizer_def.model.model_type == "BPE" else {
    raise InvalidModel(
      "expected BPE tokenzier, got \{qwen_tokenizer_def.model.model_type} }",
    )
  }

  let vocab = qwen_tokenizer_def.model.vocab
  let special_vocab : Map[String, Int32] = Map([])
  let special_ids : Map[Int32, String] = Map([])

  // merge those special tokens
  for each_special_token in qwen_tokenizer_def.added_tokens {
    vocab[each_special_token.content] = each_special_token.id
    special_ids.set(each_special_token.id, each_special_token.content)
    special_vocab.set(each_special_token.content, each_special_token.id)
  }

  // build revers one
  let ids : Map[Int32, String] = Map([])
  for token, id in vocab {
    if ids.contains(id) {
      raise InvalidFormat("duplicated id \{id} for token: \{token}")
    }
    ids[id] = token
  }

  let regex_pattern = get_regex_pattern_from_pre_tokenizer(
    qwen_tokenizer_def.pre_tokenizer,
  )

  let merges = qwen_tokenizer_def.model.merges |> parse_merges

  { vocab, ids, special_vocab, special_ids, merges, regex_pattern, }
}

///|
/// Parse the "merges" array from tokenizer.json's model section into
/// pair -> rank (rank = position in the array, lower merges first).
/// In this file merges are JSON pairs: [["Ġ", "t"], ["i", "n"], ...].
#warnings("-unused_error_type")
fn parse_merges(
  merges : Array[Array[String]],
) -> Map[(String, String), Int32] raise TokenizerError {
  let merged : Map[(String, String), Int32] = Map([])

  for (index, pair) in merges.iter2() {
    let key = match pair {
      [x, y] => (x, y)
      _ =>
        raise InvalidFormat(
          "The merges in tokenizer.json should be two element arrays",
        )
    }
    merged.set(key, index)
  }
  merged
}

///|
fn get_regex_pattern_from_pre_tokenizer(
  pre_tokenizer : PreTokenizer,
) -> String raise TokenizerError {
  match pre_tokenizer {
    Sequence([s, _]) => get_regex_pattern_from_pre_tokenizer(s)
    Split(pattern~, ..) => pattern
    ByteLevel(..) =>
      raise InvalidFormat("expected Split pre-tokenizer, found ByteLevel")
    Sequence(_) =>
      raise InvalidFormat("expected Split as first pre-tokenizer in Sequence")
  }
}

///|
/// Load tokenizer.json from disk, then parse.
pub fn Tokenizer::load(path : String) -> Tokenizer raise TokenizerError {
  let content = @fs.read_file_to_string(path) catch {
    @fs.IOError(msg) =>
      raise InvalidFormat("cannot read file \{path}, error: \{msg}")
  }
  Tokenizer::new(content)
}

///|
/// Use byte pair encoding to convert text into an integer representation (token IDs) for LLM trainning
/// Ref: https://sebastianraschka.com/blog/2025/bpe-from-scratch.html
/// Q: Where is the vocabulary coming from? Do I need to build it manually first?
/// A: The goal of BPE tokenization algorithm is to build a vocabulary of commonly occurring subwords
/// So, yes, we need to build the vocabulary first, then use it to encode the text.
/// Q: what is the Tokenizer we created from qwen's tokenizer.json file? how it is related with our algorithm
/// A: ?
/// Q: From the example: "the cat in the hat" in https://sebastianraschka.com/blog/2025/bpe-from-scratch.html#14-bpe-algorithm-example
/// "the" appears twice, so we need to split the "the cat in the hat" as our first step. How this split happened? 
/// Q: Do we use some help from the tokenizer.json or we do the split purely by our own?
/// === Following are general steps:
/// Text -> token ids.
/// Step 1: Use PreTokenizer's regex_pattern to split into chunks
/// Step 2: Byte-level mapping (bytes → alphabet chars)
/// Step 3: The BPE merge loop (per chunk)
/// Step 4: Token → id
pub fn Tokenizer::encode(self : Tokenizer, text : String) -> Array[Int32] {
  abort("TODO")
}

///|
/// Run the BPE merge loop on one chunk (starts as single-char strings):
/// repeatedly merge the adjacent pair with the lowest rank until no
/// adjacent pair exists in `merges`. Merges never cross chunks.
fn bpe_merge(
  chars : Array[String],
  merges : Map[(String, String), Int32],
) -> Array[String] {
  ignore(merges)
  chars // TODO: greedy-by-rank merge loop (placeholder: never merges)
}

///|
/// Token ids -> text. Raises UnknownToken for ids outside the vocab.
#warnings("-unused_error_type")
pub fn Tokenizer::decode(
  self : Tokenizer,
  ids : Array[Int32],
) -> String raise TokenizerError {
  ignore(self)
  ignore(ids)
  abort("TODO: byte-alphabet mapping + utf8 decode")
}

///|
test "parse builds vocab map from model.vocab" {
  let snippet =
    #|{"model": {"vocab": {"a": 0, "b": 1, "!": 2}, "type": "BPE", "merges": []}, "added_tokens": [], "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]}}
  let tok = Tokenizer::new(snippet)
  assert_eq(tok.vocab.length(), 3)
  assert_eq(tok.vocab.get("b"), Some(1))
  assert_eq(tok.vocab.get("!"), Some(2))
}

///|
test "parse builds reverse index (id -> token) for decode" {
  // NOTE: requires adding a second field to `struct Tokenizer`
  // (spec README step 4), e.g. `ids : Array[String]`.
  let snippet =
    #|{"model": {"vocab": {"a": 0, "b": 1, "!": 2}, "type": "BPE", "merges": []}, "added_tokens": [], "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]}}
  let tok = Tokenizer::new(snippet)
  assert_eq(tok.ids.length(), 3)
  assert_eq(tok.ids[0], "a")
  assert_eq(tok.ids[1], "b")
  assert_eq(tok.ids[2], "!")
}

///|
test "parse rejects invalid JSON" {
  let raised = try {
    Tokenizer::new("not json") |> ignore
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
    Tokenizer::new("{}") |> ignore
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
      #|{"model": {"vocab": {"a": 0}, "type": "BPE", "merges": []}, "added_tokens": [], "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]}}
    ),
  )
  let file : QwenTokenizerDef = @json.from_json(j)
  assert_eq(file.model.model_type, "BPE")
  assert_eq(file.model.vocab.get("a"), Some(0))
}

///|
test "parse folds added_tokens into both maps" {
  let snippet =
    #|{"model": {"vocab": {"a": 0, "b": 1}, "type": "BPE", "merges": []},
    #| "added_tokens": [
    #|   {"id": 2, "content": "<|endoftext|>", "single_word": false,
    #|    "lstrip": false, "rstrip": false, "normalized": false, "special": true}
    #| ],
    #| "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]}}
  let tok = Tokenizer::new(snippet)
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
