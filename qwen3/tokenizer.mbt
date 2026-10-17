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
/// “th” appears twice, so first, we need to split the "the cat in the hat" to do that. 
/// How the the split happended. It seems we need to use the 0-255 token value? where is it comes from? 
/// Text -> token ids.
/// Step 1: Use PreTokenizer's regex_pattern to split into chunks
/// Step 2: Byte-level mapping (bytes → alphabet chars)
/// Step 3: The BPE merge loop (per chunk)
/// Step 4: Token → id
pub fn Tokenizer::encode(self : Tokenizer, text : String) -> Array[Int32] {
  let ids : Array[Int32] = []
  for chunk in split_text(text) {
    // byte-level alphabet: " large" -> "Ġlarge"
    let word = to_byte_chars(chunk)
    // start BPE from single-char tokens
    let chars : Array[String] = []
    for c in word {
      chars.push(c.to_string())
    }
    // replay the merge table, then look up ids
    for token in bpe_merge(chars, self.merges) {
      match self.vocab.get(token) {
        Some(id) => ids.push(id)
        None =>
          // byte-level BPE covers every byte, so this means a bug or a
          // foreign tokenizer file
          abort("token not in vocab: \{token}")
      }
    }
  }
  ids
}

///|
/// Split text into pre-tokenizer chunks (the "Split" step of the pipeline).
/// NOTE: regex matches ARE the chunks (Isolated behavior), not delimiters.
/// Implement by hand-rolling the alternation branches in order:
/// contractions | optional-punct+letters | digit | space+punct-run |
/// newlines | trailing-ws | whitespace.
// --- split_text helpers: one matcher per alternation branch of the Qwen
// pre-tokenizer pattern, tried in regex order (first match wins) ---

///|
/// Approximation of \p{L}: ASCII letters plus non-ASCII non-space non-digit
/// chars (covers CJK ideographs; may misclassify rare symbols).
fn is_letter(c : Char) -> Bool {
  c.is_ascii_alphabetic() ||
  (c.to_int() > 0x7F && !c.is_whitespace() && !c.is_numeric())
}

///|
/// [^\s\p{L}\p{N}] — punctuation/symbol class.
fn is_punct(c : Char) -> Bool {
  !c.is_whitespace() && !is_letter(c) && !c.is_numeric()
}

///|
/// Case-insensitive check that cs[i..] starts with `s`.
fn matches_at(cs : Array[Char], i : Int32, s : String) -> Bool {
  if i + s.length() > cs.length() {
    return false
  }
  for k, c in s {
    if cs[i + k].to_ascii_lowercase() != c {
      return false
    }
  }
  true
}

///|
/// Branch 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
fn match_contraction(cs : Array[Char], i : Int32) -> Int32 {
  for cand in ["'re", "'ve", "'ll", "'s", "'t", "'m", "'d"] {
    if matches_at(cs, i, cand) {
      return cand.length()
    }
  }
  0
}

///|
/// Branch 2: [^\r\n\p{L}\p{N}]?\p{L}+ — optional punct/space prefix + letters.
fn match_word(cs : Array[Char], i : Int32) -> Int32 {
  let mut j = i
  if j < cs.length() &&
    cs[j] != '\r' &&
    cs[j] != '\n' &&
    !is_letter(cs[j]) &&
    !cs[j].is_numeric() &&
    j + 1 < cs.length() &&
    is_letter(cs[j + 1]) {
    j += 1
  }
  let mut n = 0
  while j + n < cs.length() && is_letter(cs[j + n]) {
    n += 1
  }
  if n == 0 {
    0
  } else {
    j - i + n
  }
}

///|
/// Branch 3: \p{N} — a single digit.
fn match_digit(cs : Array[Char], i : Int32) -> Int32 {
  if i < cs.length() && cs[i].is_numeric() {
    1
  } else {
    0
  }
}

///|
/// Branch 4: ` ?[^\s\p{L}\p{N}]+[\r\n]*` — optional space + punct run.
fn match_punct(cs : Array[Char], i : Int32) -> Int32 {
  let mut j = i
  if j < cs.length() &&
    cs[j] == ' ' &&
    j + 1 < cs.length() &&
    is_punct(cs[j + 1]) {
    j += 1
  }
  let mut n = 0
  while j + n < cs.length() && is_punct(cs[j + n]) {
    n += 1
  }
  if n == 0 {
    return 0
  }
  j += n
  while j < cs.length() && (cs[j] == '\r' || cs[j] == '\n') {
    j += 1
  }
  j - i
}

///|
/// Branch 5: \s*[\r\n]+ — whitespace including at least one newline.
fn match_newlines(cs : Array[Char], i : Int32) -> Int32 {
  let mut j = i
  while j < cs.length() &&
        cs[j].is_whitespace() &&
        cs[j] != '\r' &&
        cs[j] != '\n' {
    j += 1
  }
  let mut n = 0
  while j + n < cs.length() && (cs[j + n] == '\r' || cs[j + n] == '\n') {
    n += 1
  }
  if n == 0 {
    0
  } else {
    j + n - i
  }
}

///|
/// Branch 6: \s+(?!\S) — whitespace run at end, or all-but-last char of a
/// run (the last whitespace may attach to a following word via branch 2).
fn match_trailing_ws(cs : Array[Char], i : Int32) -> Int32 {
  let mut n = 0
  while i + n < cs.length() && cs[i + n].is_whitespace() {
    n += 1
  }
  if n == 0 {
    return 0
  }
  if i + n == cs.length() {
    return n
  }
  if n >= 2 {
    return n - 1
  }
  0
}

///|
/// Branch 7: \s+ — any other whitespace run.
fn match_whitespace(cs : Array[Char], i : Int32) -> Int32 {
  let mut n = 0
  while i + n < cs.length() && cs[i + n].is_whitespace() {
    n += 1
  }
  n
}

///|
/// Split text into pre-tokenizer chunks (the "Split" step of the pipeline).
/// Regex matches ARE the chunks (Isolated behavior), not delimiters.
fn split_text(text : String) -> Array[String] {
  // the alternation branches, in regex order (first non-zero match wins)
  let branches = [
    match_contraction, match_word, match_digit, match_punct, match_newlines, match_trailing_ws,
    match_whitespace,
  ]
  let chars = text.to_array()
  let chunks : Array[String] = []
  let mut i = 0
  while i < chars.length() {
    let len = branches
      .iter()
      .filter_map(fn(f) {
        match f(chars, i) {
          0 => None
          n => Some(n)
        }
      })
      .next()
      .unwrap_or(1) // safety net: always make progress
    chunks.push(String::from_array(chars[i:i + len].to_owned()))
    i += len
  }
  chunks
}

///|
/// Map a chunk's UTF-8 bytes to the byte-level alphabet chars:
/// printable bytes map to themselves, others (space, control, >= 0x80)
/// map to chars in the 0x100+ range (space 0x20 -> 'Ġ').
/// (The reverse mapping is what `decode` will need.)

///|
/// GPT-2 byte alphabet: byte -> printable char code.
/// Printable ranges (0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF) map to themselves;
/// the 68 remaining bytes map to 0x100+n in increasing order
/// (space 0x20 -> 0x120 = 'Ġ').
fn byte_to_char(b : Byte) -> Char {
  let code = b.to_int()
  let mapped = if (code >= 0x21 && code <= 0x7E) ||
    (code >= 0xA1 && code <= 0xAC) ||
    code >= 0xAE {
    code
  } else if code <= 0x20 {
    256 + code
  } else if code >= 0x7F && code <= 0xA0 {
    256 + 33 + (code - 0x7F)
  } else {
    256 + 67 // 0xAD
  }
  mapped.unsafe_to_char()
}

///|
/// Map a chunk's UTF-8 bytes to the byte-level alphabet chars.
/// (The reverse mapping is what `decode` will need.)
fn to_byte_chars(chunk : String) -> String {
  let buf = StringBuilder()
  for b in @encoding.encode(UTF8, chunk) {
    buf.write_char(byte_to_char(b))
  }
  buf.to_string()
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

///|
test "split_text chunks plain words and keeps leading space" {
  assert_eq(split_text("Explain large language models."), [
    "Explain", " large", " language", " models", ".",
  ])
}

///|
test "split_text splits off contractions" {
  assert_eq(split_text("don't"), ["don", "'t"])
  assert_eq(split_text("I'm"), ["I", "'m"])
}

///|
test "split_text splits digits one by one" {
  assert_eq(split_text("3.14"), ["3", ".", "1", "4"])
}

///|
test "to_byte_chars maps space to Ġ, keeps letters" {
  assert_eq(to_byte_chars(" large"), "Ġlarge")
  assert_eq(to_byte_chars("Explain"), "Explain")
}

///|
test "bpe_merge replays merges by rank" {
  // ranks: (E,x)=0 merges before (p,l)=1 etc.
  let merges : Map[(String, String), Int32] = Map([])
  merges[("E", "x")] = 0
  merges[("Ex", "p")] = 5 // must NOT win over lower ranks
  merges[("p", "l")] = 1
  merges[("pl", "a")] = 2
  merges[("pla", "i")] = 3
  merges[("plai", "n")] = 4
  assert_eq(bpe_merge(["E", "x", "p", "l", "a", "i", "n"], merges), [
    "Ex", "plain",
  ])
}

///|
test "bpe_merge merges a single known pair" {
  let merges : Map[(String, String), Int32] = Map([])
  merges[("a", "b")] = 0
  assert_eq(bpe_merge(["a", "b", "c"], merges), ["ab", "c"])
}

///|
test "encode: known ids for the reference prompt (acceptance)" {
  let tok = Tokenizer::load(@utils.qwen3_model_dir() + "/tokenizer.json")
  // verified against the Rust port and the book's notebook:
  assert_eq(tok.encode("Explain large language models."), [
    840, 20772, 3460, 4128, 4119, 13,
  ])
}
