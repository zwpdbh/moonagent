/// The BPE algorithm from https://sebastianraschka.com/blog/2025/bpe-from-scratch.htm
/// And the tokenizer.json file downloaded with Qwen3 model.
/// Q: what is "train a BPE tokenizer"
/// A: It means running the BPE learning algorithm over a large text to discover useful
/// subwords and save them as:
/// A vocabular
/// An order list of merge rules: which two symbols can be combined and their priority.
/// Q: Is tokenizer.json the result from trainning?
/// A: It is. Our job is to load the tokenizer and implement encode and decode using its saved rules.
/// tokenizer.json
/// ├── model.vocab        ← learned token-to-ID vocabulary
/// ├── model.merges       ← learned, ordered BPE merge rules
/// ├── normalizer         ← how to normalize input text
/// ├── pre_tokenizer      ← how to split text before applying BPE
/// ├── added_tokens       ← special/control tokens such as <|im_start|>
/// ├── post_processor     ← additional processing configuration
/// └── decoder            ← how to convert tokens back into text

///|
pub(all) suberror TokenizerError {
  InvalidFormat(String)
  UnknownToken(Int32)
  InvalidModel(String)
}

///|
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
  let vocab : Map[String, Int32] = @utils.extract(obj, "vocab", path)
  let merges : Array[Array[String]] = @utils.extract(obj, "merges", path)
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
/// Rules for splitting text before BPE.
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
    "Sequence" => Sequence(@utils.extract(obj, "pretokenizers", path))
    "Split" => {
      // pattern is nested: {"Regex": "..."}
      let pattern_map : Map[String, String] = @utils.extract(
        obj, "pattern", path,
      )
      guard pattern_map.get("Regex") is Some(regex) else {
        raise @json.JsonDecodeError(
          (path.add_key("pattern"), "Split: expected {\"Regex\": ...}"),
        )
      }
      Split(
        pattern=regex,
        behavior=@utils.extract(obj, "behavior", path),
        invert=@utils.extract(obj, "invert", path),
      )
    }
    "ByteLevel" =>
      ByteLevel(
        add_prefix_space=@utils.extract(obj, "add_prefix_space", path),
        trim_offsets=@utils.extract(obj, "trim_offsets", path),
        use_regex=@utils.extract(obj, "use_regex", path),
      )
    _ =>
      raise @json.JsonDecodeError(
        (path, "PreTokenizer: unknown type \"\{kind}\""),
      )
  }
}

///|
priv struct QwenTokenizerJsonFile {
  added_tokens : Array[SpecialControlToken]
  model : QwenTokenizerModel
  pre_tokenizer : PreTokenizer
} derive(FromJson)

///|
struct Tokenizer {
  // Ordinary BPE tokens
  vocab : Map[String, Int32]
  // reverse lookup of vocab
  ids : Map[Int32, String]
  // Added tokens only: ID -> literal content.
  added_ids : Map[Int32, String]
  // reverse lookup for added_ids
  added_vocabs : Map[String, Int32]
  // 151,387 ordered BPE rules. (left symbol, right symbol) -> merge rank.
  merges : Map[(String, String), Int32]
}

///|
priv enum TextString {
  Ordinay(String)
  Added(String)
}

///|
fn Tokenizer::parse_text(self : Self, text : String) -> Array[TextString] {
  let tokens = self.added_vocabs.keys().to_array()
  tokens.sort_by_key(token => -token.length())
  let regex : @string.Regex? = tokens.fold(init=None, (acc, token) => {
    let literal = @string.Regex::string(token)
    match acc {
      None => Some(literal)
      Some(current) => Some(current | literal)
    }
  })
  let result : Array[TextString] = []
  guard regex is Some(regex) else {
    if text.length() > 0 {
      result.push(Ordinay(text))
    }
    return result
  }
  let mut remaining = text[:]
  while true {
    match regex.execute(remaining) {
      None => {
        if remaining.length() > 0 {
          result.push(Ordinay(remaining.to_owned()))
        }
        break
      }
      Some(matched) => {
        let ordinary = matched.before()
        if ordinary.length() > 0 {
          result.push(Ordinay(ordinary.to_owned()))
        }
        result.push(Added(matched.content().to_owned()))
        remaining = matched.after()
      }
    }
  }
  result
}

///|
/// Build a Tokenizer from the *contents* of tokenizer.json. Pure: no I/O.
pub fn Tokenizer::new(
  tokenizer_json : String,
) -> Tokenizer raise TokenizerError {
  let parsed = @json.parse(tokenizer_json) catch {
    e => raise InvalidFormat(e.to_string())
  }
  let qwen_tokenizer_def : QwenTokenizerJsonFile = @json.from_json(parsed) catch {
    e => raise InvalidFormat(e.to_string())
  }

  guard qwen_tokenizer_def.model.model_type == "BPE" else {
    raise InvalidModel(
      "expected BPE tokenzier, got \{qwen_tokenizer_def.model.model_type} }",
    )
  }

  let vocab = qwen_tokenizer_def.model.vocab
  let added_ids : Map[Int32, String] = Map([])
  let added_vocabs : Map[String, Int32] = Map([])
  // merge those special tokens
  for each_special_token in qwen_tokenizer_def.added_tokens {
    vocab.set(each_special_token.content, each_special_token.id)
    added_vocabs.set(each_special_token.content, each_special_token.id)
    added_ids.set(each_special_token.id, each_special_token.content)
  }

  // build revers one
  let ids : Map[Int32, String] = Map([])
  for token, id in vocab {
    if ids.contains(id) {
      raise InvalidFormat("duplicated id \{id} for token: \{token}")
    }
    ids[id] = token
  }

  let merges = qwen_tokenizer_def.model.merges |> parse_merges

  { vocab, ids, added_ids, added_vocabs, merges, }
}

///|
/// Parse the "merges" array from tokenizer.json's model section into
/// pair -> rank (rank = position in the array, lower merges first).
/// In this file merges are JSON pairs: [["Ġ", "t"], ["i", "n"], ...].
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
/// A: We don't need to build the vocabulary, it is created/trained from Qwen model, the tokenizer.json file contains the vocabs.
/// Q: what is the Tokenizer we created from qwen's tokenizer.json file? how it is related with our code.
/// The tokenizer contains result of  the BPE algorithm worked on the text. Our code don't need to implement BPE, we just use the rules and vocabs to do encode and decode.
/// Q: From the example: "the cat in the hat" in https://sebastianraschka.com/blog/2025/bpe-from-scratch.html#14-bpe-algorithm-example
/// "the" appears twice, so we need to split the "the cat in the hat" as our first step. How this split happened? 
/// Q: Do we use some help from the tokenizer.json or we do the split purely by our own?
/// === Following are general steps:
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
pub fn Tokenizer::decode(
  self : Tokenizer,
  ids : Array[Int32],
) -> String raise TokenizerError {
  ignore(self)
  ignore(ids)
  abort("TODO: byte-alphabet mapping + utf8 decode")
}
