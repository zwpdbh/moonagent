// Whitebox tests run within the package scope.
// Use them to validate internal helpers or invariants.
// Keep them focused; public behavior belongs in `_test.mbt`.

///|
test "parse_text returns ordinary input when no added tokens are configured" {
  let tok : Tokenizer = {
    vocab: Map([]),
    ids: Map([]),
    added_ids: Map([]),
    added_vocabs: Map([]),
    merges: Map([]),
    pre_tokenizer: Sequence([]),
  }
  let parts = tok.parse_text("plain 世界")
  assert_eq(parts.length(), 1)
  assert_true(parts[0] is Ordinay("plain 世界"))
  assert_eq(tok.parse_text("").length(), 0)
}

///|
test "parse_text isolates adjacent added tokens and preserves ordinary text" {
  let tok = Tokenizer::load(@utils.qwen3_model_dir() + "/tokenizer.json")
  let parts = tok.parse_text("hello<|im_start|><|im_end|> 世界")
  assert_eq(parts.length(), 4)
  assert_true(parts[0] is Ordinay("hello"))
  assert_true(parts[1] is Added("<|im_start|>"))
  assert_true(parts[2] is Added("<|im_end|>"))
  assert_true(parts[3] is Ordinay(" 世界"))
}

///|
test "parse_text prefers the longest added token at the same position" {
  let added_vocabs : Map[String, Int32] = Map([])
  added_vocabs.set("<|a|>", 0)
  added_vocabs.set("<|a|>x", 1)
  let tok : Tokenizer = {
    vocab: Map([]),
    ids: Map([]),
    added_ids: Map([]),
    added_vocabs,
    merges: Map([]),
    pre_tokenizer: Sequence([]),
  }
  let parts = tok.parse_text("<|a|>x<|a|>")
  assert_eq(parts.length(), 2)
  assert_true(parts[0] is Added("<|a|>x"))
  assert_true(parts[1] is Added("<|a|>"))
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
  let file : QwenTokenizerJsonFile = @json.from_json(j)
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
test "split_string splits words contractions numbers and punctuation" {
  let tok = Tokenizer::load(@utils.qwen3_model_dir() + "/tokenizer.json")
  assert_eq(tok.split_string(""), [])
  assert_eq(tok.split_string("Explain large language models."), [
    "Explain", " large", " language", " models", ".",
  ])
  assert_eq(tok.split_string("I'll"), ["I", "'ll"])
  assert_eq(tok.split_string("I'M"), ["I", "'M"])
  assert_eq(tok.split_string("we're"), ["we", "'re"])
  assert_eq(tok.split_string("they've"), ["they", "'ve"])
  assert_eq(tok.split_string("he'd"), ["he", "'d"])
  assert_eq(tok.split_string("I'll I'M we're they've he'd"), [
    "I", "'ll", " I", "'M", " we", "'re", " they", "'ve", " he", "'d",
  ])
  assert_eq(tok.split_string("123 !?..."), ["1", "2", "3", " !?..."])
}

///|
test "split_string preserves Qwen whitespace and newline boundaries" {
  let tok = Tokenizer::load(@utils.qwen3_model_dir() + "/tokenizer.json")
  assert_eq(tok.split_string("  hello"), [" ", " hello"])
  assert_eq(tok.split_string("   !"), ["  ", " !"])
  assert_eq(tok.split_string("hello  "), ["hello", "  "])
  assert_eq(tok.split_string("Hello...\n\nWorld"), ["Hello", "...\n\n", "World"])
  assert_eq(tok.split_string("a \n  b"), ["a", " \n", " ", " b"])
  assert_eq(tok.split_string(" \tfoo"), [" ", "\tfoo"])
}

///|
test "split_string handles Unicode letters numbers and symbols" {
  let tok = Tokenizer::load(@utils.qwen3_model_dir() + "/tokenizer.json")
  assert_eq(tok.split_string("café 中文😀"), ["café", " 中文", "😀"])
  assert_eq(tok.split_string("٤٥"), ["٤", "٥"])
}

///|
test "create_pairs" {
  let paris = build_pairs([1, 2, 3])
  assert_eq([(1, 2), (2, 3)], paris)

  let paris = build_pairs([1, 2, 3, 4])
  assert_eq([(1, 2), (2, 3), (3, 4)], paris)

  let pairs : Array[(Int32, Int32)] = build_pairs([])
  assert_eq([], pairs)

  let pairs = build_pairs([1, 2])
  assert_eq([(1, 2)], pairs)
}

///|
test "bpe_merge returns unchanged symbols when no pair has a rule" {
  let merges : Map[(String, String), Int32] = Map([])
  assert_eq(bpe_merge(["a", "b", "c"], merges), ["a", "b", "c"])
}

///|
test "bpe_merge chooses the lowest rank instead of the leftmost pair" {
  let merges : Map[(String, String), Int32] = Map([])
  merges.set(("a", "b"), 10)
  merges.set(("b", "c"), 1)
  assert_eq(bpe_merge(["a", "b", "c"], merges), ["a", "bc"])
}

///|
test "bpe_merge skips pairs that have no rule" {
  let merges : Map[(String, String), Int32] = Map([])
  merges.set(("a", "b"), 0)
  assert_eq(bpe_merge(["x", "a", "b"], merges), ["x", "ab"])
}

///|
test "bpe_merge applies newly formed pairs and stops at one symbol" {
  let merges : Map[(String, String), Int32] = Map([])
  merges.set(("a", "b"), 0)
  merges.set(("ab", "c"), 1)
  assert_eq(bpe_merge(["a", "b", "c"], merges), ["abc"])
}

///|
test "bpe_merge merges non-overlapping occurrences of the selected pair" {
  let merges : Map[(String, String), Int32] = Map([])
  merges.set(("a", "a"), 0)
  assert_eq(bpe_merge(["a", "a", "a", "a"], merges), ["aa", "aa"])
  assert_eq(bpe_merge(["a", "a", "a"], merges), ["aa", "a"])
}
