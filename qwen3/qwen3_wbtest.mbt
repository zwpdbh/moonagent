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
  let parts = tok.split_text_into_token_string("plain 世界")
  assert_eq(parts.length(), 1)
  assert_true(parts[0] is Ordinay("plain 世界"))
  assert_eq(tok.split_text_into_token_string("").length(), 0)
}

///|
test "real tokenizer integration: parse, split, encode, and decode" {
  let tok = Tokenizer::load(@utils.qwen3_model_dir() + "/tokenizer.json")
  assert_eq(tok.vocab.length(), 151669) // 151643 vocab + 26 added
  assert_eq(tok.ids[151645], "<|im_end|>")
  let parts = tok.split_text_into_token_string(
    "hello<|im_start|><|im_end|> 世界",
  )
  assert_eq(parts.length(), 4)
  assert_true(parts[0] is Ordinay("hello"))
  assert_true(parts[1] is Added("<|im_start|>"))
  assert_true(parts[2] is Added("<|im_end|>"))
  assert_true(parts[3] is Ordinay(" 世界"))
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
  assert_eq(tok.split_string("  hello"), [" ", " hello"])
  assert_eq(tok.split_string("   !"), ["  ", " !"])
  assert_eq(tok.split_string("hello  "), ["hello", "  "])
  assert_eq(tok.split_string("Hello...\n\nWorld"), ["Hello", "...\n\n", "World"])
  assert_eq(tok.split_string("a \n  b"), ["a", " \n", " ", " b"])
  assert_eq(tok.split_string(" \tfoo"), [" ", "\tfoo"])
  assert_eq(tok.split_string("café 中文😀"), ["café", " 中文", "😀"])
  assert_eq(tok.split_string("٤٥"), ["٤", "٥"])
  assert_eq(tok.encode(""), [])
  assert_eq(tok.encode("Explain large language models."), [
    840, 20772, 3460, 4128, 4119, 13,
  ])
  assert_eq(tok.encode("hello world"), [14990, 1879])
  assert_eq(tok.encode("  hello"), [220, 23811])
  assert_eq(tok.encode("hello<|im_end|>"), [14990, 151645])
  assert_eq(tok.encode("<|im_start|>hello<|im_end|>"), [151644, 14990, 151645])
  assert_eq(tok.encode("café"), [924, 58858])
  assert_eq(tok.encode("中文"), [104811])
  assert_eq(tok.encode("😀"), [141334])
  assert_eq(tok.decode([]), "")
  assert_eq(tok.decode([151644]), "<|im_start|>")
  assert_eq(
    tok.decode([840, 20772, 3460, 4128, 4119, 13]),
    "Explain large language models.",
  )
  assert_eq(tok.decode([14990, 1879]), "hello world")
  assert_eq(tok.decode([220, 23811]), "  hello")
  assert_eq(tok.decode([14990, 151645]), "hello<|im_end|>")
  assert_eq(tok.decode([151644, 14990, 151645]), "<|im_start|>hello<|im_end|>")
  assert_eq(tok.decode([924, 58858]), "café")
  assert_eq(tok.decode([104811]), "中文")
  assert_eq(tok.decode([141334]), "😀")
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
  let parts = tok.split_text_into_token_string("<|a|>x<|a|>")
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

///|
test "DType parses bfloat16" {
  let parsed : DType = @json.from_json(@json.parse("\"bfloat16\""))
  assert_true(parsed is BF16)
}

///|
test "DType rejects unsupported dtype strings" {
  let raised = try {
    let parsed : DType = @json.from_json(@json.parse("\"float32\""))
    ignore(parsed)
    false
  } catch {
    _ => true
  }
  assert_true(raised)
}

///|
test "Config parses the real Qwen3 config.json" {
  let content = @fs.read_file_to_string(
    @utils.qwen3_model_dir() + "/config.json",
  )
  let config : Config = @json.from_json(@json.parse(content))
  assert_eq(config.architectures, ["Qwen3ForCausalLM"])
  assert_eq(config.hidden_size, 1024)
  assert_eq(config.num_hidden_layers, 28)
  assert_eq(config.num_attention_heads, 16)
  assert_eq(config.num_key_value_heads, 8)
  assert_eq(config.head_dim, 128)
  assert_eq(config.intermediate_size, 3072)
  assert_eq(config.vocab_size, 151936)
  assert_eq(config.rope_theta, 1000000)
  assert_true(config.rope_scaling is None)
  assert_true(config.sliding_window is None)
  assert_true(config.tie_word_embeddings)
  assert_true(config.torch_dtype is BF16)
}

///|
async test "read_tensor_entries_from_safetensors parses the real header" {
  let header = read_safetensors_header(
    @utils.qwen3_model_dir() + "/model.safetensors",
  )
  assert_eq(header.entries.length(), 311)
  guard header.entries.get("model.embed_tokens.weight") is Some(entry) else {
    abort("model.embed_tokens.weight should exist")
  }
  assert_true(entry.dtype is BF16)
  assert_eq(entry.shape, [151936, 1024])
  assert_eq(entry.data_offsets, TensorOffsets::{
    start: 311164928,
    end: 622329856,
  })
}

///|
test "TensorOffsets parses a two-element JSON array" {
  let parsed : TensorOffsets = @json.from_json(
    @json.parse("[311164928, 622329856]"),
  )
  assert_eq(parsed, TensorOffsets::{ start: 311164928, end: 622329856, })
}

///|
test "TensorOffsets rejects non-array and wrong-length inputs" {
  for input in ["{}", "[1]", "[1, 2, 3]"] {
    let raised = try {
      let parsed : TensorOffsets = @json.from_json(@json.parse(input))
      ignore(parsed)
      false
    } catch {
      _ => true
    }
    assert_true(raised)
  }
}

///|
test "rms_norm normalizes input and applies learned weights" {
  let result = rms_norm([3.0, 4.0], [1.0, 0.5], 0.0)
  assert_true((result[0] - 0.84852814).abs() < 0.000001)
  assert_true((result[1] - 0.56568542).abs() < 0.000001)
}

///|
test "rms_norm handles zero input with epsilon" {
  assert_eq(rms_norm([0.0, 0.0], [1.0, 2.0], 0.000001), [0.0, 0.0])
}

///|
test "linear computes vector-matrix dot products" {
  let result = linear([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0, 1.0, 1.0], [2, 3])
  assert_eq(result, [1.0, 5.0])
}

///|
test "linear returns one value per output row" {
  let result = linear([2.0, -1.0], [1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2])
  assert_eq(result, [2.0, -1.0, 1.0])
}

///|
test "softmax converts equal scores to equal weights" {
  assert_eq(softmax([0.0, 0.0]), [0.5, 0.5])
}

///|
test "softmax produces normalized weights for large scores" {
  let result = softmax([10000.0, 10001.0])
  assert_true((result[0] - 0.26894143).abs() < 0.000001)
  assert_true((result[1] - 0.7310586).abs() < 0.000001)
  let sum = result.fold(init=0.0F, (acc, value) => acc + value)
  assert_true((sum - 1.0).abs() < 0.000001)
}

///|
test "panic softmax rejects empty input" {
  ignore(softmax([]))
}

///|
test "attention with one position returns that position's value" {
  let result = scaled_dot_product_attention(
    [1.0, 0.0],
    [2.0, 3.0],
    [4.0, 5.0],
    1,
    2,
  )
  assert_eq(result, [4.0, 5.0])
}

///|
test "attention with equal scores averages value vectors" {
  let result = scaled_dot_product_attention(
    [0.0, 0.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    2,
    2,
  )
  assert_eq(result, [2.0, 3.0])
}

///|
test "attention applies non-uniform weights to values" {
  let result = scaled_dot_product_attention([1.0], [0.0, 1.0], [2.0, 4.0], 2, 1)
  assert_true((result[0] - 3.4621172).abs() < 0.000001)
}

///|
test "panic attention rejects inconsistent cache lengths" {
  ignore(scaled_dot_product_attention([1.0], [1.0], [1.0, 2.0], 1, 1))
}
