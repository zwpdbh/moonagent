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
async test "read_tensor_entries_from_safetensors parses the real header" {
  let header = read_safetensors_header(
    "/home/zw/.cache/models/Qwen/Qwen3-0.6B/model.safetensors",
  )
  assert_eq(header.entries.length(), 311)
  guard header.entries.get("model.embed_tokens.weight") is Some(entry) else {
    abort("model.embed_tokens.weight should exist")
  }
  assert_true(entry.dtype is BF16)
  assert_eq(entry.shape, [151936, 1024])
  assert_eq(entry.data_offsets.start, 311164928)
  assert_eq(entry.data_offsets.end, 622329856)
}

///|
test "TensorOffsets parses a two-element JSON array" {
  let parsed : TensorOffsets = @json.from_json(
    @json.parse("[311164928, 622329856]"),
  )
  assert_eq(parsed.start, 311164928)
  assert_eq(parsed.end, 622329856)
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
