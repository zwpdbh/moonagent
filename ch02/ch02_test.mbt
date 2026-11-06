///|
test "basic stream generates the expected prefix" {
  let model_dir = "/home/zw/.cache/models/Qwen/Qwen3-0.6B"
  @torch.add_python_path(
    "/home/zw/code/moonbit_programming/moonagent/.venv/lib/python3.12/site-packages",
  )
  let device = if @torch.cuda_is_available() { "cuda" } else { "cpu" }
  let model = @qwen3.Qwen3Model::load(model_dir, device~)
  let tokenizer = @qwen3.Tokenizer::load(model_dir + "/tokenizer.json") catch {
    _ => abort("failed to load tokenizer")
  }
  let prompt = "<|im_start|>user\nExplain large language models.<|im_end|>\n<|im_start|>assistant\n"
  let input_ids = tokenizer.encode(prompt).map(id => id.to_int64()) catch {
      _ => abort("failed to encode prompt")
    }
  let stream = @ch02.generate_text_basic_stream(
    model~,
    token_ids=input_ids,
    max_new_tokens=4,
    eos_token_id=None,
    device~,
  )
  let generated = stream.collect()
  let text = tokenizer.decode(generated.map(id => id.to_int())) catch {
    _ => abort("failed to decode generated tokens")
  }
  assert_eq(text, "<think>\nOkay,")
}

///|
async test "native basic and cached streams generate the same prefix" {
  let model_dir = "/home/zw/.cache/models/Qwen/Qwen3-0.6B"
  let model : @qwen3.NativeQwen3Model[@nparray.NpArray] = @qwen3.NativeQwen3Model::load(
    model_dir,
  )
  let tokenizer = @qwen3.Tokenizer::load(model_dir + "/tokenizer.json") catch {
    _ => abort("failed to load tokenizer")
  }
  let prompt = "<|im_start|>user\nExplain large language models.<|im_end|>\n<|im_start|>assistant\n"
  let input_ids = tokenizer.encode(prompt) catch {
    _ => abort("failed to encode prompt")
  }
  let basic = @ch02.generate_native_text_basic_stream(
    model~,
    token_ids=input_ids,
    max_new_tokens=4,
    eos_token_id=None,
  ).collect()
  let cached = @ch02.generate_native_text_basic_stream_cache(
    model~,
    token_ids=input_ids,
    max_new_tokens=4,
    eos_token_id=None,
  ).collect()
  assert_eq(basic, cached)
  let text = tokenizer.decode(cached) catch {
    _ => abort("failed to decode generated tokens")
  }
  assert_eq(text, "<think>\nOkay,")
}

///|
async test "native CUDA basic and cached streams generate the expected prefix" {
  let model_dir = "/home/zw/.cache/models/Qwen/Qwen3-0.6B"
  @cuda.gpu_init(0)
  let model : @qwen3.NativeQwen3Model[@gt.GpuTensor] = @qwen3.NativeQwen3Model::load(
    model_dir,
  )
  let tokenizer = @qwen3.Tokenizer::load(model_dir + "/tokenizer.json") catch {
    _ => abort("failed to load tokenizer")
  }
  let prompt = "<|im_start|>user\nExplain large language models.<|im_end|>\n<|im_start|>assistant\n"
  let input_ids = tokenizer.encode(prompt) catch {
    _ => abort("failed to encode prompt")
  }
  let basic = @ch02.generate_native_text_basic_stream(
    model~,
    token_ids=input_ids,
    max_new_tokens=4,
    eos_token_id=None,
  ).collect()
  let generated = @ch02.generate_native_text_basic_stream_cache(
    model~,
    token_ids=input_ids,
    max_new_tokens=4,
    eos_token_id=None,
  ).collect()
  assert_eq(basic, generated)
  let text = tokenizer.decode(generated) catch {
    _ => abort("failed to decode generated tokens")
  }
  assert_eq(text, "<think>\nOkay,")
}
