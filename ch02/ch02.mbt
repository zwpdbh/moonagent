///|
using @qwen3 {type Qwen3Model}

///|
using @torch {type TorchError}

///|
pub(all) suberror Ch02Error {
  InvalidArgumentValue(String)
  TorchError(@torch.TorchError)
} derive(Debug)

///|
/// Human-readable output for Ch02Error.
pub impl Show for Ch02Error with fn output(self, logger) {
  match self {
    InvalidArgumentValue(value) =>
      logger.write_string("Variant message: " + value)
    TorchError(error) => logger.write_string(error.to_string())
  }
}

///|
/// Expose Show methods explicitly.
pub extend Ch02Error with Show::{to_string, output}

///|
/// Expose the derived Debug representation explicitly.
/// Requires derive(Debug) on the type and moonbitlang/core/debug in moon.pkg.
pub extend Ch02Error with @debug.Debug::{to_repr}

///|
fn prepare_native_prompt(
  text : String,
) -> (Array[Int32], @qwen3.Tokenizer) raise Ch02Error {
  let tokenizer = @qwen3.Tokenizer::load(
    "/home/zw/.cache/models/Qwen/Qwen3-0.6B/tokenizer.json",
  ) catch {
    error => raise InvalidArgumentValue(error.to_string())
  }
  let prompt = "<|im_start|>user\n\{text}<|im_end|>\n<|im_start|>assistant\n"
  let input_ids = tokenizer.encode(prompt) catch {
    error => raise InvalidArgumentValue(error.to_string())
  }
  (input_ids, tokenizer)
}

///|
fn native_profile_enabled() -> Bool {
  match @env.get_env_var("MOONXI_LLM_PROFILE") {
    Some(value) => value == "1" || value == "true"
    None => false
  }
}

///|
fn native_max_tokens() -> Int32 raise Ch02Error {
  match @env.get_env_var("MOONXI_MAX_TOKENS") {
    Some(value) =>
      @string.parse_int(value) catch {
        _ => raise InvalidArgumentValue("MOONXI_MAX_TOKENS must be an integer")
      }
    None => 100
  }
}

///|
fn native_use_cuda() -> Bool {
  match @env.get_env_var("MOONXI_DEVICE") {
    Some("cpu") => false
    Some("cuda") => true
    _ => @cuda.device_count() > 0
  }
}

///|
async fn[T : @qwen3.NativeTensor] run_native_basic(
  model : @qwen3.NativeQwen3Model[T],
  input_ids : Array[Int32],
  tokenizer : @qwen3.Tokenizer,
) -> Unit raise Ch02Error {
  async fn print_decoded_token(id : Int32) -> Unit raise Ch02Error {
    let text = tokenizer.decode([id]) catch {
      error => raise InvalidArgumentValue(error.to_string())
    }
    @utils.print_x(text) catch {
      error => raise InvalidArgumentValue(error.to_string())
    }
  }

  let profiling = native_profile_enabled()
  @gt.llm_profiler().reset(enabled=profiling)
  let stream = generate_native_text_basic_stream(
    model~,
    token_ids=input_ids,
    max_new_tokens=native_max_tokens(),
    eos_token_id=Some(151645),
  ) catch {
    error => raise InvalidArgumentValue(error.to_string())
  }
  let started_at = @bench.monotonic_clock_start()
  let generated : Array[Int32] = []
  try {
    while stream.next() is Some(id) {
      generated.push(id)
      print_decoded_token(id)
    }
  } catch {
    error => raise InvalidArgumentValue(error.to_string())
  }
  let elapsed_seconds = @bench.monotonic_clock_end(started_at) / 1_000_000.0
  let tokens_per_second = if elapsed_seconds > 0.0 {
    generated.length().to_double() / elapsed_seconds
  } else {
    0.0
  }
  println(
    "\nGenerated \{generated.length()} tokens in \{elapsed_seconds} seconds (\{tokens_per_second} tokens/sec)",
  )
  if profiling {
    @gt.llm_profiler().print()
  }
}

///|
pub async fn run_ch02_ex01(text : String) -> Unit raise Ch02Error {
  let (input_ids, tokenizer) = prepare_native_prompt(text)
  let model_dir = "/home/zw/.cache/models/Qwen/Qwen3-0.6B"
  if native_use_cuda() {
    println("Using device: cuda")
    @cuda.gpu_init(0)
    let model : @qwen3.NativeQwen3Model[@gt.GpuTensor] = @qwen3.NativeQwen3Model::load(
      model_dir,
    ) catch {
      error => raise InvalidArgumentValue(error.to_string())
    }
    run_native_basic(model, input_ids, tokenizer)
  } else {
    println("Using device: cpu")
    let model : @qwen3.NativeQwen3Model[@nparray.NpArray] = @qwen3.NativeQwen3Model::load(
      model_dir,
    ) catch {
      error => raise InvalidArgumentValue(error.to_string())
    }
    run_native_basic(model, input_ids, tokenizer)
  }
}

///|
fn[T : @qwen3.NativeTensor] run_native_cached(
  model : @qwen3.NativeQwen3Model[T],
  input_ids : Array[Int32],
  tokenizer : @qwen3.Tokenizer,
) -> Unit raise Ch02Error {
  let profiling = native_profile_enabled()
  @gt.llm_profiler().reset(enabled=profiling)
  let started_at = @bench.monotonic_clock_start()
  let generated = generate_native_text_basic_stream_cache(
    model~,
    token_ids=input_ids,
    max_new_tokens=native_max_tokens(),
    eos_token_id=Some(151645),
  ).collect() catch {
    error => raise InvalidArgumentValue(error.to_string())
  }
  let elapsed_seconds = @bench.monotonic_clock_end(started_at) / 1_000_000.0
  let output = tokenizer.decode(generated) catch {
    error => raise InvalidArgumentValue(error.to_string())
  }
  println(output)
  let tokens_per_second = if elapsed_seconds > 0.0 {
    generated.length().to_double() / elapsed_seconds
  } else {
    0.0
  }
  println(
    "\nGenerated \{generated.length()} tokens in \{elapsed_seconds} seconds (\{tokens_per_second} tokens/sec)",
  )
  if profiling {
    @gt.llm_profiler().print()
  }
}

///|
pub async fn run_ch02_ex02(text : String) -> Unit raise Ch02Error {
  let (input_ids, tokenizer) = prepare_native_prompt(text)
  let model_dir = "/home/zw/.cache/models/Qwen/Qwen3-0.6B"
  if native_use_cuda() {
    println("Using device: cuda")
    @cuda.gpu_init(0)
    let model : @qwen3.NativeQwen3Model[@gt.GpuTensor] = @qwen3.NativeQwen3Model::load(
      model_dir,
    ) catch {
      error => raise InvalidArgumentValue(error.to_string())
    }
    run_native_cached(model, input_ids, tokenizer)
  } else {
    println("Using device: cpu")
    let model : @qwen3.NativeQwen3Model[@nparray.NpArray] = @qwen3.NativeQwen3Model::load(
      model_dir,
    ) catch {
      error => raise InvalidArgumentValue(error.to_string())
    }
    run_native_cached(model, input_ids, tokenizer)
  }
}

///|
pub fn generate_text_basic_stream(
  model~ : Qwen3Model,
  token_ids~ : Array[Int64],
  max_new_tokens~ : Int32,
  eos_token_id~ : Int64?,
  device~ : String,
) -> @utils.LazyGenerator[Int64, TorchError] raise TorchError {
  if token_ids.length() == 0 {
    raise InvalidShape("generation requires at least one input token")
  }
  if max_new_tokens < 0 {
    raise InvalidShape("max_new_tokens must not be negative")
  }

  let mut input = @torch.int_tensor(data=token_ids, shape=[
    1,
    token_ids.length(),
  ]).to(device~)

  let mut remaining = max_new_tokens

  fn next_token() -> Result[Int64?, TorchError] {
    if remaining == 0 {
      return Ok(None)
    }
    @torch.without_grad(() => {
      let logits = @torch.select(
        input=model.forward(input~, cache=None),
        dim=1,
        index=-1,
      )
      let next_token = @torch.argmax(input=logits, dim=-1)

      let next_id = next_token.item().to_int64()
      if eos_token_id is Some(eos_id) && next_id == eos_id {
        // Keep the stream exhausted if next() is called again after EOS.
        remaining = 0
        return Ok(None)
      }

      input = @torch.concat([input, next_token.reshape([1, 1])], dim=1)
      remaining -= 1
      Ok(Some(next_id))
    }) catch {
      error => Err(error)
    }
  }
  @utils.lazy_generator(next_token)
}

///|
/// Generate greedily with Qwen3's KV cache, emitting each token immediately.
pub fn generate_text_basic_stream_cache(
  model~ : Qwen3Model,
  token_ids~ : Array[Int64],
  max_new_tokens~ : Int32,
  eos_token_id~ : Int64?,
  device~ : String,
  on_token~ : (Int64) -> Unit raise TorchError,
) -> Array[Int64] raise TorchError {
  if token_ids.length() == 0 {
    raise InvalidShape("generation requires at least one input token")
  }
  if max_new_tokens < 0 {
    raise InvalidShape("max_new_tokens must not be negative")
  }

  @torch.without_grad(() => {
    let cache = model.new_cache()
    model.reset_kv_cache()
    let input = @torch.int_tensor(data=token_ids, shape=[1, token_ids.length()]).to(
      device~,
    )
    let mut logits = @torch.select(
      input=model.forward(input~, cache=Some(cache)),
      dim=1,
      index=-1,
    )
    let generated : Array[Int64] = []

    for step in 0..<max_new_tokens {
      let next_token = @torch.argmax(input=logits, dim=-1)
      let next_id = next_token.item().to_int64()
      if eos_token_id is Some(eos_id) && next_id == eos_id {
        break
      }

      generated.push(next_id)
      on_token(next_id)
      if step + 1 < max_new_tokens {
        let next_input = next_token.reshape([1, 1])
        logits = @torch.select(
          input=model.forward(input=next_input, cache=Some(cache)),
          dim=1,
          index=-1,
        )
      }
    }
    generated
  })
}

///|
fn[T : @qwen3.NativeTensor] native_next_id(logits : T) -> Int32 {
  let sequence_length = logits.shape()[1]
  logits.select(dim=1, index=sequence_length - 1).argmax_last().item_int()
}

///|
/// Generate greedily with native tensors and no KV cache.
pub fn[T : @qwen3.NativeTensor] generate_native_text_basic_stream(
  model~ : @qwen3.NativeQwen3Model[T],
  token_ids~ : Array[Int32],
  max_new_tokens~ : Int32,
  eos_token_id~ : Int32?,
) -> @utils.LazyGenerator[Int32, @qwen3.NativeQwen3Error] raise @qwen3.NativeQwen3Error {
  if token_ids.is_empty() {
    raise InvalidModel("generation requires at least one input token")
  }
  if max_new_tokens < 0 {
    raise InvalidModel("max_new_tokens must not be negative")
  }

  let input = token_ids.copy()
  let mut remaining = max_new_tokens
  fn next_token() -> Result[Int32?, @qwen3.NativeQwen3Error] {
    if remaining == 0 {
      return Ok(None)
    }
    let result = try {
      let next_id = native_next_id(model.forward(input~, cache=None))
      if eos_token_id is Some(eos_id) && next_id == eos_id {
        remaining = 0
        return Ok(None)
      }
      input.push(next_id)
      remaining -= 1
      Ok(Some(next_id))
    } catch {
      error => Err(error)
    }
    result
  }
  @utils.lazy_generator(next_token)
}

///|
/// Generate greedily with native tensors and a KV cache.
pub fn[T : @qwen3.NativeTensor] generate_native_text_basic_stream_cache(
  model~ : @qwen3.NativeQwen3Model[T],
  token_ids~ : Array[Int32],
  max_new_tokens~ : Int32,
  eos_token_id~ : Int32?,
) -> @utils.LazyGenerator[Int32, @qwen3.NativeQwen3Error] raise @qwen3.NativeQwen3Error {
  if token_ids.is_empty() {
    raise InvalidModel("generation requires at least one input token")
  }
  if max_new_tokens < 0 {
    raise InvalidModel("max_new_tokens must not be negative")
  }

  let cache = model.new_cache()
  let mut next_input = Some(token_ids)
  let mut logits : T? = None
  let mut remaining = max_new_tokens
  let mut initialized = false
  fn next_token() -> Result[Int32?, @qwen3.NativeQwen3Error] {
    if remaining == 0 {
      return Ok(None)
    }
    let result = try {
      if !initialized {
        model.reset_kv_cache()
        initialized = true
      }
      if next_input is Some(input) {
        logits = Some(model.forward(input~, cache=Some(cache)))
        next_input = None
      }
      guard logits is Some(current_logits) else {
        raise @qwen3.NativeQwen3Error::InvalidModel(
          "cached generation has no logits",
        )
      }
      let next_id = native_next_id(current_logits)
      if eos_token_id is Some(eos_id) && next_id == eos_id {
        remaining = 0
        return Ok(None)
      }
      next_input = Some([next_id])
      remaining -= 1
      Ok(Some(next_id))
    } catch {
      error => Err(error)
    }
    result
  }
  @utils.lazy_generator(next_token)
}
