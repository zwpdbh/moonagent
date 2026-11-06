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
/// Load the MoonBit Qwen3 implementation from an existing model cache.
fn load_cached_qwen3(
  model_dir~ : String,
  device~ : String,
) -> Qwen3Model raise TorchError {
  @qwen3.Qwen3Model::load(model_dir, device~)
}

///|
fn prepare_common(
  text : String,
) -> (@qwen3.Qwen3Model, Array[Int64], String, @qwen3.Tokenizer) raise Ch02Error {
  let model_dir = "/home/zw/.cache/models/Qwen/Qwen3-0.6B"
  @torch.add_python_path(
    "/home/zw/code/moonbit_programming/moonagent/.venv/lib/python3.12/site-packages",
  ) catch {
    error => raise InvalidArgumentValue(error.to_string())
  }
  let device = @torch.cuda_is_available() catch {
    error => raise InvalidArgumentValue(error.to_string())
  }
  let device = if device { "cuda" } else { "cpu" }
  println("Using device: \{device}")
  let model = load_cached_qwen3(model_dir~, device~) catch {
    error => raise InvalidArgumentValue(error.to_string())
  }
  let tokenizer = @qwen3.Tokenizer::load(model_dir + "/tokenizer.json") catch {
    _ => raise InvalidArgumentValue("failed to load tokenizer")
  }
  let prompt = "<|im_start|>user\n\{text}<|im_end|>\n<|im_start|>assistant\n"
  let input_ids = tokenizer.encode(prompt).map(id => id.to_int64()) catch {
      _ => raise InvalidArgumentValue("failed to encode prompt")
    }
  (model, input_ids, device, tokenizer)
}

///|
pub fn run_ch02_ex01(text : String) -> Unit raise Ch02Error {
  let (model, input_ids, device, tokenizer) = prepare_common(text)

  fn print_decoded_token(id : Int64) -> Unit raise @torch.TorchError {
    let text = tokenizer.decode([id.to_int()]) catch {
      _ => raise UnexpectedType("failed to decode token")
    }
    @utils.print_x(text) catch {
      _ => raise UnexpectedType("failed to print token")
    }
  }

  let stream = generate_text_basic_stream(
    model~,
    token_ids=input_ids,
    max_new_tokens=100,
    eos_token_id=Some(151645L),
    device~,
  ) catch {
    error => raise InvalidArgumentValue(error.to_string())
  }
  let started_at = @bench.monotonic_clock_start()

  let generated : Array[Int64] = []
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
}

///|
pub fn run_ch02_ex02(text : String) -> Unit raise Ch02Error {
  let (model, input_ids, device, tokenizer) = prepare_common(text)
  let generated = generate_text_basic_stream_cache(
    model~,
    token_ids=input_ids,
    max_new_tokens=100,
    eos_token_id=Some(151645L),
    device~,
    on_token=id => ignore(id),
  ) catch {
    error => raise InvalidArgumentValue(error.to_string())
  }
  let output = tokenizer.decode(generated.map(id => id.to_int())) catch {
    _ => raise InvalidArgumentValue("failed to decode output")
  }
  println(output)
  println("\nGenerated \{generated.length()} tokens")
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
fn native_next_id(logits : @nparray.NpArray) -> Int32 {
  let sequence_length = logits.shape.shape[1]
  logits.select(dim=1, index=sequence_length - 1).argmax_last().data[0].to_int()
}

///|
/// Generate greedily with native CPU tensors and no KV cache.
pub fn generate_native_text_basic_stream(
  model~ : @qwen3.NativeQwen3Model,
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
/// Generate greedily with native CPU tensors and a KV cache.
pub fn generate_native_text_basic_stream_cache(
  model~ : @qwen3.NativeQwen3Model,
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
  let mut logits : @nparray.NpArray? = None
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
