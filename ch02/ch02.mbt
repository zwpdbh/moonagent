///|
using @qwen3 {type Qwen3Model}

///|
using @torch {type TorchError}

///|
pub(all) suberror Ch02Error {
  InvalidArgumentValue(String)
} derive(Debug)

///|
/// Human-readable output for Ch02Error.
pub impl Show for Ch02Error with fn output(self, logger) {
  match self {
    InvalidArgumentValue(value) =>
      logger.write_string("Variant message: " + value)
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
  let generated = generate_text_basic_stream(
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
  on_token~ : (Int64) -> Unit raise TorchError,
) -> Array[Int64] raise TorchError {
  if token_ids.length() == 0 {
    raise InvalidShape("generation requires at least one input token")
  }
  if max_new_tokens < 0 {
    raise InvalidShape("max_new_tokens must not be negative")
  }
  ignore(model)
  ignore(eos_token_id)
  ignore(on_token)
  ignore(
    @torch.int_tensor(data=token_ids, shape=[1, token_ids.length()]).to(device~),
  )

  abort("TODO")
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
