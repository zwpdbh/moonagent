///|
using @qwen3 {type Qwen3Model}

///|
using @torch {type TorchError}

///|
/// Load the MoonBit Qwen3 implementation from an existing model cache.
pub fn load_cached_qwen3(
  model_dir~ : String,
  device~ : String,
) -> Qwen3Model raise TorchError {
  @qwen3.Qwen3Model::load(model_dir, device~)
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
