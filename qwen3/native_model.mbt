///|

///|
pub(all) suberror NativeQwen3Error {
  InvalidModel(String)
} derive(Debug)

///|
pub impl Show for NativeQwen3Error with fn output(self, logger) {
  match self {
    InvalidModel(message) => logger.write_string(message)
  }
}

///|
pub extend NativeQwen3Error with Show::{to_string, output}

///|
pub extend NativeQwen3Error with @debug.Debug::{to_repr}

///|
priv struct NativeKVLayerCache[T] {
  keys : Ref[T?]
  values : Ref[T?]
}

///|
pub struct NativeKVCache[T] {
  priv layers : Array[NativeKVLayerCache[T]]
}

///|
pub fn[T] NativeKVCache::new(n_layers~ : Int32) -> NativeKVCache[T] {
  {
    layers: Array::makei(n_layers, _ => {
      keys: { val: None, },
      values: { val: None, },
    }),
  }
}

///|
/// Qwen3 weights and reusable RoPE tables backed by native F32 tensors.
pub struct NativeQwen3Model[T] {
  config : Config
  weights : Map[String, T]
  cos : T
  sin : T
  mut current_pos : Int32
}

///|
pub fn[T] NativeQwen3Model::new_cache(
  self : NativeQwen3Model[T],
) -> NativeKVCache[T] {
  NativeKVCache::new(n_layers=self.config.num_hidden_layers)
}

///|
pub fn[T] NativeQwen3Model::reset_kv_cache(self : NativeQwen3Model[T]) -> Unit {
  self.current_pos = 0
}

///|
fn[T] NativeQwen3Model::weight(
  self : NativeQwen3Model[T],
  name : String,
) -> T raise NativeQwen3Error {
  guard self.weights.get(name) is Some(weight) else {
    raise InvalidModel("missing Qwen3 weight: \{name}")
  }
  weight
}

///|
fn[T : NativeTensor] NativeQwen3Model::linear(
  self : NativeQwen3Model[T],
  input : T,
  weight_name : String,
) -> T raise NativeQwen3Error {
  input.linear(self.weight(weight_name))
}

///|
fn[T : NativeTensor] NativeQwen3Model::norm(
  self : NativeQwen3Model[T],
  input : T,
  weight_name : String,
) -> T raise NativeQwen3Error {
  input.rms_norm(self.weight(weight_name), eps=self.config.rms_norm_eps)
}

///|
fn[T : NativeTensor] make_native_rope_tables(config : Config) -> (T, T) {
  let context_length = config.max_position_embeddings
  let head_dim = config.head_dim
  let half_dim = head_dim / 2
  let size = context_length * head_dim
  let cos_data = FixedArray::makei(size, index => {
    let position = index / head_dim
    let pair_index = index % head_dim % half_dim
    let exponent = 2.0 * pair_index.to_double() / head_dim.to_double()
    let angle = position.to_double() /
      @math.pow(config.rope_theta.to_double(), exponent)
    Float::from_double(@math.cos(angle))
  })
  let sin_data = FixedArray::makei(size, index => {
    let position = index / head_dim
    let pair_index = index % head_dim % half_dim
    let exponent = 2.0 * pair_index.to_double() / head_dim.to_double()
    let angle = position.to_double() /
      @math.pow(config.rope_theta.to_double(), exponent)
    Float::from_double(@math.sin(angle))
  })
  let shape = [context_length, head_dim]
  (
    NativeTensor::from_host(cos_data, shape),
    NativeTensor::from_host(sin_data, shape),
  )
}

///|
/// Load the cached Qwen3 checkpoint as native F32 tensors without Python.
pub async fn[T : NativeTensor] NativeQwen3Model::load(
  model_dir : String,
) -> NativeQwen3Model[T] raise NativeQwen3Error {
  let config_text = @fs.read_file_to_string(model_dir + "/config.json") catch {
    _ => raise InvalidModel("cannot read Qwen3 config.json")
  }
  let config : Config = @json.from_json(@json.parse(config_text)) catch {
    error => raise InvalidModel("invalid Qwen3 config: \{error.to_string()}")
  }
  let checkpoint = @safetensors.SafeTensors::load(
    model_dir + "/model.safetensors",
  ) catch {
    error => raise InvalidModel("cannot load safetensors: \{error.to_string()}")
  }
  let weights = Map([])
  for name in checkpoint.tensor_names() {
    let (values, shape) = checkpoint.tensor_f32(name) catch {
      error =>
        raise InvalidModel("cannot read tensor \{name}: \{error.to_string()}")
    }
    weights.set(
      name,
      NativeTensor::from_host(FixedArray::from_array(values), shape),
    )
  }
  if !weights.contains("lm_head.weight") {
    weights.set("lm_head.weight", weights["model.embed_tokens.weight"])
  }
  let (cos, sin) = make_native_rope_tables(config)
  { config, weights, cos, sin, current_pos: 0, }
}

///|
fn[T : NativeTensor] NativeQwen3Model::rope(
  self : NativeQwen3Model[T],
  input : T,
  position : Int32,
  num_tokens : Int32,
) -> T {
  let head_dim = self.config.head_dim
  let half_dim = head_dim / 2
  let cos = self.cos
    .narrow(dim=0, start=position, length=num_tokens)
    .unsqueeze(dim=0)
    .unsqueeze(dim=0)
  let sin = self.sin
    .narrow(dim=0, start=position, length=num_tokens)
    .unsqueeze(dim=0)
    .unsqueeze(dim=0)
  let first = input.narrow(dim=-1, start=0, length=half_dim)
  let second = input.narrow(dim=-1, start=half_dim, length=half_dim)
  let rotated = NativeTensor::concat([second.scale(-1.0F), first], dim=-1)
  input.mul_elem(cos) + rotated.mul_elem(sin)
}

///|
fn[T : NativeTensor] NativeQwen3Model::causal_mask(
  self : NativeQwen3Model[T],
  position : Int32,
  num_tokens : Int32,
  total_tokens : Int32,
) -> T {
  ignore(self)
  let data = FixedArray::make(num_tokens * total_tokens, 0.0F)
  for row in 0..<num_tokens {
    for column in (position + row + 1)..<total_tokens {
      data[row * total_tokens + column] = 1.0F
    }
  }
  NativeTensor::from_host(data, [1, 1, num_tokens, total_tokens])
}

///|
fn[T : NativeTensor] NativeQwen3Model::attention(
  self : NativeQwen3Model[T],
  input : T,
  mask : T,
  position : Int32,
  layer : Int32,
  cache : NativeKVLayerCache[T]?,
) -> T raise NativeQwen3Error {
  let shape = input.shape()
  let batch = shape[0]
  let num_tokens = shape[1]
  let num_heads = self.config.num_attention_heads
  let num_kv_heads = self.config.num_key_value_heads
  let head_dim = self.config.head_dim
  let prefix = "model.layers.\{layer}.self_attn"

  let queries = self
    .linear(input, prefix + ".q_proj.weight")
    .view([batch, num_tokens, num_heads, head_dim])
    .permute([0, 2, 1, 3])
  let keys_new = self
    .linear(input, prefix + ".k_proj.weight")
    .view([batch, num_tokens, num_kv_heads, head_dim])
    .permute([0, 2, 1, 3])
  let values_new = self
    .linear(input, prefix + ".v_proj.weight")
    .view([batch, num_tokens, num_kv_heads, head_dim])
    .permute([0, 2, 1, 3])

  let queries = self.rope(
    self.norm(queries, prefix + ".q_norm.weight"),
    position,
    num_tokens,
  )
  let keys_new = self.rope(
    self.norm(keys_new, prefix + ".k_norm.weight"),
    position,
    num_tokens,
  )

  let (keys, values) = match cache {
    Some(layer) => {
      let (keys, values) = match (layer.keys.val, layer.values.val) {
        (Some(previous_keys), Some(previous_values)) =>
          (
            NativeTensor::concat([previous_keys, keys_new], dim=2),
            NativeTensor::concat([previous_values, values_new], dim=2),
          )
        _ => (keys_new, values_new)
      }
      layer.keys.val = Some(keys)
      layer.values.val = Some(values)
      (keys, values)
    }
    None => (keys_new, values_new)
  }

  let group_size = num_heads / num_kv_heads
  let keys = keys.repeat_interleave(repeats=group_size, dim=1)
  let values = values.repeat_interleave(repeats=group_size, dim=1)
  let scores = queries.batched_matmul(keys.permute([0, 1, 3, 2]))
  let masked = scores.masked_fill(
    mask,
    value=Float::from_double(@double.neg_infinity),
  )
  let weights = masked
    .scale(1.0F / Float::sqrt(Float::from_int(head_dim)))
    .softmax_last()
  let context = weights
    .batched_matmul(values)
    .permute([0, 2, 1, 3])
    .view([batch, num_tokens, num_heads * head_dim])
  self.linear(context, prefix + ".o_proj.weight")
}

///|
fn[T : NativeTensor] NativeQwen3Model::mlp(
  self : NativeQwen3Model[T],
  input : T,
  layer : Int32,
) -> T raise NativeQwen3Error {
  let prefix = "model.layers.\{layer}.mlp"
  let gate = self.linear(input, prefix + ".gate_proj.weight")
  let up = self.linear(input, prefix + ".up_proj.weight")
  self.linear(gate.silu().mul_elem(up), prefix + ".down_proj.weight")
}

///|
/// Run a native F32 forward pass over integer token IDs.
pub fn[T : NativeTensor] NativeQwen3Model::forward(
  self : NativeQwen3Model[T],
  input~ : Array[Int32],
  cache~ : NativeKVCache[T]?,
) -> T raise NativeQwen3Error {
  if input.is_empty() {
    raise InvalidModel("Qwen3 input requires at least one token")
  }
  let num_tokens = input.length()
  let position = match cache {
    Some(_) => self.current_pos
    None => 0
  }
  let total_tokens = position + num_tokens
  if total_tokens > self.config.max_position_embeddings {
    raise InvalidModel("Qwen3 input exceeds the configured context length")
  }
  if cache is Some(_) {
    self.current_pos = total_tokens
  }

  let mut hidden = self
    .weight("model.embed_tokens.weight")
    .embedding(input)
    .unsqueeze(dim=0)
  let mask = self.causal_mask(position, num_tokens, total_tokens)
  for layer in 0..<self.config.num_hidden_layers {
    let prefix = "model.layers.\{layer}"
    let layer_cache = match cache {
      Some(cache) => Some(cache.layers[layer])
      None => None
    }
    let residual = hidden
    let normalized = self.norm(hidden, prefix + ".input_layernorm.weight")
    let attended = self.attention(
      normalized, mask, position, layer, layer_cache,
    )
    hidden = attended + residual

    let residual = hidden
    let normalized = self.norm(
      hidden,
      prefix + ".post_attention_layernorm.weight",
    )
    hidden = self.mlp(normalized, layer) + residual
  }
  self.linear(self.norm(hidden, "model.norm.weight"), "lm_head.weight")
}
