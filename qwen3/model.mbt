///|
/// Qwen3 model weights and reusable RoPE tables, executed through zw/torch.
pub struct Qwen3Model {
  config : Config
  weights : Map[String, Tensor]
  cos : Tensor
  sin : Tensor
  device : String
  mut current_pos : Int32
}

///|
priv struct KVLayerCache {
  keys : Ref[Tensor?]
  values : Ref[Tensor?]
}

///|
/// Per-layer key/value caches for incremental attention.
pub struct KVCache {
  priv layers : Array[KVLayerCache]
}

///|
pub fn KVCache::new(n_layers~ : Int32) -> KVCache {
  {
    layers: Array::makei(n_layers, _ => {
      keys: { val: None, },
      values: { val: None, },
    }),
  }
}

///|
pub fn Qwen3Model::new_cache(self : Qwen3Model) -> KVCache {
  KVCache::new(n_layers=self.config.num_hidden_layers)
}

///|
pub fn Qwen3Model::reset_kv_cache(self : Qwen3Model) -> Unit {
  self.current_pos = 0
}

///|
fn Qwen3Model::weight(
  self : Qwen3Model,
  name : String,
) -> Tensor raise TorchError {
  guard self.weights.get(name) is Some(weight) else {
    raise AttributeMissing("missing Qwen3 weight: \{name}")
  }
  weight
}

///|
fn Qwen3Model::linear(
  self : Qwen3Model,
  input : Tensor,
  weight_name : String,
) -> Tensor raise TorchError {
  @torch.linear_projection(input~, weight=self.weight(weight_name))
}

///|
fn Qwen3Model::norm(
  self : Qwen3Model,
  input : Tensor,
  weight_name : String,
) -> Tensor raise TorchError {
  @torch.rms_norm(
    input~,
    weight=self.weight(weight_name),
    eps=self.config.rms_norm_eps.to_double(),
  )
}

///|
fn make_rope_tables(
  config : Config,
  device : String,
) -> (Tensor, Tensor) raise TorchError {
  let context_length = config.max_position_embeddings
  let head_dim = config.head_dim
  let half_dim = head_dim / 2
  let size = context_length * head_dim
  let cos_data = Array::makei(size, index => {
    let position = index / head_dim
    let pair_index = index % head_dim % half_dim
    let exponent = 2.0 * pair_index.to_double() / head_dim.to_double()
    let angle = position.to_double() /
      @math.pow(config.rope_theta.to_double(), exponent)
    @math.cos(angle)
  })
  let sin_data = Array::makei(size, index => {
    let position = index / head_dim
    let pair_index = index % head_dim % half_dim
    let exponent = 2.0 * pair_index.to_double() / head_dim.to_double()
    let angle = position.to_double() /
      @math.pow(config.rope_theta.to_double(), exponent)
    @math.sin(angle)
  })
  let shape = [context_length, head_dim]
  let cos = @torch.tensor(data=cos_data, shape~).to(device~)
  let sin = @torch.tensor(data=sin_data, shape~).to(device~)
  (cos, sin)
}

///|
/// Load config.json and model.safetensors from an existing model directory.
pub fn Qwen3Model::load(
  model_dir : String,
  device~ : String,
) -> Qwen3Model raise TorchError {
  let config_text = @fs.read_file_to_string(model_dir + "/config.json") catch {
    _ => raise ImportFailed("cannot read Qwen3 config.json")
  }
  let config : Config = @json.from_json(@json.parse(config_text)) catch {
    error => raise InvalidShape("invalid Qwen3 config: \{error.to_string()}")
  }
  let loaded = @torch.load_safetensors(model_dir + "/model.safetensors")
  let weights = Map([])
  for name, weight in loaded {
    weights.set(name, weight.to(device~))
  }
  if !weights.contains("lm_head.weight") {
    weights.set("lm_head.weight", weights["model.embed_tokens.weight"])
  }
  let (cos, sin) = make_rope_tables(config, device)
  { config, weights, cos, sin, device, current_pos: 0, }
}

///|
fn Qwen3Model::rope(
  self : Qwen3Model,
  input : Tensor,
  position : Int32,
  num_tokens : Int32,
) -> Tensor raise TorchError {
  let head_dim = self.config.head_dim
  let half_dim = head_dim / 2
  let cos_rows = @torch.narrow(
    input=self.cos,
    dim=0,
    start=position,
    length=num_tokens,
  )
  let sin_rows = @torch.narrow(
    input=self.sin,
    dim=0,
    start=position,
    length=num_tokens,
  )
  let cos = @torch.unsqueeze(
    input=@torch.unsqueeze(input=cos_rows, dim=0),
    dim=0,
  )
  let sin = @torch.unsqueeze(
    input=@torch.unsqueeze(input=sin_rows, dim=0),
    dim=0,
  )
  let first = @torch.narrow(input~, dim=-1, start=0, length=half_dim)
  let second = @torch.narrow(input~, dim=-1, start=half_dim, length=half_dim)
  let rotated = @torch.concat(
    [@torch.scale(input=second, scalar=-1.0), first],
    dim=-1,
  )
  @torch.add(
    left=@torch.mul(left=input, right=cos),
    right=@torch.mul(left=rotated, right=sin),
  ).to_dtype(dtype="bfloat16")
}

///|
fn Qwen3Model::causal_mask(
  self : Qwen3Model,
  position : Int32,
  num_tokens : Int32,
  total_tokens : Int32,
) -> Tensor raise TorchError {
  let full = @torch.triu(
    input=@torch.ones([total_tokens, total_tokens], device=self.device),
    diagonal=1,
  )
  let rows = @torch.narrow(input=full, dim=0, start=position, length=num_tokens)
  let cropped = @torch.narrow(input=rows, dim=1, start=0, length=total_tokens)
  @torch.unsqueeze(input=@torch.unsqueeze(input=cropped, dim=0), dim=0)
}

///|
fn Qwen3Model::attention(
  self : Qwen3Model,
  input : Tensor,
  mask : Tensor,
  position : Int32,
  layer : Int32,
  cache : KVLayerCache?,
) -> Tensor raise TorchError {
  let shape = input.shape()
  let batch = shape[0]
  let num_tokens = shape[1]
  let num_heads = self.config.num_attention_heads
  let num_kv_heads = self.config.num_key_value_heads
  let head_dim = self.config.head_dim
  let prefix = "model.layers.\{layer}.self_attn"

  let queries = self
    .linear(input, prefix + ".q_proj.weight")
    .reshape([batch, num_tokens, num_heads, head_dim])
    .transpose(dim0=1, dim1=2)
  let keys_new = self
    .linear(input, prefix + ".k_proj.weight")
    .reshape([batch, num_tokens, num_kv_heads, head_dim])
    .transpose(dim0=1, dim1=2)
  let values_new = self
    .linear(input, prefix + ".v_proj.weight")
    .reshape([batch, num_tokens, num_kv_heads, head_dim])
    .transpose(dim0=1, dim1=2)

  let normalized_queries = self.norm(queries, prefix + ".q_norm.weight")
  let queries = self.rope(normalized_queries, position, num_tokens)
  let normalized_keys = self.norm(keys_new, prefix + ".k_norm.weight")
  let keys_new = self.rope(normalized_keys, position, num_tokens)

  let (keys, values) = match cache {
    Some(layer) => {
      let (keys, values) = match (layer.keys.val, layer.values.val) {
        (Some(previous_keys), Some(previous_values)) =>
          (
            @torch.concat([previous_keys, keys_new], dim=2),
            @torch.concat([previous_values, values_new], dim=2),
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
  let keys = @torch.repeat_interleave(input=keys, repeats=group_size, dim=1)
  let values = @torch.repeat_interleave(input=values, repeats=group_size, dim=1)
  let scores = @torch.matmul(left=queries, right=keys.transpose(dim0=2, dim1=3))
  let masked = @torch.masked_fill(
    input=scores,
    mask~,
    value=@double.neg_infinity,
  )
  let weights = @torch.softmax(
    input=@torch.scale(input=masked, scalar=1.0 / head_dim.to_double().sqrt()),
    dim=-1,
  )
  let context = @torch.matmul(left=weights, right=values)
    .transpose(dim0=1, dim1=2)
    .reshape([batch, num_tokens, num_heads * head_dim])
  self.linear(context, prefix + ".o_proj.weight")
}

///|
fn Qwen3Model::mlp(
  self : Qwen3Model,
  input : Tensor,
  layer : Int32,
) -> Tensor raise TorchError {
  let prefix = "model.layers.\{layer}.mlp"
  let gate = self.linear(input, prefix + ".gate_proj.weight")
  let up = self.linear(input, prefix + ".up_proj.weight")
  let hidden = @torch.mul(left=@torch.silu(input=gate), right=up)
  self.linear(hidden, prefix + ".down_proj.weight")
}

///|
/// Run the Qwen3 forward pass, optionally reading and updating a KV cache.
pub fn Qwen3Model::forward(
  self : Qwen3Model,
  input~ : Tensor,
  cache~ : KVCache?,
) -> Tensor raise TorchError {
  let input_shape = input.shape()
  if input_shape.length() != 2 {
    raise InvalidShape("Qwen3 input must have shape [batch, sequence]")
  }
  let num_tokens = input_shape[1]
  let position = match cache {
    Some(_) => self.current_pos
    None => 0
  }
  let total_tokens = position + num_tokens
  if total_tokens > self.config.max_position_embeddings {
    raise InvalidShape("Qwen3 input exceeds the configured context length")
  }
  if cache is Some(_) {
    self.current_pos = total_tokens
  }

  let mut hidden = @torch.embedding(
    weight=self.weight("model.embed_tokens.weight"),
    input~,
  )
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
    hidden = @torch.add(left=attended, right=residual)

    let residual = hidden
    let normalized = self.norm(
      hidden,
      prefix + ".post_attention_layernorm.weight",
    )
    let mlp_output = self.mlp(normalized, layer)
    hidden = @torch.add(left=mlp_output, right=residual)
  }
  hidden = self.norm(hidden, "model.norm.weight")
  hidden = hidden.to_dtype(dtype="bfloat16")
  self.linear(hidden, "lm_head.weight")
}
