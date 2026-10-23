///|
pub(all) suberror ConfigError {
  InvalidFormat(String)
}

///|
struct Config {
  architectures : Array[String]
  attention_bias : Bool
  attention_dropout : Float
  bos_token_id : Int32
  eos_token_id : Int32
  head_dim : Int32
  hidden_act : String
  hidden_size : Int32
  initializer_range : Float
  intermediate_size : Int32
  max_position_embeddings : Int32
  max_window_layers : Int32
  model_type : String
  num_attention_heads : Int32
  num_hidden_layers : Int32
  num_key_value_heads : Int32
  rms_norm_eps : Float
  rope_scaling : Int32?
  rope_theta : Int32
  sliding_window : Int32?
  tie_word_embeddings : Bool
  torch_dtype : DType
  transformers_version : String
  use_cache : Bool
  use_sliding_window : Bool
  vocab_size : Int32
}

///|
/// Parse a Qwen3 configuration JSON object.
impl FromJson for Config with fn from_json(json, path) {
  let obj = @utils.expect_object(json, what="Config", path~)

  let architectures : Array[String] = @utils.extract(obj, "architectures", path)
  let attention_bias : Bool = @utils.extract(obj, "attention_bias", path)
  let attention_dropout : Float = @utils.extract(obj, "attention_dropout", path)
  let bos_token_id : Int32 = @utils.extract(obj, "bos_token_id", path)
  let eos_token_id : Int32 = @utils.extract(obj, "eos_token_id", path)
  let head_dim : Int32 = @utils.extract(obj, "head_dim", path)
  let hidden_act : String = @utils.extract(obj, "hidden_act", path)
  let hidden_size : Int32 = @utils.extract(obj, "hidden_size", path)
  let initializer_range : Float = @utils.extract(obj, "initializer_range", path)
  let intermediate_size : Int32 = @utils.extract(obj, "intermediate_size", path)
  let max_position_embeddings : Int32 = @utils.extract(
    obj, "max_position_embeddings", path,
  )
  let max_window_layers : Int32 = @utils.extract(obj, "max_window_layers", path)
  let model_type : String = @utils.extract(obj, "model_type", path)
  let num_attention_heads : Int32 = @utils.extract(
    obj, "num_attention_heads", path,
  )
  let num_hidden_layers : Int32 = @utils.extract(obj, "num_hidden_layers", path)
  let num_key_value_heads : Int32 = @utils.extract(
    obj, "num_key_value_heads", path,
  )
  let rms_norm_eps : Float = @utils.extract(obj, "rms_norm_eps", path)
  let rope_scaling : Int32? = @utils.extract(obj, "rope_scaling", path)
  let rope_theta : Int32 = @utils.extract(obj, "rope_theta", path)
  let sliding_window : Int32? = @utils.extract(obj, "sliding_window", path)
  let tie_word_embeddings : Bool = @utils.extract(
    obj, "tie_word_embeddings", path,
  )
  let torch_dtype : DType = @utils.extract(obj, "torch_dtype", path)
  let transformers_version : String = @utils.extract(
    obj, "transformers_version", path,
  )
  let use_cache : Bool = @utils.extract(obj, "use_cache", path)
  let use_sliding_window : Bool = @utils.extract(
    obj, "use_sliding_window", path,
  )
  let vocab_size : Int32 = @utils.extract(obj, "vocab_size", path)

  {
    architectures,
    attention_bias,
    attention_dropout,
    bos_token_id,
    eos_token_id,
    head_dim,
    hidden_act,
    hidden_size,
    initializer_range,
    intermediate_size,
    max_position_embeddings,
    max_window_layers,
    model_type,
    num_attention_heads,
    num_hidden_layers,
    num_key_value_heads,
    rms_norm_eps,
    rope_scaling,
    rope_theta,
    sliding_window,
    tie_word_embeddings,
    torch_dtype,
    transformers_version,
    use_cache,
    use_sliding_window,
    vocab_size,
  }
}

///|
pub enum DType {
  BF16
}

///|
/// Parse the dtype string used by Hugging Face configuration files.
impl FromJson for DType with fn from_json(json, path) {
  let value : String = @json.from_json(json, path~)
  match value {
    "bfloat16" | "BF16" => BF16
    _ => raise @json.JsonDecodeError((path, "unsupported dtype \"\{value}\""))
  }
}
