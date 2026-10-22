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
} derive(FromJson)

///|
pub enum DType {
  BF16
}
