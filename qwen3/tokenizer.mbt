///|
pub(all) suberror TokenizerError {
  InvalidFormat(String)
  UnknowToken(Int)
}

///|
type Tokenizer

///|
pub fn Tokenizer::parse(json : String) -> Tokenizer raise TokenizerError {
  abort("todo")
}

///|
pub async fn Tokenizer::load(path : String) -> Tokenizer raise TokenizerError {
  abort("todo")
}

///|
pub fn Tokenizer::encode(self : Tokenizer, text : String) -> Array[Int] {
  abort("TODO")
}

///|
pub fn Tokenizer::decode(
  self : Tokenizer,
  ids : Array[Int],
) -> String raise TokenizerError {
  abort("TODO")
}
