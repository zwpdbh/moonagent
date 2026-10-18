///|
fn byte_to_symbol(byte : Byte) -> String {
  let value = byte.to_int()
  let codepoint = if (value >= 33 && value <= 126) ||
    (value >= 161 && value <= 172) ||
    value >= 174 {
    value
  } else {
    let index = if value <= 32 {
      value
    } else if value <= 160 {
      value - 94
    } else {
      67
    }
    256 + index
  }
  guard codepoint.to_char() is Some(symbol) else {
    abort("invalid byte-alphabet code point")
  }
  symbol.to_string()
}

///|
pub fn bytes_to_symbols(bytes : Bytes) -> Array[String] {
  let symbols : Array[String] = []
  for byte in bytes {
    symbols.push(byte_to_symbol(byte))
  }
  symbols
}
