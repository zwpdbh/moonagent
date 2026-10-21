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

///|
/// Reverse `byte_to_symbol` for one byte-alphabet symbol.
fn symbol_to_byte(symbol : Char) -> Byte {
  let codepoint = symbol.to_int()
  let value = if (codepoint >= 33 && codepoint <= 126) ||
    (codepoint >= 161 && codepoint <= 172) ||
    (codepoint >= 174 && codepoint <= 255) {
    codepoint
  } else if codepoint >= 256 && codepoint <= 288 {
    codepoint - 256
  } else if codepoint >= 289 && codepoint <= 322 {
    codepoint - 162
  } else if codepoint == 323 {
    173
  } else {
    abort("invalid byte-alphabet symbol")
  }
  value.to_byte()
}

///|
/// Convert token strings containing byte-alphabet symbols back to their
/// original bytes. UTF-8 decoding is the caller's responsibility because
/// adjacent BPE tokens may together form one UTF-8 character.
pub fn symbols_to_bytes(symbols : Array[String]) -> Bytes {
  let bytes : Array[Byte] = []
  for token in symbols {
    for symbol in token {
      bytes.push(symbol_to_byte(symbol))
    }
  }
  Bytes::from_array(bytes[:])
}
