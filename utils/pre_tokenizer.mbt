///|
let qwen_manual_fallback_pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

///|
fn contraction_length(chars : Array[Char], start : Int32) -> Int32 {
  guard start + 1 < chars.length() && chars[start] == '\'' else { return 0 }
  match chars[start + 1].to_ascii_lowercase() {
    's' | 't' | 'm' | 'd' => 2
    'r' | 'v' =>
      if start + 2 < chars.length() &&
        chars[start + 2].to_ascii_lowercase() == 'e' {
        3
      } else {
        0
      }
    'l' =>
      if start + 2 < chars.length() &&
        chars[start + 2].to_ascii_lowercase() == 'l' {
        3
      } else {
        0
      }
    _ => 0
  }
}

///|
fn split_compiled_string_regex(
  regex : @string.Regex,
  input : String,
) -> Array[String] {
  let result : Array[String] = []
  let mut remaining = input[:]
  while remaining.length() > 0 {
    match regex.execute(remaining) {
      None => {
        result.push(remaining.to_owned())
        return result
      }
      Some(matched) => {
        let content = matched.content()
        guard content.length() > 0 else {
          abort("pre-tokenizer regex produced an empty match")
        }
        let before = matched.before()
        if before.length() > 0 {
          result.push(before.to_owned())
        }
        result.push(content.to_owned())
        remaining = matched.after()
      }
    }
  }
  result
}

///|
fn try_split_with_string_regex(
  input : String,
  pattern : String,
) -> Array[String]? {
  let regex = @string.Regex(pattern) catch { _ => return None }
  Some(split_compiled_string_regex(regex, input))
}

///|
fn split_compiled_regexp(
  regex : @regexp.Regexp,
  input : String,
) -> Array[String] {
  let result : Array[String] = []
  let mut remaining = input[:]
  while remaining.length() > 0 {
    let matched = regex.execute(remaining)
    guard matched.matched() else {
      result.push(remaining.to_owned())
      return result
    }
    guard matched.get(0) is Some(content) else {
      abort("pre-tokenizer regex match is missing group 0")
    }
    guard content.length() > 0 else {
      abort("pre-tokenizer regex produced an empty match")
    }
    let before = matched.before()
    if before.length() > 0 {
      result.push(before.to_owned())
    }
    result.push(content.to_owned())
    remaining = matched.after()
  }
  result
}

///|
fn try_split_with_regexp(input : String, pattern : String) -> Array[String]? {
  let regex = @regexp.compile(pattern) catch { _ => return None }
  Some(split_compiled_regexp(regex, input))
}

///|
/// The purpose of this guard is to make sure the pattern we extracted is from 
/// qwen3 model.
/// Otherwise if we just provide a default manuall implementation, 
/// and the pattern (-> model) is not compatiable, we may have problems.
/// The manual fallback implements Qwen's rules for one exact pattern.
/// If both regex engines reject the pattern, this check prevents an
/// incompatible pattern (which its corresponding model) from silently using those Qwen-specific rules.
/// In short: The guard cannot prove where the pattern came from. 
/// It only proves that the pattern is exactly equal to the pattern whose behavior the manual fallback implements.
pub fn split_with_pre_tokenizer(
  input : String,
  pattern : String,
) -> Array[String] {
  if try_split_with_string_regex(input, pattern) is Some(result) {
    return result
  }
  if try_split_with_regexp(input, pattern) is Some(result) {
    return result
  }

  guard pattern == qwen_manual_fallback_pattern else {
    abort("pattern has no supported regex or manual implementation")
  }
  split_with_qwen_manual_fallback(input)
}

///|
fn split_with_qwen_manual_fallback(input : String) -> Array[String] {
  let chars = input.to_array()
  let result : Array[String] = []
  let letter_regex = try! @regexp.compile("^\\p{L}$")
  let letters = chars.map(c => {
    c.is_ascii_alphabetic() ||
    (!c.is_ascii() && letter_regex.execute(c.to_string()).matched())
  })
  let mut pos = 0
  while pos < chars.length() {
    let contraction = contraction_length(chars, pos)
    let mut end = if contraction > 0 { pos + contraction } else { 0 }
    if end == 0 {
      let mut first_letter = pos
      if !letters[pos] &&
        chars[pos] != '\r' &&
        chars[pos] != '\n' &&
        !chars[pos].is_numeric() &&
        pos + 1 < chars.length() &&
        letters[pos + 1] {
        first_letter = pos + 1
      }
      if first_letter < chars.length() && letters[first_letter] {
        end = first_letter + 1
        while end < chars.length() && letters[end] {
          end += 1
        }
      }
    }
    if end == 0 && chars[pos].is_numeric() {
      end = pos + 1
    }
    if end == 0 {
      let mut punctuation_start = pos
      if chars[pos] == ' ' && pos + 1 < chars.length() {
        punctuation_start = pos + 1
      }
      if punctuation_start < chars.length() &&
        !chars[punctuation_start].is_whitespace() &&
        !chars[punctuation_start].is_numeric() &&
        !letters[punctuation_start] {
        end = punctuation_start + 1
        while end < chars.length() &&
              !chars[end].is_whitespace() &&
              !chars[end].is_numeric() &&
              !letters[end] {
          end += 1
        }
        while end < chars.length() && (chars[end] == '\r' || chars[end] == '\n') {
          end += 1
        }
      }
    }
    if end == 0 {
      let mut whitespace_end = pos
      while whitespace_end < chars.length() &&
            chars[whitespace_end].is_whitespace() {
        whitespace_end += 1
      }
      let mut last_newline = -1
      for i in pos..<whitespace_end {
        if chars[i] == '\r' || chars[i] == '\n' {
          last_newline = i
        }
      }
      if last_newline >= 0 {
        end = last_newline + 1
      } else if whitespace_end < chars.length() && whitespace_end - pos > 1 {
        end = whitespace_end - 1
      } else {
        end = whitespace_end
      }
    }
    guard end > pos else { abort("Qwen pre-tokenizer did not match the input") }
    result.push(String::from_array(chars[pos:end]))
    pos = end
  }
  result
}
