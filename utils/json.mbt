///|
/// Extract required field `key` from a JSON object and decode it as `T`.
///
/// The `T : FromJson` constraint means `T` knows how to decode itself from a
/// JSON value. `path` records the current location so errors point to the
/// missing or malformed field. This raises `JsonDecodeError` when the key is
/// absent or `T`'s decoder rejects the value.
pub fn[T : FromJson] extract(
  obj : Map[String, Json],
  key : String,
  path : @json.JsonPath,
) -> T raise @json.JsonDecodeError {
  match obj.get(key) {
    Some(v) => @json.from_json(v, path=path.add_key(key))
    None =>
      raise @json.JsonDecodeError(
        (path.add_key(key), "missing \"\{key}\" field"),
      )
  }
}

///|
/// Decode a JSON value as an object, or raise a standard `JsonDecodeError`.
///
/// `what` describes the value being decoded, and `path` points to its current
/// location in the document so the error message carries useful context.
pub fn expect_object(
  j : Json,
  what~ : String,
  path~ : @json.JsonPath,
) -> Map[String, Json] raise @json.JsonDecodeError {
  guard j is Json::Object(o) else {
    raise @json.JsonDecodeError((path, "\{what}: expected JSON object"))
  }
  o
}

///|
/// Return the raw JSON value for a required object key.
///
/// Unlike `extract`, this does not decode the value. It is useful when an enum
/// needs to inspect a discriminator before choosing how to decode the value.
/// The error path points to the missing key.
pub fn expect_key(
  obj : Map[String, Json],
  key : String,
  what~ : String,
  path~ : @json.JsonPath,
) -> Json raise @json.JsonDecodeError {
  match obj.get(key) {
    Some(v) => v
    None =>
      raise @json.JsonDecodeError(
        (path.add_key(key), "\{what}: missing \"\{key}\""),
      )
  }
}
