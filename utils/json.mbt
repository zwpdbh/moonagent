///|
pub(all) suberror UtilJsonError {
  InvalidFormat(String)
}

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
/// Extract the object at a JSON value, or raise with a clear message.
pub fn expect_object(
  j : Json,
  what~ : String,
) -> Map[String, Json] raise UtilJsonError {
  guard j is Json::Object(o) else {
    raise InvalidFormat("\{what}: expected JSON object")
  }
  o
}

///|
/// Extract a required key from a JSON object, or raise with the key name.
pub fn expect_key(
  obj : Map[String, Json],
  key : String,
  what~ : String,
) -> Json raise UtilJsonError {
  match obj.get(key) {
    Some(v) => v
    None => raise InvalidFormat("\{what}: missing \"{key}\"")
  }
}
