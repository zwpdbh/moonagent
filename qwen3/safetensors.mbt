///|
/// For download files:
/// zw@wei:~/code/moonbit_programming/moonagent$ ll ~/.cache/models/Qwen/Qwen3-0.6B/
// total 1483604
// drwxr-xr-x 2 zw zw       4096 Oct  8 21:18 ./
// drwxr-xr-x 3 zw zw       4096 Oct  8 21:14 ../
// -rw-r--r-- 1 zw zw        726 Oct  8 21:14 config.json
// -rw-r--r-- 1 zw zw        239 Oct  8 21:14 generation_config.json
// -rw-r--r-- 1 zw zw    1671853 Oct  8 21:16 merges.txt
// -rw-r--r-- 1 zw zw 1503300328 Oct  8 21:57 model.safetensors
// -rw-r--r-- 1 zw zw   11422654 Oct  8 21:15 tokenizer.json
// -rw-r--r-- 1 zw zw       9732 Oct  8 21:15 tokenizer_config.json
// -rw-r--r-- 1 zw zw    2776833 Oct  8 21:18 vocab.json
/// When "load model" in the LLM context, it means:
/// preparing the trained model so your program can run inference with it.
/// A download model usually contain following parts
/// 1. architecture configuration (config.json). It describe the model shape, like blueprint
/// 2. Learned weight (model.safetensors). It stores the numerical parameters learned during trainning.
/// 3. Tokenizer. (tokenizer.json, vocab.json, merges.text). It have rules for converts text into token IDs and back to text.
/// 
/// So, "loading" means multiple level
/// 1. loading the file, read the safetensors bytes from disk (bytes in memory)
/// 2. parse the safetensors header (bytes -> tensor name, shapes, dtypes, offsets), this file's job
/// 3. converting tensors (raw BF6 bytes -> Array[Float])
/// Convert raw BF16 bytes into usable numeric arrays, such as F32 for the CPU reference implementation:
/// 4. Constructing the model, using config.json to create Qwen3 architecture and bind each named 
/// tensor to the corrct model (configuration + tensors → runnable Qwen3 model)
/// 5. Moving to device: Later, loading can also mean copying weights to GPU memory. That is a separate optimization phase.
/// 
/// General step in this file
/// 1. Read config.json to know the architecture.
/// 2. Create the corresponding model structure.
/// 3. Read model.safetensors.
/// 4. Assign each named weight tensor to the correct component.
/// 5. Validate that tensor shapes match the configuration.
/// 
///

///|
pub(all) suberror SafeTensorsError {
  InvalidFormat(String)
  UnknowTensor(String)
  UnsupportedDtype(String)
} derive(Debug)

///|
pub extend SafeTensorsError with @debug.Debug::{to_repr}

///|
/// Mirrors one tensor entry inside the safetensors JSON header.
/// Only used during parsing
/// It is metadata about one tensor, like a catalog card.
/// It does not contain the learned number themselves. It tells
/// where and how to read them.
priv struct TensorHeaderEntry {
  // how each number is encoded
  dtype : DType
  // the dimensions of the tensor
  shape : Array[Int32]
  // where its bytes are stored
  data_offsets : TensorOffsets
} derive(FromJson)

///|
priv struct TensorOffsets {
  start : Int32
  end : Int32
} derive(Eq, Debug)

///|
/// Parse a two-element safetensors `data_offsets` array.
impl FromJson for TensorOffsets with fn from_json(json, path) {
  guard json is Json::Array([start, end]) else {
    raise @json.JsonDecodeError(
      (path, "TensorOffsets: expected a two-element array"),
    )
  }
  let start : Int32 = @json.from_json(start, path=path.add_index(0))
  let end : Int32 = @json.from_json(end, path=path.add_index(1))
  { start, end, }
}

///|
/// Normalize runtime metadata for one tensor
/// Store absolute file offsets, not the header's relative offsets
priv struct TensorInfo {
  dtype : DType
  shape : Array[Int32]
  data_begin : Int32
  data_end : Int32
}

///|
struct SafeTensorsHeader {
  entries : Map[String, TensorHeaderEntry]
  // each TensorHeaderEntry's data_offsets is relative to data_start_at
  data_start_at : Int32
}

///|
struct SafeTensors {
  tensors : Map[String, TensorInfo]
  path : String
}

///|
pub fn SafeTensors::load(path : String) -> SafeTensors raise SafeTensorsError {
  abort("TODO")
}

///|
pub fn SafeTensors::tensor_names(self : SafeTensors) -> Array[String] {
  abort("TODO")
}

///|
pub fn SafeTensors::tensor_f32(
  self : SafeTensors,
  name : String,
) -> (Array[Float], Array[Int32]) raise SafeTensorsError {
  abort("TODO")
}

///|
fn SafeTensors::tensor_bytes(
  self : SafeTensors,
  name : String,
) -> BytesView raise SafeTensorsError {
  abort("TODO")
}

///|
/// Load model.safetensors header parts parse it as Map[String, TensorHeaderEntry]
/// The key is 
async fn read_safetensors_header(
  path : String,
) -> SafeTensorsHeader raise SafeTensorsError {
  let file = @async_fs.open(path, mode=@async_fs.Mode::ReadOnly) catch {
    _ => raise InvalidFormat("cannot open safetensors file")
  }
  defer file.close()

  let header_length_bytes = file.read_exactly_at(8, position=0) catch {
    _ => raise InvalidFormat("cannot read safetensors header length")
  }
  let header_size = read_u64_le(header_length_bytes)
  let header_bytes = file.read_exactly_at(header_size.to_int(), position=8) catch {
    _ => raise InvalidFormat("cannot read safetensors header bytes")
  }

  let entries = header_bytes |> parse_tensor_header_entries()
  SafeTensorsHeader::{ entries, data_start_at: 8 + header_size.to_int(), }
}

///|
/// 1. From Bytes to String (Json Serilized String)
/// 2. String to Json 
/// 3. Use Json::Object to extract entries which is name, value entries.
/// Ignore the name which is "__metadata__"
/// 4. added name, and value which is TensorHeaderEntry into Map
/// 5. return final result 
fn parse_tensor_header_entries(
  header_bytes : Bytes,
) -> Map[String, TensorHeaderEntry] raise SafeTensorsError {
  // decode bytes into UTF8 string
  let decoder = @encoding.decoder(UTF8)
  let json_string = decoder.decode(header_bytes) catch {
    _ => raise InvalidFormat("failed to decode header bytes")
  }
  // from string to json obj
  let json = @json.parse(json_string) catch {
    _ => raise InvalidFormat("failed to parse header string into json object")
  }

  // from json obj get top level pairs
  guard json is Json::Object(entries) else {
    raise InvalidFormat("failed to get top level json entries")
  }

  let header_entries : Map[String, TensorHeaderEntry] = Map([])
  for (key, entry) in entries {
    if key == "__metadata__" {
      continue
    }
    let header_entry : TensorHeaderEntry = @json.from_json(entry) catch {
      e =>
        raise InvalidFormat(
          "failed to parse tensor header entry \{key}: \{e.to_string()}",
        )
    }
    header_entries.set(key, header_entry)
  }
  header_entries
}

///|
/// u64le(value) means 
/// u64 means “read a 64-bit unsigned integer.”
/// le means little-endian.
/// value is the extracted integer.
fn read_u64_le(bytes : Bytes) -> UInt64 raise SafeTensorsError {
  match bytes[:] {
    [u64le(value)] => value
    _ =>
      raise InvalidFormat(
        "expected exactly 8 bytes for the safetensors header length",
      )
  }
}
