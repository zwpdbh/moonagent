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
/// Q: what are we doing.
/// Q: how to check the eplore file model.safetensors

///|
pub(all) suberror SafeTensorsError {
  InvalidFormat(String)
  UnknowTensor(String)
  UnsupportedDtype(String)
}

///|
priv enum DType {
  BF16
}

///|
/// Mirrors one tensor entry inside the safetensors JSON header.
/// Only used during parsing
priv struct TensorHeaderEntry {
  dtype : String
  shape : Array[Int32]
  data_offsets : Array[Int32]
} derive(FromJson)

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
struct SafeTensors {
  tensors : Map[String, TensorInfo]
  bytes : Bytes
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
