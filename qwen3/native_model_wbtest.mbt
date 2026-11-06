///|
async test "native Qwen3 forward predicts the first chapter token" {
  let model_dir = "/home/zw/.cache/models/Qwen/Qwen3-0.6B"
  let model = NativeQwen3Model::load(model_dir)
  let tokenizer = Tokenizer::load(model_dir + "/tokenizer.json") catch {
    _ => abort("failed to load tokenizer")
  }
  let prompt = "<|im_start|>user\nExplain large language models.<|im_end|>\n<|im_start|>assistant\n"
  let input = tokenizer.encode(prompt) catch {
    _ => abort("failed to encode prompt")
  }
  let logits = model.forward(input~, cache=None)
  assert_eq(logits.shape.shape[0], 1)
  assert_eq(logits.shape.shape[1], input.length())
  assert_eq(logits.shape.shape[2], 151936)
  let next_id = logits.select(dim=1, index=input.length() - 1).argmax_last().data[0].to_int()
  assert_eq(next_id, 151667)
}
