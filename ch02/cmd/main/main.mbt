///|
using @torch {type TorchError}

///|
fn main raise {
  let case = @utils.parse_case()
  let result = run(case) catch {
    error => {
      println(error.to_string())
      return
    }
  }
  println("\nGenerated \{result.length()} tokens")
}

///|
fn run(case : String) -> Array[Int64] raise TorchError {
  let model_dir = "/home/zw/.cache/models/Qwen/Qwen3-0.6B"
  @torch.add_python_path(
    "/home/zw/code/moonbit_programming/moonagent/.venv/lib/python3.12/site-packages",
  )
  let device = if @torch.cuda_is_available() { "cuda" } else { "cpu" }
  println("Using device: \{device}")
  let model = @lib.load_cached_qwen3(model_dir~, device~)
  let tokenizer = @qwen3.Tokenizer::load(model_dir + "/tokenizer.json") catch {
    _ => raise ImportFailed("failed to load Qwen3 tokenizer")
  }
  let prompt = "<|im_start|>user\nExplain large language models.<|im_end|>\n<|im_start|>assistant\n"
  let input_ids = tokenizer.encode(prompt).map(id => id.to_int64()) catch {
      _ => raise ImportFailed("failed to encode Qwen3 prompt")
    }

  match case {
    "case01" => abort("TODO")
    "case02" =>
      @lib.generate_text_basic_stream_cache(
        model~,
        token_ids=input_ids,
        max_new_tokens=32,
        eos_token_id=Some(151645L),
        device~,
        on_token=id => {
          let text = tokenizer.decode([id.to_int()]) catch {
            _ => raise ImportFailed("failed to decode Qwen3 token")
          }
          println(text)
        },
      )
    _ => abort("Not supported case: \{case}")
  }
}
