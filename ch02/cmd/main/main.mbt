///|
fn main {
  let command = parse_args() catch {
    error => {
      println(error)
      return
    }
  }
  match command {
    Some(Ex00(text~)) => run_ex00(text) catch { error => println(error) }
    None => println("No subcommand selected; use --help for usage")
  }
}

///|
fn run_ex00(text : String) -> Unit raise @clap.ParserError {
  let model_dir = "/home/zw/.cache/models/Qwen/Qwen3-0.6B"
  @torch.add_python_path(
    "/home/zw/code/moonbit_programming/moonagent/.venv/lib/python3.12/site-packages",
  ) catch {
    error => raise @clap.ParserError::InvalidArgumentValue(error.to_string())
  }
  let device = @torch.cuda_is_available() catch {
    error => raise @clap.ParserError::InvalidArgumentValue(error.to_string())
  }
  let device = if device { "cuda" } else { "cpu" }
  println("Using device: \{device}")
  let model = @lib.load_cached_qwen3(model_dir~, device~) catch {
    error => raise @clap.ParserError::InvalidArgumentValue(error.to_string())
  }
  let tokenizer = @qwen3.Tokenizer::load(model_dir + "/tokenizer.json") catch {
    _ =>
      raise @clap.ParserError::InvalidArgumentValue("failed to load tokenizer")
  }
  let prompt = "<|im_start|>user\n\{text}<|im_end|>\n<|im_start|>assistant\n"
  let input_ids = tokenizer.encode(prompt).map(id => id.to_int64()) catch {
      _ =>
        raise @clap.ParserError::InvalidArgumentValue("failed to encode prompt")
    }
  let generated = @lib.generate_text_basic_stream_cache(
    model~,
    token_ids=input_ids,
    max_new_tokens=32,
    eos_token_id=Some(151645L),
    device~,
    on_token=id => ignore(id),
  ) catch {
    error => raise @clap.ParserError::InvalidArgumentValue(error.to_string())
  }
  let output = tokenizer.decode(generated.map(id => id.to_int())) catch {
    _ =>
      raise @clap.ParserError::InvalidArgumentValue("failed to decode output")
  }
  println(output)
  println("\nGenerated \{generated.length()} tokens")
}
