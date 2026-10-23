// Command-line parsing and dispatch for the playground CLI.
//
// Add a new experiment: create its Command in `command()` below and a
// branch in `run()`'s match.

///|
/// The playground command tree — one subcommand per experiment.
fn command() -> @argparse.Command {
  @argparse.Command(
    "playground",
    about="MoonBit feature playground",
    subcommand_required=true,
    subcommands=[
      @argparse.Command("hello", about="Print a greeting", options=[
        @argparse.OptionArg("name", short='n', about="Who to greet", default_values=[
          "world",
        ]),
      ]),
      @argparse.Command("echo", about="Echo the arguments back", positionals=[
        @argparse.PositionArg(
          "words",
          about="Words to echo",
          num_args=@argparse.ValueRange(lower=1, upper=99),
        ),
      ]),
      @argparse.Command(
        "download",
        about="Download a file (skips files already complete in the target dir)",
        options=[
          @argparse.OptionArg(
            "out",
            short='o',
            about="Output directory (default: the shared Qwen3 model cache)",
          ),
          @argparse.OptionArg(
            "backup",
            short='b',
            about="Backup URL to try if the primary fails",
          ),
        ],
        positionals=[
          @argparse.PositionArg(
            "url",
            about="URL to download",
            num_args=@argparse.ValueRange::single(),
          ),
        ],
      ),
    ],
  )
}

///|
async fn run_download(m : @argparse.Matches) -> Unit {
  let url = match m.values.get("url") {
    Some([url, ..]) => url
    _ => {
      println("error: no URL given")
      return
    }
  }
  let out = match m.values.get("out") {
    Some([dir, ..]) => dir
    _ => @utils.qwen3_model_dir()
  }
  let backup = match m.values.get("backup") {
    Some([b, ..]) => Some(b)
    _ => None
  }
  try {
    let dest = @utils.download_file(url, out, backup_url?=backup)
    println("saved: \{dest}")
  } catch {
    e => println("download failed: \{e}")
  }
}

///|
/// Parse argv and dispatch to the subcommand handler.
async fn run() -> Unit {
  let cmd = command()
  let matches = cmd.parse() catch {
    e => {
      println(e)
      println(cmd.render_help())
      return
    }
  }
  match matches.subcommand {
    Some(("hello", m)) => {
      let name = match m.values.get("name") {
        Some([name, ..]) => name
        _ => "world"
      }
      println(@playground.greet(name))
    }
    Some(("echo", m)) =>
      match m.values.get("words") {
        Some(words) => println(words.join(" "))
        None => println("(nothing to echo)")
      }
    Some(("download", m)) => run_download(m)
    _ => println(cmd.render_help())
  }
}
