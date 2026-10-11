///|
/// Feature playground CLI — run experiments from the workspace root:
///
///   moon run playground/cmd/main --target native -- hello --name MoonBit
///   moon run playground/cmd/main --target native -- echo some words here
///
/// Add a new experiment: create its Command below and a branch in the match.
fn main {
  let cmd = @argparse.Command(
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
    ],
  )
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
    _ => println(cmd.render_help())
  }
}
