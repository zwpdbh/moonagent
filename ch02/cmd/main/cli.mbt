///|
struct AppArgs {
  mut command : Command?
}

///|
pub enum Command {
  Ex00(text~ : String)
}

///|
impl @clap.Value for AppArgs with fn add_value(self, name, value, _positional) -> Unit raise @clap.ParserError {
  match self.command {
    Some(Ex00(..)) => {
      guard name == "text" else {
        raise @clap.ParserError::InvalidArgumentName(
          "expected text, got: \{name}",
        )
      }
      self.command = Some(Ex00(text=value))
    }
    None =>
      raise @clap.ParserError::InvalidArgumentName(
        "value before a subcommand: \{name}",
      )
  }
}

///|
impl @clap.Value for AppArgs with fn select_subcmd(self, name) -> AppArgs raise @clap.ParserError {
  guard name == "ex00" else {
    raise @clap.ParserError::InvalidSubCommandName(
      "unknown subcommand: \{name}",
    )
  }
  self.command = Some(Ex00(text=""))
  self
}

///|
/// Parse command-line arguments into the command enum without running business logic.
pub fn parse_args() -> Command? raise @clap.ParserError {
  let parser = @clap.Parser::new(
    prog="ch02",
    description="Generate text with Qwen3",
    subcmds={
      "ex00": @clap.SubCommand::new(
        args={
          "text": @clap.Arg::positional(
            nargs=@clap.Nargs::One,
            help="Prompt text",
          ),
        },
        help="Generate text from a prompt",
      ),
    },
  )
  let value = AppArgs::{ command: None, }
  let help_message = parser.parse(value, @env.args()[1:])
  if help_message is Some(help) {
    println(help)
    return None
  }
  value.command
}
