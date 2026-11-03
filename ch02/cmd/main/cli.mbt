///|
struct AppArgs {
  mut command : Command?
}

///|
pub enum SubCommand {
  Case01
  Case02(input~ : String)
}

///|
pub enum Command {
  Ex00
  Ex01(text~ : String)
  Ex03(SubCommand?)
}

///|
impl @clap.Value for AppArgs with fn add_value(self, name, value, _positional) -> Unit raise @clap.ParserError {
  match self.command {
    Some(Ex00) =>
      raise @clap.ParserError::InvalidArgumentName(
        "ex00 does not accept a value: \{name}",
      )
    Some(Ex01(..)) => {
      guard name == "text" else {
        raise @clap.ParserError::InvalidArgumentName(
          "expected text, got: \{name}",
        )
      }
      self.command = Some(Ex01(text=value))
    }
    Some(Ex03(None)) =>
      raise @clap.ParserError::InvalidArgumentName(
        "ex03 requires case01 or case02 before a value",
      )
    Some(Ex03(Some(Case01))) =>
      raise @clap.ParserError::InvalidArgumentName(
        "case01 does not accept a value: \{name}",
      )
    Some(Ex03(Some(Case02(..)))) => {
      guard name == "input" else {
        raise @clap.ParserError::InvalidArgumentName(
          "expected input, got: \{name}",
        )
      }
      self.command = Some(Ex03(Some(Case02(input=value))))
    }
    None =>
      raise @clap.ParserError::InvalidArgumentName(
        "value before a subcommand: \{name}",
      )
  }
}

///|
impl @clap.Value for AppArgs with fn select_subcmd(self, name) -> AppArgs raise @clap.ParserError {
  match name {
    "ex00" => self.command = Some(Ex00)
    "ex01" => self.command = Some(Ex01(text=""))
    "ex03" => self.command = Some(Ex03(None))
    "case01" if self.command is Some(Ex03(_)) =>
      self.command = Some(Ex03(Some(Case01)))
    "case02" if self.command is Some(Ex03(_)) =>
      self.command = Some(Ex03(Some(Case02(input=""))))
    _ =>
      raise @clap.ParserError::InvalidSubCommandName(
        "unknown subcommand: \{name}",
      )
  }
  self
}

///|
/// Parse command-line arguments into the command enum without running business logic.
pub fn parse_args() -> Command? raise @clap.ParserError {
  let parser = @clap.Parser::new(
    prog="ch02",
    description="Generate text with Qwen3",
    subcmds={
      "ex00": @clap.SubCommand::new(help="Dummy command without arguments"),
      "ex01": @clap.SubCommand::new(
        args={
          "text": @clap.Arg::positional(
            nargs=@clap.Nargs::One,
            help="Prompt text",
          ),
        },
        help="Generate text from a prompt",
      ),
      "ex03": @clap.SubCommand::new(
        subcmds={
          "case01": @clap.SubCommand::new(help="Dummy case01"),
          "case02": @clap.SubCommand::new(
            args={
              "input": @clap.Arg::positional(
                nargs=@clap.Nargs::One,
                help="Dummy input",
              ),
            },
            help="Dummy case02 with input",
          ),
        },
        help="Dummy nested subcommands",
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
