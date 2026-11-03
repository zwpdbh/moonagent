///|
/// Typed clap.mbt example: parse arguments, then dispatch in main.
/// Published dependency: moon add TheWaWaR/clap
/// Local-fork alternative: add "../forked/clap.mbt" to moon.work members.
/// Keep TheWaWaR/clap@0.2.6 in moon.mod; the workspace resolves the local member.
/// moon.pkg imports: "TheWaWaR/clap" @clap and "moonbitlang/core/env"
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
  Ex01(text~ : String)
  Ex03(SubCommand?)
}

///|
/// Map argument values into the currently selected command variant.
impl @clap.Value for AppArgs with fn add_value(self, name, value, _positional) -> Unit raise @clap.ParserError {
  match self.command {
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
        "case01 takes no value: \{name}",
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
/// Select the command or nested subcommand to populate.
impl @clap.Value for AppArgs with fn select_subcmd(self, name) -> AppArgs raise @clap.ParserError {
  match name {
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
/// Parse arguments without running application logic.
pub fn parse_args() -> Command? raise @clap.ParserError {
  let parser = @clap.Parser::new(
    prog="myapp",
    description="Command-line examples",
    subcmds={
      "ex00": @clap.SubCommand::new(help="Show the no-argument example"),
      "ex01": @clap.SubCommand::new(
        args={
          "text": @clap.Arg::positional(
            nargs=@clap.Nargs::One,
            help="Text to process",
          ),
        },
        help="Run the text-input example",
      ),
      "ex03": @clap.SubCommand::new(
        subcmds={
          "case01": @clap.SubCommand::new(help="Show the first nested example"),
          "case02": @clap.SubCommand::new(
            args={
              "input": @clap.Arg::positional(
                nargs=@clap.Nargs::One,
                help="Text for the second nested example",
              ),
            },
            help="Run the second nested example with text",
          ),
        },
        help="Run a nested-command example",
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
