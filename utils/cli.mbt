///|
pub(all) suberror CliError {
  MissingCase
  InvalidCase(String)
}

///|
impl Show for CliError with fn output(self, logger) -> Unit {
  match self {
    MissingCase => logger.write_string("expected case01 or case02")
    InvalidCase(value) =>
      logger.write_string("invalid case '\{value}': expected case01 or case02")
  }
}

///|
fn parse_case_from(args : Array[String]) -> String raise CliError {
  guard args.length() >= 2 else { raise MissingCase }
  let case = args[1]
  if case == "case01" || case == "case02" {
    case
  } else {
    raise InvalidCase(case)
  }
}

///|
pub fn parse_case() -> String raise CliError {
  parse_case_from(@env.args())
}
