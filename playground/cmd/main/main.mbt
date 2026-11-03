///|
/// Entry point for the CLI or runnable program in this template.
/// Keep main focused; move logic into the library package.
/// Run with `moon run cmd/main` from the project root.
fn main {
  let command = parse_args() catch {
    error => {
      println(error)
      return
    }
  }
  match command {
    Some(Ex01(text~)) =>
      @ch02.run_ch02_ex01(text) catch {
        error => println(error)
      }
    Some(Ex03(Some(Case01))) => println("Ex03 Case01 selected")
    Some(Ex03(Some(Case02(input~)))) =>
      println("Ex03 Case02 received: \{input}")
    None | Some(Ex03(None)) => ()
  }
}
