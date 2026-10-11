///|
/// Feature playground CLI — run experiments from the workspace root:
///
///   moon run playground/cmd/main --target native -- hello --name MoonBit
///   moon run playground/cmd/main --target native -- echo some words here
///   moon run playground/cmd/main --target native -- download <url>
///
/// The command tree and dispatch live in `cli.mbt`.
async fn main {
  run()
}
