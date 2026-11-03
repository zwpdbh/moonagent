// Learn more about moon.mod configuration:
// https://docs.moonbitlang.com/en/latest/toolchain/moon/module.html
//
// To add a dependency, run this command in your terminal:
//   moon add moonbitlang/x
//
// Or manually declare it in `import`, for example:
// import {
//   "moonbitlang/x@0.4.6",
// }

name = "zw/ch02"

version = "0.1.0"

readme = "README.mbt.md"

repository = ""

license = "Apache-2.0"

keywords = [ ]

preferred_target = "native"

description = ""

import {
  "zw/torch@0.1.0",
  "zw/qwen3@0.1.0",
  "TheWaWaR/clap@0.2.6",
  "zw/utils@0.1.0",
}
