# Learn Reasoning From Scratch — in MoonBit

Follow the book [Reasoning From Scratch](https://github.com/rasbt/reasoning-from-scratch)
by Sebastian Raschka, but implement everything in [MoonBit](https://www.moonbitlang.com)
instead of Python/PyTorch.

**The point is learning, twice over.** By the end you will (1) understand
deeply how an LLM reasoning agent works — tokenization, inference,
evaluation, reasoning techniques, RL-based training — well enough to build one
yourself, and (2) learn MoonBit (a Rust-flavored, GC'd, functional-first
language) by building something real with it.

**Audience:** this repo assumes you know *neither* ML nor MoonBit. Docs here
explain both sides as we go.

## Ground rules

- **No big downloads.** The Qwen3-0.6B model files already live in the shared
  cache at `~/.cache/models/Qwen/Qwen3-0.6B` (shared with parallel Rust/Python
  ports). All code must skip files that already exist (see
  `utils/download.mbt`).
- **Stubs first, implement one by one.** New modules are scaffolded with
  `abort("TODO")` bodies so everything compiles; then we fill them in a
  learning order, each with a known-answer test.
- **Correctness before speed.** CPU reference implementation first; GPU
  (CUDA via C FFI) is a later optimization with the CPU version as the test
  oracle.

## Road map

1. Start with a pretrained conventional LLM as base model (ch02)
   - load pretrained weights
   - text generation
   - detailed plan: [docs/roadmap/ch02.md](docs/roadmap/ch02.md)
2. Implement techniques to evaluate the performance of the LLMs with and
   without reasoning techniques (ch03)
   - benchmark-based evaluation
   - judgement-based evaluation
3. Implement reasoning-enhancing techniques that do not require training the
   LLM (ch04, ch05)
   - advanced text generation and voting
   - self-refinement
4. Implement techniques that improve reasoning via training (ch06 ~ ch08)
   - reinforcement learning
   - distillation

## References

- Download model: `~/.cache/models/Qwen/Qwen3-0.6B`

```sh
zw@wei:~/code/moonbit_programming/moonagent$ ll ~/.cache/models/Qwen/Qwen3-0.6B
total 1483604
drwxr-xr-x 2 zw zw       4096 Oct  8 21:18 ./
drwxr-xr-x 3 zw zw       4096 Oct  8 21:14 ../
-rw-r--r-- 1 zw zw        726 Oct  8 21:14 config.json
-rw-r--r-- 1 zw zw        239 Oct  8 21:14 generation_config.json
-rw-r--r-- 1 zw zw    1671853 Oct  8 21:16 merges.txt
-rw-r--r-- 1 zw zw 1503300328 Oct  8 21:57 model.safetensors
-rw-r--r-- 1 zw zw   11422654 Oct  8 21:15 tokenizer.json
-rw-r--r-- 1 zw zw       9732 Oct  8 21:15 tokenizer_config.json
-rw-r--r-- 1 zw zw    2776833 Oct  8 21:18 vocab.json
```

- `/home/zw/code/python_programming/reasoning-from-scratch`
  Python code for book: Build a Reasoning Model (From Scratch)
- Related blog articles

  - [mplementing A Byte Pair Encoding (BPE) Tokenizer From Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)
