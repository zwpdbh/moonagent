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
- **The Rust port is the oracle.** `~/code/rust_programming/learn_reasoning`
  is a working reference port; when behavior is ambiguous, match it
  byte-for-byte (token IDs, generated text).
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

## Workspace layout

```
moonagent/                  workspace root (moon.work)
├── utils/                  zw/utils — download_file, shared model cache dir
├── qwen3/                  zw/qwen3  — tokenizer, model, generation (ch02)
└── playground/             CLI + scratchpad: try MoonBit features here
    └── cmd/main            moon run playground/cmd/main --target native -- <cmd>
```

Useful commands (from the workspace root):

```bash
moon check --target native        # compile everything
moon test --target native         # run all tests
moon run playground/cmd/main --target native -- download <url>   # fetch a model file
moon ide peek-def --loc <file>:<line>:<col>   # go-to-definition in the terminal
```

## Current task — `Tokenizer::parse` (ch02, step 1)

File: `qwen3/tokenizer.mbt`. This spec tells you *exactly what to build* —
the implementation is yours to write.

### Background: what a tokenizer even is

LLMs don't read text; they read sequences of integer ids. The tokenizer is the
dictionary between the two: `encode("Explain large language models.")` →
`[840, 20772, 3460, 4128, 4119, 13]`, and `decode` goes back. Qwen3 uses
**byte-level BPE** (byte-pair encoding): a vocabulary of 151,936 slots (151,669 known tokens) where
common character sequences merge into single tokens, and any byte can always
be represented (so encoding *never* fails on weird input).

The dictionary ships as `tokenizer.json` (already at
`~/.cache/models/Qwen/Qwen3-0.6B/tokenizer.json`). Inspect it:

```bash
jq 'keys'                  ~/.cache/models/Qwen/Qwen3-0.6B/tokenizer.json
jq '.model | keys'         ~/.cache/models/Qwen/Qwen3-0.6B/tokenizer.json
jq '.model.vocab | length' ~/.cache/models/Qwen/Qwen3-0.6B/tokenizer.json   # 151643
```

### What `parse` must do

Signature (already scaffolded):

```moonbit
pub fn Tokenizer::parse(json : String) -> Tokenizer raise TokenizerError
```

Input: the *text contents* of `tokenizer.json` (no file I/O — that's `load`'s
job; keeping `parse` pure makes it unit-testable).

Steps:

1. **Parse the string into a `Json` value** with `@json.parse`.
   - Requires `"moonbitlang/core/json"` in `qwen3/moon.pkg` (the compiler warns
     `core_package_not_imported` until you add it — that's your reminder).
   - If the string is not valid JSON: `raise InvalidFormat(...)`. Convert the
     library's error into *your* typed error at the boundary:

     ```moonbit
     let parsed = @json.parse(json) catch {
       e => raise InvalidFormat(e.to_string())
     }
     ```

2. **Navigate to `model.vocab`.** `Json` is an enum; navigate by pattern
   matching. House style for "extract or fail" is `guard ... else raise`:

   ```moonbit
   guard parsed is Json::Object(root) else {
     raise InvalidFormat("top level must be a JSON object")
   }
   ```

   Chain the same pattern for the `"model"` key, then the `"vocab"` key inside
   it. Every missing key / wrong shape → `InvalidFormat` with a message saying
   *what* was missing.

3. **Build the vocab map.** `model.vocab` is a JSON object: token string →
   numeric id. Produce `Map[String, Int]`.
   - MoonBit notes: create with `Map([])` (a bare `{}` literal is ambiguous);
     iterate a JSON object with `for token, id in vocab_json`; each value must
     be `Json::Number` — guard it too.
   - Sanity check: `model.vocab` has **151,643** entries (the BPE-learned
     vocab, ids 0–151642). Note this is *not* `config.json`'s
     `vocab_size = 151936` — see step 4.

4. **Fold in `added_tokens`.** The top-level `added_tokens` array holds 26
   *special* tokens (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, …) that
   were registered after BPE training and are NOT in `model.vocab`. Each entry
   is `{"id": N, "content": "<|...|>", "special": true, ...}` with ids
   151643–151668. For each one: `vocab[content] = id` and `ids[id] = content`.
   (Encoding treats them specially — matched verbatim, bypassing BPE — but
   that's `encode`'s problem, later. `parse` just records them. If
   `added_tokens` is missing, treat it as an empty list, not an error.)

5. **Build the reverse index for `decode`**: an `Array[String]` where position
   `i` holds the token whose id is `i`. Size it to **151,936** (the model's
   padded `vocab_size`), fill from vocab + added_tokens; the unused tail
   (151669+) may stay empty strings — `decode` will `raise UnknownToken` for
   those anyway. (`encode` needs token→id, `decode` needs id→token — the JSON
   gives you each fact once; store both directions. Generation will *emit*
   `<|endoftext|>` (151643) as its stop signal, so decode must know it.)

6. **Return the struct**: `{ vocab, ids }` — struct literal syntax; fields are
   visible because you're inside the defining package. Outside, the struct
   stays opaque (that's the point of it not being `pub`).

### What `parse` must NOT do (yet)

- No `merges`, no `pre_tokenizer` — those belong to `encode`, later.
  Implementing them now means debugging more code before your first green test.
- No file I/O, no `async` — `parse` is pure.
- Don't suppress the remaining stub warnings by deleting the `raise`
  annotations; the signatures are the design.

### How to know you're done

Add an inline test at the bottom of `tokenizer.mbt` (inline tests see private
items and don't ship in release builds):

- A small hand-written snippet `{"model":{"vocab":{"a":0,"b":1,"!":2}}}`
  parses to a 3-entry map with the right ids.
- `"not json"` raises `InvalidFormat` (catch it in the test and assert).
- `{}` (missing `model`) raises `InvalidFormat`.

Then the real file: `Tokenizer::load` + `model.vocab` length = 151643, plus
26 added tokens = 151669 known ids (that's the next stub after `parse`).

### ML concept you're actually learning here

The vocab you parse **was itself produced by training** — BPE learned which
byte sequences are common in the corpus and merged them greedily. When you
later implement `encode`, you'll replay those merges in rank order. A
tokenizer is *bound to its model*: Qwen3's ids are meaningless to Llama. Our
`tokenizer.json` is only for Qwen3-0.6B (and its siblings that share the
tokenizer).
