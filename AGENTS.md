# AGENTS.md

This file provides guidance to the agent when working with code in this repository.

## What this is

A single Rust crate (`laya`) that runs the Laya non-autoregressive typed-decision model
(ModernBERT-large encoder + RL decision head) on candle. No Python, no torch, no ONNX.
One project, not a monorepo. The README is the authoritative user-facing doc; read it before
touching behavior.

## Commands

```sh
cargo build --release                          # CPU: library + laya CLI
cargo build --release --features serve         # also the laya-serve web console
cargo build --release --features metal         # Apple GPU
cargo build --release --features cuda          # NVIDIA
cargo test --release                           # all tests
cargo run --release --example exfil_triage -- models/laya-typed   # worked example
cargo run --release --features serve --bin laya-serve             # web console, :8080
cargo run --release --bin laya -- --state-file examples/ticket.txt --questions examples/questions.json
```

There is no CI, Makefile, or custom lint/format config. Default `rustfmt`/`clippy` apply;
`cargo fmt` and `cargo clippy` are the expected checks.

## Setup

The checkpoint is ~847 MB and is gitignored. Fetch it with the commands in the README into
`models/laya-base` (or point `--model` at any sibling variant directory). `models/` and
`target/` are ignored; do not commit them.

## Tests

- `cargo test --release` runs everything.
- `tests/rendering.rs` is model-free and runs anywhere.
- `tests/inference.rs` skips itself unless a checkpoint exists at `models/laya-base`, or at
  the path in `LAYA_MODEL_DIR`. Set that env var to run inference tests against another
  checkpoint.

## Gotchas

- **f16 needs an accelerator.** Candle has no f16 CPU kernels; `Agent::from_dir` bails if you
  ask for f16 on CPU. f16 also costs bit-exact agreement between a batched answer and the same
  question asked alone (they drift ~1e-3). f32 is the default and doubles resident memory
  (1.69 GB vs 0.84 GB) because weights are f16 on disk.
- **`src/modernbert.rs` is vendored** from candle-transformers 0.9.2 with one deliberate change:
  attention masks follow the model's dtype instead of hardcoded f32, so an f16 backbone runs.
  Do not re-vendor it blindly; that change is the reason f16 works at all.
- **Checkpoint key rename:** the backbone is stored under `encoder.*`; `Agent::from_dir`
  renames those to `model.*` before loading. Keep that mapping if you touch loading.
- **`head_max_len` is a shared budget.** Options share it; too many or too long options fail
  loudly with an explicit `head_max_len` error rather than silently dropping options. For more
  than ~20 options, raise both `max_len` and `head_max_len` or split into a coarse-to-fine pair.
- **A literal `[MASK]` in user text is scrubbed** to a space before encoding, because the head
  reads markers by position. Don't remove that scrub.
- **`serde_json` is built with `preserve_order`**, so `choice` option order follows the JSON
  author's order and is what probabilities are reported against. Don't sort criteria.
- **`rl_agent.act_probability` reads 1.0000 on the published checkpoints** — use the
  entropy-derived `confidence` instead.
- **Calibration is distribution-specific.** The bundled temperatures were fitted on the
  authors' data; refit per (question type, option count) bucket before trusting probabilities.
- **The base checkpoint is near chance** on the authors' typed-decisions benchmark; the
  fine-tuned `typed-decisions` variant is the sensible default for real decisions.
- **Question phrasing matters more than checkpoint choice:** give prose, not a struct; make
  options phrases, not enum tokens; ask about the world, not about a policy the model cannot see.
- The web console binds loopback only and has no auth — it is a dev tool, not a deployment.

## Repo etiquette

Commit directly to `main`; there is no PR workflow in this repo. Keep commits focused and
self-contained. The `serve` feature is opt-in (pulls axum + tokio); don't make it default.
