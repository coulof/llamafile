# AGENT_INSTRUCTIONS.md

## Port `localstore` from C to Python (LocalScore / llamafile)

### 0) Context

This repository includes **LocalScore**, a CLI benchmarking tool for measuring how fast Large Language Models (LLMs) run on a given machine and (optionally) submitting results to a public database. The LocalScore CLI documents supported output formats (`csv|json|md`) and runtime options.

A C utility named **`localstore`** (as referred to by the project owner) exists in the repo and needs to be **rewritten in Python**.

> **Source of truth:** the existing C implementation (and any data it produces/consumes). Do not guess behavior—derive it from the code.

---

### 1) Goal (definition of done)

Implement a Python replacement for the C `localstore` utility such that:

1. **Feature parity** with the C version
   - Same subcommands/flags, defaults, and exit codes.
   - Same storage format and on-disk layout (unless explicitly approved to change).

2. **Compatibility** with LocalScore artifacts
   - If `localstore` reads/writes benchmark outputs, it must remain compatible with LocalScore’s documented formats and conventions.

3. **Maintainability**
   - Clear module structure, minimal dependencies, tests, and short docs.

---

### 2) Non-goals

- Do **not** alter LocalScore’s benchmarking/scoring logic.
- Do **not** redesign the database schema or output format.
- Do **not** require network access unless the C version already does.
- Do **not** introduce heavy dependencies without a strong reason.

---

### 3) First actions (must be done before coding)

1. **Locate the C utility**
   - Search the repo for: `localstore`, `store`, `db`, `schema`, `sqlite`, `results`, `.llamafile`, `localscore`.
   - Identify:
     - Entry point (`main`) and CLI parsing.
     - Storage backend (SQLite vs filesystem vs custom).
     - Data model and schema (tables/fields/indexes) or file formats.
     - Default paths and environment variables.

2. **Write a short behavior inventory** (1–2 pages)
   - Commands/subcommands and options.
   - Inputs/outputs.
   - Default file locations and resolution rules.
   - Error handling + exit codes.
   - Any concurrency/locking behaviors.

3. Commit that inventory as:
   - `localscore/doc/localstore_port_notes.md` (or similar location consistent with repo structure).

---

### 4) Constraints

- **Python**: target Python **3.11+** unless the repo dictates otherwise.
- **Cross-platform**: must work on Linux/macOS/Windows (same general platforms as LocalScore).
- **Dependencies**: prefer stdlib:
  - `argparse`, `json`, `csv`, `sqlite3`, `pathlib`, `os`, `platform`, `hashlib`, `datetime`, `logging`, `tempfile`, `subprocess`.
  - If third-party libs are added, keep them optional and justify.
- **Determinism**: identical inputs → identical outputs.

---

### 5) Recommended Python project layout

Add a small module that supports both script execution and import:

```
localscore/
  tools/
    localstore/
      __init__.py
      cli.py          # argparse/entrypoint
      store.py        # backend operations
      schema.py       # (if SQLite) schema + migrations
      formats.py      # json/csv helpers and normalization
      paths.py        # default path resolution
      errors.py       # typed exceptions -> exit codes
  tests/
    test_localstore_cli.py
    test_localstore_store.py
    test_localstore_migrations.py  # if applicable
```

Entrypoint choices (pick the one that matches repo conventions):

- `python -m localscore.tools.localstore ...`
- a thin wrapper script named `localstore` placed where the build expects it.

---

### 6) Functional requirements (derive specifics from C)

#### 6.1 CLI parity

- Match C utility’s flags/subcommands and help output as closely as practical.
- Preserve default values and behaviors.
- Preserve exit codes:
  - `0`: success
  - nonzero: failure (map each C failure case to a stable Python exit code).

If the C tool supports `--help`, `--version`, `--verbose`, etc., implement them.

#### 6.2 Storage backend parity

After you inspect the C implementation, implement the same backend:

**If SQLite**
- Use stdlib `sqlite3`.
- Keep schema identical:
  - table/column names
  - indexes
  - constraints
- Use transactions.
- If migrations exist in C, replicate them.

**If filesystem store** (JSON files, CSV files, etc.)
- Keep directory layout and naming identical.
- Writes must be atomic:
  - write to temp file in same dir
  - fsync
  - rename/replace

#### 6.3 Data format compatibility

- Ensure any data written by Python can be read by existing consumers (and vice versa).
- If it touches LocalScore output formats:
  - LocalScore supports `csv|json|md`.
  - Parse flexibly (whitespace, ordering, BOM) but emit canonically.

#### 6.4 Logging and verbosity

- Implement verbosity flags consistent with the C tool.
- Default should mirror current behavior (quiet vs progress output).
- Use `logging` module; avoid ad-hoc prints except for user-visible output.

#### 6.5 Path handling

- Replicate C tool’s default store location(s).
- Use `pathlib.Path`.
- Respect environment overrides (if any exist in C).
- Avoid hardcoded path separators.

#### 6.6 Concurrency and locking

- If the C tool uses file locks or SQLite locking patterns, replicate them.
- If the store might be written concurrently:
  - SQLite: rely on transactions and consider `busy_timeout`.
  - Filesystem: lock file or atomic rename strategy.

---

### 7) Testing requirements

Tests must run in CI with **no GPU** and **no network**.

Minimum coverage:

1. **CLI smoke tests**
   - `--help` returns `0` and includes expected keywords.
   - invalid flags return nonzero.

2. **Read/write roundtrip**
   - Create temp store
   - Insert record
   - Retrieve record
   - Assert equality

3. **Migration tests** (if applicable)
   - Create an “old” store fixture (or build it via SQL)
   - Run migration
   - Validate schema and that records survive.

4. **Golden tests**
   - Add fixtures where output text must match exactly (stable fields only).

**If feasible**: provide a local developer script that runs both C and Python implementations on the same fixtures and diffs outputs.

---

### 8) Performance and correctness requirements

- Avoid unnecessary overhead:
  - SQLite: prepared statements, batch writes inside one transaction.
  - JSON: streaming or incremental where possible.

- Ensure atomicity:
  - file store: temp + fsync + rename.
  - sqlite: transactions; explicit commit/rollback.

- Make error messages actionable; keep them compatible with any C tooling expectations.

---

### 9) Deliverables

1. Python implementation of `localstore` with CLI parity.
2. Documentation:
   - `localstore_port_notes.md` (behavior inventory + differences)
   - `README_localstore.md` (how to run, examples)
3. Test suite.
4. Optional: thin wrapper preserving old invocation name/path if build scripts require it.

---

### 10) Acceptance criteria

Work is complete when:

- The Python tool produces the **same observable behavior** as the C tool for representative inputs.
- All tests pass.
- No regressions to LocalScore workflows or formats.
- Any intentional deviations are documented with rationale.

---

### 11) Suggested step-by-step implementation plan

1. Inspect C sources and write behavior inventory doc.
2. Implement backend layer (`store.py`) and unit tests.
3. Implement CLI (`cli.py`) and integration tests.
4. Add golden fixtures; optionally add C-vs-Python conformance runner.
5. Wire into repo build/run conventions.
6. Final pass: docs + cleanup + lint/format.

---

### 12) Notes for the agent

- If you cannot find `localstore` in the repo, stop and report:
  - the search paths you used
  - candidate files that look related
  - what you suspect `localstore` might have been called instead

- Do not introduce behavior changes “because Python makes it easier.” Preserve existing behavior unless explicitly instructed.

