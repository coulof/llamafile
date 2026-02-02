# LocalScore Python Port - Behavior Inventory

## Overview

This document describes the behavior of the original C++ LocalScore implementation and the Python port.

## Source Files (C++ Original)

| File | Lines | Purpose |
|------|-------|---------|
| `localscore/main.cpp` | 5 | Entry point |
| `localscore/localscore.cpp` | 566 | Main CLI logic |
| `localscore/cmd.cpp` | 205 | CLI argument parsing |
| `localscore/benchmark.cpp` | 363 | Benchmark execution |
| `localscore/system.cpp` | 323 | System info collection |
| `localscore/printer.cpp` | 410 | Output formatting |

## CLI Commands and Options

### Commands/Subcommands

LocalScore is a single-command CLI (no subcommands).

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-h, --help` | flag | - | Print help and exit |
| `-m, --model <file>` | string | required | Model file to benchmark |
| `-c, --cpu` | flag | - | Disable GPU (alias for --gpu=disabled) |
| `-g, --gpu <backend>` | enum | auto | GPU backend: auto\|amd\|apple\|nvidia\|disabled |
| `-i, --gpu-index <i>` | int | 0 | Select GPU by index |
| `--list-gpus` | flag | - | List available GPUs and exit |
| `-o, --output <fmt>` | enum | console | Output format: csv\|json\|console |
| `-v, --verbose` | flag | false | Verbose output |
| `--plaintext` | flag | false | Plaintext output (no ANSI colors) |
| `-y, --send-results` | flag | - | Auto-send results without confirmation |
| `-n, --no-send-results` | flag | - | Don't send results |
| `-e, --extended` | flag | - | Run 4 repetitions (shortcut for --reps=4) |
| `--long` | flag | - | Run 16 repetitions (shortcut for --reps=16) |
| `--reps <N>` | int | 1 | Custom repetition count |

### Hidden/Internal Options (C++ only)

| Flag | Description |
|------|-------------|
| `--recompile` | Force recompilation |
| `--localscore` | No-op (accepted but ignored) |

## Inputs/Outputs

### Input Files

- **Model file** (required): GGUF format model file specified via `-m`

### Output Formats

#### Console (default)
```
+----------------------------------------------------------------------+
|                    GPU Name - X.X GiB                                |
|                    Model Name - Q4_K_M                               |
+----------------------------------------------------------------------+
|     test      | run number |   avg time   | tokens processed | pp t/s | tg t/s |     ttft     |
| ------------- | ---------- | ------------ | ---------------- | ------ | ------ | ------------ |
| pp1024+tg16   |    1/1     |   123.45 ms  |    1040 / 1040   | 123.45 |  67.89 |   45.67 ms   |
...
+----------------------------------------------------------------------+
```

#### JSON
```json
{
  "runtime_info": {
    "name": "localscore-python",
    "version": "1.0.0",
    "commit": ""
  },
  "system_info": {
    "cpu_name": "...",
    "cpu_arch": "x86_64",
    "ram_gb": 32.0,
    "kernel_type": "Linux",
    "kernel_release": "...",
    "version": "..."
  },
  "accelerator_info": {
    "name": "NVIDIA GeForce RTX 4090",
    "manufacturer": "NVIDIA",
    "memory_gb": 24.0,
    "type": "GPU"
  },
  "results": [
    {
      "n_prompt": 1024,
      "n_gen": 16,
      "prompt_tps": 1234.56,
      "gen_tps": 78.90,
      "ttft_ms": 45.67,
      ...
    }
  ],
  "results_summary": {
    "avg_prompt_tps": 1000.0,
    "avg_gen_tps": 75.0,
    "avg_ttft_ms": 50.0,
    "performance_score": 890
  }
}
```

#### CSV
Standard CSV with headers matching JSON field names.

## Default File Locations

### Database (C++ only - for chat history)
```
1. FLAG_db (command-line override)
2. $HOME/.llamafile/llamafile.sqlite3
3. ./llamafile.sqlite3 (fallback)
```

### Python Port
No local database - results are output to stdout and optionally submitted to API.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid argument, missing model, load failure) |

## Error Handling

### C++ Implementation
- Invalid parameter: prints error to stderr, exits 1
- Missing model: prints error to stderr, exits 1
- Model load failure: prints error to stderr, exits 1
- API submission failure: prints error, continues (non-fatal)

### Python Implementation
- Same behavior replicated

## Baseline Test Configurations

9 tests with varying prompt/generation ratios:

| n_prompt | n_gen | Ratio | Use Case |
|----------|-------|-------|----------|
| 1024 | 16 | 64:1 | Title generation |
| 4096 | 256 | 16:1 | Content summarization |
| 2048 | 256 | 8:1 | Code fix |
| 2048 | 768 | 3:1 | Standard code chat |
| 1024 | 1024 | 1:1 | Code back-and-forth |
| 1280 | 3072 | 1:3 | Reasoning over code |
| 384 | 1152 | 1:3 | Code gen with chat |
| 64 | 1024 | 1:16 | Code gen/ideation |
| 16 | 1536 | 1:96 | QA, Storytelling |

## Performance Score Formula

```
score = 10 × ∛(avg_prompt_tps × avg_gen_tps × (1000 / avg_ttft_ms))
```

Where:
- `avg_prompt_tps`: Average prompt tokens per second across all tests
- `avg_gen_tps`: Average generation tokens per second across all tests
- `avg_ttft_ms`: Average time to first token in milliseconds

## API Submission

### Endpoint
```
POST https://www.localscore.ai/api/results
Content-Type: application/json
```

### Retry Logic
- Maximum 3 attempts
- Exponential backoff: 2^attempt seconds between retries

### Response
```json
{"id": 12345}
```

### Result URL
```
https://www.localscore.ai/result/{id}
```

## Concurrency/Locking

### C++ Implementation
- Uses `pthread_setcancelstate()` to disable cancellation during SQLite operations
- SQLite with WAL mode for concurrent access

### Python Implementation
- Power sampling runs in a separate daemon thread
- No database locking needed (no local database)

## Differences Between C++ and Python Implementations

### Intentional Differences

1. **No local database**: Python version doesn't use SQLite for local storage
2. **llama-cpp-python**: Uses Python bindings instead of direct C++ llama.cpp
3. **Runtime name**: Reports as "localscore-python" instead of "llamafile"
4. **No chat history**: Database schema for chats/messages not implemented

### Behavioral Parity

1. **CLI flags**: All flags from C++ are implemented
2. **Output formats**: All three formats (console, json, csv) supported
3. **Baseline tests**: Same 9 test configurations
4. **Score formula**: Identical calculation
5. **API submission**: Same endpoint, retry logic, and payload format

## Testing

### CLI Smoke Tests
```bash
python -m localscore --help        # Should exit 0, print help
python -m localscore --version     # Should exit 0, print version
python -m localscore --list-gpus   # Should exit 0, list GPUs
```

### Benchmark Test
```bash
python -m localscore -m model.gguf --reps 1 -n  # Quick test, no submission
```

### Output Format Tests
```bash
python -m localscore -m model.gguf -o json -n
python -m localscore -m model.gguf -o csv -n
```

### Unit Tests
```bash
cd localscore/python
pip install -e ".[dev]"
pytest tests/ -v
```

**Test Status**: 50 passed, 1 skipped (skipped test requires llama-cpp-python with a model)

## Packaging and Distribution

### Building a Wheel

```bash
cd localscore/python
pip install build
python -m build --wheel
```

Creates `dist/localscore-1.0.0-py3-none-any.whl`.

### Installing on Another Machine

```bash
# Basic install
pip install localscore-1.0.0-py3-none-any.whl

# With NVIDIA power monitoring support
pip install "localscore-1.0.0-py3-none-any.whl[nvidia]"
```

### GPU Support

The target machine needs `llama-cpp-python` compiled with GPU support:

```bash
# NVIDIA CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# AMD ROCm
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python

# Apple Metal (macOS)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

## Implementation Status

| Component | Status |
|-----------|--------|
| CLI argument parsing | ✅ Complete |
| System info detection | ✅ Complete |
| GPU detection (NVIDIA/AMD/Apple) | ✅ Complete |
| Benchmark execution | ✅ Complete |
| Console output | ✅ Complete |
| JSON output | ✅ Complete |
| CSV output | ✅ Complete |
| API submission | ✅ Complete |
| Power monitoring (NVIDIA) | ✅ Complete |
| Unit tests | ✅ 50 tests passing |
| Packaging | ✅ Wheel build ready |
