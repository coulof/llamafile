# LocalScore Python

A Python implementation of the LocalScore CLI benchmarking tool for measuring LLM inference performance.

## Installation

```bash
# Install from source
cd localscore/python
pip install -e .

# Or install with NVIDIA power monitoring support
pip install -e ".[nvidia]"

# Or install with development dependencies
pip install -e ".[dev]"
```

## Requirements

- Python 3.11+
- llama-cpp-python
- psutil
- requests
- pynvml (optional, for NVIDIA power monitoring)

## Usage

```bash
# Run benchmark on a model
python -m localscore -m path/to/model.gguf

# Run with specific options
python -m localscore -m model.gguf --reps 4 --output json

# Run in CPU-only mode
python -m localscore -m model.gguf --cpu

# List available GPUs
python -m localscore --list-gpus

# Auto-submit results without confirmation
python -m localscore -m model.gguf -y

# Skip result submission
python -m localscore -m model.gguf -n
```

## Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | Model file to benchmark (required) | - |
| `-c, --cpu` | Disable GPU acceleration | - |
| `-g, --gpu` | GPU backend: auto\|amd\|apple\|nvidia\|disabled | auto |
| `-i, --gpu-index` | Select GPU by index | 0 |
| `--list-gpus` | List available GPUs and exit | - |
| `-o, --output` | Output format: csv\|json\|console | console |
| `-v, --verbose` | Verbose output | off |
| `--plaintext` | Plaintext output (no colors) | off |
| `-y, --send-results` | Auto-send results | - |
| `-n, --no-send-results` | Don't send results | - |
| `-e, --extended` | Run 4 repetitions | - |
| `--long` | Run 16 repetitions | - |
| `--reps N` | Custom repetition count | 1 |
| `--n-gpu-layers N` | GPU layers to offload | 9999 |
| `--n-batch N` | Batch size | 2048 |
| `--threads N` | Number of threads | CPU count |
| `--flash-attn` | Enable flash attention | off |

## Benchmark Tests

LocalScore runs 9 baseline tests with different prompt/generation token ratios:

| Prompt Tokens | Gen Tokens | Ratio | Use Case |
|---------------|------------|-------|----------|
| 1024 | 16 | 64:1 | Title generation |
| 4096 | 256 | 16:1 | Content summarization |
| 2048 | 256 | 8:1 | Code fix |
| 2048 | 768 | 3:1 | Standard code chat |
| 1024 | 1024 | 1:1 | Code back-and-forth |
| 1280 | 3072 | 1:3 | Reasoning over code |
| 384 | 1152 | 1:3 | Code gen with chat |
| 64 | 1024 | 1:16 | Code gen/ideation |
| 16 | 1536 | 1:96 | QA, Storytelling |

## Performance Score

The performance score is calculated as:

```
score = 10 × ∛(avg_prompt_tps × avg_gen_tps × (1000 / avg_ttft_ms))
```

Where:
- `avg_prompt_tps`: Average prompt tokens per second
- `avg_gen_tps`: Average generation tokens per second
- `avg_ttft_ms`: Average time to first token in milliseconds

## Output Formats

### Console (default)

Displays a formatted table with real-time progress.

### JSON

```json
{
  "runtime_info": { ... },
  "system_info": { ... },
  "accelerator_info": { ... },
  "results": [ ... ],
  "results_summary": {
    "avg_prompt_tps": 123.45,
    "avg_gen_tps": 67.89,
    "avg_ttft_ms": 45.67,
    "performance_score": 890
  }
}
```

### CSV

Standard CSV format with headers matching the JSON field names.

## API Submission

Results can be submitted to https://localscore.ai for public benchmarking comparisons.

- Use `-y` to auto-submit without confirmation
- Use `-n` to skip submission entirely
- By default, you'll be prompted for confirmation

## Building a Distributable Package

To build a wheel for installation on another machine:

```bash
cd localscore/python
pip install build
python -m build --wheel
```

This creates `dist/localscore-1.0.0-py3-none-any.whl`.

**To install on another machine:**

```bash
# Basic install
pip install localscore-1.0.0-py3-none-any.whl

# With NVIDIA power monitoring support
pip install "localscore-1.0.0-py3-none-any.whl[nvidia]"
```

**Note:** The target machine needs `llama-cpp-python` with GPU support. For NVIDIA GPUs:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

## License

Apache-2.0
