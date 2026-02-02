"""Command-line interface for LocalScore."""

import argparse
import os
import sys
from typing import List, Optional

from . import __version__
from .models import CmdParams, OutputFormat, SendResultsMode, ResultsSummary


def get_default_threads() -> int:
    """Get default number of threads (number of CPU cores)."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def parse_args(args: Optional[List[str]] = None) -> CmdParams:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="localscore",
        description="CLI benchmarking tool for measuring LLM inference performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        metavar="FILENAME",
        help="Model file to benchmark (required)",
    )

    parser.add_argument(
        "-c", "--cpu",
        action="store_true",
        help="Disable GPU acceleration (alias for --gpu=disabled)",
    )

    parser.add_argument(
        "-g", "--gpu",
        type=str,
        choices=["auto", "amd", "apple", "nvidia", "disabled"],
        default="auto",
        metavar="BACKEND",
        help="GPU backend: auto|amd|apple|nvidia|disabled (default: auto)",
    )

    parser.add_argument(
        "-i", "--gpu-index",
        type=int,
        default=0,
        metavar="INDEX",
        help="Select GPU by index (default: 0)",
    )

    parser.add_argument(
        "--list-gpus",
        action="store_true",
        help="List available GPUs and exit",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        choices=["csv", "json", "console"],
        default="console",
        metavar="FORMAT",
        help="Output format: csv|json|console (default: console)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--plaintext",
        action="store_true",
        help="Plaintext output (no colors or ASCII art)",
    )

    parser.add_argument(
        "-y", "--send-results",
        action="store_true",
        help="Send results without confirmation",
    )

    parser.add_argument(
        "-n", "--no-send-results",
        action="store_true",
        help="Disable sending results",
    )

    parser.add_argument(
        "-e", "--extended",
        action="store_true",
        help="Run 4 reps (shortcut for --reps=4)",
    )

    parser.add_argument(
        "--long",
        action="store_true",
        help="Run 16 reps (shortcut for --reps=16)",
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        metavar="N",
        help="Set custom number of repetitions (default: 1)",
    )

    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=9999,
        metavar="N",
        help="Number of layers to offload to GPU (default: 9999 = all)",
    )

    parser.add_argument(
        "--n-batch",
        type=int,
        default=2048,
        metavar="N",
        help="Batch size for prompt processing (default: 2048)",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=get_default_threads(),
        metavar="N",
        help=f"Number of threads (default: {get_default_threads()})",
    )

    parser.add_argument(
        "--no-mmap",
        action="store_true",
        help="Disable memory mapping",
    )

    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Enable flash attention",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parsed = parser.parse_args(args)

    # Validate arguments
    if parsed.list_gpus:
        # List GPUs mode - model not required
        pass
    elif not parsed.model:
        parser.error("the following arguments are required: -m/--model")

    if parsed.send_results and parsed.no_send_results:
        parser.error("cannot use both --send-results and --no-send-results")

    # Build CmdParams
    params = CmdParams()
    params.model = parsed.model or ""
    params.main_gpu = parsed.gpu_index
    params.n_batch = parsed.n_batch
    params.n_threads = parsed.threads
    params.use_mmap = not parsed.no_mmap
    params.flash_attn = parsed.flash_attn
    params.verbose = parsed.verbose
    params.plaintext = parsed.plaintext
    params.list_gpus = parsed.list_gpus

    # GPU backend
    if parsed.cpu:
        params.gpu_backend = "disabled"
        params.n_gpu_layers = 0
    else:
        params.gpu_backend = parsed.gpu
        params.n_gpu_layers = parsed.n_gpu_layers if parsed.gpu != "disabled" else 0

    # Output format
    params.output_format = OutputFormat(parsed.output)

    # Send results mode
    if parsed.send_results:
        params.send_results = SendResultsMode.YES
    elif parsed.no_send_results:
        params.send_results = SendResultsMode.NO
    else:
        params.send_results = SendResultsMode.ASK

    # Repetitions
    if parsed.long:
        params.reps = 16
    elif parsed.extended:
        params.reps = 4
    else:
        params.reps = max(1, parsed.reps)

    return params


def print_runtime_info(verbose: bool = False) -> None:
    """Print runtime information banner."""
    from .system import get_runtime_info
    runtime = get_runtime_info()

    print("\033[0;35m")
    print("=" * 70)
    print(f"{'LocalScore Runtime Information':^70}")
    print("=" * 70)
    print(f"\033[1m{'localscore version:':<20}\033[22m {runtime.version}")
    print(f"{'llama-cpp-python:':<20} (bundled)")
    print("=" * 70)
    print("\033[0m")


def print_system_info() -> None:
    """Print system information banner."""
    from .system import get_sys_info
    sys_info = get_sys_info()

    print("=" * 70)
    print(f"{'System Information':^70}")
    print("=" * 70)
    print(f"{'Kernel Type:':<20} {sys_info.kernel_type}")
    print(f"{'Kernel Release:':<20} {sys_info.kernel_release}")
    print(f"{'Version:':<20} {sys_info.version[:50]}...")
    print(f"{'System Architecture:':<20} {sys_info.cpu_arch}")
    print(f"{'CPU:':<20} {sys_info.cpu_name}")
    print(f"{'RAM:':<20} {sys_info.ram_gb:.1f} GiB")
    print("=" * 70)
    print()


def print_accelerator_info(params: CmdParams) -> None:
    """Print accelerator/GPU information banner."""
    from .system import detect_gpu_info
    accel = detect_gpu_info(params)

    color = "\033[0;32m" if accel.type == "GPU" else "\033[0;90m"
    print(color)
    print("=" * 70)
    if accel.type == "GPU":
        print(f"{'Active GPU Information':^70}")
    else:
        print(f"{'CPU Mode (No GPU Acceleration)':^70}")
    print("=" * 70)
    print(f"{'Name:':<26} {accel.name}")
    print(f"{'Manufacturer:':<26} {accel.manufacturer}")
    if accel.memory_gb > 0:
        label = "VRAM:" if accel.type == "GPU" else "RAM:"
        print(f"{label:<26} {accel.memory_gb:.1f} GiB")
    if accel.core_count > 0:
        print(f"{'Core Count:':<26} {accel.core_count}")
    if accel.capability > 0:
        print(f"{'Capability:':<26} {accel.capability:.1f}")
    print("=" * 70)
    print("\033[0m")
    print()


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for LocalScore CLI."""
    try:
        params = parse_args(args)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1

    # Handle --list-gpus (before importing heavy dependencies)
    if params.list_gpus:
        from .system import list_available_gpus
        list_available_gpus()
        return 0

    # Import heavy dependencies only when needed
    from .system import get_runtime_info, get_sys_info, detect_gpu_info
    from .benchmark import Benchmark
    from .printer import get_printer, display_results, JSONPrinter
    from .http import submit_results, prepare_submission_payload

    # Print banners
    if not params.plaintext:
        print_runtime_info(params.verbose)
        print_system_info()
        print_accelerator_info(params)

    # Create output printer
    use_color = not params.plaintext
    printer = get_printer(params.output_format.value, use_color)

    # Also create JSON printer for API submission
    json_printer = JSONPrinter()

    # Run benchmark
    benchmark = Benchmark(params)

    if not benchmark.load_model():
        return 1

    benchmark.warmup()

    # Get system info for headers
    runtime = get_runtime_info()
    sys_info = get_sys_info()
    accel = detect_gpu_info(params)

    # Print headers
    printer.print_header(params, accel, runtime, sys_info, benchmark.model_info)
    json_printer.print_header(params, accel, runtime, sys_info, benchmark.model_info)

    # Run tests
    from .models import BASELINE_TESTS
    from .power import get_power_sampler

    results = []
    sampler = get_power_sampler(100, params.main_gpu)

    for config in BASELINE_TESTS:
        result = benchmark.run_single_test(config, sampler)
        if result:
            results.append(result)
            printer.print_test(result)
            json_printer.print_test(result)

    # Print footers
    printer.print_footer()
    json_printer.print_footer()

    # Cleanup
    benchmark.cleanup()

    # Calculate summary
    summary = ResultsSummary.from_results(results)

    # Display results
    display_results(summary, params.plaintext)

    # Submit results if requested
    if params.send_results != SendResultsMode.NO:
        json_data = json_printer.get_data()
        summary_dict = {
            "avg_prompt_tps": summary.avg_prompt_tps,
            "avg_gen_tps": summary.avg_gen_tps,
            "avg_ttft_ms": summary.avg_ttft_ms,
            "performance_score": summary.performance_score,
        }
        payload = prepare_submission_payload(json_data, summary_dict)

        submit_results(
            payload,
            auto_submit=(params.send_results == SendResultsMode.YES),
            skip_submit=(params.send_results == SendResultsMode.NO),
            verbose=params.verbose,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
