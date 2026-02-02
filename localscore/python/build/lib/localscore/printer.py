"""Output formatters for LocalScore benchmark results."""

import csv
import io
import json
import sys
from abc import ABC, abstractmethod
from typing import TextIO, Optional, List

from .models import (
    CmdParams,
    BenchmarkResult,
    RuntimeInfo,
    SystemInfo,
    AcceleratorInfo,
    ModelInfo,
    ResultsSummary,
)
from .utils import format_time


class Printer(ABC):
    """Abstract base class for result printers."""

    def __init__(self):
        self.output: TextIO = sys.stdout
        self._string_output: Optional[io.StringIO] = None

    def set_file_output(self, f: TextIO) -> None:
        """Set output to a file."""
        self.output = f

    def set_string_output(self) -> io.StringIO:
        """Set output to a string buffer and return it."""
        self._string_output = io.StringIO()
        self.output = self._string_output
        return self._string_output

    def get_string_output(self) -> str:
        """Get the string output if using string buffer."""
        if self._string_output:
            return self._string_output.getvalue()
        return ""

    def write(self, text: str) -> None:
        """Write text to output."""
        self.output.write(text)
        self.output.flush()

    @abstractmethod
    def print_header(
        self,
        params: CmdParams,
        accelerator: AcceleratorInfo,
        runtime: RuntimeInfo,
        system: SystemInfo,
        model: ModelInfo,
    ) -> None:
        """Print header information."""
        pass

    @abstractmethod
    def print_test(self, result: BenchmarkResult) -> None:
        """Print a single test result."""
        pass

    @abstractmethod
    def print_footer(self) -> None:
        """Print footer."""
        pass


class CSVPrinter(Printer):
    """CSV format output printer."""

    FIELDS = [
        "build_commit", "build_number", "model_name", "model_quant_str",
        "model_params_str", "model_filename", "model_type", "model_size",
        "model_n_params", "n_prompt", "n_gen", "test_time", "avg_time_ms",
        "stddev_time_ms", "prompt_tps", "prompt_tps_watt", "prompt_tps_stddev",
        "gen_tps", "gen_tps_watt", "gen_tps_stddev", "name", "power_watts",
        "ttft_ms", "main_gpu",
    ]

    def __init__(self):
        super().__init__()
        self._writer: Optional[csv.writer] = None

    def _escape_csv(self, field: str) -> str:
        """Escape a CSV field."""
        return f'"{field.replace(chr(34), chr(34) + chr(34))}"'

    def print_header(
        self,
        params: CmdParams,
        accelerator: AcceleratorInfo,
        runtime: RuntimeInfo,
        system: SystemInfo,
        model: ModelInfo,
    ) -> None:
        """Print CSV header row."""
        self.write(",".join(self.FIELDS) + "\n")

    def print_test(self, result: BenchmarkResult) -> None:
        """Print a test result as CSV row."""
        values = [
            result.build_commit,
            str(result.build_number),
            result.model_name,
            result.model_quant_str,
            result.model_params_str,
            result.model_filename,
            result.model_type,
            str(result.model_size),
            str(result.model_n_params),
            str(result.n_prompt),
            str(result.n_gen),
            result.test_time,
            f"{result.avg_time_ms():.2f}",
            f"{result.stdev_time_ms():.2f}",
            f"{result.avg_prompt_tps():.2f}",
            f"{result.prompt_tps_watt():.4f}",
            f"{result.stdev_prompt_tps():.2f}",
            f"{result.avg_gen_tps():.2f}",
            f"{result.gen_tps_watt():.4f}",
            f"{result.stdev_gen_tps():.2f}",
            result.name,
            f"{result.power_watts:.2f}",
            f"{result.ttft_ms():.2f}",
            str(result.main_gpu),
        ]
        escaped = [self._escape_csv(v) for v in values]
        self.write(",".join(escaped) + "\n")

    def print_footer(self) -> None:
        """No footer for CSV."""
        pass


class JSONPrinter(Printer):
    """JSON format output printer."""

    def __init__(self):
        super().__init__()
        self._first_test = True
        self._data = {}

    def print_header(
        self,
        params: CmdParams,
        accelerator: AcceleratorInfo,
        runtime: RuntimeInfo,
        system: SystemInfo,
        model: ModelInfo,
    ) -> None:
        """Start JSON output with header info."""
        self._first_test = True
        self._data = {
            "runtime_info": {
                "name": runtime.name,
                "version": runtime.version,
                "commit": runtime.commit,
            },
            "system_info": {
                "cpu_name": system.cpu_name,
                "cpu_arch": system.cpu_arch,
                "ram_gb": system.ram_gb,
                "kernel_type": system.kernel_type,
                "kernel_release": system.kernel_release,
                "version": system.version,
            },
            "accelerator_info": {
                "name": accelerator.name,
                "manufacturer": accelerator.manufacturer,
                "memory_gb": accelerator.memory_gb,
                "type": accelerator.type,
            },
            "results": [],
        }

    def print_test(self, result: BenchmarkResult) -> None:
        """Add test result to JSON data."""
        test_data = {
            "build_commit": result.build_commit,
            "build_number": result.build_number,
            "model_name": result.model_name,
            "model_quant_str": result.model_quant_str,
            "model_params_str": result.model_params_str,
            "model_filename": result.model_filename,
            "model_type": result.model_type,
            "model_size": result.model_size,
            "model_n_params": result.model_n_params,
            "n_prompt": result.n_prompt,
            "n_gen": result.n_gen,
            "test_time": result.test_time,
            "avg_time_ms": result.avg_time_ms(),
            "stddev_time_ms": result.stdev_time_ms(),
            "prompt_tps": result.avg_prompt_tps(),
            "prompt_tps_watt": result.prompt_tps_watt(),
            "prompt_tps_stddev": result.stdev_prompt_tps(),
            "gen_tps": result.avg_gen_tps(),
            "gen_tps_watt": result.gen_tps_watt(),
            "gen_tps_stddev": result.stdev_gen_tps(),
            "name": result.name,
            "power_watts": result.power_watts,
            "ttft_ms": result.ttft_ms(),
            "main_gpu": result.main_gpu,
            "samples_ns": result.get_samples_ns(),
        }
        self._data["results"].append(test_data)

    def print_footer(self) -> None:
        """Write complete JSON output."""
        self.write(json.dumps(self._data, indent=2))

    def get_data(self) -> dict:
        """Get the JSON data dictionary."""
        return self._data


class ConsolePrinter(Printer):
    """Console/Markdown table format output printer."""

    FIELDS = ["test", "run number", "avg time", "tokens processed", "pp t/s", "tg t/s", "ttft"]

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color
        self._total_width = 0

    def _color(self, code: str) -> str:
        """Return ANSI color code if colors enabled."""
        return code if self.use_color else ""

    def _get_field_width(self, field: str) -> int:
        """Get display width for a field."""
        widths = {
            "test": 13,
            "run number": 10,
            "avg time": 12,
            "tokens processed": 18,
            "pp t/s": 10,
            "tg t/s": 10,
            "ttft": 12,
        }
        return widths.get(field, 10)

    def _calculate_total_width(self) -> int:
        """Calculate total table width."""
        total = 1  # Starting |
        for field in self.FIELDS:
            total += self._get_field_width(field) + 3  # " value |"
        return total

    def print_header(
        self,
        params: CmdParams,
        accelerator: AcceleratorInfo,
        runtime: RuntimeInfo,
        system: SystemInfo,
        model: ModelInfo,
    ) -> None:
        """Print console table header."""
        self._total_width = self._calculate_total_width()
        border = "+" + "-" * (self._total_width - 2) + "+"

        self.write(border + "\n")

        # GPU/Accelerator info line
        gpu_info = f"{accelerator.name} - {accelerator.memory_gb:.1f} GiB"
        padding = self._total_width - 2 - len(gpu_info)
        left_pad = padding // 2
        right_pad = padding - left_pad
        self.write("|" + " " * left_pad + gpu_info + " " * right_pad + "|\n")

        # Model info line
        model_info = f"{model.name} - {model.quant}" if model.name else model.filename
        padding = self._total_width - 2 - len(model_info)
        left_pad = padding // 2
        right_pad = padding - left_pad
        self.write("|" + " " * left_pad + model_info + " " * right_pad + "|\n")

        self.write(border + "\n")

        # Header row
        self.write("|")
        for field in self.FIELDS:
            width = self._get_field_width(field)
            self.write(f" {field:^{width}} |")
        self.write("\n")

        # Separator row
        self.write("|")
        for field in self.FIELDS:
            width = self._get_field_width(field)
            self.write(" " + "-" * width + " |")
        self.write("\n")

    def print_test(self, result: BenchmarkResult) -> None:
        """Print a test result row."""
        self.write("|")

        for field in self.FIELDS:
            width = self._get_field_width(field)
            value = self._get_field_value(field, result)
            self.write(f" {value:>{width}} |")

        self.write("\n")

    def _get_field_value(self, field: str, result: BenchmarkResult) -> str:
        """Get formatted value for a field."""
        if field == "test":
            return result.name
        elif field == "run number":
            return f"{result.reps}/{result.reps}"
        elif field == "avg time":
            return format_time(result.avg_time_ms())
        elif field == "tokens processed":
            total = (result.n_prompt + result.n_gen) * result.reps
            return f"{total} / {total}"
        elif field == "pp t/s":
            return f"{result.avg_prompt_tps():.2f}"
        elif field == "tg t/s":
            return f"{result.avg_gen_tps():.2f}"
        elif field == "ttft":
            ttft = result.ttft_ms()
            if ttft < 1000:
                return f"{ttft:.2f} ms"
            else:
                return f"{ttft / 1000:.2f} s"
        else:
            return ""

    def print_footer(self) -> None:
        """Print table footer."""
        border = "+" + "-" * (self._total_width - 2) + "+"
        self.write(border + "\n")


def display_results(summary: ResultsSummary, plaintext: bool = False) -> None:
    """Display the final benchmark results summary."""
    if plaintext:
        print(f"\nLocalScore: \t\t {int(summary.performance_score)}")
    else:
        print(f"\n\033[1;35mLocalScore: {int(summary.performance_score)}\033[0m")

    print(f"\033[32mToken Generation: \t \033[1;32m{summary.avg_gen_tps:.2f}\033[0m \033[3;32mtok/s\033[0m")
    print(f"\033[36mPrompt Processing: \t \033[1;36m{summary.avg_prompt_tps:.2f}\033[0m \033[3;36mtok/s\033[0m")
    print(f"\033[33mTime to First Token:\t \033[1;33m{summary.avg_ttft_ms:.2f}\033[0m \033[3;33mms\033[0m")


def get_printer(output_format: str, use_color: bool = True) -> Printer:
    """Get appropriate printer for output format."""
    if output_format == "csv":
        return CSVPrinter()
    elif output_format == "json":
        return JSONPrinter()
    else:
        return ConsolePrinter(use_color=use_color)
