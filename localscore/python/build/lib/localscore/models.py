"""Data models for LocalScore benchmark results."""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class OutputFormat(Enum):
    """Output format options."""
    CSV = "csv"
    JSON = "json"
    CONSOLE = "console"


class SendResultsMode(Enum):
    """Result submission mode."""
    ASK = "ask"
    YES = "yes"
    NO = "no"


@dataclass
class RuntimeInfo:
    """Runtime environment information."""
    name: str = "localscore-python"
    version: str = "1.0.0"
    commit: str = ""


@dataclass
class SystemInfo:
    """System hardware and OS information."""
    cpu_name: str = ""
    cpu_arch: str = ""
    ram_gb: float = 0.0
    kernel_type: str = ""
    kernel_release: str = ""
    version: str = ""


@dataclass
class AcceleratorInfo:
    """GPU/Accelerator information."""
    name: str = ""
    manufacturer: str = ""
    memory_gb: float = 0.0
    core_count: int = 0
    capability: float = 0.0
    type: str = "CPU"  # "GPU" or "CPU"


@dataclass
class ModelInfo:
    """Model metadata."""
    name: str = ""
    filename: str = ""
    type: str = ""
    quant: str = ""
    size_label: str = ""
    size: int = 0
    params: int = 0


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark test."""
    n_prompt: int
    n_gen: int

    @property
    def name(self) -> str:
        """Generate test name like 'pp1024+tg16'."""
        if self.n_prompt > 0 and self.n_gen == 0:
            return f"pp{self.n_prompt}"
        elif self.n_gen > 0 and self.n_prompt == 0:
            return f"tg{self.n_gen}"
        else:
            return f"pp{self.n_prompt}+tg{self.n_gen}"


@dataclass
class TimeInterval:
    """Time interval for benchmark measurements."""
    start_ns: int = 0
    end_ns: int = 0

    @property
    def duration_ns(self) -> int:
        """Get duration in nanoseconds."""
        if self.end_ns == 0:
            return 0
        return self.end_ns - self.start_ns

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        return self.duration_ns / 1e6


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test."""
    # Test configuration
    n_prompt: int = 0
    n_gen: int = 0
    reps: int = 1
    test_time: str = ""

    # Model info
    model_name: str = ""
    model_filename: str = ""
    model_type: str = ""
    model_quant_str: str = ""
    model_params_str: str = ""
    model_size: int = 0
    model_n_params: int = 0

    # Build info
    build_commit: str = ""
    build_number: int = 0

    # Timing data
    test_intervals: List[TimeInterval] = field(default_factory=list)
    prompt_intervals: List[TimeInterval] = field(default_factory=list)
    gen_intervals: List[TimeInterval] = field(default_factory=list)
    time_to_first_token_ns: List[int] = field(default_factory=list)

    # Power data
    power_watts: float = 0.0

    # GPU info
    main_gpu: int = 0

    @property
    def name(self) -> str:
        """Generate test name."""
        if self.n_prompt > 0 and self.n_gen == 0:
            return f"pp{self.n_prompt}"
        elif self.n_gen > 0 and self.n_prompt == 0:
            return f"tg{self.n_gen}"
        else:
            return f"pp{self.n_prompt}+tg{self.n_gen}"

    def avg_ns(self) -> float:
        """Average test duration in nanoseconds."""
        durations = [i.duration_ns for i in self.test_intervals if i.end_ns > 0]
        return sum(durations) / len(durations) if durations else 0.0

    def avg_time_ms(self) -> float:
        """Average test duration in milliseconds."""
        return self.avg_ns() / 1e6

    def stdev_ns(self) -> float:
        """Standard deviation of test duration in nanoseconds."""
        durations = [i.duration_ns for i in self.test_intervals if i.end_ns > 0]
        if len(durations) < 2:
            return 0.0
        mean = sum(durations) / len(durations)
        variance = sum((d - mean) ** 2 for d in durations) / len(durations)
        return variance ** 0.5

    def stdev_time_ms(self) -> float:
        """Standard deviation of test duration in milliseconds."""
        return self.stdev_ns() / 1e6

    def _get_tps(self, intervals: List[TimeInterval], n_tokens: int) -> List[float]:
        """Calculate tokens per second for each interval."""
        tps_list = []
        for interval in intervals:
            if interval.end_ns > 0 and interval.duration_ns > 0:
                tps = 1e9 * n_tokens / interval.duration_ns
                tps_list.append(tps)
        return tps_list

    def prompt_tps_list(self) -> List[float]:
        """Get list of prompt tokens/sec for each rep."""
        return self._get_tps(self.prompt_intervals, self.n_prompt)

    def gen_tps_list(self) -> List[float]:
        """Get list of generation tokens/sec for each rep."""
        return self._get_tps(self.gen_intervals, self.n_gen)

    def avg_prompt_tps(self) -> float:
        """Average prompt tokens per second."""
        tps = self.prompt_tps_list()
        return sum(tps) / len(tps) if tps else 0.0

    def avg_gen_tps(self) -> float:
        """Average generation tokens per second."""
        tps = self.gen_tps_list()
        return sum(tps) / len(tps) if tps else 0.0

    def stdev_prompt_tps(self) -> float:
        """Standard deviation of prompt tokens/sec."""
        tps = self.prompt_tps_list()
        if len(tps) < 2:
            return 0.0
        mean = sum(tps) / len(tps)
        variance = sum((t - mean) ** 2 for t in tps) / len(tps)
        return variance ** 0.5

    def stdev_gen_tps(self) -> float:
        """Standard deviation of generation tokens/sec."""
        tps = self.gen_tps_list()
        if len(tps) < 2:
            return 0.0
        mean = sum(tps) / len(tps)
        variance = sum((t - mean) ** 2 for t in tps) / len(tps)
        return variance ** 0.5

    def ttft_ms(self) -> float:
        """Average time to first token in milliseconds."""
        if not self.time_to_first_token_ns:
            return 0.0
        return (sum(self.time_to_first_token_ns) / len(self.time_to_first_token_ns)) / 1e6

    def prompt_tps_watt(self) -> float:
        """Prompt tokens per second per watt."""
        if self.power_watts <= 0:
            return 0.0
        return self.avg_prompt_tps() / self.power_watts

    def gen_tps_watt(self) -> float:
        """Generation tokens per second per watt."""
        if self.power_watts <= 0:
            return 0.0
        return self.avg_gen_tps() / self.power_watts

    def get_samples_ns(self) -> List[int]:
        """Get all test duration samples in nanoseconds."""
        return [i.duration_ns for i in self.test_intervals if i.end_ns > 0]


@dataclass
class ResultsSummary:
    """Summary of all benchmark results."""
    avg_prompt_tps: float = 0.0
    avg_gen_tps: float = 0.0
    avg_ttft_ms: float = 0.0
    performance_score: float = 0.0

    @classmethod
    def from_results(cls, results: List[BenchmarkResult]) -> "ResultsSummary":
        """Calculate summary from list of test results."""
        if not results:
            return cls()

        total_prompt_tps = 0.0
        total_gen_tps = 0.0
        total_ttft_ms = 0.0
        valid_count = 0

        for result in results:
            prompt_tps = result.avg_prompt_tps()
            gen_tps = result.avg_gen_tps()
            ttft = result.ttft_ms()

            if prompt_tps > 0 and gen_tps > 0 and ttft > 0:
                total_prompt_tps += prompt_tps
                total_gen_tps += gen_tps
                total_ttft_ms += ttft
                valid_count += 1

        if valid_count == 0:
            return cls()

        avg_prompt_tps = total_prompt_tps / valid_count
        avg_gen_tps = total_gen_tps / valid_count
        avg_ttft_ms = total_ttft_ms / valid_count

        # Performance score: geometric mean scaled by 10
        # score = 10 * (avg_prompt_tps * avg_gen_tps * (1000 / avg_ttft_ms)) ^ (1/3)
        performance_score = 10 * (avg_prompt_tps * avg_gen_tps * (1000 / avg_ttft_ms)) ** (1/3)

        return cls(
            avg_prompt_tps=avg_prompt_tps,
            avg_gen_tps=avg_gen_tps,
            avg_ttft_ms=avg_ttft_ms,
            performance_score=performance_score,
        )


@dataclass
class CmdParams:
    """Command-line parameters."""
    model: str = ""
    n_gpu_layers: int = 9999
    n_batch: int = 2048
    n_ubatch: int = 512
    n_threads: int = 0
    main_gpu: int = 0
    use_mmap: bool = True
    flash_attn: bool = False
    reps: int = 1
    verbose: bool = False
    plaintext: bool = False
    send_results: SendResultsMode = SendResultsMode.ASK
    output_format: OutputFormat = OutputFormat.CONSOLE
    gpu_backend: str = "auto"  # auto|amd|apple|nvidia|disabled
    list_gpus: bool = False


# Baseline test configurations matching C++ implementation
BASELINE_TESTS: List[BenchmarkConfig] = [
    BenchmarkConfig(1024, 16),    # 64:1 title generation
    BenchmarkConfig(4096, 256),   # 16:1 content summarization
    BenchmarkConfig(2048, 256),   # 8:1 code fix
    BenchmarkConfig(2048, 768),   # 3:1 standard code chat
    BenchmarkConfig(1024, 1024),  # 1:1 code back-and-forth
    BenchmarkConfig(1280, 3072),  # 1:3 reasoning over code
    BenchmarkConfig(384, 1152),   # 1:3 code gen with chat
    BenchmarkConfig(64, 1024),    # 1:16 code gen/ideation
    BenchmarkConfig(16, 1536),    # 1:96 QA, Storytelling
]
