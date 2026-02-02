"""Benchmark execution for LocalScore using llama-cpp-python."""

import random
import sys
from pathlib import Path
from typing import Optional, Callable

from llama_cpp import Llama

from .models import (
    CmdParams,
    BenchmarkConfig,
    BenchmarkResult,
    TimeInterval,
    ModelInfo,
    BASELINE_TESTS,
)
from .power import PowerSampler, get_power_sampler
from .utils import get_time_ns, get_rfc3339_timestamp


class Benchmark:
    """Benchmark runner for LLM inference performance."""

    def __init__(
        self,
        params: CmdParams,
        on_test_update: Optional[Callable[[BenchmarkResult], None]] = None,
    ):
        self.params = params
        self.on_test_update = on_test_update
        self.model: Optional[Llama] = None
        self.model_info = ModelInfo()

    def load_model(self) -> bool:
        """Load the model for benchmarking."""
        print("Loading model... ", end="", flush=True)

        try:
            # Determine GPU layers
            n_gpu_layers = self.params.n_gpu_layers
            if self.params.gpu_backend == "disabled":
                n_gpu_layers = 0

            # Load model with initial context size for warmup
            self.model = Llama(
                model_path=self.params.model,
                n_gpu_layers=n_gpu_layers,
                n_ctx=1024 + 16,  # Initial context for warmup
                n_batch=self.params.n_batch,
                use_mmap=self.params.use_mmap,
                flash_attn=self.params.flash_attn,
                verbose=self.params.verbose,
            )

            # Extract model info
            self._extract_model_info()

            print("Model loaded.")
            return True

        except Exception as e:
            print(f"\nError: Failed to load model: {e}", file=sys.stderr)
            return False

    def _extract_model_info(self) -> None:
        """Extract model information from loaded model."""
        if not self.model:
            return

        self.model_info.filename = Path(self.params.model).name

        # Try to get metadata
        try:
            metadata = self.model.metadata or {}
            self.model_info.name = metadata.get("general.name", "")
            self.model_info.size_label = metadata.get("general.size_label", "")

            # Get quantization type from filename or metadata
            filename = self.model_info.filename.lower()
            quant_types = ["q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q6_k", "q8_0", "f16", "f32"]
            for qt in quant_types:
                if qt in filename:
                    self.model_info.quant = qt.upper()
                    break

        except Exception:
            pass

        # Get model size from file
        try:
            self.model_info.size = Path(self.params.model).stat().st_size
        except Exception:
            pass

    def warmup(self) -> None:
        """Run warmup to initialize GPU and caches."""
        if not self.model:
            return

        print("Warming up...... ", end="", flush=True)

        n_prompt = 1024
        n_gen = 16
        n_vocab = self.model.n_vocab()

        try:
            # Reset context
            self.model.reset()

            # Warmup prompt processing
            tokens = [random.randint(0, n_vocab - 1) for _ in range(n_prompt)]
            # Use BOS token for first position if available
            if hasattr(self.model, "token_bos"):
                tokens[0] = self.model.token_bos()

            self.model.eval(tokens)

            # Warmup token generation
            for _ in range(n_gen):
                # Sample and evaluate
                token = random.randint(0, n_vocab - 1)
                self.model.eval([token])

            print("Warmup complete.\n")

        except Exception as e:
            print(f"Warmup failed: {e}", file=sys.stderr)

    def run_all_tests(self) -> list[BenchmarkResult]:
        """Run all baseline benchmark tests."""
        results = []

        sampler = get_power_sampler(100, self.params.main_gpu)

        for test_config in BASELINE_TESTS:
            result = self.run_single_test(test_config, sampler)
            if result:
                results.append(result)
                if self.on_test_update:
                    self.on_test_update(result)

        return results

    def run_single_test(
        self,
        config: BenchmarkConfig,
        sampler: PowerSampler,
    ) -> Optional[BenchmarkResult]:
        """Run a single benchmark test configuration."""
        if not self.model:
            return None

        n_prompt = config.n_prompt
        n_gen = config.n_gen
        reps = self.params.reps

        result = BenchmarkResult(
            n_prompt=n_prompt,
            n_gen=n_gen,
            reps=reps,
            test_time=get_rfc3339_timestamp(),
            model_name=self.model_info.name,
            model_filename=self.model_info.filename,
            model_type=self.model_info.type,
            model_quant_str=self.model_info.quant,
            model_params_str=self.model_info.size_label,
            model_size=self.model_info.size,
            model_n_params=self.model_info.params,
            main_gpu=self.params.main_gpu,
        )

        try:
            # Recreate context with appropriate size
            n_ctx = n_prompt + n_gen
            self.model = Llama(
                model_path=self.params.model,
                n_gpu_layers=self.params.n_gpu_layers if self.params.gpu_backend != "disabled" else 0,
                n_ctx=n_ctx,
                n_batch=self.params.n_batch,
                use_mmap=self.params.use_mmap,
                flash_attn=self.params.flash_attn,
                verbose=self.params.verbose,
            )
        except Exception as e:
            print(f"Error creating context: {e}", file=sys.stderr)
            return None

        n_vocab = self.model.n_vocab()

        # Start power sampling
        sampler.start()

        for rep in range(reps):
            # Reset for each repetition
            self.model.reset()

            # Overall test interval
            test_interval = TimeInterval(start_ns=get_time_ns())
            result.test_intervals.append(test_interval)

            # Prompt processing
            if n_prompt > 0:
                prompt_interval = TimeInterval(start_ns=get_time_ns())

                tokens = [random.randint(0, n_vocab - 1) for _ in range(n_prompt)]
                if hasattr(self.model, "token_bos"):
                    tokens[0] = self.model.token_bos()

                # Process in batches
                batch_size = self.params.n_batch
                for i in range(0, n_prompt, batch_size):
                    batch = tokens[i:i + batch_size]
                    self.model.eval(batch)

                prompt_interval.end_ns = get_time_ns()
                result.prompt_intervals.append(prompt_interval)

            # Token generation
            if n_gen > 0:
                gen_interval = TimeInterval(start_ns=get_time_ns())

                for i in range(n_gen):
                    # Sample next token (simple random for benchmarking)
                    token = random.randint(0, n_vocab - 1)
                    self.model.eval([token])

                    # Record time to first token
                    if i == 0:
                        ttft_ns = get_time_ns() - test_interval.start_ns
                        result.time_to_first_token_ns.append(ttft_ns)

                gen_interval.end_ns = get_time_ns()
                result.gen_intervals.append(gen_interval)

            # Complete test interval
            test_interval.end_ns = get_time_ns()

        # Stop power sampling
        power_result = sampler.stop()
        result.power_watts = power_result.power_watts

        return result

    def cleanup(self) -> None:
        """Clean up resources."""
        self.model = None


def run_benchmark(params: CmdParams, printer) -> list[BenchmarkResult]:
    """Run the complete benchmark suite."""
    benchmark = Benchmark(
        params=params,
        on_test_update=lambda r: printer.print_test(r) if printer else None,
    )

    # Load model
    if not benchmark.load_model():
        return []

    # Run warmup
    benchmark.warmup()

    # Print header
    if printer:
        from .system import get_runtime_info, get_sys_info, detect_gpu_info
        printer.print_header(
            params,
            detect_gpu_info(params),
            get_runtime_info(),
            get_sys_info(),
            benchmark.model_info,
        )

    # Run tests
    results = benchmark.run_all_tests()

    # Print footer
    if printer:
        printer.print_footer()

    # Cleanup
    benchmark.cleanup()

    return results
