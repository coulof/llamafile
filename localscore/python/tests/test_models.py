"""Tests for LocalScore data models."""

import pytest

from localscore.models import (
    BenchmarkConfig,
    TimeInterval,
    BenchmarkResult,
    ResultsSummary,
    BASELINE_TESTS,
)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_name_prompt_only(self):
        """Test name for prompt-only test."""
        config = BenchmarkConfig(n_prompt=1024, n_gen=0)
        assert config.name == "pp1024"

    def test_name_gen_only(self):
        """Test name for generation-only test."""
        config = BenchmarkConfig(n_prompt=0, n_gen=256)
        assert config.name == "tg256"

    def test_name_mixed(self):
        """Test name for mixed test."""
        config = BenchmarkConfig(n_prompt=1024, n_gen=16)
        assert config.name == "pp1024+tg16"


class TestTimeInterval:
    """Tests for TimeInterval."""

    def test_duration_ns(self):
        """Test duration calculation in nanoseconds."""
        interval = TimeInterval(start_ns=1000, end_ns=5000)
        assert interval.duration_ns == 4000

    def test_duration_ns_incomplete(self):
        """Test duration for incomplete interval."""
        interval = TimeInterval(start_ns=1000, end_ns=0)
        assert interval.duration_ns == 0

    def test_duration_ms(self):
        """Test duration calculation in milliseconds."""
        interval = TimeInterval(start_ns=0, end_ns=1_000_000)
        assert interval.duration_ms == 1.0


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_name_property(self):
        """Test name property."""
        result = BenchmarkResult(n_prompt=1024, n_gen=16)
        assert result.name == "pp1024+tg16"

    def test_avg_time_ms(self):
        """Test average time calculation."""
        result = BenchmarkResult(n_prompt=1024, n_gen=16)
        result.test_intervals = [
            TimeInterval(start_ns=0, end_ns=1_000_000),
            TimeInterval(start_ns=0, end_ns=2_000_000),
        ]
        assert result.avg_time_ms() == 1.5

    def test_avg_time_ms_empty(self):
        """Test average time with no intervals."""
        result = BenchmarkResult()
        assert result.avg_time_ms() == 0.0

    def test_prompt_tps(self):
        """Test prompt tokens per second calculation."""
        result = BenchmarkResult(n_prompt=1000, n_gen=0)
        result.prompt_intervals = [
            TimeInterval(start_ns=0, end_ns=1_000_000_000),  # 1 second
        ]
        assert result.avg_prompt_tps() == 1000.0

    def test_gen_tps(self):
        """Test generation tokens per second calculation."""
        result = BenchmarkResult(n_prompt=0, n_gen=100)
        result.gen_intervals = [
            TimeInterval(start_ns=0, end_ns=1_000_000_000),  # 1 second
        ]
        assert result.avg_gen_tps() == 100.0

    def test_ttft_ms(self):
        """Test time to first token calculation."""
        result = BenchmarkResult()
        result.time_to_first_token_ns = [10_000_000, 20_000_000]  # 10ms, 20ms
        assert result.ttft_ms() == 15.0

    def test_ttft_ms_empty(self):
        """Test TTFT with no samples."""
        result = BenchmarkResult()
        assert result.ttft_ms() == 0.0


class TestResultsSummary:
    """Tests for ResultsSummary."""

    def test_from_results_empty(self):
        """Test summary from empty results."""
        summary = ResultsSummary.from_results([])
        assert summary.performance_score == 0.0

    def test_from_results_single(self):
        """Test summary from single result."""
        result = BenchmarkResult(n_prompt=1000, n_gen=100)
        result.prompt_intervals = [
            TimeInterval(start_ns=0, end_ns=1_000_000_000),  # 1000 tok/s
        ]
        result.gen_intervals = [
            TimeInterval(start_ns=0, end_ns=1_000_000_000),  # 100 tok/s
        ]
        result.time_to_first_token_ns = [100_000_000]  # 100ms

        summary = ResultsSummary.from_results([result])
        assert summary.avg_prompt_tps == 1000.0
        assert summary.avg_gen_tps == 100.0
        assert summary.avg_ttft_ms == 100.0
        # score = 10 * (1000 * 100 * (1000/100))^(1/3) = 10 * 1000 = 10000^(1/3) * 10
        assert summary.performance_score > 0

    def test_performance_score_formula(self):
        """Test performance score formula."""
        # score = 10 * (prompt_tps * gen_tps * (1000/ttft_ms))^(1/3)
        summary = ResultsSummary(
            avg_prompt_tps=1000.0,
            avg_gen_tps=100.0,
            avg_ttft_ms=100.0,  # 1000/100 = 10
        )
        # Expected: 10 * (1000 * 100 * 10)^(1/3) = 10 * (1_000_000)^(1/3) = 10 * 100 = 1000
        expected = 10 * (1000 * 100 * 10) ** (1/3)
        summary.performance_score = expected
        assert abs(summary.performance_score - 1000) < 1


class TestBaselineTests:
    """Tests for baseline test configurations."""

    def test_baseline_count(self):
        """Test correct number of baseline tests."""
        assert len(BASELINE_TESTS) == 9

    def test_baseline_has_valid_configs(self):
        """Test all baseline tests have valid configurations."""
        for config in BASELINE_TESTS:
            assert config.n_prompt >= 0
            assert config.n_gen >= 0
            assert config.n_prompt + config.n_gen > 0
