"""Tests for LocalScore CLI."""

import pytest
import sys
from io import StringIO

from localscore.cli import parse_args, main
from localscore.models import OutputFormat, SendResultsMode


class TestParseArgs:
    """Tests for argument parsing."""

    def test_help_flag(self):
        """Test --help flag exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_version_flag(self):
        """Test --version flag exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_model_required(self):
        """Test that model is required unless --list-gpus."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args([])
        assert exc_info.value.code != 0

    def test_list_gpus_no_model(self):
        """Test --list-gpus doesn't require model."""
        params = parse_args(["--list-gpus"])
        assert params.list_gpus is True

    def test_model_flag(self):
        """Test -m/--model flag."""
        params = parse_args(["-m", "test.gguf"])
        assert params.model == "test.gguf"

        params = parse_args(["--model", "test2.gguf"])
        assert params.model == "test2.gguf"

    def test_cpu_flag(self):
        """Test -c/--cpu flag disables GPU."""
        params = parse_args(["-m", "test.gguf", "-c"])
        assert params.gpu_backend == "disabled"
        assert params.n_gpu_layers == 0

    def test_gpu_flag(self):
        """Test -g/--gpu flag."""
        params = parse_args(["-m", "test.gguf", "-g", "nvidia"])
        assert params.gpu_backend == "nvidia"

        params = parse_args(["-m", "test.gguf", "--gpu", "disabled"])
        assert params.gpu_backend == "disabled"
        assert params.n_gpu_layers == 0

    def test_output_format(self):
        """Test -o/--output flag."""
        params = parse_args(["-m", "test.gguf", "-o", "json"])
        assert params.output_format == OutputFormat.JSON

        params = parse_args(["-m", "test.gguf", "--output", "csv"])
        assert params.output_format == OutputFormat.CSV

    def test_send_results_flags(self):
        """Test send results flags."""
        params = parse_args(["-m", "test.gguf", "-y"])
        assert params.send_results == SendResultsMode.YES

        params = parse_args(["-m", "test.gguf", "-n"])
        assert params.send_results == SendResultsMode.NO

        params = parse_args(["-m", "test.gguf"])
        assert params.send_results == SendResultsMode.ASK

    def test_conflicting_send_results(self):
        """Test -y and -n together is an error."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["-m", "test.gguf", "-y", "-n"])
        assert exc_info.value.code != 0

    def test_reps_flags(self):
        """Test repetition flags."""
        params = parse_args(["-m", "test.gguf", "--reps", "5"])
        assert params.reps == 5

        params = parse_args(["-m", "test.gguf", "-e"])
        assert params.reps == 4

        params = parse_args(["-m", "test.gguf", "--long"])
        assert params.reps == 16

    def test_reps_minimum(self):
        """Test reps is at least 1."""
        params = parse_args(["-m", "test.gguf", "--reps", "0"])
        assert params.reps == 1

    def test_verbose_flag(self):
        """Test -v/--verbose flag."""
        params = parse_args(["-m", "test.gguf", "-v"])
        assert params.verbose is True

    def test_plaintext_flag(self):
        """Test --plaintext flag."""
        params = parse_args(["-m", "test.gguf", "--plaintext"])
        assert params.plaintext is True

    def test_gpu_index(self):
        """Test -i/--gpu-index flag."""
        params = parse_args(["-m", "test.gguf", "-i", "2"])
        assert params.main_gpu == 2

    def test_batch_size(self):
        """Test --n-batch flag."""
        params = parse_args(["-m", "test.gguf", "--n-batch", "512"])
        assert params.n_batch == 512

    def test_flash_attn(self):
        """Test --flash-attn flag."""
        params = parse_args(["-m", "test.gguf", "--flash-attn"])
        assert params.flash_attn is True

    def test_no_mmap(self):
        """Test --no-mmap flag."""
        params = parse_args(["-m", "test.gguf", "--no-mmap"])
        assert params.use_mmap is False


class TestMain:
    """Tests for main function."""

    def test_list_gpus_returns_zero(self):
        """Test --list-gpus returns exit code 0."""
        result = main(["--list-gpus"])
        assert result == 0

    def test_missing_model_file(self):
        """Test missing model file returns exit code 1."""
        try:
            import llama_cpp
        except ImportError:
            pytest.skip("llama-cpp-python not installed")
        result = main(["-m", "/nonexistent/model.gguf", "-n"])
        assert result == 1
