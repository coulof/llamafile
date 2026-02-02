"""Tests for LocalScore system information collection."""

import platform
import pytest

from localscore.system import (
    get_runtime_info,
    get_cpu_info,
    get_ram_gb,
    get_sys_info,
    get_cpu_manufacturer,
    detect_gpu_info,
)
from localscore.models import CmdParams


class TestRuntimeInfo:
    """Tests for runtime info collection."""

    def test_runtime_info_has_name(self):
        """Test runtime info has a name."""
        info = get_runtime_info()
        assert info.name == "localscore-python"

    def test_runtime_info_has_version(self):
        """Test runtime info has a version."""
        info = get_runtime_info()
        assert info.version
        assert len(info.version) > 0


class TestCpuInfo:
    """Tests for CPU info collection."""

    def test_get_cpu_info_returns_string(self):
        """Test get_cpu_info returns a non-empty string."""
        cpu = get_cpu_info()
        assert isinstance(cpu, str)
        assert len(cpu) > 0

    def test_get_cpu_info_no_trademark_symbols(self):
        """Test trademark symbols are removed."""
        cpu = get_cpu_info()
        assert "(TM)" not in cpu
        assert "(R)" not in cpu


class TestRamInfo:
    """Tests for RAM info collection."""

    def test_get_ram_gb_returns_positive(self):
        """Test get_ram_gb returns a positive value."""
        ram = get_ram_gb()
        assert ram > 0

    def test_get_ram_gb_reasonable_value(self):
        """Test RAM is a reasonable value (> 0.5 GB, < 10 TB)."""
        ram = get_ram_gb()
        assert 0.5 < ram < 10000


class TestSysInfo:
    """Tests for system info collection."""

    def test_sys_info_has_kernel_type(self):
        """Test system info has kernel type."""
        info = get_sys_info()
        assert info.kernel_type
        assert info.kernel_type in ["Linux", "Darwin", "Windows"]

    def test_sys_info_has_cpu_arch(self):
        """Test system info has CPU architecture."""
        info = get_sys_info()
        assert info.cpu_arch
        assert info.cpu_arch in ["x86_64", "aarch64", "arm64", "AMD64"]

    def test_sys_info_has_cpu_name(self):
        """Test system info has CPU name."""
        info = get_sys_info()
        assert info.cpu_name

    def test_sys_info_has_ram(self):
        """Test system info has RAM."""
        info = get_sys_info()
        assert info.ram_gb > 0


class TestCpuManufacturer:
    """Tests for CPU manufacturer detection."""

    def test_get_cpu_manufacturer_returns_known(self):
        """Test manufacturer is a known value."""
        manufacturer = get_cpu_manufacturer()
        assert manufacturer in ["AMD", "Intel", "Apple", "ARM", "Unknown"]


class TestGpuDetection:
    """Tests for GPU detection."""

    def test_detect_gpu_info_cpu_mode(self):
        """Test GPU detection in CPU mode."""
        params = CmdParams()
        params.gpu_backend = "disabled"
        params.n_gpu_layers = 0

        info = detect_gpu_info(params)
        assert info.type == "CPU"
        assert info.name  # Should have CPU name

    def test_detect_gpu_info_has_manufacturer(self):
        """Test GPU detection returns manufacturer."""
        params = CmdParams()
        params.gpu_backend = "auto"

        info = detect_gpu_info(params)
        assert info.manufacturer
