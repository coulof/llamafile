"""System information collection for LocalScore."""

import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

from .models import RuntimeInfo, SystemInfo, AcceleratorInfo, ModelInfo, CmdParams
from . import __version__


def get_runtime_info() -> RuntimeInfo:
    """Get runtime environment information."""
    return RuntimeInfo(
        name="localscore-python",
        version=__version__,
        commit="",
    )


def get_cpu_info() -> str:
    """Get CPU model name and architecture info."""
    cpu_name = ""

    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name") or line.startswith("Model\t\t:"):
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            cpu_name = parts[1].strip()
                            break
        except (IOError, OSError):
            pass

    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                cpu_name = result.stdout.strip()

            # Get performance/efficiency core counts on Apple Silicon
            try:
                perf_result = subprocess.run(
                    ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                eff_result = subprocess.run(
                    ["sysctl", "-n", "hw.perflevel1.logicalcpu"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if perf_result.returncode == 0:
                    cpu_name += f" {perf_result.stdout.strip()}P"
                if eff_result.returncode == 0:
                    cpu_name += f"+{eff_result.stdout.strip()}E"
            except (subprocess.TimeoutExpired, OSError):
                pass
        except (subprocess.TimeoutExpired, OSError):
            pass

    elif system == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
        except (ImportError, OSError):
            pass

    if not cpu_name:
        cpu_name = platform.processor() or "Unknown CPU"

    # Clean up common suffixes
    cpu_name = cpu_name.replace(" 96-Cores", "")
    cpu_name = cpu_name.replace("(TM)", "")
    cpu_name = cpu_name.replace("(R)", "")

    return cpu_name


def get_ram_gb() -> float:
    """Get total system RAM in GiB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return round(mem.total / (1024 ** 3), 1)
    except ImportError:
        pass

    # Fallback for Linux
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            kb = int(parts[1])
                            return round(kb / (1024 ** 2), 1)
        except (IOError, OSError, ValueError):
            pass

    return 0.0


def get_sys_info() -> SystemInfo:
    """Get system hardware and OS information."""
    uname = platform.uname()

    return SystemInfo(
        cpu_name=get_cpu_info(),
        cpu_arch=uname.machine,
        ram_gb=get_ram_gb(),
        kernel_type=uname.system,
        kernel_release=uname.release,
        version=uname.version,
    )


def get_cpu_manufacturer() -> str:
    """Get CPU manufacturer name."""
    cpu_info = get_cpu_info().lower()

    if "amd" in cpu_info:
        return "AMD"
    elif "intel" in cpu_info:
        return "Intel"
    elif "apple" in cpu_info or platform.system() == "Darwin":
        return "Apple"
    elif "arm" in cpu_info:
        return "ARM"
    else:
        return "Unknown"


def detect_gpu_info(params: CmdParams) -> AcceleratorInfo:
    """Detect GPU/accelerator information."""
    info = AcceleratorInfo()

    # Check if GPU is disabled
    if params.gpu_backend == "disabled" or params.n_gpu_layers == 0:
        info.name = get_cpu_info()
        info.manufacturer = get_cpu_manufacturer()
        info.memory_gb = get_ram_gb()
        info.type = "CPU"
        return info

    # Try to detect NVIDIA GPU
    nvidia_info = _detect_nvidia_gpu(params.main_gpu)
    if nvidia_info:
        return nvidia_info

    # Try to detect AMD GPU
    amd_info = _detect_amd_gpu()
    if amd_info:
        return amd_info

    # Try to detect Apple Metal
    if platform.system() == "Darwin":
        apple_info = _detect_apple_metal()
        if apple_info:
            return apple_info

    # Fallback to CPU
    info.name = get_cpu_info()
    info.manufacturer = get_cpu_manufacturer()
    info.memory_gb = get_ram_gb()
    info.type = "CPU"
    return info


def _detect_nvidia_gpu(gpu_index: int = 0) -> Optional[AcceleratorInfo]:
    """Detect NVIDIA GPU using pynvml or nvidia-smi."""
    # Try pynvml first
    try:
        import pynvml
        pynvml.nvmlInit()

        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return None

        idx = min(gpu_index, device_count - 1)
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_gb = round(memory.total / (1024 ** 3), 1)

        # Get CUDA compute capability
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        capability = float(f"{major}.{minor}")

        pynvml.nvmlShutdown()

        return AcceleratorInfo(
            name=name,
            manufacturer="NVIDIA",
            memory_gb=memory_gb,
            capability=capability,
            type="GPU",
        )
    except (ImportError, Exception):
        pass

    # Fallback to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                idx = min(gpu_index, len(lines) - 1)
                parts = lines[idx].split(", ")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    memory_mb = float(parts[1].strip())
                    return AcceleratorInfo(
                        name=name,
                        manufacturer="NVIDIA",
                        memory_gb=round(memory_mb / 1024, 1),
                        type="GPU",
                    )
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass

    return None


def _detect_amd_gpu() -> Optional[AcceleratorInfo]:
    """Detect AMD GPU using rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            output = result.stdout
            name = "AMD GPU"
            memory_gb = 0.0

            for line in output.split("\n"):
                if "GPU" in line and ":" in line:
                    name = line.split(":")[-1].strip()
                if "Total Memory" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p.isdigit():
                            memory_gb = float(p) / 1024  # Assuming MB
                            break

            return AcceleratorInfo(
                name=name,
                manufacturer="AMD",
                memory_gb=round(memory_gb, 1),
                type="GPU",
            )
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass

    return None


def _detect_apple_metal() -> Optional[AcceleratorInfo]:
    """Detect Apple Metal GPU."""
    if platform.system() != "Darwin":
        return None

    cpu_info = get_cpu_info()

    # Get GPU core count
    core_count = 0
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "Total Number of Cores:" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        core_count = int(parts[1].strip())
                        break
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass

    name = cpu_info
    if core_count > 0:
        name += f"+{core_count}GPU"

    return AcceleratorInfo(
        name=name,
        manufacturer="Apple",
        memory_gb=get_ram_gb(),  # Unified memory
        core_count=core_count,
        type="GPU",
    )


def list_available_gpus() -> None:
    """Print list of available GPUs."""
    print("\n==================== Available GPUs ====================\n")

    # Try NVIDIA
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_gb = memory.total / (1024 ** 3)
            print(f"{i}: {name} - {memory_gb:.2f} GiB")
        pynvml.nvmlShutdown()
        if count > 0:
            print("\n" + "=" * 56)
            return
    except (ImportError, Exception):
        pass

    # Fallback to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = line.split(", ")
                if len(parts) >= 3:
                    idx = parts[0].strip()
                    name = parts[1].strip()
                    memory_mb = float(parts[2].strip())
                    print(f"{idx}: {name} - {memory_mb / 1024:.2f} GiB")
            print("\n" + "=" * 56)
            return
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Check for Apple Metal
    if platform.system() == "Darwin":
        print("Apple Metal")
        print("\n" + "=" * 56)
        return

    print("No GPU accelerator support available")
    print("\n" + "=" * 56)


def get_model_info_from_llama(llm) -> ModelInfo:
    """Extract model info from llama-cpp-python model."""
    info = ModelInfo()

    try:
        # Get model path
        if hasattr(llm, "model_path"):
            info.filename = Path(llm.model_path).name

        # Get metadata
        metadata = {}
        if hasattr(llm, "metadata"):
            metadata = llm.metadata or {}

        info.name = metadata.get("general.name", "")
        info.size_label = metadata.get("general.size_label", "")

        # Get model description
        if hasattr(llm, "desc"):
            info.type = llm.desc() if callable(llm.desc) else str(llm.desc)

        # Get quantization info
        if hasattr(llm, "scores") and hasattr(llm, "n_vocab"):
            # Try to infer from metadata
            quant = metadata.get("general.file_type", "")
            if quant:
                info.quant = quant

        # Get size and params
        if hasattr(llm, "n_params"):
            info.params = llm.n_params() if callable(llm.n_params) else llm.n_params

    except Exception:
        pass

    return info
