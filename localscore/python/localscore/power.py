"""Power sampling for LocalScore (optional NVIDIA support via pynvml)."""

import threading
import time
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PowerSample:
    """Power measurement sample."""
    power_mw: float = 0.0  # Power in milliwatts
    timestamp_ns: int = 0

    @property
    def power_watts(self) -> float:
        """Get power in watts."""
        return self.power_mw / 1000.0


class PowerSampler:
    """Base class for power sampling."""

    def __init__(self, interval_ms: int = 100, gpu_index: int = 0):
        self.interval_ms = interval_ms
        self.gpu_index = gpu_index
        self._samples: List[PowerSample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start power sampling."""
        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> PowerSample:
        """Stop sampling and return average power."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

        with self._lock:
            if not self._samples:
                return PowerSample()

            avg_power = sum(s.power_mw for s in self._samples) / len(self._samples)
            return PowerSample(
                power_mw=avg_power,
                timestamp_ns=time.perf_counter_ns(),
            )

    def get_latest_sample(self) -> PowerSample:
        """Get the most recent power sample."""
        with self._lock:
            if self._samples:
                return self._samples[-1]
            return PowerSample()

    def _sample_loop(self) -> None:
        """Sampling loop - override in subclass."""
        while self._running:
            sample = self._take_sample()
            if sample.power_mw > 0:
                with self._lock:
                    self._samples.append(sample)
            time.sleep(self.interval_ms / 1000.0)

    def _take_sample(self) -> PowerSample:
        """Take a single power sample - override in subclass."""
        return PowerSample()


class NvidiaPowerSampler(PowerSampler):
    """Power sampler for NVIDIA GPUs using pynvml."""

    def __init__(self, interval_ms: int = 100, gpu_index: int = 0):
        super().__init__(interval_ms, gpu_index)
        self._handle = None
        self._nvml_initialized = False

    def start(self) -> None:
        """Start NVIDIA power sampling."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except (ImportError, Exception) as e:
            self._nvml_initialized = False
            self._handle = None

        super().start()

    def stop(self) -> PowerSample:
        """Stop sampling and shutdown NVML."""
        result = super().stop()

        if self._nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False

        return result

    def _take_sample(self) -> PowerSample:
        """Take a power sample from NVIDIA GPU."""
        if not self._handle:
            return PowerSample()

        try:
            import pynvml
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            return PowerSample(
                power_mw=float(power_mw),
                timestamp_ns=time.perf_counter_ns(),
            )
        except Exception:
            return PowerSample()


class DummyPowerSampler(PowerSampler):
    """Dummy power sampler that returns no data."""

    def _take_sample(self) -> PowerSample:
        return PowerSample()


def get_power_sampler(interval_ms: int = 100, gpu_index: int = 0) -> PowerSampler:
    """Get appropriate power sampler for the system."""
    # Try NVIDIA first
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        if count > 0:
            return NvidiaPowerSampler(interval_ms, gpu_index)
    except (ImportError, Exception):
        pass

    # Return dummy sampler if no GPU power sampling available
    return DummyPowerSampler(interval_ms, gpu_index)
