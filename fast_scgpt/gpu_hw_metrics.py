"""Sample GPU compute utilization and SM activity via nvidia-smi (dmon).

Uses ``nvidia-smi dmon -s u``, which reports per-GPU ``gpu`` (kernel-active %) and
``sm`` (warp-active on SM %) over each sampling interval. For multiple GPUs, one
dmon period emits one line per GPU; we average across GPUs for that period, then
average across periods. For a single GPU, each line is one period.
"""

from __future__ import annotations

import re
import subprocess
import threading


def _parse_dmon_util_line(line: str) -> tuple[int, float, float] | None:
    """Parse one dmon utilization data line; return (gpu_idx, gpu_pct, sm_pct)."""
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    parts = s.split()
    if len(parts) < 2:
        return None
    try:
        a = int(float(parts[0]))
        b = float(parts[1])
    except ValueError:
        return None
    # Most drivers: idx gpu% sm% mem% ...
    if len(parts) >= 3 and 0 <= a <= 64 and 0 <= b <= 100:
        try:
            sm_pct = float(parts[2])
        except ValueError:
            return None
        if 0 <= sm_pct <= 100:
            return a, b, sm_pct
    # Fallback: gpu% sm% ... (single-GPU or no index column)
    try:
        sm_pct = float(parts[1]) if len(parts) >= 2 else b
        gpu_pct = float(parts[0])
    except ValueError:
        return None
    if not (0 <= gpu_pct <= 100 and 0 <= sm_pct <= 100):
        return None
    return 0, gpu_pct, sm_pct


def _fallback_gpu_util_from_query(
    gpu_indices: list[int] | None,
) -> tuple[float | None, float | None]:
    """Use ``nvidia-smi --query-gpu`` for GPU util only (no SM column)."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(
            cmd, text=True, timeout=10, stderr=subprocess.DEVNULL
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None, None
    vals: list[float] = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\s*(\d+)\s*,\s*(\d+)", line)
        if not m:
            continue
        idx = int(m.group(1))
        pct = float(m.group(2))
        if gpu_indices is not None and idx not in gpu_indices:
            continue
        vals.append(pct)
    if not vals:
        return None, None
    return sum(vals) / len(vals), None


class DmonUtilSampler:
    """Background sampler for GPU and SM utilization using nvidia-smi dmon."""

    def __init__(
        self,
        n_gpus: int,
        gpu_indices: list[int] | None = None,
        interval_sec: int = 1,
    ) -> None:
        self._n_gpus = max(1, n_gpus)
        self._gpu_indices = gpu_indices
        self._interval = max(1, int(interval_sec))
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._tick_gpu: list[float] = []
        self._tick_sm: list[float] = []
        self._samples_gpu: list[float] = []
        self._samples_sm: list[float] = []
        self._running = False

    def start(self) -> bool:
        """Start dmon; return False if subprocess could not be started."""
        cmd = [
            "nvidia-smi",
            "dmon",
            "-s",
            "u",
            "-d",
            str(self._interval),
        ]
        if self._gpu_indices is not None:
            cmd.extend(["-i", ",".join(str(i) for i in self._gpu_indices)])
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except (FileNotFoundError, OSError):
            return False
        if self._proc.stdout is None:
            return False
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def _flush_tick(self) -> None:
        if not self._tick_gpu:
            return
        self._samples_gpu.append(sum(self._tick_gpu) / len(self._tick_gpu))
        self._samples_sm.append(sum(self._tick_sm) / len(self._tick_sm))
        self._tick_gpu.clear()
        self._tick_sm.clear()

    def _on_row(self, _idx: int, gpu_pct: float, sm_pct: float) -> None:
        if self._n_gpus <= 1:
            self._samples_gpu.append(gpu_pct)
            self._samples_sm.append(sm_pct)
            return
        self._tick_gpu.append(gpu_pct)
        self._tick_sm.append(sm_pct)
        if len(self._tick_gpu) >= self._n_gpus:
            self._flush_tick()

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if not self._running:
                break
            parsed = _parse_dmon_util_line(line)
            if parsed is None:
                continue
            idx, gpu_pct, sm_pct = parsed
            with self._lock:
                self._on_row(idx, gpu_pct, sm_pct)

        with self._lock:
            self._flush_tick()

    def stop(self) -> tuple[float | None, float | None]:
        """Stop dmon and return (mean gpu %, mean sm %) over collected samples."""
        self._running = False
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=6)
            self._thread = None

        with self._lock:
            self._flush_tick()
            g = self._samples_gpu
            s = self._samples_sm
            mean_g = sum(g) / len(g) if g else None
            mean_s = sum(s) / len(s) if s else None

        if mean_g is None:
            return _fallback_gpu_util_from_query(self._gpu_indices)
        if mean_s is None:
            return mean_g, None
        return mean_g, mean_s
