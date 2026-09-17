"""
benchmarks/registry.py — Map benchmark names to classes.
"""

from pathlib import Path
from benchmarks.base import Benchmark

# Shipped in this repo. terminal_bench is dispatched but not vendored here, so
# it is listed separately and only reported as available once importable.
_SHIPPED = ("fire_bench", "autolab")


def load_benchmark(name: str, benchmark_path: str | None = None) -> Benchmark:
    path = Path(benchmark_path) if benchmark_path else None
    if name == "fire_bench":
        from benchmarks.fire_bench import FireBench  # benchmarks/fire_bench/__init__.py
        return FireBench(path)
    elif name == "autolab":
        from benchmarks.autolab import AutoLab
        return AutoLab(path)
    elif name == "terminal_bench":
        try:
            from benchmarks.terminal_bench import TerminalBench
        except ImportError as e:
            raise ValueError(
                "benchmark 'terminal_bench' is not included in this repository "
                f"(import failed: {e}). Available: {', '.join(_SHIPPED)}."
            ) from e
        return TerminalBench(path)
    else:
        raise ValueError(
            f"Unknown benchmark: {name!r}. Available: {', '.join(_SHIPPED)}."
        )
