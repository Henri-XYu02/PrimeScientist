"""
benchmarks/registry.py — Map benchmark names to classes.
"""

from pathlib import Path
from benchmarks.base import Benchmark


def load_benchmark(name: str, benchmark_path: str | None = None) -> Benchmark:
    if name == "fire_bench":
        from benchmarks.fire_bench import FireBench  # benchmarks/fire_bench/__init__.py
        return FireBench(Path(benchmark_path) if benchmark_path else None)
    elif name == "autolab":
        from benchmarks.autolab import AutoLab
        return AutoLab(Path(benchmark_path) if benchmark_path else None)
    elif name == "terminal_bench":
        from benchmarks.terminal_bench import TerminalBench
        return TerminalBench(Path(benchmark_path) if benchmark_path else None)
    else:
        raise ValueError(f"Unknown benchmark: {name!r}. Available: fire_bench, autolab, terminal_bench")
