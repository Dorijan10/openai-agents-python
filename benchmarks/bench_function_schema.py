"""Artemis Discovery benchmark for openai-agents-python (CPU-bound, offline)."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Literal, Optional

from agents import Agent, function_tool
from agents.function_schema import function_schema

SAMPLES = 50
WARMUP = 5
INNER = 20


def simple_tool(city: str, days: int = 3) -> str:
    """Get a weather forecast.

    Args:
        city: The city to look up.
        days: How many days ahead.
    """
    return ""


def complex_tool(
    query: str,
    limit: int = 10,
    offset: Optional[int] = None,
    mode: Literal["fast", "thorough"] = "fast",
    tags: Optional[list[str]] = None,
    verbose: bool = False,
) -> str:
    """Search a corpus with a number of options.

    Args:
        query: The search query.
        limit: Maximum results to return.
        offset: Result offset for pagination.
        mode: Search strategy to use.
        tags: Restrict results to these tags.
        verbose: Emit extra detail.
    """
    return ""


def _time_schema_build() -> float:
    t0 = time.perf_counter()
    for _ in range(INNER):
        function_schema(simple_tool)
        function_schema(complex_tool)
    return (time.perf_counter() - t0) * 1000.0


def _time_agent_build() -> float:
    t0 = time.perf_counter()
    for _ in range(INNER):
        tools = [function_tool(simple_tool), function_tool(complex_tool)]
        Agent(name="bench-agent", instructions="benchmark", tools=tools)
    return (time.perf_counter() - t0) * 1000.0


def main() -> None:
    for _ in range(WARMUP):
        _time_schema_build()
        _time_agent_build()

    samples: list[dict[str, float]] = []
    for _ in range(SAMPLES):
        samples.append(
            {
                "function_schema_ms": _time_schema_build(),
                "agent_construction_ms": _time_agent_build(),
            }
        )

    out_path = Path(__file__).resolve().parents[1] / "artemis_results.json"
    out_path.write_text(json.dumps(samples, indent=2))

    schema_ms = [s["function_schema_ms"] for s in samples]
    agent_ms = [s["agent_construction_ms"] for s in samples]
    print(
        f"wrote {out_path} ({len(samples)} samples) | "
        f"function_schema median {statistics.median(schema_ms):.3f} ms | "
        f"agent_construction median {statistics.median(agent_ms):.3f} ms"
    )


if __name__ == "__main__":
    main()
