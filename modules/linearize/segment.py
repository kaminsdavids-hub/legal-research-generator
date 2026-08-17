"""Cutting a fixed order into sections. Exact, by dynamic programming.

Once the order is fixed, segmentation is not hard: sections are contiguous runs,
so the optimal cut into ``k`` parts is a shortest path over prefix boundaries and
falls out of an O(n²k) DP. Searching it would be slower, non-deterministic, and
worse.

That split -- search only where the problem is genuinely hard, exact methods
everywhere else -- is the design point worth keeping. An annealer that also
chose the section breaks would make the whole result stochastic for no gain, and
"the section boundaries moved because the seed changed" is not something an
author can work with.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .problem import LinearizeProblem, _cosine

__all__ = ["Segmentation", "SegmentConfig", "segment", "sweep_k"]

INFEASIBLE = float("inf")


@dataclass(frozen=True)
class SegmentConfig:
    min_words: int = 150
    max_words: int = 4_000
    #: Weight on section-length imbalance. Incoherence is the primary signal;
    #: balance stops the DP from answering "one enormous section and five stubs",
    #: which is coherent by construction and unreadable.
    balance: float = 1.0


@dataclass
class Segmentation:
    #: Half-open [start, end) index ranges into the order.
    bounds: list[tuple[int, int]]
    cost: float
    words: list[int]

    @property
    def k(self) -> int:
        return len(self.bounds)

    def section_of(self, index: int) -> int:
        for section, (start, end) in enumerate(self.bounds):
            if start <= index < end:
                return section
        raise IndexError(index)  # pragma: no cover - bounds cover the order


def _incoherence(order: Sequence[str], embeddings: Mapping[str, Sequence[float]]) -> list[list[float]]:
    """``cost[i][j]`` = incoherence of the run ``order[i:j]``.

    One minus the mean cosine between consecutive members: a section whose
    neighbours are unrelated reads as a list rather than an argument. A run of
    one node has no adjacency and so no incoherence.
    """

    n = len(order)
    cost = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        running = 0.0
        for j in range(i + 1, n + 1):
            if j - i >= 2:
                a = embeddings.get(order[j - 2])
                b = embeddings.get(order[j - 1])
                running += 1.0 - _cosine(a, b) if a is not None and b is not None else 1.0
            cost[i][j] = running / max(1, (j - i - 1))
    return cost


def segment(
    problem: LinearizeProblem,
    order: Sequence[str],
    k: int,
    config: SegmentConfig | None = None,
) -> Segmentation:
    """The optimal cut of ``order`` into ``k`` sections.

    ``dp[j][s]`` is the best cost of covering the first ``j`` nodes with ``s``
    sections; the transition tries every previous boundary. O(n²k), exact, and
    deterministic for a given input -- no seed appears anywhere in this file.
    """

    cfg = config or SegmentConfig()
    n = len(order)
    if k < 1 or k > n:
        raise ValueError(f"cannot cut {n} node(s) into {k} section(s)")

    words = [problem.words.get(node, 0) for node in order]
    prefix = [0]
    for w in words:
        prefix.append(prefix[-1] + w)

    incoherence = _incoherence(order, problem.embeddings)
    target = prefix[n] / k

    def run_cost(i: int, j: int) -> float:
        total_words = prefix[j] - prefix[i]
        if total_words < cfg.min_words or total_words > cfg.max_words:
            return INFEASIBLE
        imbalance = abs(total_words - target) / max(1.0, target)
        return incoherence[i][j] + cfg.balance * imbalance

    dp = [[INFEASIBLE] * (k + 1) for _ in range(n + 1)]
    back = [[-1] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for j in range(1, n + 1):
        for s in range(1, k + 1):
            best, best_i = INFEASIBLE, -1
            for i in range(s - 1, j):
                if dp[i][s - 1] == INFEASIBLE:
                    continue
                candidate = dp[i][s - 1] + run_cost(i, j)
                if candidate < best:
                    best, best_i = candidate, i
            dp[j][s], back[j][s] = best, best_i

    if dp[n][k] == INFEASIBLE:
        # Rather than silently relaxing the word bounds -- which would return a
        # segmentation that violates what the author asked for while looking
        # like one that does not.
        raise ValueError(
            f"no segmentation into {k} sections satisfies min_words={cfg.min_words} / "
            f"max_words={cfg.max_words} for {prefix[n]} words; change k or the bounds"
        )

    bounds: list[tuple[int, int]] = []
    j, s = n, k
    while s > 0:
        i = back[j][s]
        bounds.append((i, j))
        j, s = i, s - 1
    bounds.reverse()

    return Segmentation(
        bounds=bounds,
        cost=dp[n][k],
        words=[prefix[end] - prefix[start] for start, end in bounds],
    )


def sweep_k(
    problem: LinearizeProblem,
    order: Sequence[str],
    candidates: Sequence[int],
    config: SegmentConfig | None = None,
) -> tuple[Segmentation, dict[int, float]]:
    """Try each ``k`` and pick the elbow.

    The elbow rather than the minimum: cost falls monotonically with more
    sections (smaller runs are more coherent and easier to balance), so taking
    the argmin always returns the largest ``k`` offered, which is not a choice.
    The elbow is the last ``k`` that bought a material improvement.
    """

    curve: dict[int, float] = {}
    solutions: dict[int, Segmentation] = {}
    for k in candidates:
        try:
            solution = segment(problem, order, k, config)
        except ValueError:
            continue
        curve[k], solutions[k] = solution.cost, solution
    if not solutions:
        raise ValueError("no feasible segmentation for any k in the sweep")

    ordered = sorted(solutions)
    best = ordered[0]
    for previous, current in zip(ordered, ordered[1:], strict=False):
        gain = curve[previous] - curve[current]
        if gain <= 0.05 * max(abs(curve[previous]), 1e-9):
            break
        best = current
    return solutions[best], curve


def section_titles(order: Sequence[str], segmentation: Segmentation) -> list[str]:
    """Roman-numbered placeholders. Naming a section is the author's job."""

    return [f"Part {_roman(i + 1)}" for i in range(segmentation.k)]


def _roman(n: int) -> str:
    numerals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for value, symbol in numerals:
        while n >= value:
            out, n = out + symbol, n - value
    return out


def words_per_section(problem: LinearizeProblem, order: Sequence[str], seg: Segmentation) -> list[int]:
    return [sum(problem.words.get(order[i], 0) for i in range(a, b)) for a, b in seg.bounds]
