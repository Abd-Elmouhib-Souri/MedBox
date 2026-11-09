"""Wilson score interval for a proportion k / n (95 % by default).

Why Wilson and not "p ± 1.96 * sqrt(p(1-p)/n)": with small n or p close to 0 or 1, the simple formula
gives impossible intervals (below 0 or above 1, or zero width for 0/n). Wilson stays inside [0, 1].
"""

from math import sqrt


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Return (low, high) for k successes out of n, or None if n == 0 (proportion undefined)."""
    if n == 0:
        return None                                   # protocol: report "undefined", never 0 % or 100 %
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def proportion(k: int, n: int) -> dict:
    """k / n with its Wilson interval, in the format written to the results JSON."""
    ci = wilson(k, n)
    return {
        "k": k,
        "n": n,
        "value": None if n == 0 else round(k / n, 4),
        "ci95": None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
        "text": "undefined (n = 0)" if n == 0 else f"{k}/{n} (95% CI {ci[0]:.2f}-{ci[1]:.2f})",
    }
