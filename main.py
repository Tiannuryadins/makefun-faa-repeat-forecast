#!/usr/bin/env python3
"""Minute-by-minute quantile forecasts of Polymarket Yes prices, 4 hours ahead.

For every market in a case this program carries the last observed Yes price
forward as the median forecast -- prediction-market prices behave like a
near-martingale over a few hours -- and wraps it in a band whose width comes
from how much the market actually moved in the last two hours. The band
grows sub-diffusively with horizon and its shape is Laplace, rescaled so the
forecast distribution's variance equals the realised variance.

Input:  every <id>.in (makefun/polymarket-case/v1) in $MAKEFUN_INPUT_DIR
        (default /input). Each case lists markets with one Yes price per
        minute for up to 72 hours ending at the checkpoint.
Output: one <id>.out per case in $MAKEFUN_OUTPUT_DIR (default /output):
        {"forecasts": [{"id": <market id>, "q": [[5 quantiles] x horizon]}]},
        240 rows (or the case's horizon_minutes) at levels
        0.05, 0.25, 0.5, 0.75, 0.95, compact JSON with a trailing newline.

Standard library only. Deterministic: no clock, no network, no randomness.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Quantile levels, in the order each output row lists them.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# How many of the most recent minutes feed the volatility estimate.
RECENT_WINDOW = 120
# A market needs at least this many one-minute changes to estimate its own
# volatility; below it, the case-wide fallback applies.
MIN_CHANGES = 10
# Volatility floor and cap, in price units per minute.
VOL_FLOOR = 0.0007
VOL_CAP = 0.01
# Fallback volatility when no market in the case can estimate one.
DEFAULT_VOL = 0.0025
# Band width grows as h ** GROWTH with horizon h: slower than a random
# walk's sqrt(h), because minute-to-minute moves mean-revert a little.
GROWTH = 0.4


def _laplace_offset(tau):
    """Laplace (double-exponential) quantile at level tau, rescaled by
    1/sqrt(2). A Laplace with scale b has variance 2*b**2, so dividing by
    sqrt(2) makes the forecast distribution's variance equal the realised
    variance the width is built from."""
    if tau < 0.5:
        return math.log(2.0 * tau) / math.sqrt(2.0)
    if tau > 0.5:
        return -math.log(2.0 * (1.0 - tau)) / math.sqrt(2.0)
    return 0.0


OFFSETS = tuple(_laplace_offset(tau) for tau in LEVELS)


def _clean_prices(raw):
    """Keep only usable prices: real numbers, finite, not booleans. Anything
    else is dropped rather than allowed to poison an estimate."""
    good = []
    for value in raw or []:
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            good.append(number)
    return good


def _recent_volatility(prices):
    """Root-mean-square of the one-minute price changes over the most recent
    RECENT_WINDOW minutes. Returns None when the series is too short to
    estimate its own volatility (fewer than MIN_CHANGES changes)."""
    changes = []
    first = max(0, len(prices) - 1 - RECENT_WINDOW)
    for i in range(first, len(prices) - 1):
        changes.append(prices[i + 1] - prices[i])
    if len(changes) < MIN_CHANGES:
        return None
    squares = sum(d * d for d in changes)
    return math.sqrt(squares / len(changes))


def _clamp_round(value):
    """A price into [0, 1], rounded to 6 decimals, with -0.0 normalised
    away so the output never carries a negative zero."""
    rounded = round(min(1.0, max(0.0, value)), 6)
    return 0.0 if rounded == 0 else rounded


def forecast_market(prices, horizon, fallback_vol):
    """The horizon rows of five quantiles for one market.

    The median stays at the last observed price. Every other quantile sits
    at anchor + offset * width, where width is the market's realised
    volatility grown sub-diffusively with horizon. A market with no prices
    at all is centred on 0.5 instead of failing the whole case.
    """
    series = _clean_prices(prices)
    if not series:
        anchor = 0.5
        vol = fallback_vol
    else:
        anchor = series[-1]
        own = _recent_volatility(series)
        vol = fallback_vol if own is None else own
    vol = min(VOL_CAP, max(VOL_FLOOR, vol))

    rows = []
    for step in range(1, horizon + 1):
        width = vol * (step ** GROWTH)
        row = []
        lowest = 0.0
        for offset in OFFSETS:
            value = _clamp_round(anchor + offset * width)
            # Rounding can only break monotonicity by a hair; keep each row
            # non-decreasing by carrying the running maximum forward.
            if value < lowest:
                value = lowest
            lowest = value
            row.append(value)
        rows.append(row)
    return rows


def _median(values):
    """Plain median of a non-empty list of numbers."""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def forecast_case(case):
    """The full forecast object for one case.

    First pass: every market's own volatility, where it can be estimated.
    The fallback for short-history markets is the median of those own
    volatilities (DEFAULT_VOL when no market in the case can estimate one).
    An empty markets list yields an empty forecasts list.
    """
    horizon = int(case.get("horizon_minutes") or 240)
    markets = case.get("markets") or []

    own_vols = []
    for market in markets:
        own_vols.append(_recent_volatility(_clean_prices(market.get("prices"))))

    estimable = [v for v in own_vols if v is not None]
    fallback_vol = _median(estimable) if estimable else DEFAULT_VOL

    forecasts = []
    for market in markets:
        forecasts.append({
            "id": market["id"],
            "q": forecast_market(market.get("prices"), horizon, fallback_vol),
        })
    return {"forecasts": forecasts}


def run(input_dir, output_dir):
    """Read every *.in in input_dir (name order) and write one *.out per
    case to output_dir, creating it when it is missing."""
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(input_dir) if name.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        with open(os.path.join(output_dir, case_id + ".out"), "w",
                  encoding="utf-8") as handle:
            handle.write(text)


def main():
    run(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()
