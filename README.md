# FAA outage attribution forecast

Forecasts the Yes price of every market in Polymarket's FAA outage
attribution family, minute by minute, four hours ahead — as five quantiles
(5%, 25%, 50%, 75%, 95%) per minute.

## The idea

Prediction-market prices are close to martingales over a few hours, so the
best guess for where a price will be is where it is now: the median
forecast is simply the last observed price (persistence).

What changes is how far the price might wander. The program measures how
much each market actually moved in its most recent two hours — the
root-mean-square of its one-minute price changes — and uses that as the
spread. The band then widens with horizon, but slower than a random walk
(`h ** 0.4` instead of `sqrt(h)`), because minute-to-minute moves
mean-revert a little. The shape of the band is Laplace (double-exponential,
fat-tailed, which fits the step-and-jump way these prices move), rescaled by
`1/sqrt(2)` so the forecast distribution's variance matches the realised
variance.

A few guardrails:

- Volatility is floored at 0.0007 and capped at 0.01 price units per minute,
  so a dead-quiet market still gets a sane band and a wild one doesn't
  explode it.
- A market with fewer than 10 one-minute changes can't estimate its own
  volatility, so it borrows the median volatility of the case's other
  markets (or 0.0025 if none can). A market with no prices at all is centred
  on 0.5 rather than failing the case.
- Every value is clamped to [0, 1], rounded to 6 decimals, and each row is
  kept non-decreasing.

## Running it

Python 3, standard library only — nothing to install, no network:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `<id>.in` in the input directory and writes one
`<id>.out` per case. Each run takes well under a second per case.

## Tests

```sh
python3 -m unittest discover -s tests
```

## Practice score

On the kit's six practice cases (cut from Polymarket's "Next CEO of
NVIDIA?" event), the local pinball scorer reports a mean loss of about
**0.266**, clearly below the house baseline's 0.363589. The practice is
another event's data, so it is no guide to the final — but the margin
shows the volatility-adaptive band is doing real work over the fixed
0.0025 spread of the baseline.
