# Build transcript — faa-repeat-forecast (fresh rebuild, post-opening session)

Built in a new session started after the competition opened (2026-11-15
04:00 WIB). Every file here was authored from scratch in this session, in
my own words and structure, from the kit's format spec only. I never read
the previous entry's files.

## What I built

- `main.py` — Python 3, standard library only. Reads every `<id>.in` from
  `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one compact-JSON
  `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`).
- `tests/test_main.py` — 20 unit tests.
- `README.md` — what the entry does and how to run it.
- `LICENSE` — MIT, SPDX identifier first.
- `.gitignore` — `bundle/`, `__pycache__/`, `*.pyc`.

## The approach

Persistence median: each market's forecast median is its last observed
price, because prediction-market prices are near-martingales over a few
hours.

Volatility-adaptive band: the spread is the root-mean-square of the
one-minute price changes over only the most recent 120 minutes, floored at
0.0007 and capped at 0.01 price units per minute. The band width at horizon
h is that volatility times `h ** 0.4` — sub-diffusive growth, slower than a
random walk's `sqrt(h)`.

Laplace shape: quantiles at 0.05 / 0.25 / 0.5 / 0.75 / 0.95 use the
double-exponential (Laplace) quantile function, each multiplied by
`1/sqrt(2)`. A Laplace with scale b has variance `2b^2`, so the rescaling
makes the forecast distribution's variance equal the realised variance.
Values are clamped to [0, 1], rounded to 6 decimals, and each row is kept
non-decreasing.

Fallbacks: a market with fewer than 10 one-minute changes borrows the
case's cross-sectional median volatility (0.0025 if no market can estimate
one). A market with no prices at all is centred on 0.5. Non-finite and
boolean prices are skipped. An empty markets list yields an empty
forecasts list.

## Validation (this session)

- Practice run on the kit's six practice cases, scored with the local
  pinball scorer: **overall loss 0.266110** (median error 0.767503, 90%
  coverage 0.9084) — well below the house baseline's 0.363589. Per case:
  p00 0.464635, p01 0.310920, p02 0.333897, p03 0.232744, p04 0.200556,
  p05 0.117333.
- Unit tests: **20/20 pass** (`python3 -m unittest discover -s tests`).
- Determinism: two full runs over all six practice cases are byte-identical
  (`diff -r` clean).
- Timing: all six practice cases run in ~0.5 s total, far inside the 8 s
  per-case cap; a synthetic full-size case (24 markets x 4320 prices)
  forecasts in well under 2 s.
- Output shape checked: exactly one `forecasts` key, one row per market,
  240 rows x 5 quantiles, all finite in [0, 1], non-decreasing, no `-0.0`,
  each file under the 1 MiB cap.
