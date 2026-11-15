"""Tests for the FAA attribution forecaster (main.py), written from scratch.

Covers the output contract, the persistence median, band growth, edge-case
markets, the short-history fallback, empty cases, determinism, timing and
the end-to-end command line.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_PY = os.path.join(ROOT, "main.py")

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


def make_market(market_id, prices, horizon=None):
    """One market dict in the case format, with as many prices as given."""
    market = {"id": market_id, "name": market_id, "slug": market_id,
              "yes_token": "0" * 64, "start": "2026-11-12T00:00:00.000Z",
              "prices": list(prices)}
    return market


def make_case(markets, horizon=240, case_id="t00"):
    case = {"schema": "makefun/polymarket-case/v1",
            "competition": "polymarket-faa-attribution-01",
            "case": case_id, "at": "2026-11-12T06:00:00.000Z",
            "horizon_minutes": horizon, "quantiles": list(LEVELS),
            "event": {"slug": "test", "title": "Test"},
            "markets": markets}
    return case


def flat_prices(value, count):
    return [value] * count


def noisy_prices(seed, count, start=0.5, step=0.002):
    """A deterministic pseudo-random walk (no random module: fixed LCG)."""
    state = seed
    price = start
    out = []
    for _ in range(count):
        state = (1103515245 * state + 12345) % (1 << 31)
        move = ((state / (1 << 31)) - 0.5) * 2 * step
        price = min(0.999, max(0.001, price + move))
        out.append(round(price, 6))
    return out


class TestOutputShape(unittest.TestCase):
    def test_exactly_one_key_and_one_row_per_market(self):
        case = make_case([make_market("m1", flat_prices(0.4, 200)),
                          make_market("m2", flat_prices(0.6, 200))])
        result = main.forecast_case(case)
        self.assertEqual(list(result.keys()), ["forecasts"])
        self.assertEqual(len(result["forecasts"]), 2)
        self.assertEqual({row["id"] for row in result["forecasts"]},
                         {"m1", "m2"})
        for row in result["forecasts"]:
            self.assertEqual(list(row.keys()), ["id", "q"])

    def test_rows_columns_and_ranges(self):
        case = make_case([make_market("m1", noisy_prices(7, 1500))])
        result = main.forecast_case(case)
        rows = result["forecasts"][0]["q"]
        self.assertEqual(len(rows), 240)
        for row in rows:
            self.assertEqual(len(row), 5)
            for value in row:
                self.assertIsInstance(value, float)
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
            for a, b in zip(row, row[1:]):
                self.assertLessEqual(a, b)

    def test_horizon_comes_from_the_case(self):
        case = make_case([make_market("m1", flat_prices(0.3, 100))],
                         horizon=12)
        result = main.forecast_case(case)
        self.assertEqual(len(result["forecasts"][0]["q"]), 12)

    def test_no_negative_zero_anywhere(self):
        case = make_case([make_market("m1", flat_prices(0.0, 300)),
                          make_market("m2", flat_prices(1.0, 300))])
        result = main.forecast_case(case)
        text = json.dumps(result)
        self.assertNotIn("-0.0", text)

    def test_rounding_to_six_decimals(self):
        case = make_case([make_market("m1", noisy_prices(3, 500))])
        result = main.forecast_case(case)
        for row in result["forecasts"][0]["q"]:
            for value in row:
                self.assertEqual(value, round(value, 6))


class TestForecastBehaviour(unittest.TestCase):
    def test_median_is_the_last_price(self):
        """Persistence: every row's middle quantile is the checkpoint price."""
        prices = noisy_prices(11, 800, start=0.42)
        case = make_case([make_market("m1", prices)])
        rows = main.forecast_case(case)["forecasts"][0]["q"]
        for row in rows:
            self.assertEqual(row[2], round(prices[-1], 6))

    def test_bands_widen_with_horizon(self):
        """The 5%-95% spread grows monotonically as the horizon lengthens."""
        case = make_case([make_market("m1", noisy_prices(5, 1200))])
        rows = main.forecast_case(case)["forecasts"][0]["q"]
        spreads = [row[4] - row[0] for row in rows]
        for earlier, later in zip(spreads, spreads[1:]):
            self.assertLessEqual(earlier, later)
        self.assertGreater(spreads[-1], spreads[0])
        # And the first row is much narrower than the last.
        self.assertLess(spreads[0] * 2, spreads[-1])

    def test_flat_market_has_minimum_width(self):
        """A market that never moves still gets the floored volatility band,
        symmetric around its price."""
        case = make_case([make_market("m1", flat_prices(0.5, 500))])
        rows = main.forecast_case(case)["forecasts"][0]["q"]
        for row in rows:
            self.assertEqual(row[2], 0.5)
            self.assertLess(row[0], 0.5)
            self.assertGreater(row[4], 0.5)
            self.assertAlmostEqual(row[4] - 0.5, 0.5 - row[0], places=6)

    def test_single_price_market(self):
        """One price: no changes, so the case fallback applies and the
        median is still that single price."""
        case = make_case([make_market("m1", [0.33])])
        rows = main.forecast_case(case)["forecasts"][0]["q"]
        self.assertEqual(len(rows), 240)
        for row in rows:
            self.assertEqual(row[2], 0.33)

    def test_empty_price_list_centres_on_half(self):
        case = make_case([make_market("m1", [])])
        rows = main.forecast_case(case)["forecasts"][0]["q"]
        for row in rows:
            self.assertEqual(row[2], 0.5)
            self.assertLess(row[0], 0.5)
            self.assertGreater(row[4], 0.5)

    def test_prices_at_zero_and_one(self):
        case = make_case([make_market("low", flat_prices(0.0, 200)),
                          make_market("high", flat_prices(1.0, 200))])
        result = main.forecast_case(case)
        low = result["forecasts"][0]["q"]
        high = result["forecasts"][1]["q"]
        for row in low:
            self.assertEqual(row[2], 0.0)
            self.assertEqual(row[0], 0.0)
        for row in high:
            self.assertEqual(row[2], 1.0)
            self.assertEqual(row[4], 1.0)

    def test_nonfinite_and_boolean_prices_are_skipped(self):
        dirty = [0.5, float("nan"), True, False, None, "0.6",
                 float("inf"), 0.52, 0.53]
        case = make_case([make_market("m1", dirty)])
        rows = main.forecast_case(case)["forecasts"][0]["q"]
        # Only 0.5, 0.52, 0.53 survive, so the median is the last of them.
        for row in rows:
            self.assertEqual(row[2], 0.53)

    def test_volatile_market_is_wider_than_quiet_one(self):
        quiet = make_market("quiet", flat_prices(0.5, 1000))
        loud = make_market("loud", noisy_prices(9, 1000, step=0.02))
        case = make_case([quiet, loud])
        result = main.forecast_case(case)
        q_spread = result["forecasts"][0]["q"][-1][4] - \
            result["forecasts"][0]["q"][-1][0]
        l_spread = result["forecasts"][1]["q"][-1][4] - \
            result["forecasts"][1]["q"][-1][0]
        self.assertGreater(l_spread, q_spread)

    def test_short_history_uses_case_median_volatility(self):
        """A market with 5 prices (< 10 changes) borrows the median
        volatility of the markets that can estimate their own."""
        estimable = make_market("big", noisy_prices(21, 1000, step=0.01))
        short = make_market("short", [0.4, 0.41, 0.4, 0.42, 0.41])
        mixed = make_case([estimable, short])
        alone = make_case([short])
        mixed_rows = main.forecast_case(mixed)["forecasts"][1]["q"]
        alone_rows = main.forecast_case(alone)["forecasts"][0]["q"]
        # With a volatile neighbour the band is wider than with the default
        # 0.0025 fallback alone.
        self.assertGreater(mixed_rows[-1][4] - mixed_rows[-1][0],
                           alone_rows[-1][4] - alone_rows[-1][0])
        for row in mixed_rows:
            self.assertEqual(row[2], 0.41)

    def test_no_markets_gives_empty_forecasts(self):
        result = main.forecast_case(make_case([]))
        self.assertEqual(result, {"forecasts": []})


class TestDeterminismAndCli(unittest.TestCase):
    def test_two_runs_are_identical(self):
        case = make_case([make_market("m1", noisy_prices(13, 4320)),
                          make_market("m2", noisy_prices(17, 4320))])
        first = main.forecast_case(case)
        second = main.forecast_case(case)
        self.assertEqual(json.dumps(first, sort_keys=True),
                         json.dumps(second, sort_keys=True))

    def test_full_72h_case_runs_fast(self):
        """A full-size case (24 markets x 4320 prices) must finish well
        inside the 8-second per-case cap."""
        markets = [make_market(f"m{i}", noisy_prices(100 + i, 4320))
                   for i in range(24)]
        case = make_case(markets)
        started = time.perf_counter()
        main.forecast_case(case)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 2.0)

    def test_cli_end_to_end_on_two_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(in_dir)
            cases = {
                "p00": make_case([make_market("a", noisy_prices(31, 3000)),
                                  make_market("b", noisy_prices(37, 3000))],
                                 case_id="p00"),
                "p01": make_case([make_market("a", noisy_prices(41, 3200))],
                                 case_id="p01"),
            }
            for case_id, case in cases.items():
                with open(os.path.join(in_dir, case_id + ".in"), "w") as fh:
                    json.dump(case, fh)
            env = dict(os.environ, MAKEFUN_INPUT_DIR=in_dir,
                       MAKEFUN_OUTPUT_DIR=out_dir)
            proc = subprocess.run([sys.executable, MAIN_PY],
                                  env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(sorted(os.listdir(out_dir)),
                             ["p00.out", "p01.out"])
            for case_id, case in cases.items():
                with open(os.path.join(out_dir, case_id + ".out")) as fh:
                    text = fh.read()
                self.assertTrue(text.endswith("\n"))
                result = json.loads(text)
                self.assertEqual(len(result["forecasts"]),
                                 len(case["markets"]))
                for row in result["forecasts"]:
                    self.assertEqual(len(row["q"]), 240)
                # Compact JSON: no spaces after separators.
                self.assertNotIn('": ', text)
                self.assertLessEqual(len(text.encode("utf-8")), 1048576)

    def test_cli_creates_output_dir_and_handles_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "deep", "out")
            os.makedirs(in_dir)
            env = dict(os.environ, MAKEFUN_INPUT_DIR=in_dir,
                       MAKEFUN_OUTPUT_DIR=out_dir)
            proc = subprocess.run([sys.executable, MAIN_PY],
                                  env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(os.path.isdir(out_dir))
            self.assertEqual(os.listdir(out_dir), [])

    def test_cli_two_runs_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_a = os.path.join(tmp, "out_a")
            out_b = os.path.join(tmp, "out_b")
            os.makedirs(in_dir)
            case = make_case([make_market("a", noisy_prices(53, 4320)),
                              make_market("b", noisy_prices(59, 4320))])
            with open(os.path.join(in_dir, "p00.in"), "w") as fh:
                json.dump(case, fh)
            for out_dir in (out_a, out_b):
                env = dict(os.environ, MAKEFUN_INPUT_DIR=in_dir,
                           MAKEFUN_OUTPUT_DIR=out_dir)
                proc = subprocess.run([sys.executable, MAIN_PY],
                                      env=env, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(os.path.join(out_a, "p00.out"), "rb") as fh:
                first = fh.read()
            with open(os.path.join(out_b, "p00.out"), "rb") as fh:
                second = fh.read()
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
