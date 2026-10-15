"""Portable-template contracts, with synthetic inputs only."""

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s2_lstm_b_train import prepare_lstm_b_data
from backtest import calc_metrics, run_topk_strategy, validate_backtest_inputs
from run_original_architecture import original_regime_ensemble
from demo_data import generate_demo
from s1_gnn_train import IndustryGAT, build_monthly_graphs
from s2_lstm_b_train import LSTMBModel
import config


class LstmBoundaryTests(unittest.TestCase):
    @staticmethod
    def fixture(periods=8):
        dates = pd.date_range("2020-01-31", periods=periods, freq="ME")
        tech = pd.DataFrame({"ts_code": "A", "date": dates,
                             "momentum": np.arange(periods, dtype=float)})
        market = pd.DataFrame({"ts_code": "A", "date": dates,
                               "ret": np.arange(periods, dtype=float) / 100})
        return tech, market, dates

    def test_signal_month_targets_next_month_and_last_signal_is_unlabeled(self):
        tech, market, dates = self.fixture(5)
        _, labels, meta, _ = prepare_lstm_b_data(
            tech, market, ["A"], (dates[0], dates[-1]), lookback=2,
            include_unlabeled=True,
        )
        self.assertEqual([pd.Timestamp(item[1]) for item in meta], list(dates[1:]))
        np.testing.assert_allclose(labels[:3], [.02, .03, .04])
        self.assertTrue(np.isnan(labels[-1]))

    def test_label_boundary_excludes_not_yet_mature_returns(self):
        tech, market, dates = self.fixture(6)
        _, labels, meta, _ = prepare_lstm_b_data(
            tech, market, ["A"], (dates[0], dates[3]), lookback=2,
            label_end=dates[3],
        )
        self.assertEqual([pd.Timestamp(item[1]) for item in meta], list(dates[1:3]))
        np.testing.assert_allclose(labels, [.02, .03])
        self.assertTrue(all(pd.Timestamp(item[1]) + pd.offsets.MonthEnd(1) <= dates[3]
                            for item in meta))

    def test_missing_month_is_not_silently_treated_as_one_month(self):
        tech, market, dates = self.fixture(6)
        tech = tech.loc[tech.date != dates[2]].copy()
        market = market.loc[market.date != dates[2]].copy()
        with self.assertRaises(ValueError):
            prepare_lstm_b_data(tech, market, ["A"], (dates[0], dates[-1]),
                                lookback=2, include_unlabeled=True)

    def test_missing_factor_month_is_not_silently_inner_joined_away(self):
        tech, market, dates = self.fixture(6)
        tech = tech.loc[tech.date != dates[2]].copy()
        with self.assertRaises(ValueError):
            prepare_lstm_b_data(tech, market, ["A"], (dates[0], dates[-1]),
                                lookback=2, include_unlabeled=True)

    def test_explicit_empty_scalers_never_refit_prediction_data(self):
        tech, market, dates = self.fixture()
        with self.assertRaises((ValueError, KeyError)):
            prepare_lstm_b_data(
                tech, market, ["A"], (dates[3], dates[-1]), lookback=2,
                scaler_dict={}, include_unlabeled=True,
            )

    def test_reused_training_scaler_does_not_fit_validation_extremes(self):
        tech, market, dates = self.fixture()
        _, _, _, scalers = prepare_lstm_b_data(
            tech, market, ["A"], (dates[0], dates[3]), lookback=2,
            label_end=dates[3],
        )
        minimum = scalers["A"].data_min_.copy()
        maximum = scalers["A"].data_max_.copy()
        tech.loc[tech.date >= dates[4], "momentum"] = 100000.
        x, _, _, reused = prepare_lstm_b_data(
            tech, market, ["A"], (dates[2], dates[-1]), target_start=dates[4],
            lookback=2, scaler_dict=scalers, include_unlabeled=True,
        )
        self.assertGreater(len(x), 0)
        np.testing.assert_array_equal(reused["A"].data_min_, minimum)
        np.testing.assert_array_equal(reused["A"].data_max_, maximum)
        np.testing.assert_array_equal(x[:, :, 1], np.ones(x.shape[:2]))

    def test_future_append_does_not_change_training_samples_or_scaler(self):
        tech, market, dates = self.fixture()
        kwargs = dict(industries=["A"], date_range=(dates[0], dates[4]),
                      lookback=2, label_end=dates[4])
        before = prepare_lstm_b_data(tech.iloc[:5], market.iloc[:5], **kwargs)
        tech.loc[tech.date > dates[4], "momentum"] = 99999.
        market.loc[market.date > dates[4], "ret"] = -99999.
        after = prepare_lstm_b_data(tech, market, **kwargs)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2], after[2])
        np.testing.assert_array_equal(before[3]["A"].data_min_, after[3]["A"].data_min_)
        np.testing.assert_array_equal(before[3]["A"].data_max_, after[3]["A"].data_max_)


class CoreModelTests(unittest.TestCase):
    def test_reference_model_shapes_and_hmm_configuration(self):
        self.assertEqual((config.HMM_N_STATES, config.HMM_COVARIANCE, config.HMM_TRAIN_WINDOW),
                         (4, "full", 60))
        gnn = IndustryGAT(14 + 2 + 4)
        self.assertEqual(tuple(gnn.W.weight.shape), (16, 20))
        self.assertEqual(tuple(gnn.a_src.shape), (16, 1))
        lstm = LSTMBModel(12 + 29)
        self.assertEqual(tuple(lstm.lstm1.weight_ih_l0.shape), (256, 41))
        self.assertEqual(tuple(lstm.lstm2.weight_ih_l0.shape), (128, 64))
        self.assertEqual(tuple(lstm.fc.weight.shape), (1, 32))

    def test_graph_uses_only_strictly_preceding_month_returns(self):
        dates = pd.date_range("2020-01-31", periods=14, freq="ME")
        frame = pd.DataFrame([
            {"ts_code": code, "date": date, "ret": (i + j) / 100}
            for i, date in enumerate(dates) for j, code in enumerate(("A", "B"))
        ])
        cutoff = dates[11]
        captured = []

        def record_history(values):
            captured.append(values.copy())
            return np.eye(values.shape[1])

        with patch("s1_gnn_train.build_glasso_graph", side_effect=record_history):
            build_monthly_graphs(frame, ["A", "B"], [cutoff])
            frame.loc[frame.date >= cutoff, "ret"] = 99999.
            build_monthly_graphs(frame, ["A", "B"], [cutoff])
        np.testing.assert_array_equal(captured[0], captured[1])
        self.assertEqual(captured[0].shape, (11, 2))


class PackagingTests(unittest.TestCase):
    def test_source_contains_no_machine_specific_drive_paths(self):
        pattern = re.compile(r"[A-Za-z]:[\\/]")
        for path in sorted((ROOT / "src").rglob("*.py")):
            with self.subTest(source=path.name):
                self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")))

    def test_import_does_not_create_directories_or_replace_stdout(self):
        # Dependencies may initialize their own state. The template modules may not.
        script = """
import sys, os, pathlib
import numpy, pandas, torch, scipy, sklearn, hmmlearn
import scipy.stats, sklearn.preprocessing, sklearn.covariance, hmmlearn.hmm
sys.path.insert(0, sys.argv[1])
original_stdout = sys.stdout
def forbidden(*args, **kwargs):
    raise AssertionError('Import attempted to create a directory')
os.makedirs = forbidden
pathlib.Path.mkdir = forbidden
import config
import s1_gnn_train
import s2_lstm_b_train
import run_original_architecture
assert sys.stdout is original_stdout, 'Import replaced caller stdout'
"""
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
            result = subprocess.run(
                [sys.executable, "-B", "-c", script, str(ROOT / "src")],
                cwd=directory, env=environment, capture_output=True, text=True,
                encoding="utf-8", timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])


class BacktestBoundaryTests(unittest.TestCase):
    @staticmethod
    def predictions():
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        market = pd.DataFrame([
            {"ts_code": f"I{i}", "date": date, "ret": i / 100}
            for date in dates for i in range(6)
        ])
        predictions = pd.DataFrame([
            {"ts_code": f"I{i}", "date": date, "pred_gnn": float(i),
             "pred_lstm_b": float(5 - i),
             "actual_ret": i / 100 if date != dates[-1] else np.nan}
            for date in dates for i in range(6)
        ])
        return predictions, market, dates

    def test_first_month_loss_counts_in_drawdown(self):
        metrics = calc_metrics(pd.Series([-.10, .05]))
        self.assertAlmostEqual(metrics["max_drawdown"], -.10)
        self.assertAlmostEqual(metrics["total_return"], .90 * 1.05 - 1)

    def test_unknown_last_signal_is_pending_not_an_extra_return(self):
        predictions, _, dates = self.predictions()
        returns, holdings = run_topk_strategy(predictions, "pred_gnn", k=2)
        self.assertEqual(len(returns), 2)
        np.testing.assert_allclose(returns, [.045, .045])
        self.assertEqual(holdings.iloc[-1]["status"], "pending")
        self.assertEqual(holdings.iloc[-1]["holding_month"], dates[-1] + pd.offsets.MonthEnd(1))
        self.assertEqual(holdings.iloc[-1]["holdings"], ["I5", "I4"])

    def test_selection_cannot_replace_a_future_missing_label(self):
        predictions, _, dates = self.predictions()
        predictions.loc[(predictions.date == dates[0]) & (predictions.ts_code == "I5"), "actual_ret"] = np.nan
        with self.assertRaisesRegex(ValueError, "selected index"):
            run_topk_strategy(predictions, "pred_gnn", k=2)

    def test_missing_prediction_month_fails_validation(self):
        predictions, market, dates = self.predictions()
        lstm = predictions.loc[predictions.date != dates[1]].copy()
        with self.assertRaisesRegex(ValueError, "missing prediction months"):
            validate_backtest_inputs(predictions, lstm, market, as_of="2020-05-01")

    def test_same_month_or_multimonth_labels_fail_validation(self):
        predictions, market, dates = self.predictions()
        predictions.loc[predictions.date == dates[0], "actual_ret"] += .1
        with self.assertRaisesRegex(ValueError, "next-calendar-month"):
            validate_backtest_inputs(predictions, predictions, market, as_of="2020-05-01")
        predictions, market, dates = self.predictions()
        market = market.loc[~((market.ts_code == "I5") & (market.date == dates[1]))].copy()
        with self.assertRaises(ValueError):
            validate_backtest_inputs(predictions, predictions, market, as_of="2020-05-01")

    def test_delayed_realized_labels_delay_hmm_weight_updates(self):
        predictions, _, dates = self.predictions()
        predictions["label_available_at"] = predictions.date + pd.offsets.MonthEnd(1)
        predictions.loc[predictions.date == dates[0], "label_available_at"] = dates[2]
        regimes = pd.DataFrame({"date": dates, "regime": 0})
        result = original_regime_ensemble(predictions.drop(columns="pred_lstm_b"), predictions, regimes)
        np.testing.assert_allclose(result.groupby("date").w_gnn.first(), [.5, .5, 1.])

    def test_forward_label_cannot_claim_availability_before_holding_month_end(self):
        predictions, _, dates = self.predictions()
        predictions["label_available_at"] = predictions.date
        regimes = pd.DataFrame({"date": dates, "regime": 0})
        with self.assertRaisesRegex(ValueError, "cannot mature"):
            original_regime_ensemble(predictions.drop(columns="pred_lstm_b"), predictions, regimes)


class SyntheticDemoTests(unittest.TestCase):
    def test_same_seed_reproduces_all_six_input_files(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            first = generate_demo(Path(directory) / "first", months=74, assets=10)
            second = generate_demo(Path(directory) / "second", months=74, assets=10)
            self.assertEqual(first["files"], second["files"])
            self.assertEqual(len(first["files"]), 6)
            self.assertTrue(first["synthetic"])
            self.assertFalse(first["contains_real_market_data"])
            self.assertEqual(len(first["technical_features"]) + len(first["pattern_features"]), 12)
            self.assertTrue(all(code.startswith("SYNTH") for code in first["industry_codes"]))

    def test_demo_rejects_existing_directory_without_writing(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            with self.assertRaises(FileExistsError):
                generate_demo(Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_invalid_demo_dimensions_do_not_create_destination(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            target = Path(directory) / "invalid"
            for kwargs in ({"months": 73}, {"assets": 9}, {"seed": -1}):
                with self.assertRaises(ValueError):
                    generate_demo(target, **kwargs)
            self.assertFalse(target.exists())

    def test_check_rejects_missing_fundamental_node_in_final_usable_quarter(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory) / "data"
            manifest = generate_demo(root)
            with patch.object(config, "LOCAL_DATA_RAW", str(root / "raw")), \
                 patch.object(config, "LOCAL_DATA_PROCESSED", str(root / "processed")):
                config.check_data()
                path = root / "processed/prosperity_indicators_clean.pkl"
                frame = pd.read_pickle(path)
                required_quarter = pd.Period(manifest["end"], freq="Q") - 1
                absent = ((frame.l1_code == manifest["industry_codes"][0]) &
                          (frame.year == required_quarter.year) &
                          (frame.quarter == required_quarter.quarter))
                self.assertEqual(int(absent.sum()), 1)
                frame.loc[~absent].to_pickle(path)
                with self.assertRaises(ValueError):
                    config.check_data()

    def test_check_rejects_one_missing_technical_node_at_final_month(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory) / "data"
            manifest = generate_demo(root)
            with patch.object(config, "LOCAL_DATA_RAW", str(root / "raw")), \
                 patch.object(config, "LOCAL_DATA_PROCESSED", str(root / "processed")):
                config.check_data()
                path = root / "processed/price_volume_factors.pkl"
                frame = pd.read_pickle(path)
                absent = ((frame.ts_code == manifest["industry_codes"][0]) &
                          (frame.date == manifest["end"]))
                self.assertEqual(int(absent.sum()), 1)
                frame.loc[~absent].to_pickle(path)
                with self.assertRaises(ValueError):
                    config.check_data()


if __name__ == "__main__":
    unittest.main()
