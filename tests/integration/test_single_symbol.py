"""T014 - single-symbol run reports baseline vs tuned and is deterministic."""
from optimization import runner


def test_run_reports_baseline_and_tuned(real_or_synth_csv):
    input_dir, symbol, csv_path = real_or_synth_csv
    res = runner.run_symbol(symbol, csv_path, forward_days=[5, 10],
                            tune="weights", seed=42)
    assert res.status in ("improved", "no_improvement")
    # Both baseline and tuned held-out metrics are present (FR-006, SC-003).
    assert res.baseline_metrics is not None and res.tuned_metrics is not None
    assert "objective_score" in res.baseline_metrics
    assert "objective_score" in res.tuned_metrics
    # Window carries an embargo (Principle II).
    assert res.window["embargo_bars"] >= 10
    # Accepted configs are never worse than baseline on held-out (SC-004).
    if res.accepted:
        assert (res.tuned_metrics["objective_score"]
                > res.baseline_metrics["objective_score"])
        assert res.selected_config is not None


def test_run_is_deterministic(real_or_synth_csv):
    input_dir, symbol, csv_path = real_or_synth_csv
    a = runner.run_symbol(symbol, csv_path, tune="weights", seed=7)
    b = runner.run_symbol(symbol, csv_path, tune="weights", seed=7)
    assert a.status == b.status
    assert a.baseline_metrics == b.baseline_metrics
    assert a.tuned_metrics == b.tuned_metrics
    assert a.selected_config == b.selected_config
