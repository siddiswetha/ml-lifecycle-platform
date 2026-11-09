from ml_platform import data as data_module
from ml_platform.drift import baseline_stats, drift_report


def test_generate_is_deterministic():
    a = data_module.generate_synthetic(n=50, seed=42)
    b = data_module.generate_synthetic(n=50, seed=42)
    assert a == b


def test_generate_different_seed_differs():
    a = data_module.generate_synthetic(n=50, seed=1)
    b = data_module.generate_synthetic(n=50, seed=2)
    assert a != b


def test_generate_size_and_labels():
    rows = data_module.generate_synthetic(n=100, seed=42)
    assert len(rows) == 100
    assert set(r["will_fail_30d"] for r in rows) <= {0, 1}
    assert 0 < sum(r["will_fail_30d"] for r in rows) < 100


def test_split_sizes():
    rows = data_module.generate_synthetic(n=100, seed=42)
    train, held = data_module.split(rows, test_size=0.2, seed=42)
    assert len(train) == 80
    assert len(held) == 20


def test_split_is_deterministic():
    rows = data_module.generate_synthetic(n=100, seed=42)
    assert data_module.split(rows) == data_module.split(rows)


def test_csv_round_trip(tmp_path):
    rows = data_module.generate_synthetic(n=20, seed=3)
    path = tmp_path / "data.csv"
    data_module.write_csv(rows, str(path))
    back = data_module.read_csv(str(path))
    assert len(back) == 20
    assert back[0]["temperature"] == rows[0]["temperature"]


def test_baseline_stats():
    rows = data_module.generate_synthetic(n=50, seed=42)
    stats = baseline_stats(rows)
    assert stats["temperature"]["n"] == 50
    assert stats["temperature"]["std"] > 0


def test_drift_report_same_distribution_not_drifted():
    rows = data_module.generate_synthetic(n=200, seed=42)
    train, held = data_module.split(rows, test_size=0.2, seed=42)
    report = drift_report(train, held)
    assert report["drifted"] is False


def test_drift_report_shifted_is_drifted():
    rows = data_module.generate_synthetic(n=200, seed=42)
    shifted = [dict(r, temperature=r["temperature"] + 40.0) for r in rows]
    report = drift_report(rows, shifted)
    assert report["drifted"] is True
    assert report["features"]["temperature"]["drifted"] is True
