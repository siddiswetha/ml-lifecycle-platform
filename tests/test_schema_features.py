from ml_platform.features import engineer, engineer_all
from ml_platform.schema import feature_statistics, validate_dataset, validate_row


def _row(**overrides):
    row = {"temperature": 72.0, "vibration": 2.5, "pressure": 100.0,
           "operating_hours": 1000.0, "load": 0.6}
    row.update(overrides)
    return row


def test_valid_row_has_no_errors():
    assert validate_row(_row()) == []


def test_missing_feature_is_error():
    row = _row()
    del row["temperature"]
    assert any("temperature" in e for e in validate_row(row))


def test_out_of_range_is_error():
    assert validate_row(_row(temperature=200.0))
    assert validate_row(_row(load=2.0))


def test_non_numeric_is_error():
    assert validate_row(_row(vibration="high"))


def test_validate_dataset_splits_valid_and_invalid():
    valid, errors = validate_dataset([_row(), _row(temperature=999.0), _row()])
    assert len(valid) == 2
    assert len(errors) == 1
    assert errors[0]["index"] == 1


def test_feature_statistics():
    stats = feature_statistics([_row(), _row(temperature=82.0)])
    assert stats["temperature"]["mean"] == 77.0
    assert stats["temperature"]["n"] == 2


def test_engineer_adds_four_features():
    out = engineer(_row())
    assert out["temp_vibration"] == 72.0 * 2.5 / 100.0
    assert out["pressure_deviation"] == 0.0
    assert out["load_hours"] == 0.6
    assert abs(out["vibration_per_load"] - 2.5 / 0.7) < 0.001


def test_engineer_all_preserves_count():
    assert len(engineer_all([_row(), _row()])) == 2
