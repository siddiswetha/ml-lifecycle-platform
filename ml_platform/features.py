"""Feature engineering for the sensor readings.

The five raw sensor features are kept as-is, and four derived features
are computed deterministically from them before training or prediction:

- temp_vibration: temperature x vibration / 100 — hot, shaking equipment
  is a stronger failure signal than either reading alone.
- load_hours: load x operating_hours / 1000 — cumulative work done.
- pressure_deviation: pressure - 100 — signed distance from the
  nominal 100 bar operating point.
- vibration_per_load: vibration / (load + 0.1) — vibration relative to
  how hard the equipment is working.

Engineering is applied identically in training, evaluation and serving,
so a model never sees a different feature set at prediction time than
it was trained on. The transforms use only the raw readings — no
information from the label.
"""

RAW_FEATURES = ["temperature", "vibration", "pressure", "operating_hours", "load"]

ENGINEERED_FEATURES = [
    "temp_vibration",
    "load_hours",
    "pressure_deviation",
    "vibration_per_load",
]

ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


def engineer(row):
    """Return a new dict with the raw features plus the derived ones."""
    out = dict(row)
    temperature = float(row["temperature"])
    vibration = float(row["vibration"])
    pressure = float(row["pressure"])
    operating_hours = float(row["operating_hours"])
    load = float(row["load"])
    out["temp_vibration"] = round(temperature * vibration / 100.0, 4)
    out["load_hours"] = round(load * operating_hours / 1000.0, 4)
    out["pressure_deviation"] = round(pressure - 100.0, 4)
    out["vibration_per_load"] = round(vibration / (load + 0.1), 4)
    return out


def engineer_all(rows):
    return [engineer(r) for r in rows]
