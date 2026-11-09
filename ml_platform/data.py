"""Deterministic synthetic data for predictive maintenance.

Domain: industrial equipment sensor readings. Target: will the equipment
fail within 30 days (1) or not (0)? Features are temperature, vibration,
pressure, operating hours and load. The label is drawn from a known
logistic function of the features, so a logistic model can learn real
signal — but the data is entirely synthetic, small, and not a substitute
for real equipment data.

Default size: 600 rows, seed 42, split 480 train / 120 held-out.
"""

import csv
import random

from .schema import LABEL

HEADER = ["temperature", "vibration", "pressure", "operating_hours", "load", LABEL]


def _clip(value, low, high):
    return max(low, min(high, value))


def generate_synthetic(n=600, seed=42):
    """Generate n labelled rows deterministically from the given seed."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        temperature = _clip(rng.gauss(72.0, 8.0), 50.0, 110.0)
        vibration = _clip(rng.gauss(2.5, 1.2), 0.2, 8.0)
        pressure = _clip(rng.gauss(100.0, 15.0), 60.0, 150.0)
        operating_hours = rng.uniform(0.0, 8000.0)
        load = rng.uniform(0.3, 1.0)
        logit = (
            -5.0
            + 0.06 * (temperature - 70.0)
            + 0.9 * vibration
            + 0.02 * (pressure - 100.0)
            + 0.0003 * operating_hours
            + 1.5 * load
        )
        import math

        prob = 1.0 / (1.0 + math.exp(-logit))
        label = 1 if rng.random() < prob else 0
        rows.append(
            {
                "temperature": round(temperature, 2),
                "vibration": round(vibration, 2),
                "pressure": round(pressure, 2),
                "operating_hours": round(operating_hours, 1),
                "load": round(load, 3),
                LABEL: label,
            }
        )
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for raw in reader:
            row = {}
            for key, value in raw.items():
                if key == LABEL:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
        return rows


def split(rows, test_size=0.2, seed=42):
    """Deterministic shuffle split into (train_rows, held_out_rows)."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1.0 - test_size))
    return shuffled[:cut], shuffled[cut:]
