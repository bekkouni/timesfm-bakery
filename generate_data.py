"""Generate the deterministic bakery history and the held-out, original-plan outcome."""
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
CONTEXT = 256
HORIZON = 28
START = date(2026, 11, 2)  # Monday
ORIGINAL = 2  # Wednesday, zero-based within the forecast
CLOSURE = 17
SEED = 1126


def write_csv(name, columns, records):
    with (DATA / name).open("w", newline="", encoding="utf-8") as dest:
        writer = csv.writer(dest, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(records)


def main():
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    count = CONTEXT + HORIZON
    dates = [START + timedelta(days=i - CONTEXT) for i in range(count)]
    weekday = np.array([day.weekday() for day in dates])
    index = np.arange(count)
    sale = np.zeros(count, dtype=int)
    day = 4
    while day < CONTEXT:
        sale[day] = 1
        day += int(rng.integers(7, 16))
    sale[CONTEXT + ORIGINAL] = 1
    demand = np.zeros(count)
    for i in range(1, count):
        demand[i] = 0.89 * demand[i-1] + rng.normal(0, 4)
    # A recurring wave of foot traffic leads bread demand by two days.
    footfall = 115 + 23 * np.sin(index * 2 * np.pi / 23) + demand
    visitors = np.rint(footfall + 23 * (weekday >= 5) + rng.normal(0, 3, count)).astype(int)
    lag = footfall[np.maximum(index - 2, 0)] - 115
    loaf = np.rint(65 + 0.65 * lag + 14 * (weekday >= 5) + 47 * sale + rng.normal(0, 3, count)).astype(int)
    rolls = np.rint(32 + 0.50 * (visitors - 110) + rng.normal(0, 3, count)).astype(int)
    baguettes = np.rint(24 + 0.33 * (visitors - 110) + rng.normal(0, 2, count)).astype(int)
    loaf[CONTEXT + CLOSURE] = 0
    write_csv("history.csv", ["date", "loaf", "roll", "baguette", "visitors", "sale"],
              ((dates[i], loaf[i], rolls[i], baguettes[i], visitors[i], sale[i]) for i in range(CONTEXT)))
    write_csv("actual.csv", ["date", "loaf", "event"],
              ((dates[i], loaf[i], "臨時休業" if i == CONTEXT + CLOSURE else "") for i in range(CONTEXT, count)))


if __name__ == "__main__":
    main()
