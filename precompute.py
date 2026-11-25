"""Generate all 29 sale calendars and time warm inference on the chosen device."""
import argparse
import json
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path

from forecast import RESULTS, load_model, predict, read_history, runtime_info, signature
from generate_data import HORIZON, ORIGINAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    data = read_history()
    start = time.perf_counter()
    model = load_model(args.device)
    load_seconds = time.perf_counter() - start
    bundle = {"input": signature(), "C": {}}
    for condition in ("A", "B"):
        bundle[condition], seconds = predict(model, data, condition)
        print(f"{condition}: {seconds:.2f} s", flush=True)
    elapsed = []
    for day in range(-1, HORIZON):
        values, seconds = predict(model, data, "C", day)
        bundle["C"][str(day)] = values
        elapsed.append(seconds)
        print(f"C {day:2d}: {seconds:.2f} s", flush=True)
    bundle["benchmark"] = {"device": args.device, "processor": platform.processor(),
                           "packages": runtime_info(), "load_seconds": round(load_seconds, 3),
                           "median_seconds": round(statistics.median(elapsed), 3),
                           "max_seconds": round(max(elapsed), 3), "n": len(elapsed)}
    with tempfile.NamedTemporaryFile(dir=RESULTS.parent, suffix=".tmp", mode="w", encoding="utf-8",
                                     delete=False) as writer:
        json.dump(bundle, writer, allow_nan=False, separators=(",", ":"))
        temp_path = Path(writer.name)
    os.replace(temp_path, RESULTS)
    print(bundle["benchmark"])
    old, new = (bundle["C"][str(i)]["p50"] for i in (ORIGINAL, 4))
    print("Wednesday / Friday forecast changes:", [round(new[i] - old[i], 1) for i in (ORIGINAL, 4)])


if __name__ == "__main__":
    main()
