"""gta6-twitch-ccu-forecast-01 entry: kernel-regression forecaster (tuned v2).

Model, in one paragraph: for each hour h = 2..24 of release day, take every
comparable past launch's log-ratio log(viewers_h / viewers_1) on its own
release day and fit it against that launch's log hour-1 print with a
Gaussian-kernel weighted local-linear regression (bandwidth 2.0 in log
space), evaluated at this case's log hour-1 print. That fitted value is the
median log-ratio. The spread is the kernel-weighted root-mean-square
residual of the same fit, scaled by 0.55 and floored at 0.10 log units, and
the five quantiles are the median times exp(z_tau * spread) for the standard
normal z at each level. Everything is computed from the case file at run
time; nothing about any past case is baked in.

Why these numbers: on the four practice cases the unscaled residual spread
is far too wide (the 90% band covered 100% of scored hours), so the spread
is scaled down; the wider bandwidth smooths the median across launches and
keeps the fit stable when hour 1 is larger than any past launch's. Both
choices were picked by leave-one-out validation over the practice cases,
not by fitting the practice scores directly.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only, no network. It reads every
<id>.in in the input directory and writes <id>.out to the output directory:
{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}, exactly 23
rows, hours 2..24 each once, five finite numbers above 0 and at most
1e9, non-decreasing. Outside the grader, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry ships no other module.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Kernel bandwidth in log hour-1 space for the local-linear fit.
BANDWIDTH = 2.0
# The residual spread is scaled down (the raw cross-launch spread is far
# wider than the conditional spread) and floored, in log units.
SPREAD_SCALE = 0.55
SPREAD_FLOOR = 0.10
# Fallback spread (persistence baseline) if an hour has no donors at all.
FALLBACK_SIGMA = 0.15


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0). Rounding also keeps two runs
    byte-identical.
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def release_day_ratios(case: dict) -> list[tuple[float, list]]:
    """Per comparable launch: (log hour-1 print, [log ratio h=2..24]).

    A launch counts only if its release day's first hour has a print; an
    hour with no print is left as None for that launch. The Grand Theft
    Auto VI category history has no release-day hour 1 (it starts at the
    premiere), so it is naturally left out: a trailer spike is not a
    launch day, and its shape would not transfer.
    """
    out = []
    for comp in case["comparables"]:
        day_start = comp["day_start"]
        by_hour = {}
        for row in comp["hourly"]:
            by_hour[row["from"]] = row["viewers"]
        first = by_hour.get(day_start)
        if not first or first <= 0:
            continue
        ratios = []
        for h in range(2, 25):
            # Rebuild the ISO hour stamp for day_start + (h - 1) hours.
            stamp = shift_hour(day_start, h - 1)
            viewers = by_hour.get(stamp)
            if viewers and viewers > 0:
                ratios.append(math.log(viewers / first))
            else:
                ratios.append(None)
        out.append((math.log(first), ratios))
    return out


def shift_hour(iso: str, hours: int) -> str:
    """'2022-02-25T00:00:00Z' plus `hours` hours, same format."""
    year = int(iso[0:4])
    month = int(iso[5:7])
    day = int(iso[8:10])
    hour = int(iso[11:13])
    # Days in month, with leap years, so the shift stays exact.
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    month_days = [31, 29 if leap else 28, 31, 30, 31, 30,
                  31, 31, 30, 31, 30, 31]
    hour += hours
    while hour >= 24:
        hour -= 24
        day += 1
        if day > month_days[month - 1]:
            day = 1
            month += 1
            if month > 12:
                month = 1
                year += 1
                leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
                month_days = [31, 29 if leap else 28, 31, 30, 31, 30,
                              31, 31, 30, 31, 30, 31]
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00Z"


def local_linear(xs: list[float], ys: list[float], ws: list[float],
                 x0: float) -> tuple[float, float]:
    """Gaussian-kernel weighted local-linear fit.

    Returns (prediction at x0, weighted RMS residual around the fit).
    With no usable slope the fit falls back to the weighted mean.
    """
    total_w = sum(ws)
    mean_x = sum(w * x for w, x in zip(ws, xs)) / total_w
    mean_y = sum(w * y for w, y in zip(ws, ys)) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mean_x) * (y - mean_y)
              for w, x, y in zip(ws, xs, ys))
    if sxx <= 1e-12:
        slope = 0.0
    else:
        slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    prediction = intercept + slope * x0
    variance = sum(w * (y - (intercept + slope * x)) ** 2
                   for w, x, y in zip(ws, xs, ys)) / total_w
    return prediction, math.sqrt(variance)


def hour_quantiles(p1: float, median_log_ratio: float, spread: float) -> list:
    """Five quantiles around hour 1's print, non-decreasing, in range."""
    quantiles = []
    for z in Z:
        q = p1 * math.exp(median_log_ratio + z * spread)
        q = round2(q)
        if q <= 0:
            q = 1
        if q > 1e9:
            q = 1e9
        quantiles.append(q)
    for i in range(1, 5):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    return quantiles


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    x0 = math.log(p1)
    donors = release_day_ratios(case)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        xs, ys, ws = [], [], []
        for log_h1, ratios in donors:
            ratio = ratios[h - 2]
            if ratio is None:
                continue
            weight = math.exp(-0.5 * ((log_h1 - x0) / BANDWIDTH) ** 2)
            xs.append(log_h1)
            ys.append(ratio)
            ws.append(weight)
        if not xs:
            # No donor has this hour: fall back to the persistence baseline.
            spread = FALLBACK_SIGMA * math.sqrt(h - 1)
            hours.append({"hour": h,
                          "q": hour_quantiles(p1, 0.0, spread)})
            continue
        # Never extrapolate the fitted line past the donors: if hour 1 is
        # larger (or smaller) than every past launch's, evaluate the fit at
        # the edge of the donor range instead of running the slope on.
        # The weights still use the true hour 1, so the nearest donors
        # dominate. On the practice cases hour 1 always lies inside the
        # donor range, so this changes nothing there.
        x_eval = min(max(x0, min(xs)), max(xs))
        median_log_ratio, resid_sd = local_linear(xs, ys, ws, x_eval)
        spread = max(SPREAD_FLOOR, SPREAD_SCALE * resid_sd)
        hours.append({"hour": h,
                      "q": hour_quantiles(p1, median_log_ratio, spread)})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
