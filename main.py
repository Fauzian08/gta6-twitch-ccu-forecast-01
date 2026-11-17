#!/usr/bin/env python3
"""Forecast GTA VI launch-day Twitch viewers hour by hour (hours 2-24).

Method
------
The hour-1 print is the anchor. For every comparable past launch in the case
file we take its release-day hourly curve and form, for each hour h, the
log-ratio log(viewers_h / viewers_hour1).  The shape of that curve depends
strongly on how big the hour-1 print already is: launches whose category
already had a large audience at 00:00 UTC stay almost flat, while launches
whose hour-1 print was tiny (new category, rolling midnight unlock still
ahead) rise many-fold as time zones unlock.  So for each hour we fit a
local-linear (Gaussian kernel, bandwidth 1.5 in log hour-1 space) regression
of the log-ratio on log hour-1 print, using the comparables in this very case
file as the training set.  The predictive spread is the kernel-weighted
residual standard deviation (floored at 0.25 log units), and the five
quantiles are the normal quantiles of that local distribution.  The median
log-ratio is floored at the 5th percentile of the observed log-ratios for
that hour, so a giant launch is never forecast to collapse far below its
hour-1 print.  Everything is deterministic, stdlib only, and fast.

If anything about the data is unexpected the program falls back to the house
persistence baseline for that case, so the output is always valid.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
BANDWIDTH = 1.5
SD_FLOOR = 0.25
MIN_TRAIN = 8


def parse_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def release_day_series(comp):
    """Hour -> viewers for a comparable's release day (hours 1..24)."""
    try:
        day_start = parse_time(comp["day_start"])
    except Exception:
        return {}
    out = {}
    for row in comp.get("hourly") or []:
        try:
            viewers = row.get("viewers")
            if not viewers or viewers <= 0:
                continue
            t = parse_time(row["from"])
        except Exception:
            continue
        delta = (t - day_start).total_seconds() / 3600.0
        if 0.0 <= delta < 24.0:
            out[int(delta) + 1] = float(viewers)
    return out


def hour1_print(case):
    """Hour-1 print from the known rows, with fallbacks."""
    known = case.get("known") or []
    if isinstance(known, dict):
        known = [known]
    for row in known:
        if isinstance(row, dict) and row.get("hour") == 1:
            value = row.get("viewers")
            if value and value > 0:
                return float(value)
    for row in known:
        if isinstance(row, dict):
            value = row.get("viewers")
            if value and value > 0:
                return float(value)
    for key in ("history_hourly", "history"):
        rows = case.get(key) or []
        for row in reversed(rows):
            value = row.get("viewers")
            if value and value > 0:
                return float(value)
    return 1000.0


def quantile(sorted_values, q):
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def local_linear(points, x, bandwidth):
    """Kernel-weighted local-linear fit; returns (mu, weighted residual sd)."""
    weights = []
    for px, _py in points:
        d = (px - x) / bandwidth
        weights.append(math.exp(-0.5 * d * d))
    total_w = sum(weights)
    if total_w <= 0.0:
        return 0.0, 1.0
    mean_x = sum(w * px for w, (px, _py) in zip(weights, points)) / total_w
    mean_y = sum(w * py for w, (_px, py) in zip(weights, points)) / total_w
    sxx = sum(w * (px - mean_x) ** 2 for w, (px, _py) in zip(weights, points))
    sxy = sum(
        w * (px - mean_x) * (py - mean_y)
        for w, (px, py) in zip(weights, points)
    )
    slope = sxy / sxx if sxx > 1e-9 else 0.0
    intercept = mean_y - slope * mean_x
    mu = intercept + slope * x
    var = sum(
        w * (py - (intercept + slope * px)) ** 2
        for w, (px, py) in zip(weights, points)
    ) / total_w
    return mu, math.sqrt(var) if var > 0.0 else 0.0


def forecast_case(case):
    """Return the 23 forecast rows for one case."""
    p1 = hour1_print(case)
    x = math.log(p1)

    # Training set: one (log hour-1, log-ratio) point per comparable per hour.
    training = []
    for comp in case.get("comparables") or []:
        series = release_day_series(comp)
        if 1 not in series:
            continue
        present = sum(1 for h in range(2, 25) if h in series)
        if present < 10:
            continue
        training.append(series)

    rows = []
    for h in range(2, 25):
        points = []
        for series in training:
            if h in series:
                points.append((math.log(series[1]), math.log(series[h] / series[1])))
        if len(points) >= MIN_TRAIN:
            ratios = sorted(py for _px, py in points)
            floor = quantile(ratios, 0.05)
            mu, sd = local_linear(points, x, BANDWIDTH)
            mu = max(mu, floor)
            sd = max(sd, SD_FLOOR)
            qs = [p1 * math.exp(mu + z * sd) for z in Z]
        else:
            # Too few comparables: fall back to persistence with a widening
            # spread, exactly the house baseline shape.
            sigma = 0.15 * math.sqrt(h - 1)
            qs = [p1 * math.exp(z * sigma) for z in Z]
        clean = []
        prev = 1.0
        for value in qs:
            value = min(max(value, 1.0), 1e9)
            value = max(value, prev)
            clean.append(round(value))
            prev = value
        rows.append({"hour": h, "q": clean})
    return {"hours": rows}


def baseline_case(case):
    """House persistence baseline (used only if the model path fails)."""
    p1 = hour1_print(case)
    rows = []
    for h in range(2, 25):
        sigma = 0.15 * math.sqrt(h - 1)
        qs = [p1 * math.exp(z * sigma) for z in Z]
        clean = []
        prev = 1.0
        for value in qs:
            value = min(max(value, 1.0), 1e9)
            value = max(value, prev)
            clean.append(round(value))
            prev = value
        rows.append({"hour": h, "q": clean})
    return {"hours": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(
        name for name in os.listdir(input_dir) if name.endswith(".in")
    )
    for name in names:
        case_id = name[: -len(".in")]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        try:
            out = forecast_case(case)
            # Defensive validity check; fall back rather than emit bad output.
            if len(out["hours"]) != 23:
                raise ValueError("bad row count")
            for row in out["hours"]:
                q = row["q"]
                if len(q) != 5 or any(v <= 0 or v > 1e9 for v in q):
                    raise ValueError("bad quantiles")
                if any(q[i + 1] < q[i] for i in range(4)):
                    raise ValueError("decreasing quantiles")
        except Exception:
            out = baseline_case(case)
        with open(
            os.path.join(output_dir, case_id + ".out"), "w", encoding="utf-8"
        ) as fh:
            json.dump(out, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
