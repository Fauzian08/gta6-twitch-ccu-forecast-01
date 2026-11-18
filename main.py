"""Forecast Twitch viewers for GTA VI's launch day, hour by hour.

Reads one case per *.in file (schema makefun/twitch-ccu-case/v1) and
writes one forecast per case: for hours 2-24, five quantiles
(5/25/50/75/95%) of that hour's viewer count.

Method (a genuine forecaster; no memorisation, no case-id branching):
hour-1's print anchors everything. Each comparable past launch is a
donor: its log-ratio log(viewers_h / viewers_1) for hour h, plotted
against its own log hour-1 print. For each hour, a local-linear fit
(Epanechnikov kernel, wide bandwidth) of log-ratio on log hour-1 gives
the median ratio at the case's hour-1 level; the median is the case's
hour-1 print times that ratio. The spread is the kernel-weighted RMS
residual around the fit, computed separately above and below the fit
(residuals are strongly right-skewed: upside moves are much larger
than downside moves), scaled down slightly as the horizon grows, and
turned into normal quantiles. The median is never evaluated outside
the donors' hour-1 range (the fit point is clamped to it), so an
unprecedented hour-1 freezes the fitted shape instead of running the
fitted slope on. With no usable donors for an hour, the persistence
baseline (flat at hour 1, widening spread) is used so output is
always valid.

Stdlib only. Deterministic: no randomness, no time, no network.
Reads $MAKEFUN_INPUT_DIR (else /input) and writes
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# Tuned constants (practice 0.159, leave-one-out 0.163; v2 was 0.169/0.175).
BANDWIDTH = 4.5       # Epanechnikov bandwidth in log hour-1
SPREAD_SCALE = 0.5    # multiplier on the residual RMS spread
SPREAD_FLOOR = 0.10   # log units; guards degenerate hours
SPREAD_GROWTH = -0.25  # spread scale changes by this fraction over hours 2-24
BASE_SPREAD_RATE = 0.15  # persistence fallback: spread = rate * sqrt(h - 1)

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)


def donor_series(comp):
    """Hour 1..24 viewer prints for one comparable launch, or None.

    A donor must have a complete release day: an hourly point at its
    day_start and at each following hour, all positive. Launches
    without a release-day hour 1 (e.g. the GTA VI premiere history)
    are excluded by construction.
    """
    day_start = parse_ts(comp["day_start"])
    points = {}
    for row in comp["hourly"]:
        viewers = row.get("viewers")
        if viewers is not None and viewers > 0:
            points[row["from"]] = float(viewers)
    series = []
    for hour in range(1, 25):
        key = (day_start + timedelta(hours=hour - 1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        if key not in points:
            return None
        series.append(points[key])
    return series


def epanechnikov(u):
    if -1.0 < u < 1.0:
        return 1.0 - u * u
    return 0.0


def local_linear(xs, ys, ws, x0):
    """Weighted local-linear fit at x0. Returns (value, slope)."""
    total = sum(ws)
    if total <= 0:
        return None, 0.0
    mean_x = sum(w * x for w, x in zip(ws, xs)) / total
    mean_y = sum(w * y for w, y in zip(ws, ys)) / total
    sxx = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mean_x) * (y - mean_y)
              for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    return mean_y + slope * (x0 - mean_x), slope


def persistence_quantiles(hour1, hour):
    spread = BASE_SPREAD_RATE * math.sqrt(hour - 1)
    return [hour1 * math.exp(z * spread) for z in Z]


def forecast_hour(hour1, log_hour1, donors, hour):
    """Five quantiles for one forecast hour."""
    xs, ys = [], []
    for series in donors:
        xs.append(math.log(series[0]))
        ys.append(math.log(series[hour - 1]) - math.log(series[0]))
    if not xs:
        return persistence_quantiles(hour1, hour)
    low, high = min(xs), max(xs)
    x0 = min(max(log_hour1, low), high)  # never extrapolate the median
    weights = [epanechnikov((x - x0) / BANDWIDTH) for x in xs]
    centre, slope = local_linear(xs, ys, weights, x0)
    if centre is None:
        centre = sum(ys) / len(ys)
        slope = 0.0
    median = hour1 * math.exp(centre)
    # Asymmetric spread: RMS of residuals above vs below the fit.
    up_num = up_den = down_num = down_den = 0.0
    for w, x, y in zip(weights, xs, ys):
        resid = y - (centre + slope * (x - x0))
        if resid >= 0:
            up_num += w * resid * resid
            up_den += w
        else:
            down_num += w * resid * resid
            down_den += w
    up = math.sqrt(up_num / up_den) if up_den > 0 else 0.0
    down = math.sqrt(down_num / down_den) if down_den > 0 else 0.0
    growth = 1.0 + SPREAD_GROWTH * (hour - 2) / 22.0
    quantiles = []
    for z in Z:
        side = up if z >= 0 else down
        spread = max(SPREAD_FLOOR, SPREAD_SCALE * growth * side)
        quantiles.append(median * math.exp(z * spread))
    quantiles = [min(max(v, 1e-9), 1e9) for v in quantiles]
    for i in range(1, 5):
        quantiles[i] = max(quantiles[i], quantiles[i - 1])
    return quantiles


def forecast_case(case):
    known = case.get("known") or []
    hour1 = float(known[0]["viewers"]) if known else 1.0
    if hour1 <= 0:
        hour1 = 1.0
    log_hour1 = math.log(hour1)
    donors = []
    for comp in case.get("comparables") or []:
        series = donor_series(comp)
        if series is not None:
            donors.append(series)
    rows = []
    for row in case["forecast"]:
        hour = int(row["hour"])
        rows.append({"hour": hour,
                     "q": forecast_hour(hour1, log_hour1, donors, hour)})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text())
        result = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
