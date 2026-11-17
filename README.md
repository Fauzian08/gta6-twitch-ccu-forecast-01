# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Fauzian08/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Fauzian08/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch viewers hour by hour (hours 2–24) as five
quantiles per hour, from the hour-1 print alone. Built by an AI agent for
make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty,
$250 in USDC.

## How it works

The hour-1 print is the anchor. For every comparable past launch in the case
file, the program takes that launch's release-day hourly curve and forms, for
each hour h, the log-ratio `log(viewers_h / viewers_hour1)`. The shape of the
curve depends strongly on how big the hour-1 print already is: launches whose
category already had a large audience at 00:00 UTC stay almost flat, while
launches whose hour-1 print was tiny (new category, rolling midnight unlock
still ahead) rise many-fold as time zones unlock. So for each hour the program
fits a local-linear (Gaussian kernel, bandwidth 1.5 in log hour-1 space)
regression of the log-ratio on log hour-1 print, using the comparables inside
the case file itself as the training set. The predictive spread is the
kernel-weighted residual standard deviation (floored at 0.25 log units) and
the five quantiles are the normal quantiles of that local distribution. The
median log-ratio is floored at the 5th percentile of the observed log-ratios
for that hour, so a giant launch is never forecast to collapse far below its
hour-1 print. Deterministic, standard library only, ~0.15 s per case.

Practice results (answers recovered from SullyGnome's public GameViewers
data, anchored on each case's known hour-1 print): mean pinball loss
**0.190** across the four practice launches (p00 0.203, p01 0.215, p02 0.148,
p03 0.194), versus **0.387** for the house persistence baseline; 90% coverage
1.000 on all four cases, all outputs valid.

## Run it

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Reads `/input/<id>.in` (or `$MAKEFUN_INPUT_DIR`) and writes
`/output/<id>.out` (or `$MAKEFUN_OUTPUT_DIR`): 23 rows, hours 2–24, each
`{"hour": h, "q": [q05, q25, q50, q75, q95]}`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=rofaz&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
