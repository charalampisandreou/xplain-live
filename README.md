# xplain_live

Expectation versus outcome, live. A studio copilot for the Microsoft
"Inside the Game" hackathon: it ingests synthetic match events, compares what a trained model
expected with what happened, explains the gap, and emits timed, machine-readable
overlay cards that a graphics system can render alongside the match.

> Status: work in progress (spine only: schema, feature builder, generator, `/score`, replay).
> Results, demo and deployment sections are filled in as they are measured.

## What it does

_To be written once the end-to-end demo works. It will lead with measured results._

## Honest limits

- **Synthetic data only.** All matches are produced by a seeded generator with fictional clubs
  and players. No real Premier League data is used or redistributed.
- **xP is situational pass difficulty.** It is computed from start-of-pass information only
  (location, pass height, body part, play pattern, pressure). It does not know the intended
  target, so it does not say whether a particular pass was a good choice.
- **Generated outcomes are not ground truth about football.** Pass outcomes in generated matches
  come from a mechanism independent of the xP model; how well xP matches them is measured and
  reported, not assumed.
- **"Progressive" and "difficulty" are defined from available fields.** No claims that need
  freeze-frame data (e.g. line-breaking passes) are made.

## Attribution

- The pass-completion model (xPass) was trained on the [StatsBomb Open Data](https://github.com/statsbomb/open-data)
  under its user agreement. Source repo: [xPass-Project](https://github.com/charalampisandreou/xPass-Project).
  StatsBomb credit is required wherever the model or its derived analysis is shown.
- The model is used behind a pluggable scoring interface, so it can be replaced without changing
  the rest of the system.
- Fictional clubs and players in the generator are invented and unrelated to real people or clubs.

## Run

Windows (PowerShell), Python 3.13:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
```

The xPass model is not in git. Place `xpass_production_*.joblib` and `feature_list.json`
in `models/xpass/` (see `models/README.md`).

## License

MIT (see `LICENSE`).
