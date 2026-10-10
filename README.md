# xplain-live

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
  and players. No real match data is used at run time or redistributed. The xPass model was
  trained on StatsBomb open data, part of which is real English top-flight matches, and the
  generator's outcome weights were estimated from the same data (see Attribution).
- **xP is situational pass difficulty.** It is computed from start-of-pass information only
  (location, pass height, body part, play pattern, pressure). It does not know the intended
  target, so it does not say whether a particular pass was a good choice.
- **Generated outcomes are not ground truth about football.** Pass outcomes in generated matches
  come from a mechanism independent of the xP model; how well xP matches them is measured and
  reported, not assumed.
- **"Progressive" and "difficulty" are defined from available fields.** No claims that need
  freeze-frame data (e.g. line-breaking passes) are made.

## Known simplifications

Measured with `scripts/compare_variants.py` on generated matches:

- On 40 generated matches (seeds 1-40), mean xP and actual completion agree within 0.4-1.4
  percentage points.
- xP's predictions are more spread out than the generated outcomes (calibration slope about
  0.62). Passes that xP rates below 0.4 complete 22-27 points more often than predicted.
- The generated pass mix differs from real football (seeds 1-20 against StatsBomb open data):
  mean pass length 16.2 vs 21.2, high passes 10.9% vs 21.8%, under pressure 20.2% vs 15.6%.
  Goal kicks, throw-ins and passes of 45+ complete more often than in real data.
- xP measures situational pass difficulty and does not know the pass target.
- Difficulty bands and outlier thresholds are provisional.
- The model's real-world quality is its hold-out result (ROC AUC 0.8328, log-loss 0.4007,
  Brier 0.1266), not the reliability numbers on synthetic passes.

## Synthetic data

`src/xplain/generator.py` produces full matches from a seed: two fictional clubs (4-4-2, invented
names), passes, pressure, tackles, shots, possession changes and periodic player-load events, on
the 120 x 80 StatsBomb pitch with every event in its own team's attacking frame. The same seed
always gives a byte-identical file.

- **Outcomes are independent of the xP model.** Pass completion comes from a logistic rule
  (length, forward gain, pressure, height, restart type, start location). Its weights were
  computed by fitting a logistic regression to real passes: 3,836,550 passes from StatsBomb open
  data, each labelled completed or not (`scripts/fit_outcome_weights.py`, results in
  `reports/outcome_fit.json`). The fitted values are in `src/xplain/config.py`. The player skill
  term is synthetic, and the pass mix (length, height, pressure, flow of play) is hand-designed.
  The generator never imports or loads the xP model.
- **Hidden skill stays hidden.** Each player's latent pass skill is written only to a separate
  `<name>.truth.json` file, never to the events file.

```powershell
python -m xplain.generator --seed 42 --out data/match_seed42.jsonl
```

This writes `data/match_seed42.jsonl` (one event per line) and `data/match_seed42.truth.json`.
Generated files are not committed. Run the command from the repo root with the venv active and
`src` on the path (for example `$env:PYTHONPATH = "src"`).

## Run the API and replay

```powershell
$env:PYTHONPATH = "src"
python -m xplain.generator --seed 42 --out data/match_seed42.jsonl
uvicorn xplain.api:app --port 8000
# in a second terminal (same venv and PYTHONPATH):
python -m xplain.replay --file data/match_seed42.jsonl --url http://127.0.0.1:8000 --speed 0
```

xP is situational pass difficulty (completion probability from start-of-pass information only),
not pass quality; the difficulty band and outlier thresholds are provisional.

## Attribution

- The pass-completion model (xPass) was trained on the [StatsBomb Open Data](https://github.com/statsbomb/open-data)
  under its user agreement. Source repo: [xPass-Project](https://github.com/charalampisandreou/xPass-Project).
  StatsBomb credit is required wherever the model or its derived analysis is shown.
- **Model training data:** the xPass model was trained on StatsBomb Open Data
  (3,961 matches across more than 20 competitions). 418 of those matches
  (about 10.6%) are real Premier League matches from the 2003/04 and 2015/16
  seasons. The demo and all generated matches use synthetic data only; no real
  match data is used at run time or redistributed.
- **Generator outcome weights:** the weights of the generator's pass-completion rule were
  computed with a logistic regression on real passes from StatsBomb Open Data (same source:
  3,961 matches, 3,836,550 passes). Only aggregate results are stored
  (`reports/outcome_fit.json`); no real events are stored in this repository. To rerun the fit
  you need a local copy of the StatsBomb open-data event files, passed as an argument (the data
  is not included):

  ```powershell
  python scripts/fit_outcome_weights.py --data-dir <path to the StatsBomb event JSON files>
  python scripts/compare_variants.py
  ```
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
