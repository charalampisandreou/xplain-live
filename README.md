# xplain-live

Expectation versus outcome, live. A studio copilot for the Microsoft
"Inside the Game" hackathon: it ingests synthetic match events, compares what a trained model
expected with what happened, explains the gap, and emits timed, machine-readable
overlay cards that a graphics system can render alongside the match.

> Status: work in progress. Built: the synthetic match generator, the xPass scoring API (`/score`),
> replay, and an offline deterministic analysis pipeline (outlier moments, SHAP-based drivers,
> per-half commentary gate, half reports, timeline, pass marks). Not built yet: the LLM agents
> (narration, personalization), the live overlay UI and the Azure deployment.
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
- The second-half clock starts at the first period-2 event (the kick-off), since the event stream
  has no explicit half-time marker. Added time shows as "45+N" and "90+N".
- Pitch-position labels use editorial thresholds (thirds of the pitch length, 0.85 of the length
  for "close to the opposition goal", 0.15 / 0.325 of the width) and never say left or right.
- The commentary settings (top 3 per kind per half, floor 0.25 / 0.96) were chosen from a
  calibration over 10 generated matches and are provisional.
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

## Run the pipeline

Set the path once per terminal: `export PYTHONPATH=src` in Git Bash, or `$env:PYTHONPATH = "src"`
in PowerShell. Then, from the repo root with the venv active:

```bash
python -m xplain.generator --seed 7 --out data/match_seed7.jsonl
python -m xplain.pipeline --file data/match_seed7.jsonl
python -m pytest -q
```

The pipeline prints a summary line and the half-time and full-time reports. Optional flags:

| Flag | What it does |
|---|---|
| `--seed N` | Seed used to generate `--file` when the file does not exist (default 42) |
| `--full` | Also print every moment (Moment, Explanation, commentary flag) as indented JSON |
| `--timeline` | Print the timeline of commentated moments at the end of the match |
| `--as-of-minute N` | Print the timeline as it stood at match minute N (45 includes first-half added time) |
| `--player PLAYER_ID` | Print that player's pass summary and passes (up to `--as-of-minute` if given) |
| `--out PATH` | Write one JSON line per moment, plus `*_reports` and `*_marks` files next to it |
| `--marks-out PATH` | Write one JSON line per pass (every pass, not only outliers) |

This path is deterministic and offline: it makes no LLM or Azure calls, and the same seed gives
byte-identical JSON.

## How moments are chosen

- An **outlier** is a pass that was a surprise: a completed pass with xP <= 0.35, or a failed pass
  with xP >= 0.92 (provisional thresholds).
- Every outlier gets a hover card; only some get commentary.
- Commentary goes to moments that enter the running **top 3 of their kind within their half**, or
  that clear an always-commentate floor (success xP <= 0.25, failure xP >= 0.96). Successes and
  failures are ranked separately, and the ranking resets at half-time. The half-time and
  full-time reports are the final top 3 per kind per half.
- This is a **live** ranking: an early mild moment can be commentated and later displaced, and a
  stronger one that arrives later can miss the cut. The timeline keeps every commentated moment.
- Each moment carries SHAP-based drivers computed with XGBoost's built-in TreeSHAP; they explain
  the model's raw score in log-odds, not percentage points of xP.

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
