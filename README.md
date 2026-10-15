# xplain-live

Expectation versus outcome, live. A studio copilot for the Microsoft
"Inside the Game" hackathon: it ingests synthetic match events, compares what a trained model
expected with what happened, explains the gap, and emits timed, machine-readable
overlay cards that a graphics system can render alongside the match.

> Status: work in progress. Built: the synthetic match generator, the pass model trained in this
> repository (xpass_v2) behind a scoring API (`/score`), replay, an offline deterministic analysis
> pipeline (outlier moments, SHAP-based drivers, per-half commentary gate, half reports, timeline,
> pass marks), the Narrator with a code Verifier, and a first Agent Framework workflow that runs
> them end to end (Analyst -> Explainer -> Narrator -> Verifier), and a per-match JSON bundle
> for the web page. Not built yet: personalization, the web page and the Azure deployment.
> Results, demo and deployment sections are filled in as they are measured.

## What it does

_To be written once the end-to-end demo works. It will lead with measured results._

## Honest limits

- **Synthetic data only.** All matches come from a seeded generator with fictional clubs and
  players, and the generator's rules are designed, not fitted to real data. No third-party data is
  used at run time, in training or in the repository. The scoring model is trained in this
  repository on the generator's own matches, so its quality figures check the pipeline's
  consistency; they say nothing about accuracy on real football.
- **xP is situational pass difficulty.** It is computed from start-of-pass information only
  (location, pass height, body part, play pattern, pressure). It does not know the intended
  target, so it does not say whether a particular pass was a good choice.
- **Generated outcomes are not ground truth about football.** Pass outcomes in generated matches
  come from a mechanism independent of the xP model; how well xP matches them is measured and
  reported, not assumed.
- **"Progressive" and "difficulty" are defined from available fields.** No claims that need
  freeze-frame data (e.g. line-breaking passes) are made.
- **Spanish is not reviewed by a native speaker.** The English and Greek lines are reviewed by the
  author (a native Greek speaker). The Spanish lines and the Spanish phrase table were drafted with
  an AI assistant and have not been reviewed by a native speaker. The code checks difficulty
  wording and unsupported claims in all three languages, but it cannot judge naturalness.
- **The outcome-cause check is a heuristic.** A causal word must sit next to a difficulty word to
  be accepted; the check can miss some phrasings and paraphrases, and does not read the sentence.

## Known simplifications

Measured on generated matches only (generator 0.2.1; `reports/xpass_v2_metrics.json`):

- xpass_v2 is trained on 250 generated matches (231,412 passes) and evaluated on 100 hold-out
  generated matches (92,396 passes): ROC AUC 0.770, log-loss 0.4326 (constant rate 0.5168),
  Brier 0.1372 (constant rate 0.1671), Brier skill 0.179, calibration slope 1.002. Completion on
  the hold-out is 78.8% against a mean xP of 78.8%. These are results on generated passes.
- The generator's completion rule also uses pass length, forward gain and a hidden player skill,
  which xP cannot see (it uses start-of-pass information only), so xP cannot rank passes
  perfectly. Generator 0.2.1 makes the context xP does see matter more, by design (position and
  pressure effects doubled, height and header effects 1.5 times, more pressured and lofted
  passes), so xP covers a wide range: on the hold-out its 5th percentile is 0.40 and its 95th is
  0.96; 5.1% of passes have xP <= 0.40 and 23.0% have xP >= 0.92.
- xP measures situational pass difficulty and does not know the pass target.
- Difficulty bands, outlier thresholds (completed with xP <= 0.35, failed with xP >= 0.92),
  difficulty tiers and the commentary settings (top 3 per kind per half, floor 0.25 / 0.96) are
  editorial settings. At generator 0.2.1 they give about 19 outliers per match on the hold-out
  (12.0 surprise failures and 7.3 surprise successes), and 17 commentated moments out of 19
  outliers on seed 42.
- Base-rate check (`scripts/base_rate_check.py`, seeds 1001-1010, 9,068 passes): passes with
  xP >= 0.92 have 114.1 expected failures and 109 observed; passes with xP <= 0.35 have 72.9
  expected successes and 68 observed. This is a measured check that the outliers occur about as
  often as a calibrated model implies.
- The second-half clock starts at the first period-2 event (the kick-off), since the event stream
  has no explicit half-time marker. Added time shows as "45+N" and "90+N".
- Pitch-position labels use editorial thresholds (thirds of the pitch length, 0.85 of the length
  for "close to the opposition goal", 0.15 / 0.325 of the width) and never say left or right.

## Synthetic data

`src/xplain/generator.py` produces full matches from a seed: two fictional clubs (4-4-2, invented
names), passes, pressure, tackles, shots, possession changes and periodic player-load events, on
a 120 x 80 pitch with every event in its own team's attacking frame. The same seed
always gives a byte-identical file.

- **Outcomes come from a designed rule.** Pass completion is a logistic rule over the passer's
  hidden skill, start position (how far up the pitch, how wide), pressure, pass height, header,
  restart type, length and forward gain. Every weight is a round value with a sign from football
  reasoning and a one-line rationale in `src/xplain/config.py`; none was fitted to data. The pass
  mix (length, height, pressure, flow of play) is designed too. The generator never imports or
  loads the xP model.
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

## Run the workflow

The Agent Framework workflow (`src/xplain/workflow.py`) chains Analyst -> Explainer -> Narrator ->
Verifier and narrates every commentated moment. It calls the Foundry model, so it needs `az login`:

```bash
python -m xplain.workflow --file data/match_seed42.jsonl --max-moments 5
```

Per moment it prints the minute, player, xP percent (from code), the EN/ES/EL lines and the
Verifier verdict (accepted, retried, or rejected with reasons), then a summary. A moment that is
rejected, times out or fails gets no line and never stops the others. `--mode strict|free|hints`
overrides `XPLAIN_PHRASE_MODE`.

## Match bundle

`src/xplain/bundle.py` writes one JSON file per match for the web page, which plays it on the
match clock without calling any model and without computing numbers or minutes:

```bash
python -m xplain.bundle --file data/match_seed42.jsonl --out data/bundle_seed42.json --no-llm
```

The bundle holds the match meta (teams, players, model and generator versions, phrase mode), a
clock track (one `minute_label` per match minute of each half, added time included, from the
match clock), every pass mark, every outlier moment with its line and narration status
(`accepted`, `retried`, `rejected`, `not_commentated`, `not_run`), the half reports as moment ids,
each player's full-time pass summary, and the score after each goal.

Without `--no-llm` it runs the workflow for the narration (needs `az login`). A line is written
once per moment: if `--out` already holds a bundle of the same match and model, its lines are
reused and the model is not called for those moments (the final Verifier check still runs).
`--fresh` ignores the old file. The command prints a one-line summary with the goals and the
final score.

## Phrase modes

`XPLAIN_PHRASE_MODE` (or `--mode` on the workflow CLI) sets how the Narrator model is briefed. The
default is `hints`.

| Mode | What the model is given for each driver | What the Verifier additionally checks |
|---|---|---|
| `strict` | The approved phrase in English, Spanish and Greek (a position driver also its depth and width pieces), to be used verbatim | Every driver the line declares must appear as its approved phrase in that language |
| `free` | The driver's English label only; the model words it naturally in each language | No phrase check |
| `hints` (default) | The English label plus suggested approved wordings per language, which it may use, adapt or replace | No phrase check |

The other Verifier checks run in all three modes. `free` and `hints` also add a small variation
nudge (sentence shape, how many drivers to mention, outcome wording) chosen from a hash of the
moment id, so the same moment always gets the same nudge.

## Narration contract

- One spoken sentence per language (English, Spanish, Greek) for each commentated moment.
- No numbers and no minute in any line. The card shows them, computed by code.
- A line may give a **reason for the model's rating** ("difficult because it was lofted high") or
  a **concession** ("not completed, even though it was along the ground"). It never gives a cause
  of the outcome. Drivers explain why the *model* rated the pass as it did, not why the pass
  succeeded or failed.
- A code Verifier (deterministic, no LLM) checks every line before it is shown. It rejects a line
  that:
  - is not one non-empty sentence, or is over the word limit of its language;
  - contains a digit, a percent sign or a percent or minute word;
  - uses a difficulty word that contradicts the moment's rating;
  - makes a banned claim: penalty area or box, left or right, defender counts, line-breaking,
    intent or credit verbs (tries, attempts, manages to and their Spanish and Greek equivalents),
    model internals, or "throw";
  - names a driver that was not shown to the model, uses a driver that points the wrong way for
    the rating, or says a phrase-table fact (for example "under pressure") that is not among the
    declared drivers;
  - uses "mainly" or a similar word without driver 1 declared, or declares no driver although one
    points the way of the rating;
  - uses a causal word (because, due to, thanks to and their Spanish and Greek equivalents) away
    from a difficulty word, which reads as a cause of the outcome.
- A rejected line is retried once with the Verifier's reasons. If the retry is also rejected, or
  the model times out or fails, the moment simply has no line. The card is still shown.

## How moments are chosen

- An **outlier** is a pass that was a surprise: a completed pass with xP <= 0.35, or a failed pass
  with xP >= 0.92 (provisional thresholds).
- Every outlier gets a selectable card; only some get commentary.
- Commentary goes to moments that enter the running **top 3 of their kind within their half**, or
  that clear an always-commentate floor (success xP <= 0.25, failure xP >= 0.96). Successes and
  failures are ranked separately, and the ranking resets at half-time. The half-time and
  full-time reports are the final top 3 per kind per half.
- This is a **live** ranking: an early mild moment can be commentated and later displaced, and a
  stronger one that arrives later can miss the cut. The timeline keeps every commentated moment.
- Each moment carries SHAP-based drivers computed with XGBoost's built-in TreeSHAP; they explain
  the model's raw score in log-odds, not percentage points of xP.

## Provenance

No third-party data and no pre-built scoring model is used anywhere in the current version. The
generator's rules are designed and the scoring model is trained in this repository on generated
matches only. The only third-party model is the Microsoft Foundry language model (gpt-4.1-mini)
that phrases the commentary lines; it never sees or produces a football statistic (the only
numbers it handles are the driver ranks 1 to 3).

History: an earlier version of the generator used pass-completion weights fitted on StatsBomb Open
Data. After correspondence with the hackathon support team (13 October 2026), who asked entrants
to stay within the published rules on third-party data, the fitted weights were removed and the
scoring model was retrained from scratch on generated data. The earlier commits remain in the
repository history. The generator is now fully designed (version 0.2.1).

An earlier personal pass-model project (xPass, June 2026) informed the choice of start-only
features: the 17-column feature list (`xplain.features.FEATURE_ORDER`) and its angle-to-goal
definition follow that project's design, and the pitch frame and category names follow the common
120 x 80 event-data convention. Two feature tests taken from that project were replaced with tests
written here.

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

The xPass model is trained in this repo on generated matches and is not in git (see
`models/README.md`). To build it (about 40 s):

```powershell
python scripts/make_dataset.py
python -m xplain.train --train data/xpass_train.csv --holdout data/xpass_holdout.csv `
    --out models/xpass_v2 --report reports/xpass_v2_metrics.json
```

## License

MIT (see `LICENSE`).
