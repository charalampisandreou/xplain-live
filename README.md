# xplain-live

[![tests](https://github.com/charalampisandreou/xplain-live/actions/workflows/tests.yml/badge.svg)](https://github.com/charalampisandreou/xplain-live/actions/workflows/tests.yml)

Expectation versus outcome, live. A studio copilot for the Microsoft
"Inside the Game" hackathon: it ingests synthetic match events, compares what a trained model
expected with what happened, explains the gap, and emits timed, machine-readable
overlay cards that a graphics system can render alongside the match.

> Status: work in progress. **Live demo:** the Studio page playing a synthetic match (seed 3001) at
> https://xplain-studio.whiteocean-b5bbd7fb.germanywestcentral.azurecontainerapps.io/
>
> Built: the synthetic match generator, the pass model trained in this repository (xpass_v2) behind a
> scoring API (`/score`), replay, an offline deterministic analysis pipeline (outlier moments,
> SHAP-based drivers, per-half commentary gate, half reports, timeline, pass marks), the Narrator with
> a code Verifier (five rotating sentence frames, see "Narration contract"), an Agent Framework
> workflow that runs them end to end (Analyst -> Explainer -> Narrator -> Verifier), a per-match JSON
> bundle, the Studio web page that plays the bundle on the match clock (goal flash, stepping through
> narrated moments, an in-plate narration language switch, half-time and full-time recap cards), the
> Recap Editor (the recap texts, with every number filled in by code), Players mode, the player-focus
> personalization mode (one player's pass map, numbers and moments up to the match clock; from a recap it
> opens on the star player), the Producer view (the control room: every line the clock has reached, the
> agents' steps behind it, the checks the Verifier ran, model and latency), a script that compares Narrator models
> (`scripts/compare_models.py`, see "Model comparison"), and the Azure deployment (Container Apps from
> an ARM template, see "Deploy to Azure"). The deployed page serves a precomputed bundle: its lines were
> generated with Foundry (gpt-6.1-sol) and checked by the Verifier when the bundle was built.
> The model files and the demo bundle are published in the open-licence release
> [data-v1](https://github.com/charalampisandreou/xplain-live/releases/tag/data-v1) (model files MIT,
> bundle CC BY 4.0). The bundle there was replaced after the narration retune; its SHA-256 is
> `48f18c2bac40ca3cf91a08768c655d6d3f77585602b41910f52c38fae49d5a1b`.
>
> Not built yet: the Analyst screen, the other personalization modes (analyst, casual fan, favourite club or player), and live mode (the deployed app
> narrating with Foundry itself; so far only a one-call live check, `/api/live/health`, see "Deploy to Azure").
> Results and demo sections are filled in as they are measured.

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
- **Greek is checked for correctness, not polished.** The author, a native Greek speaker, reviews the
  Greek phrase table and tier words for correctness; the generated Greek lines are not edited line by
  line and may still read stiffly.
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
(`accepted`, `retried`, `rejected`, `not_commentated`, `not_run`) and its difficulty tier words,
the half reports as moment ids, each player's full-time pass summary, the score after each goal,
and the live rankings: for each commentated moment, two snapshots of its kind as they stood at that
moment, one for scope `all` (the whole match so far) and one for its half (`1` or `2`). Each holds
the top 3 of all moments of that kind in the scope, by |PVA| with ties to the earlier pass (the
same sort as the commentary gate), and each entry has its rank in the previous snapshot of the
same scope and kind (none when new). Every ranked moment was commentated: a moment in the top 3
of the whole match is also in the top 3 of its half, so it passed the gate when it happened. The
page never ranks anything itself. Every pass mark also carries display coordinates for one fixed pitch: the
home team attacks right in the first half and left in the second, the away team the opposite
(the other team's passes are turned 180 degrees, (x, y) -> (120 - x, 80 - y));
`meta.attacks_right` names the team attacking right in each half. The stored coordinates, which
the model uses, stay in the passer's own frame (always attacking toward x = 120). Schema version 4.

Without `--no-llm` it runs the workflow for the narration (needs `az login`). A line is written
once per moment: if `--out` already holds a bundle of the same match and model, its lines are
reused and the model is not called for those moments (the final Verifier check still runs, and records its
checks; the reused Narrator steps keep what they recorded).
Lines are also read from a schema 1, 2 or 3 bundle, so rebuilding after a schema change keeps them.
With `--no-llm`, a moment whose line was rejected before is written as `not_run`.
`--fresh` ignores the old file. The command prints a one-line summary with the goals and the
final score.

`python scripts/refresh_recap_cards.py data/bundle_seed3001.json` fills in recap-card fields added
after a bundle was built (for now the two-decimal `expected_xp` and `vs_expected_xp`), from the
bundle's own passes and with no model call. Every field the card already had must come out the
same. Each recap text is then filled again from the model's raw text (with its placeholders) with the
card's two-decimal values and checked again by the recap verifier; a change is recorded as a
`recap_filler` step in the trace. It prints the new SHA-256.

## Run the web page

The Studio screen (`src/xplain/web/`: plain HTML, CSS and JS, no build step) is served by the same
FastAPI app and plays a match bundle on the match clock. In Git Bash, from the repo root with the
venv active and a bundle in `data/`:

```bash
export PYTHONPATH=src PYTHONIOENCODING=utf-8
python -m uvicorn xplain.api:app --port 8000
```

Then open http://127.0.0.1:8000/ (it plays `data/bundle_seed3001.json`, the demo match; the
development 0-0 match is `?match=seed42`). `?match=NAME` plays
`data/bundle_NAME.json`, `?t=T` starts playback at event time T, `?lang=en|es|el` picks the
commentary language and `?scope=all|1|2` the tables' starting scope. T is the bundle's event time `t` in seconds, which runs on through both halves
(first-half added time included) and is not the minute on the clock: in seed42, `t=2400` is 41',
the first half ends near `t=2841` (45+3) and the second half starts at `t=2844.15` (46'). The page
shows the `minute_label` of the clock entry for that t. `GET /api/bundle/NAME` serves the bundle (names match `^[a-z0-9_]+$`);
`XPLAIN_BUNDLE_DIR` changes the folder. Controls: play/pause, speed 1x / 12x / 60x, previous and
next pass (left and right arrows), Back to live (L), space for play/pause, and All / 1st / 2nd
above the tables (default All: the whole match so far; 1st stays frozen after half-time; 2nd says
"Starts in the 2nd half" until then). Clicking a table row (or Enter/Space on it) replays that moment:
the pitch, the plate and the possession tab show it while the clock, score and tables keep
running. A "Replay · 41'" badge above the pitch's top-right shows whenever the view is not live
(a replayed moment, or step mode with the arrows, showing the selected pass's minute). The
bottom-right Back to live, L or Esc return to the live pass. `?replay=MOMENT_ID` (e.g.
`?replay=moment:m42-00817`) opens in replay; if the moment is later than the start time, the
clock starts at it.

The page only formats: every number, minute and ranking comes from the bundle (Players mode's numbers so
far are the one exception, see below). The title reads "Live pass map" ("Half Time" or "Full Time" while a
recap is shown, "Control room" in Producer). The tables and the
plate show PVA (pass value added, the bundle's `delta` = outcome - xP: 1 - xP when completed, -xP
when not) in points of completion chance, e.g. "+85%" or "−94%" (sign, round(100 * |PVA|)); the
bar length is |PVA| on a 0..1 scale. xP is shown small in the plate as a percent with the same
rule as the code, round(100 * xP). The pitch draws the display coordinates. A tab on the pitch's
top edge shows the team of the selected pass on the side it attacks from, with cyan chevrons
toward the goal it attacks (from `meta.attacks_right`; still under reduced motion). The score bug
shows the minute and the half, and HT or FT at half time and full time; the possession tab stays at the pitch's top-left, with the code
then right-pointing chevrons when attacking right and left-pointing chevrons then the code when
attacking left. The lower third is a cyan name tab (shirt number, player, team code and minute) on a dark body
with the commentator's line in the chosen language, the tier and xP, and PVA on the right. The Analyst mode is not built yet.

Also in the Studio:

- **Goal flash.** The score bug pulses and the plate shows "GOAL · team · minute" (minute from the score
  track) until play restarts, for at least four seconds of real time at any speed.
- **Stepping through narrated moments.** Chevrons with a counter, or the `[` and `]` keys, move to the
  previous or next narrated moment; only moments the match has already reached are offered, so nothing
  is spoiled.
- **Fixed-size plate** that does not jump with line length, and a **NARRATION EN / ES / EL switch inside
  the plate** (the header no longer has a language switch).
- **PVA column in every state** (a dash while waiting for a pass, "GOAL" over the minute on a goal) and
  "xP" written with that casing throughout.
- **Half-time and full-time recaps.** When live playback reaches half time (the last first-half pass) or
  full time, the recap card takes the pitch's place: the result, the goals, each team's passing
  (completed, expected and vs expected, the last two to two decimals), the star player (their row opens
  them in Players) and the recap text in the narration language, whose numbers are the card's, to two
  decimals too ("19.45" in English, "19,45" in Spanish and Greek). The score bug reads HT or FT. A Recap /
  Map switch shows the card or the pitch; the plate's chevrons step from the recap into the narrated
  moments and back, and stepping or replaying keeps the recap. Half time resumes by itself after 10
  seconds unless you step away; full time stays. `?recap=ht` or `?recap=ft` opens one, paused. A recap
  that live playback passes while you are in Producer or Players (neither pauses or shows it) is kept as
  reached and unseen: switching to Studio while live opens the full-time one, and the half-time one only if
  no second-half pass has been played yet.

**Players mode** (the player-focus personalization mode: the Players button in the header, or
`?mode=player&player=PLAYER_ID`, e.g.
`?mode=player&player=dsk-p10`) follows one player's match on the same clock, score and narration
language as the Studio:

- **Pass map.** Every pass the player has made so far, drawn in the passer's own frame (the bundle's
  `pass_geometry`, so the player always attacks right in both halves): green arrows completed, red
  dashed lines ending in a cross not completed, a cyan ring at the start of each surprise moment and a
  white dot at the start of the latest pass. All / 1st / 2nd filter it; it grows as the player passes.
- **Numbers so far.** Passes, completed (with its rate), expected (the sum of xP, with its rate: the mean
  xP), vs expected (completed minus expected), both to two decimals as on the recap card, a bar of
  the completed passes per difficulty tier (the narration's five words), and the player's
  surprise moments with their PVA (hover or focus a row to bring its pass forward on the map).
- **Narration plate** with only this player's narrated moments (the latest; chevrons or `[` `]` page
  back), or "No commentated moment for NAME yet."
- **Change player** opens both teams side by side: shirt number, name, position and passes so far
  (arrows, Enter, Esc). From a recap, Players opens on its star player; the star's row on the recap
  card opens the star directly.

These numbers are the one place the page computes anything (the browser computation exception): sums,
counts, means and rates over the bundle's marks at the clock, shown to two decimals with `toFixed(2)`,
in `src/xplain/web/player.js`, each checked against its Python twin (`pipeline.player_summary`,
`recap.xp_sum`, `recap.round2` and others) in `tests/test_player_js.py`, which runs
`player.js` under Node and compares exactly: both sides add floats left to right (`analytics.seq_sum`;
Python's own `sum()` corrects for rounding since 3.12 and would differ in the last bits). The test is
skipped when `node` is not installed.

**Producer** (the Producer button, or `?mode=producer`) is the control room: for the person who puts AI
lines on air and must say why a line went out, what was checked, which model and how long. It shares the
clock, score, plate and controls with the Studio; switching never pauses the match, and recaps do not open
there.

- **Log** (newest first): every commentated moment and recap the clock has reached, nothing later: minute,
  player, kind, "On air" or "No line", attempts, and "Retried" when a line took more than one.
- **Steps** of the selected line, in order, from the bundle's trace: Analyst and Explainer (code), Narrator
  or Recap Editor (LLM: the model and the latency as stored), the Verifier and the final check (code: "18
  checks passed" with a button that lists them in plain words), and for a recap one "Numbers filled by code"
  step from the last fill, in plain words ("DUS vs expected 19.45 · MAR vs expected 19.11 · Star Nevek
  Iversani, 5.48"). A failed step always shows its reasons. Then the line that went on air, in the language of
  the plate's NARRATION switch (a recap: the filled text). The panel stops 24 px above the playback controls;
  an open check list scrolls inside it.
- **Selection** is the Studio's: a log row replays its moment (Back to live, Esc or L return); the Studio's
  replay or plate moment is selected in the log. A recap row is picked in Producer only.
- The checks shown are the ones each Verifier step recorded when the bundle was built (`checks`, labels in
  `trace.check_labels`, from the registries `verifier.CHECKS` and `recap_verifier.RECAP_CHECKS` that the
  verifiers loop over): a check listed is a check that ran. A bundle built before checks were recorded says
  "Checks not recorded in this bundle." on those steps (the demo bundle until its next rebuild); a bundle
  without a trace shows "No agent trace in this match bundle." Every value shown is a bundle field; the only
  count is the length of a step's stored check list ("18 checks passed").

## Docker

The image serves the Studio page, `/api/bundle`, `/score` and (with `XPLAIN_LIVE=1`) `/api/live/health`. It needs the trained model
(`models/xpass_v2/`, git-ignored: train it first) and the demo bundle `data/bundle_seed3001.json`;
no Azure or OpenAI settings are needed to serve. Build and run from this folder:

```bash
docker build -t xplain-live:local .
docker run --rm -p 8000:8000 xplain-live:local
```

Then open http://localhost:8000/. Runtime pins are in `requirements-runtime.txt` (a test keeps them
equal to `requirements.txt`); `XPLAIN_BUNDLE_DIR` (default `/app/data` in the image) is the bundle
folder.

## Deploy to Azure

The Studio runs on Azure Container Apps in `germanywestcentral`:
https://xplain-studio.whiteocean-b5bbd7fb.germanywestcentral.azurecontainerapps.io/

`infra/main.json` (ARM, Microsoft.App API 2026-07-01) describes the environment `cae-xplain-std` and the
app `xplain-studio` (0.5 vCPU / 1 GiB, 0 to 1 replicas, so it scales to zero). The app pulls
`xplainlive.azurecr.io/xplain-live:<tag>` with the user-assigned identity `id-xplain-live` (AcrPull role),
so it holds no registry password, token or other secret. Logs go to the Log Analytics workspace
`workspace-rgplhackdesV2i` (daily ingestion cap 0.1 GB). The template reads the workspace key with
`listKeys()` itself, so the key never passes through the CLI.

One command deploys the current commit. It needs `az login`, Docker running and a clean working tree:

```powershell
./scripts/deploy_azure.ps1
```

The script builds the image with local Docker as a single linux/amd64 manifest, tags it with the git short
sha and pushes it to the registry. It then prints a what-if preview and asks y/n before running
`az deployment group create`, and ends by printing the URL and `/health`. ACR Tasks (`az acr build`) is
not used, because it is paused on credit subscriptions. The model and the demo bundle are git-ignored and
are copied from the local checkout.

**Warning:** the full `az deployment group what-if` output contains the evaluated workspace key. The
script prints it with `--result-format ResourceIdOnly`; if you run what-if by hand, do the same.

Measured on 2026-10-16: a request after scale to zero (cold start) took 4.95 s; warm requests take under 1 s.

The environment runs in Container Apps **Express** mode (the platform picks it even though the template
does not set `environmentMode`). Express facts that matter here:
- no custom domains; the app uses its `azurecontainerapps.io` URL;
- user-assigned managed identities only (no system-assigned);
- pulling from ACR with the identity works when the identity and the registry are sent in one PUT, as the
  template does; adding the registry to an existing app afterwards failed with `BuildFailed: Authentication failed`;
- `az containerapp logs show` (Azure CLI 2.91) fails on Express apps, so read logs with a query on
  `ContainerAppConsoleLogs_CL` in the workspace.

**Live check (spike).** `GET /api/live/health` makes one real model call from the running app: a tiny chat
completion ("Reply with OK", at most 5 output tokens) on `XPLAIN_LIVE_DEPLOYMENT` (default `gpt-4.1-mini`) at
`XPLAIN_FOUNDRY_ENDPOINT`. It is off (404) unless `XPLAIN_LIVE=1`. Auth is Entra ID only:
`DefaultAzureCredential` with `AZURE_CLIENT_ID`, so in Azure the app's user-assigned identity `id-xplain-live`
(Cognitive Services OpenAI User on the Foundry resource) gives the token and locally `az login` does; it gets the
token first, on its own (stage `token`), then makes the request with it as the bearer (stage `request`), never a
key. At most one real call per 60 s for the whole server; in between
it returns the last result with `"cached": true` and its `age_s` (never older than 60 s; the response is
`Cache-Control: no-store`). It answers `{ok, model, latency_s, credential, endpoint_host, cached, age_s}`:
`credential` is the class that actually returned the token (null if none did), `endpoint_host` the host only. On
an error `ok: false` adds `stage`, the exception class, a short message and `causes` (up to three of its
cause / context chain, each `{error, message}`: the openai client reports any failure while sending as
"Connection error."), all with anything token- or key-like removed; the same chain is logged as a warning.
The template sets `AZURE_CLIENT_ID` from the identity itself (`reference(...).clientId`), `XPLAIN_LIVE=1`, the
endpoint and the deployment. The Studio page still plays the precomputed bundle and makes no model calls.

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
- Moment lines carry no numbers; recap numbers are inserted by code, never written by the model. The
  RecapEditor writes placeholders ({home_vs_expected}, {away_vs_expected}, {star_name}, {star_pva}); the recap
  verifier checks the raw text, code fills the placeholders from the recap card, and the filled text is checked
  again.
- A line may give a **reason for the model's rating** ("difficult because it was lofted high") or
  a **concession** ("not completed, even though it was along the ground"). It never gives a cause
  of the outcome. Drivers explain why the *model* rated the pass as it did, not why the pass
  succeeded or failed.
- **Five sentence frames.** Each moment gets one frame, chosen in code (not by the model) and rotated
  through the match from a start fixed by the match id, so neighbouring lines never share a frame and
  a re-run gives the same frames: `reason` (the pass, its difficulty and the reason for the rating, then
  the outcome), `concession` (the pass and outcome, then what it was conceded against), `concession_first`
  (opens with the concession), `player_first` (the player as the subject, concession last) and
  `tier_first` (opens with the difficulty word and its reason, then a contrast word, then the outcome).
  The two frames that start with the pass alternate between two openers ("X's pass" / "The pass from X";
  Spanish and Greek have their own pair), and the outcome phrase rotates through four per kind
  (for example "does not find a teammate", "is lost", "goes astray", "does not reach a teammate").
  The difficulty tier words are fixed per language; the Greek ones are πολύ εύκολη, εύκολη, περίπλοκη,
  δύσκολη and πολύ δύσκολη.
- **Repetition, measured** with `scripts/line_variety.py` (share of near-duplicate lines within a
  bundle) on two synthetic matches, one run each: on seed 3001 (17 commentated moments), near-duplicate
  lines fell from 76% / 76% / 88% (EN / ES / EL) before the frames to 24% / 53% / 47% after; 17 of 17
  lines were accepted on the first try and none rejected. On seed 42, 16 of 17 were accepted on the
  first try and none rejected. These are measurements on two generated matches, not a general claim
  about the Narrator.
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
    from a difficulty word, which reads as a cause of the outcome;
  - (hints mode) leaves a declared driver unnamed in any language, gives the drivers a different role
    (reason for the rating or concession on the outcome) in the three languages, names a position driver
    by both its depth and its width, or puts a concession and a causal marker in one clause;
  - opens with the difficulty word (the `tier_first` frame) without a contrast word before the outcome;
  - says "success" or its Spanish and Greek forms (con éxito, επιτυχ*); the card shows the outcome.
  Phrase-table facts such as a zone ("in the middle third") are accepted only as facts of a declared driver.
- A rejected line is retried up to twice, each time with the Verifier's reasons (three attempts in
  all). If every attempt is rejected, or the model times out or fails, the moment simply has no
  line. The card is still shown.
- Length: the prompt asks for about 18 words; the Verifier rejects more than 24 words in any
  language (`verifier.WORD_LIMIT`, one value; the plate has room for two lines).

## Model comparison

`scripts/compare_models.py` narrates the same 17 commentated moments of seed 3001 with several Foundry
deployments (same prompt, frames, phrase table and Verifier, hints mode, up to 3 attempts) and writes
`reports/model_compare/report.md`: first try / retried / rejected, repetition, latency, tokens, a rough
cost and every line side by side. It is a measurement only; it changes nothing in the Narrator. Results:
gpt-4.1-mini, gpt-4.1 and gpt-6.1-sol all gave 17 accepted on the first try, 0 retried and 0 rejected.
gpt-4.1 produced nearly identical lines at about five times the estimated cost per bundle (0.13 against
0.03 USD, from the script's price table), so gpt-4.1 was dropped. A second run on 2026-10-18
(`reports/model_compare/2026-10-18/`, seed3001 and seed42, gpt-6.1-sol priced at 2.00 / 10.00 USD per 1M
input / output tokens) gave gpt-6.1-sol 17 of 17 on the first try on both matches against 15 and 16 for
gpt-4.1-mini, at about 0.15 against 0.03 USD per bundle and with no gain in variety. gpt-6.1-sol is now the
default for building bundles; gpt-4.1-mini stays selectable with `XPLAIN_FOUNDRY_DEPLOYMENT`. Caveats: one run per model, and the same model gives
different repetition numbers in different runs (gpt-4.1-mini: 24% / 65% / 65% near-duplicates in this
run, 24% / 53% / 47% in the run above), so small differences between models mean little; and gpt-6.1-sol
rejects any temperature but 1 (it ran with `reasoning_effort=low`) while the gpt-4.1 models ran at 0.2,
so that row is not a like-for-like comparison.

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
matches only. The only third-party model is the Microsoft Foundry language model (gpt-6.1-sol by
default, gpt-4.1-mini selectable) that phrases the commentary lines and the recaps; it never sees or produces a
football statistic (the only numbers it handles are the driver ranks 1 to 3; recap numbers are put into its
placeholders by code).

History: an earlier version of the generator used pass-completion weights fitted on StatsBomb Open
Data. After correspondence with the hackathon support team (13 October 2026), who asked entrants
to stay within the published rules on third-party data, the fitted weights were removed and the
scoring model was retrained from scratch on generated data. The earlier commits remain in the
repository history. The generator is now fully designed (version 0.2.1).

The trained model and the demo match bundle are published openly in the
[data-v1 release](https://github.com/charalampisandreou/xplain-live/releases/tag/data-v1): the model
files (`xpass_v2.joblib`, `feature_list.json`, `metrics.json`) under the MIT licence and the demo
bundle (`bundle_seed3001.json`) under CC BY 4.0. Everything is synthetic, produced by generator
0.2.1, and can be regenerated and retrained from this repository (`python -m xplain.train`).

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
`models/README.md`). The published model and demo data are in the
[data-v1 release](https://github.com/charalampisandreou/xplain-live/releases/tag/data-v1). To build
it yourself (about 40 s):

```powershell
python scripts/make_dataset.py
python -m xplain.train --train data/xpass_train.csv --holdout data/xpass_holdout.csv `
    --out models/xpass_v2 --report reports/xpass_v2_metrics.json
```

CI (`.github/workflows/tests.yml`) runs the same suite on every push and pull request to `main`
(Ubuntu, Python 3.13). It downloads `xpass_v2.joblib`, `feature_list.json` and `bundle_seed3001.json`
from the data-v1 release, checks their SHA-256 against the values pinned in the workflow, and fails on
a mismatch before any test runs.

## License

MIT (see `LICENSE`).
