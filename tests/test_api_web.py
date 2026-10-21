"""The web routes: the Studio page, its static files and the match bundle endpoint."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from xplain import api
from xplain.api import create_app

from .test_scoring import StubScorer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle_dir(tmp_path):
    (tmp_path / "bundle_demo_1.json").write_text('{"meta": {"schema_version": "2"}}', encoding="utf-8")
    (tmp_path / "secret.json").write_text('{"secret": true}', encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(bundle_dir):
    with TestClient(create_app(StubScorer(), bundles=bundle_dir)) as c:
        yield c


def test_index_is_the_studio_page(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "/static/app.js" in r.text and "/static/app.css" in r.text


def test_static_files_are_served(client):
    js = client.get("/static/app.js")
    css = client.get("/static/app.css")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert "--bg: #0A1B2E" in css.text


def test_a_bundle_is_served_from_the_bundle_dir(client):
    r = client.get("/api/bundle/demo_1")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert r.json() == {"meta": {"schema_version": "2"}}


def test_missing_bundle_is_404(client):
    assert client.get("/api/bundle/nope").status_code == 404


@pytest.mark.parametrize("name", ["Demo_1", "demo-1", "demo.1", "..%2Fsecret", "%2e%2e", "a b", "x" * 65])
def test_bad_names_never_reach_the_disk(client, name):
    r = client.get("/api/bundle/" + name)
    assert r.status_code in (404, 422)
    assert "secret" not in r.text


def test_bundle_dir_env_override(bundle_dir, monkeypatch):
    monkeypatch.setenv("XPLAIN_BUNDLE_DIR", str(bundle_dir))
    assert api.bundle_dir() == bundle_dir
    with TestClient(create_app(StubScorer())) as c:
        assert c.get("/api/bundle/demo_1").status_code == 200


def test_narration_nav_is_on_the_page(client):
    html = client.get("/").text
    assert 'id="narrnav"' in html and 'hidden' in html.split('id="narrnav"')[1].split(">")[0]
    assert 'id="narr-prev" aria-label="Previous narrated moment"' in html
    assert 'id="narr-next" aria-label="Next narrated moment"' in html
    assert 'id="narr-count" aria-live="polite"' in html


def test_narration_keys_and_replay_are_wired(client):
    js = client.get("/static/app.js").text
    assert "e.key === '['" in js and "narrStep(-1)" in js
    assert "e.key === ']'" in js and "narrStep(1)" in js
    assert "enterReplay(target.bundle.moment.moment_id)" in js   # the chevrons reuse replay mode


def test_goal_state_text_uses_the_minute_label(client):
    js = client.get("/static/app.js").text
    body = js.split("function goalTabText(s)")[1].split("}")[0]
    assert "s.minute_label" in body and "'Goal · '" in body
    # fires only from the playback advance in frame(); the goal plate only while live
    frame = js.split("function frame(ts)")[1].split("function isLive")[0]
    assert "goalFlash(after, ts)" in frame and "state.goal.restartT" in frame
    flash = js.split("function goalFlash(i, ts)")[1].split("function pulse(")[0]
    assert flash.index("pulse(3") < flash.index("if (!isLive()) return") < flash.index("state.goal = {")


def test_xp_keeps_its_casing(client):
    html = client.get("/").text
    js = client.get("/static/app.js").text
    css = client.get("/static/app.css").text
    for text in (html, js):
        assert "XP " not in text and "XP<" not in text and "'XP" not in text
    assert ".xp { text-transform: none; }" in css
    assert '<span class="xp">xP</span>' in html.split('id="lower-xp"')[1]   # the upper-case plate meta
    plate = js.split("function renderPlate(")[1].split("async function fitLines")[0]
    assert "' · xP" not in plate and "'xP" not in plate   # the meta no longer writes xP as text


def test_plate_has_one_fixed_height(client):
    css = client.get("/static/app.css").text
    js = client.get("/static/app.js").text
    body = css.split(".body {")[1].split("}")[0]
    assert "height: 102px" in body and "min-height" not in body
    line = css.split(".line {")[1].split("}")[0]
    assert "line-height: 24px" in line and "font-size: 18px" in line
    assert "height: 48px" in line   # two lines reserved, so the meta row never moves
    assert ".line:empty" not in css
    assert "MIN_LINE_PX = 15" in js and "TWO_LINES_PX = 48" in js


def test_plate_transitions_are_light(client):
    js = client.get("/static/app.js").text
    css = client.get("/static/app.css").text
    assert "offsetWidth" not in js   # no forced layout to restart a fade
    plate = js.split("function renderPlate(")[1].split("// --- stage scaling")[0]
    assert "innerHTML" not in plate and "textContent" in plate
    assert "prefers-reduced-motion: reduce" in js and "reducedMotion.matches" in js
    assert "tabular-nums" in css and "min-width: 7ch" in css


def test_pitch_is_pinned_and_plate_columns_are_scoped(client):
    css = client.get("/static/app.css").text
    # a bare .left rule also styled .possession.left and pushed the pitch drawing down
    assert "\n.left {" not in css and "\n.right {" not in css
    assert ".body > .left {" in css and ".body > .right {" in css
    assert "position: absolute; top: 0; left: 0;" in css.split(".pitch > #pitch-svg {")[1].split("}")[0]
    # a rule for every svg in .pitch also caught the possession tab's chevrons
    assert ".pitch svg" not in css


def test_language_switch_is_in_the_plate_not_the_header(client):
    html = client.get("/").text.replace("\r\n", "\n")
    header = html.split('class="topright"')[1].split('<h1 class="title"')[0]
    assert "data-lang" not in header and "language" not in header.lower()
    plate = html.split('id="plate"')[1].split('<div class="controls">')[0]
    group = plate.split('role="group" aria-label="Narration language"')[1].split("</button>\n            </div>")[0]
    assert ">Narration</span>" in group
    buttons = group.split("<button")[1:]
    assert [b.split('data-lang="')[1][:2] for b in buttons] == ["en", "es", "el"]
    assert all('aria-pressed="' in b for b in buttons)
    assert 'data-lang="en" class="on" aria-pressed="true"' in group
    assert html.count("data-lang=") == 3   # one switch on the page
    # the meta row: facts on the left, the switch on the right, under a hairline
    assert plate.index('class="line"') < plate.index('class="rule"') < plate.index('class="metarow"')
    assert plate.index('id="lower-facts"') < plate.index('aria-label="Narration language"')


def test_language_switch_keeps_its_behaviour(client):
    js = client.get("/static/app.js").text
    assert "LANGS.includes(params.get('lang'))" in js   # ?lang=en|es|el
    assert "$('lang').querySelectorAll('button')" in js
    # the plate content fades on a language change; the switch itself does not
    fade = js.split("function fadePlate()")[1].split("}")[0]
    assert "lower-line" in fade and "lang" not in fade.replace("lower-line", "")


def test_pva_column_shows_a_dash_or_the_goal(client):
    html = client.get("/").text
    js = client.get("/static/app.js").text
    css = client.get("/static/app.css").text
    assert '<div class="pvalabel" id="lower-pva-label">PVA</div>' in html
    goal = js.split("function renderGoalPlate(i)")[1].split("function renderNarrNav")[0]
    assert "$('lower-pva').textContent = 'GOAL'" in goal
    assert "$('lower-pva-label').textContent = s.minute_label + \"'\"" in goal   # never computed here
    plate = js.split("function renderPlate(")[1].split("async function fitLines")[0]
    assert "$('lower-pva').textContent = '–'" in plate and plate.count("'lower-pva-label').textContent = 'PVA'") == 2
    # the column is visible in every state (it used to be hidden while waiting and on a goal)
    assert ".body > .right { visibility: hidden" not in css and ".right { visibility: hidden" not in css


def test_health_and_score_routes_are_kept(client):
    assert client.get("/health").json()["status"] == "ok"
    assert any(getattr(r, "path", None) == "/score" for r in client.app.routes)


# --- Player mode (src/xplain/web/player.js) ---

def test_players_mode_is_on_the_page(client):
    html = client.get("/").text.replace("\r\n", "\n")
    js = client.get("/static/player.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert html.index("/static/player.js") < html.index("/static/app.js")   # app.js mounts it
    modes = html.split('id="mode"')[1].split("</div>")[0]
    assert '<button type="button" data-mode="studio" class="on" aria-pressed="true">Studio</button>' in modes
    assert '<button type="button" data-mode="player" aria-pressed="false">Player</button>' in modes
    assert '<button type="button" data-mode="analyst" aria-pressed="false">Analyst</button>' in modes
    assert 'id="player-pick" aria-haspopup="listbox" aria-expanded="false" aria-controls="player-picker"' in html
    assert 'id="player-picker" role="listbox" aria-label="Players" hidden' in html
    view = html.split('id="player-view"')[1].split("</section>")[0]
    for node in ('id="pv-passes"', 'id="pv-completed"', 'id="pv-expected"', 'id="pv-vs"', 'id="pv-bandbar"',
                 'id="pv-rows"', 'aria-label="Pass map key"'):
        assert node in view
    assert '<div class="pv-label">Expected</div>' in view   # the recap's word: a count, not mean xP
    assert "recap" not in view.lower()   # Player mode has no recap


def test_players_mode_draws_the_passers_frame_and_keeps_studio_rules(client):
    js = client.get("/static/player.js").text
    app = client.get("/static/app.js").text
    draw = js.split("function drawPass(")[1].split("function renderMap(")[0]
    assert "mark.pass_geometry" in draw and ".display" not in draw   # the passer's own frame, no mirroring
    assert "class: 'newest'" in draw and "p.start_x, cy: p.start_y, r: 0.8" in draw   # the latest pass: a dot at its start
    assert "fetch(" not in js and "XMLHttpRequest" not in js
    for text in (js, app):
        assert "XP " not in text and "'XP" not in text
    # the plate shows the player's own narrated moments, or says there are none yet
    assert "'No commentated moment for ' + (who ? who.name : 'this player') + ' yet.'" in app
    assert "XplainPlayer.narratedFor(state.bundle.moments, state.player, state.now, null)" in app
    # Player mode keeps only space and the plate's [ ] as keys: nothing steps or replays
    assert "state.mode === 'player' && ![' ', '[', ']'].includes(e.key)" in app


def luminance(hex_colour):
    channels = [int(hex_colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_players_mode_layout_and_contrast(client):
    css = client.get("/static/app.css").text
    assert "#stage.mode-player .title, #stage.mode-player .tables { display: none; }" in css
    assert ("#stage.mode-player .replaybar, #stage.mode-player .controls .live, "
            "#stage.mode-player .controls .steps { display: none; }") in css
    css = css.replace("\r\n", "\n")
    view = css.split("\n.player-view { position")[1].split("}")[0]
    assert "top: 196px" in view and "height: 520px" in view   # the pitch's top and bottom lines
    picker = css.split("\n.player-picker {")[1].split("}")[0]
    assert "top: 196px" in picker and "height: 520px" in picker
    # the tier colours: brighter as the tier gets harder; each at 3:1 or more on --bg, or (the very dark purples)
    # drawn with the light --tier-edge, which is 3:1 or more on --bg itself
    bg = css.split("--bg: ")[1].split(";")[0]
    colours = [css.split(f"--tier-{i}: ")[1].split(";")[0] for i in range(5)]
    lums = [luminance(c) for c in colours]
    assert lums == sorted(lums) and len(set(colours)) == 5
    edge = css.split("--tier-edge: inset 0 0 0 1px rgba(255, 255, 255, ")[1].split(")")[0]
    bg_rgb = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
    edge_hex = "#" + "".join(f"{round(c + float(edge) * (255 - c)):02X}" for c in bg_rgb)   # the edge over --bg
    assert contrast(edge_hex, bg) >= 3.0
    for i, c in enumerate(colours):
        if contrast(c, bg) < 3.0:
            assert f".band.t{i}, .key.t{i} {{ background: var(--tier-{i}); box-shadow: var(--tier-edge); }}" in css
    html = client.get("/").text
    assert "<h2>Completed passes' difficulty</h2>" in html and "How hard the passes were" not in html


def test_tier_words_replace_the_band_words(client):
    html = client.get("/").text
    js = client.get("/static/player.js").text
    for word in ("Very easy", "Easy", "Tricky", "Difficult", "Very difficult"):
        assert f"{word} <b id=" in html
    for word in ("Routine", "Moderate", "Exceptional", "difficulty band"):
        assert word not in html
    assert "m.band" not in js and "m.tier.en" in js


def test_players_stats_show_count_and_percent_on_one_grid(client):
    html = client.get("/").text.replace("\r\n", "\n")
    css = client.get("/static/app.css").text.replace("\r\n", "\n")
    js = client.get("/static/player.js").text
    assert '<span id="pv-completed">–</span><span class="pv-sub" id="pv-completed-pct"></span>' in html
    assert "setText('pv-completed', s ? String(s.completed) : '–');" in js
    # Expected (the sum of xP) with its rate (the mean xP), and vs expected: two decimals, as on the recap card
    assert '<span id="pv-expected">–</span><span class="pv-sub" id="pv-expected-pct"></span>' in html
    assert "setDecimal('pv-expected', s ? fixed2(s.expectedXp, false) : '–');" in js
    assert "setText('pv-expected-pct', s ? api.percent(s.meanXp) + '%' : '');" in js
    assert "setDecimal('pv-vs', text);" in js and "fixed2(s.vsExpectedXp, true)" in js and "signed1" not in js
    assert ".dec { font-size: 0.62em;" in css   # the decimals a step smaller
    grid = css.split("\n.pv-stats {")[1].split("}")[0]
    assert "repeat(4, 1fr)" in grid and "align-items: baseline" in grid
    assert "tabular-nums" in css.split("\n.pv-value {")[1].split("}")[0]


def test_picker_shows_positions(client):
    js = client.get("/static/player.js").text
    assert "pos.className = 'pk-pos';" in js and "pl.position" in js
    assert ".pk-pos {" in client.get("/static/app.css").text


def test_producer_mode_is_on_the_page(client):
    html = client.get("/").text.replace("\r\n", "\n")
    js = client.get("/static/producer.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert html.index("/static/producer.js") < html.index("/static/app.js")   # app.js mounts it
    modes = html.split('id="mode"')[1].split("</div>")[0]
    assert modes.index('data-mode="player"') < modes.index(
        '<button type="button" data-mode="producer" aria-pressed="false">Producer</button>')
    assert "Coming soon" not in modes and "disabled" not in modes   # every mode is built
    log = html.split('<section class="producer-log" id="producer-log"')[1].split("</section>")[0]
    assert '<h2>Log</h2><div class="tsub">newest first</div>' in log and 'id="pr-rows"' in log
    assert '<div class="empty" id="pr-empty" hidden>This match was built without a record of the AI steps.</div>' in log
    assert 'id="pr-detail"' in html


def test_producer_mode_is_wired_and_shares_studio_state(client):
    app = client.get("/static/app.js").text.replace("\r\n", "\n")
    js = client.get("/static/producer.js").text
    assert "else if (params.get('mode') === 'producer') setMode('producer');" in app
    assert "$('stage').classList.toggle('mode-producer', mode === 'producer');" in app
    # a log row replays its moment as a Studio table row does; a recap row is Producer's own pick
    assert "selectMoment: (id) => { if (idx.markOf.has(id)) enterReplay(id); }" in app
    assert "selectRecap: (scope) => { state.producerRecap = scope; }" in app
    sel = app.split("function producerSelected(plate) {")[1].split("\n  }\n")[0]
    assert sel.index("state.producerRecap") < sel.index("state.replay") < sel.index("state.recap") < sel.index("plate ?")
    assert "title.dataset.producer" in app
    # recaps open (and pause) only in Studio: Producer keeps the match running; a recap crossed there is remembered
    assert "if (state.mode !== 'studio') {" in app and "state.unseenRecap = scope;" in app
    # no spoilers: only commentated moments and reached recaps at or before the clock
    assert "if (!m.bundle.qualifies_live || mo.t > now) continue;" in js
    assert "if (!entry || at === undefined || at === null || at > now) continue;" in js
    # checks only from the step's record; collapsed by default; a failed step shows its reasons
    assert "if (!s.checks) {" in js and "const NOT_RECORDED = 'Checks not recorded in this bundle.';" in js
    assert "let openChecks = new Set();" in js and "if (!s.ok && s.reasons.length) wrap.appendChild(reasonsBlock" in js
    assert "s.latency_s + ' s'" in js   # latency as stored
    # On air: one language, the plate's NARRATION switch (state.lang is in render()'s key, so a switch redraws it)
    assert "lang: state.lang," in app and "state.lang, state.follow" in app
    assert "if (m && m.line) text = m.line['line_' + code];" in js and "if (e.text) text = e.text['text_' + code];" in js
    assert "LANGS.map(" not in js.split("function onAir(")[1]
    # the recap's numbers: one step from the last filler, team codes from app.js
    assert "teamCode: (id) => teamCode(id, '')," in app
    assert "if (s.agent === 'recap_filler' && i !== lastFiller) return;" in js


def test_producer_layout_keeps_studio_rules(client):
    css = client.get("/static/app.css").text.replace("\r\n", "\n")
    block = css.split("/* --- Producer mode")[1]
    assert "#stage.mode-producer .pitch, #stage.mode-producer .tables, #stage.mode-producer .scope," in block
    for slot, box in ((".producer-log {", "top: 196px; left: 40px; width: 780px; height: 520px"),
                      (".producer-detail {", "top: 196px; left: 860px; width: 540px; height: 496px")):
        rule = block.split("\n" + slot)[1].split("}")[0]
        assert box in rule and "border:" not in rule and "box-shadow" not in rule and "background" not in rule
    assert re.findall(r"#[0-9A-Fa-f]{3,6}\b", block) == []   # tokens only


def test_a_retake_goes_on_air_only_by_the_producers_choice_and_only_in_this_tab(client):
    app = client.get("/static/app.js").text.replace("\r\n", "\n")
    js = client.get("/static/producer.js").text.replace("\r\n", "\n")
    css = client.get("/static/app.css").text.replace("\r\n", "\n")
    # the choice: under a line that passed the Verifier only; "on", "keep" and "back" all close the retake
    box = js.split("function retakeBox(id, result, lang) {")[1].split("\n  }\n")[0]
    assert "if (o.text) wrap.appendChild(airButtons(id, [['on', 'Put on air', 'pr-retake'], ['keep', 'Keep stored line', 'pr-toggle']]));" in box
    assert "if (a.dataset.air === 'on' && retake.id === id && retake.result && retake.result.line) {" in js
    assert "else if (a.dataset.air === 'back') ctx.air(id, null);" in js
    # app.js keeps it in memory (nothing is sent or stored) and the plate reads it for that moment, in every tab
    assert "aired: new Map()," in app and "fetch(" not in app.split("air: (id, aired) => {")[1].split("},")[0]
    assert "line.textContent = aired ? aired.line[state.lang] : (entry.line && entry.line['line_' + state.lang]) || '';" in app
    assert "state.analystPick, state.airedRev].join('/');" in app   # an air or an undo redraws
    # Show checks: a check a failed step names is marked with a cross
    assert "node('li', failed.has(name) ? 'fail' : ''," in js
    assert ".pr-checklist li.fail::before { content: '✕ '; color: var(--failed-text); }" in css
    # the Log and Steps headings: one height, so their titles and rules line up
    assert ".producer-log > .thead, .producer-detail > .thead { height: 40px; box-sizing: border-box; flex: none; }" in css


def test_a_recap_crossed_in_producer_or_players_opens_on_switching_to_studio(client):
    app = client.get("/static/app.js").text.replace("\r\n", "\n")
    frame = app.split("function frame(ts) {")[1].split("\n  }\n")[0]
    # crossed by live playback only (not in replay or step mode); outside Studio: marked reached-unseen, no pause
    assert "if (at === undefined || !(prevNow < at && state.now >= at) || !isLive()) continue;" in frame
    outside = frame.split("if (state.mode !== 'studio') {")[1].split("continue;")[0]
    assert "state.unseenRecap = scope;" in outside and "setPlaying" not in outside
    assert "unseenRecap" not in frame.split("state.unseenRecap = scope;")[1]   # opened only by the mode switch
    # switching to Studio while live: FT opens, HT only before any second-half pass; opening marks it seen
    mode = app.split("function setMode(mode) {")[1].split("\n  }\n")[0]
    assert mode.rstrip().endswith("if (mode === 'studio') showUnseenRecap();")
    show = app.split("function showUnseenRecap() {")[1].split("\n  }\n")[0]
    assert show.index("state.unseenRecap = null;") < show.index("if (!scope || !isLive() || state.recap) return;")
    assert "if (scope === 'half_time' && live >= 0 && idx.markT[live] > idx.recapT.half_time) return;" in show
    assert show.rstrip().endswith("setPlaying(false);\n    state.pendingRecap = scope;")
    # a restart from kick-off forgets it
    restart = app.split("function setPlaying(on) {")[1].split("\n  }\n")[0]
    assert "state.unseenRecap = null;" in restart


def test_leaving_studio_keeps_the_recap_and_its_switch(client):
    # bug 2026-10-19: the half-time recap's 10 s auto-resume kept running in Player (the star's row opens it)
    # and Producer; play resumed unseen, closed the recap, and back in Studio the Recap / Map switch was gone
    app = client.get("/static/app.js").text.replace("\r\n", "\n")
    mode = app.split("function setMode(mode) {")[1].split("\n  }\n")[0]
    off = "if (mode !== 'studio' && state.recap) state.recap.resumeAt = null;"
    assert off in mode and "closeRecap" not in mode
    assert mode.index(off) < mode.index("if (mode === 'studio') showUnseenRecap();")
    # the auto-resume is the only timed path to closeRecap: it fires only while resumeAt is set
    frame = app.split("function frame(ts) {")[1].split("\n  }\n")[0]
    assert "if (state.recap && state.recap.resumeAt !== null && ts >= state.recap.resumeAt) setPlaying(true);" in frame
    # the star's row goes through setMode, so it is covered too
    star = app.split("function openStar() {")[1].split("\n  }\n")[0]
    assert "setMode('player');" in star


def test_web_files_check_out_with_lf():
    assert "src/xplain/web/* text eol=lf" in (ROOT / ".gitattributes").read_text(encoding="utf-8")


def test_empty_states_end_with_a_full_stop(client):
    app = client.get("/static/app.js").text
    js = client.get("/static/player.js").text
    for text in (app, js):
        assert "const NOT_STARTED = 'Starts in the 2nd half.';" in text
    assert "emptyText = 'No commentated moment yet.'" in app
    assert "'No surprises yet.'" in js
    producer = client.get("/static/producer.js").text
    for text in ("This match was built without a record of the AI steps.", "Checks not recorded in this bundle.", "No line went on air.",
                 "No commentated moment yet.", "Pick a line in the log.", "No steps stored for this line."):
        assert "'" + text + "'" in producer


def test_title_and_recap_card_are_on_the_page(client):
    html = client.get("/").text
    assert ('data-live="Live pass map" data-ht="Half Time" data-ft="Full Time" data-producer="Control room" '
            'data-analyst="Match analysis">Live pass map</h1>') in html
    assert "Pass outliers" not in html
    recap = html.split('<section class="recap" id="recap"')[1].split("</section>")[0]
    for node in ('id="rc-score"', 'id="rc-goal-rows"', 'id="rc-team-rows"', 'id="rc-star"', 'id="rc-text"'):
        assert node in recap
    assert 'id="lower-drain"' in html
    css = client.get("/static/app.css").text.replace("\r\n", "\n")
    box = css.split("\n.recap { position")[1].split("}")[0]
    assert "top: 196px" in box and "left: 40px" in box and "width: 780px" in box and "height: 520px" in box
    assert "border:" not in box and "box-shadow" not in box and "background" not in box   # no box: Studio rules


def test_recap_plate_state_and_drain(client):
    app = client.get("/static/app.js").text
    css = client.get("/static/app.css").text
    # the card's .recap rule (absolute, 780 x 520) also caught a plate with the class "recap"
    assert "classList.add('recap')" not in app and "classList.add('recap-plate')" in app
    assert ".lower.recap-plate .pvanum" in css and ".lower.recap " not in css
    # the half-time drain is a Web Animation, cancelled with the plate, none under reduced motion
    plate = app.split("function renderRecapPlate()")[1].split("function stopDrain()")[0]
    assert "drainLine.animate(" in plate and "duration: left" in plate
    assert "const left = Math.max(0, r.resumeAt - performance.now());" in plate   # goes on where it was
    assert "!reducedMotion.matches" in plate
    assert "drain.cancel()" in app and "@keyframes drain" not in css
    # Passing rows carry the team code (the tables' token); the full names sit under the score
    assert "span('rc-team', teamCode(t.team_id, t.name))" in app


def test_recap_opens_only_on_live_playback_and_by_link(client):
    app = client.get("/static/app.js").text
    assert "const RECAP_KEYS = { ht: 'half_time', ft: 'full_time' };" in app
    assert "if (at === undefined || !(prevNow < at && state.now >= at) || !isLive()) continue;" in app
    assert "if (state.mode !== 'studio') {" in app   # Producer / Player: reached-unseen (unseenRecap), not opened
    assert "openRecap(scope, scope === 'half_time' ? ts + RECAP_PAUSE_MS : null);" in app   # FT stays paused
    assert "const RECAP_PAUSE_MS = 10000;" in app
    link = app.split("const rs = RECAP_KEYS[params.get('recap')];")[1].split("requestAnimationFrame(frame);")[0]
    assert "openRecap(rs, null);" in link   # ?recap: static, no auto-resume
    assert "if (state.recap) { closeRecap(); e.preventDefault(); }" in app   # Esc closes
    # Player mode shows no recap but keeps it open: back in Studio it is there again
    mode = app.split("function setMode(mode) {")[1].split("history.replaceState")[0]
    assert "closeRecap" not in mode
    render = app.split("function renderRecap() {")[1].split("function renderRecapPlate()")[0]
    for field in ("c.score_home", "c.goals", "t.completed", "t.expected", "t.vs_expected", "s.passes",
                  "s.completed", "s.expected", "s.pva", "who.position", "who.shirt_number",
                  "entry.text['text_' + state.lang]"):
        assert field in render
    assert "Math." not in render and "reduce(" not in render   # numbers from the card only


def test_recap_star_text_and_view_switch(client):
    html = client.get("/").text
    app = client.get("/static/app.js").text
    css = client.get("/static/app.css").text.replace("\r\n", "\n")
    # the star: a .thead like Goals and Passing, then one row (shirt number, name, meta, the card's numbers)
    star = html.split('id="rc-starblock"')[1].split('<p class="rc-text"')[0]
    assert "<h2>Star player</h2>" in star and 'id="rc-star"' in star
    assert "for (const label of ['Passes', 'Completed', 'Expected', 'vs expected'])" in app
    assert "c.goals.filter((g) => g.player_name === s.name)" in app   # the star's goals: the card's minutes
    # the text follows the star; it is not pushed to the card's bottom
    text = css.split("\n.rc-text {")[1].split("}")[0]
    assert "margin: 24px 0 0;" in text and "auto" not in text
    assert "while (size > RECAP_MIN_PX && card.scrollHeight > card.clientHeight)" in app
    # Recap / Map: shown while a recap is open in Studio; the title stays the recap's in both views
    switch = html.split('id="recap-view"')[1].split("</div>")[0]
    assert 'data-view="recap" class="on" aria-pressed="true">Recap<' in switch
    assert 'data-view="map" aria-pressed="false">Map<' in switch
    assert ".seg.recapview { position: absolute; top: 110px; right: 620px;" in css   # the title's row, clear of the tabs
    assert "state.recap = { scope, resumeAt, view: 'recap' };" in app
    render = app.split("function renderRecap() {")[1].split("function renderStar(")[0]
    assert "const on = open && state.recap.view === 'recap';" in render
    assert "$('recap-view').hidden = !open;" in render
    assert "      : open ? title.dataset[state.recap.scope === 'half_time' ? 'ht' : 'ft'] : title.dataset.live;" in render


def test_score_bug_reads_ht_and_ft(client):
    app = client.get("/static/app.js").text
    css = client.get("/static/app.css").text
    brk = app.split("function matchBreak() {")[1].split("\n  }\n")[0]
    assert "if (state.now >= c.last_event_t) return 'FT';" in brk
    assert "state.now >= idx.h1End && state.now < h2) return 'HT';" in brk
    assert "$('minute').textContent = brk || (tick >= 0 ? idx.ticks[tick].minute_label" in app   # minutes: the clock's
    assert "$('half').textContent = !brk && half ? HALF_LABEL[half] : '';" in app
    assert ".scorebug.break .live-dot { display: none; }" in css


def test_recap_stays_open_while_stepping_and_the_plate_keeps_its_arrows(client):
    app = client.get("/static/app.js").text
    for fn in ("function step(dir) {", "function enterReplay(id) {"):
        body = app.split(fn)[1].split("\n  }\n")[0]
        assert "recapToMap();" in body and "closeRecap" not in body
    nav = app.split("function narrStep(dir) {")[1].split("\n  }\n")[0]
    assert "if (dir < 0 && narration(liveIndex()).reached > 0) { recapToMap(); backToLive(); }" in nav
    assert "else if (dir > 0 && r) setRecapView('recap');" in nav
    render = app.split("function render() {")[1].split("// --- transitions")[0]
    assert "renderNarrNav({ reached: narr.reached, pos: 0, prev: narr.reached > 0, next: null });" in render
    assert "$('narrnav').hidden = true" not in render
    assert "if (view === 'recap') { backToLive(); state.recap.view = 'recap'; } else recapToMap();" in app
    # Esc: replay and step mode first; a recap closes only from live
    esc = app.split("if (e.key === 'Escape') {")[1].split("else if (e.key === 'ArrowLeft')")[0]
    assert esc.index("leaveReplay()") < esc.index("closeRecap()")


def test_recap_header_star_link_and_shirt_numbers(client):
    html = client.get("/").text
    app = client.get("/static/app.js").text
    js = client.get("/static/player.js").text
    css = client.get("/static/app.css").text
    recap = html.split('<section class="recap" id="recap"')[1].split("</section>")[0]
    assert '<div class="rc-result">' in recap and 'id="rc-home-name"' in recap and 'id="rc-away-name"' in recap
    assert "rc-when" not in recap and "rc-home-code" not in recap   # the title says HT / FT
    # Expected and vs expected from the card's two-decimal values
    assert "decimal('rc-n', t.expected_xp ?? t.expected, false)" in app
    assert "decimal('st-val', s.vs_expected_xp ?? s.pva, true)" in app
    assert "const text = XplainPlayer.fixed2(x, signed);" in app
    # the star row opens the star in Player; from a recap Player opens on the star until a player is picked
    assert '<button type="button" class="st" id="rc-star"></button>' in html
    assert "$('rc-star').addEventListener('click', openStar);" in app
    mode = app.split("function setMode(mode) {")[1].split("history.replaceState")[0]
    assert "if (mode === 'player' && state.recap && !state.playerPicked) {" in mode
    assert "state.playerPicked = true; setMode('player');" in app
    # shirt numbers: the plate's name tab, the Player title, the picker
    assert '<span class="tab-num" id="lower-num"></span>' in html
    assert "$('lower-num').textContent = shirt(m.player_id);" in app
    assert ".tab-num:empty, .tab-num:empty + .tab-div { display: none; }" in css
    assert 'id="player-num" hidden><span class="num-hash" aria-hidden="true">#</span>' in html
    assert "num.textContent = pl.shirt_number != null ? '#' + pl.shirt_number : '';" in js
    assert "head.appendChild(div('pk-sub', 'Passes'));" in js and "pk-check" not in js + css


def test_picker_columns_stay_in_the_panel_and_numbers_line_up(client):
    css = client.get("/static/app.css").text.replace("\r\n", "\n")
    rule = lambda sel: css.split("\n" + sel + " {")[1].split("}")[0]   # noqa: E731
    # a long name is cut, it never widens its column past the panel
    assert "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);" in rule(".player-picker")
    assert "min-width: 0" in rule(".pk-col") and "text-overflow: ellipsis" in rule(".pk-name")
    assert "flex: 1 1 auto; min-width: 0" in rule(".pk-name")   # the one cell that gives
    # fixed cells: the shirt numbers and the passes right-aligned, the positions in one column
    for sel, width in ((".pk-num", "26px"), (".pk-pos", "30px"), (".pk-n", "24px")):
        assert f"flex: none; width: {width}" in rule(sel), sel
    assert "text-align: right" in rule(".pk-num") and "text-align: right" in rule(".pk-n")
    assert "tabular-nums" in rule(".pk-num") and "tabular-nums" in rule(".pk-n")
    assert "font-stretch: 78%" in rule(".pk-team")   # the full team name fits beside PASSES


def test_no_league_names_in_the_page(client):
    for path in ("/", "/static/app.js", "/static/player.js", "/static/analyst.js", "/static/app.css"):
        text = client.get(path).text
        assert "Premier League" not in text and "Premier" not in text


# --- Analyst mode (src/xplain/web/analyst.js) ---

def test_analyst_mode_is_on_the_page(client):
    html = client.get("/").text.replace("\r\n", "\n")
    js = client.get("/static/analyst.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert html.index("/static/analyst.js") < html.index("/static/app.js")   # app.js mounts it
    modes = html.split('id="mode"')[1].split("</div>")[0]
    assert modes.index('data-mode="studio"') < modes.index('data-mode="analyst"') < modes.index('data-mode="player"')
    left = html.split('<section class="analyst-left" id="analyst-left"')[1].split("</section>")[0]
    for node in ('<h2>Match swing</h2>', 'id="an-chart"', 'id="an-chart-empty"', 'id="an-shap-title"',
                 'id="an-shap-rows"', 'id="an-shap-foot"', 'id="an-replay"'):
        assert node in left
    right = html.split('<section class="analyst-right" id="analyst-right"')[1].split("</section>")[0]
    for node in ('<h2>Passing Stats</h2>', 'id="an-team-rows"', '<h2>By difficulty</h2>', 'id="an-tier-rows"'):
        assert node in right
    assert '<span class="rc-n">vs expected</span>' in right   # the recap card's words


def test_analyst_mode_is_wired_and_shares_studio_state(client):
    app = client.get("/static/app.js").text.replace("\r\n", "\n")
    js = client.get("/static/analyst.js").text
    assert "else if (params.get('mode') === 'analyst') {" in app
    assert "$('stage').classList.toggle('mode-analyst', mode === 'analyst');" in app
    assert "XplainAnalyst.invalidate();" in app and "title.dataset.analyst" in app
    # the scope selector and the clock are Studio's; the plate shows the surprise the swing shows
    assert "period: XplainPlayer.scopePeriod(state.scope)" in app and "renderPlate(an.shown, 'No surprise yet.');" in app
    # the plate's chevrons and [ ] page through the surprises on the swing; the newest follows live
    assert "if (state.mode === 'analyst') {   // Analyst mode: the plate pages through the surprises" in app
    assert "else if (e.key === '[' || e.key === ']') { narrStep(e.key === '[' ? -1 : 1); e.preventDefault(); }" in app
    # a pick only moves the ring: the dots are rebuilt only when a surprise is added, so a click always lands
    assert "markShown();" in js and "if (String(found.length) !== view.dotSig) {" in js
    assert "key t" not in js   # By difficulty has no colour keys (they belong to Player's band bar)
    # Replay in Studio leaves Analyst for Studio's replay of that moment
    assert "replay: (id) => { if (idx.markOf.has(id)) { setMode('studio'); enterReplay(id); } }" in app
    # no spoilers: only surprises and passes at or before the clock
    assert "return m.t <= limit &&" in js and "m.t <= limit && (period === null" in js
    assert "s.goal && s.t <= ctx.now" in js
    # SHAP in its own unit, never as points of xP; the values as stored
    assert "(model scale, not %)" in js and "x.drivers.map(" in js and "x.rest" in js
    assert "fetch(" not in js and "XMLHttpRequest" not in js


def test_analyst_layout_keeps_studio_rules(client):
    css = client.get("/static/app.css").text.replace("\r\n", "\n")
    block = css.split("/* --- Analyst mode")[1]
    assert "#stage.mode-analyst .pitch, #stage.mode-analyst .tables, #stage.mode-analyst .recap," in block
    for slot, box in ((".analyst-left {", "top: 196px; left: 40px; width: 780px; height: 520px"),
                      (".analyst-right {", "top: 196px; left: 860px; width: 540px; height: 520px")):
        rule = block.split("\n" + slot)[1].split("}")[0]
        assert box in rule and "border:" not in rule and "box-shadow" not in rule and "background" not in rule
    assert re.findall(r"#[0-9A-Fa-f]{3,6}\b", block) == []   # tokens only


def test_analyst_empty_states_end_with_a_full_stop(client):
    js = client.get("/static/analyst.js").text
    app = client.get("/static/app.js").text
    for text in ("Starts in the 2nd half.", "No passes yet."):
        assert "'" + text + "'" in js
    assert "const NO_PICK = 'No surprise yet. " in js and "as it did.';" in js
    assert "'No surprise yet.'" in app
    assert "const NOT_COMMENTATED = \"Not commentated: when it happened, it was not among the half's top 3 surprises of its kind.\";" in app
