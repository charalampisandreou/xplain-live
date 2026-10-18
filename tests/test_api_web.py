"""The web routes: the Studio page, its static files and the match bundle endpoint."""
import pytest
from fastapi.testclient import TestClient

from xplain import api
from xplain.api import create_app

from .test_scoring import StubScorer


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


# --- Players mode (src/xplain/web/player.js) ---

def test_players_mode_is_on_the_page(client):
    html = client.get("/").text.replace("\r\n", "\n")
    js = client.get("/static/player.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert html.index("/static/player.js") < html.index("/static/app.js")   # app.js mounts it
    modes = html.split('id="mode"')[1].split("</div>")[0]
    assert '<button type="button" data-mode="studio" class="on" aria-pressed="true">Studio</button>' in modes
    assert '<button type="button" data-mode="player" aria-pressed="false">Players</button>' in modes
    assert 'disabled aria-disabled="true" title="Coming soon">Analyst</button>' in modes   # still to come
    assert 'id="player-pick" aria-haspopup="listbox" aria-expanded="false" aria-controls="player-picker"' in html
    assert 'id="player-picker" role="listbox" aria-label="Players" hidden' in html
    view = html.split('id="player-view"')[1].split("</section>")[0]
    for node in ('id="pv-passes"', 'id="pv-completed"', 'id="pv-expected"', 'id="pv-vs"', 'id="pv-bandbar"',
                 'id="pv-rows"', 'aria-label="Pass map key"'):
        assert node in view
    assert '<div class="pv-label">Expected</div>' in view   # the recap's word: a count, not mean xP
    assert "recap" not in view.lower()   # Players mode has no recap


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
    # Players mode keeps only space and the plate's [ ] as keys: nothing steps or replays
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
    # the tier colours: each at 3:1 or more on --bg, and brighter as the tier gets harder
    bg = css.split("--bg: ")[1].split(";")[0]
    colours = [css.split(f"--tier-{i}: ")[1].split(";")[0] for i in range(5)]
    lums = [luminance(c) for c in colours]
    assert lums == sorted(lums) and len(set(colours)) == 5
    assert all(contrast(c, bg) >= 3.0 for c in colours), [round(contrast(c, bg), 2) for c in colours]
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
    # Expected as the recap counts it, with its rate over the passes; vs expected a signed whole number
    assert '<span id="pv-expected">–</span><span class="pv-sub" id="pv-expected-pct"></span>' in html
    assert "setText('pv-expected', s ? String(s.expected) : '–');" in js
    assert "setText('pv-expected-pct', s ? api.percent(s.expectedRate) + '%' : '');" in js
    assert "api.signed(s.vsExpected)" in js and "signed1" not in js
    grid = css.split("\n.pv-stats {")[1].split("}")[0]
    assert "repeat(4, 1fr)" in grid and "align-items: baseline" in grid
    assert "tabular-nums" in css.split("\n.pv-value {")[1].split("}")[0]


def test_picker_shows_positions(client):
    js = client.get("/static/player.js").text
    assert "pos.className = 'pk-pos';" in js and "pl.position" in js
    assert ".pk-pos {" in client.get("/static/app.css").text


def test_empty_states_end_with_a_full_stop(client):
    app = client.get("/static/app.js").text
    js = client.get("/static/player.js").text
    for text in (app, js):
        assert "const NOT_STARTED = 'Starts in the 2nd half.';" in text
    assert "emptyText = 'No commentated moment yet.'" in app
    assert "'No surprises yet.'" in js


def test_title_and_recap_card_are_on_the_page(client):
    html = client.get("/").text
    assert 'data-live="Live pass map" data-ht="HT Recap" data-ft="FT Recap">Live pass map</h1>' in html
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
    assert "!(prevNow < at && state.now >= at) || !isLive() || state.mode !== 'studio'" in app
    assert "openRecap(scope, scope === 'half_time' ? ts + RECAP_PAUSE_MS : null);" in app   # FT stays paused
    assert "const RECAP_PAUSE_MS = 10000;" in app
    link = app.split("const rs = RECAP_KEYS[params.get('recap')];")[1].split("requestAnimationFrame(frame);")[0]
    assert "openRecap(rs, null);" in link   # ?recap: static, no auto-resume
    assert "if (state.recap) { closeRecap(); e.preventDefault(); }" in app   # Esc closes
    # Players mode shows no recap but keeps it open: back in Studio it is there again
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
    assert ".seg.recapview { position: absolute; top: 150px; right: 620px;" in css
    assert "state.recap = { scope, resumeAt, view: 'recap' };" in app
    render = app.split("function renderRecap() {")[1].split("function renderStar(")[0]
    assert "const on = open && state.recap.view === 'recap';" in render
    assert "$('recap-view').hidden = !open;" in render
    assert "title.textContent = open ?" in render


def test_no_league_names_in_the_page(client):
    for path in ("/", "/static/app.js", "/static/player.js", "/static/app.css"):
        text = client.get(path).text
        assert "Premier League" not in text and "Premier" not in text
