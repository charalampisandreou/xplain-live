"""The welcome screen and tour (src/xplain/web/intro.js, its markup in index.html, its styles in app.css).

node runs intro.js's pure functions: when a visit shows the layer (a plain visit; never ?tour=0 or a deep link), the
order of its screens, and the arrows' geometry. The markup checks pin the approved texts, the four screenshots and
the app's own tokens (never the team colours).
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "src" / "xplain" / "web"
INTRO_JS = WEB / "intro.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not on PATH: the intro.js tests need it")

HARNESS = r"""
const I = require(process.argv[1]);
const cases = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify({
  show: cases.show.map((s) => I.shouldShow(s)),
  reload: (cases.reload || []).map((s) => I.reloadSearch(s)),
  next: cases.next.map(([s, a]) => I.nextScreen(s, a)),
  arrows: cases.arrows.map((n) => I.arrowPath(n)),
}));
"""


def run_node(cases):
    done = subprocess.run([NODE, "-e", HARNESS, str(INTRO_JS), json.dumps(cases)], capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@needs_node
def test_shown_on_a_plain_visit_only():
    plain = ["", "?", "?lang=es", "?scope=2", "?match=seed3001", "?tour=1"]
    skipped = ["?tour=0", "?t=5520", "?mode=analyst", "?replay=m12", "?recap=ft", "?moment=m3", "?player=p7",
               "?lang=el&tour=0", "?scope=1&mode=player"]
    out = run_node({"show": plain + skipped, "next": [], "arrows": []})["show"]
    assert out == [True] * len(plain) + [False] * len(skipped)


@needs_node
def test_a_reload_starts_over():
    """A reload keeps only ?tour and ?match: Studio at kick-off, with the welcome screen unless ?tour=0."""
    searches = ["", "?mode=player&player=dsk-p6", "?mode=analyst&moment=m3&t=5520", "?replay=m12&recap=ft",
                "?tour=0&mode=producer", "?lang=el&scope=2&tour=0", "?match=seed3001&mode=player&player=p7", "?tour=1"]
    out = run_node({"show": [], "reload": searches, "next": [], "arrows": []})["reload"]
    assert out == ["", "", "", "", "?tour=0", "?tour=0", "?match=seed3001", "?tour=1"]
    assert run_node({"show": out, "next": [], "arrows": []})["show"] == [True, True, True, True, False, False,
                                                                         True, True]


@needs_node
def test_screen_order():
    steps = [["welcome", "next"], ["1", "next"], ["2", "next"], ["3", "next"], ["4", "next"], ["end", "next"],
             ["welcome", "skip"], ["1", "skip"], ["3", "skip"], ["end", "start"],
             ["1", "back"], ["2", "back"], ["4", "back"], ["welcome", "back"]]
    out = run_node({"show": [], "next": steps, "arrows": []})["next"]
    assert out == ["1", "2", "3", "4", "end", "app",
                   "app", "end", "end", "app",
                   "1", "1", "3", "welcome"]


@needs_node
def test_arrow_geometry():
    notes = [{"side": "L", "top": 296, "x": 90, "y": 360},
             {"side": "R", "top": 286, "x": 1400, "y": 228},
             {"side": "L", "top": 246, "x": 860, "y": 162, "bend": True}]
    a, b, c = run_node({"show": [], "next": [], "arrows": notes})["arrows"]
    # the point in the 1440 x 900 screenshot, shown at 840 x 525 from (300, 220)
    assert (a["ex"], a["ey"]) == (352.5, 430) and a["d"] == "M262 308 C 312 308, 312 430, 352.5 430"
    assert (b["ex"], b["ey"]) == (1116.7, 353) and b["d"] == "M1178 298 C 1128 298, 1128 353, 1116.7 353"
    assert c["d"] == "M262 258 C 284 258, 278 314.5, 302 314.5 L 801.7 314.5"


HTML = (WEB / "index.html").read_text(encoding="utf-8")
CSS = (WEB / "app.css").read_text(encoding="utf-8")

SLIDES = {
    "Studio": ("The match as it happens: every pass rated, the biggest surprises narrated.",
               ["Every pass, live", "The commentary", "Surprises", "PVA"]),
    "Analyst": ("How each team passed against the model, and how it rated each surprise.",
                ["Passing Swing", "Why the model rated it", "Passing Stats", "By difficulty"]),
    "Player": ("One player's match: every pass, their numbers and their surprises.",
               ["Their passes", "Change player", "Their numbers", "Their moments"]),
    "Producer": ("The control room: every AI step behind each line, checked before air.",
                 ["The log", "Live retake", "The AI agents", "Checked before air"]),
}


def intro_html():
    start = HTML.index('<section class="intro"')
    return HTML[start:HTML.index("</section>", start)]


def test_welcome_and_start_match():
    html = intro_html()
    assert html.count("Every pass rated. Every surprise explained.") == 2
    assert html.count("xPlain <span>Live</span>") == 3   # welcome, the slides' mark, Start match
    for words in ("Quick tour (1 min)", "Skip to match", "Start match", ">Back<", ">Next<", ">Skip<"):
        assert words in html
    assert 'role="dialog"' in html and 'aria-modal="true"' in html


def test_the_synthetic_line_is_on_welcome_and_start_match():
    html = intro_html()
    welcome = html[html.index('data-screen="welcome"'):html.index('data-screen="1"')]
    end = html[html.index('data-screen="end"'):html.index('id="in-nav"')]
    line = '<p class="in-synth">A synthetic match, generated for this demo.</p>'
    assert line in welcome and line in end and html.count("synthetic") == 2


def test_slides_texts_and_screenshots():
    html = intro_html()
    for i, (mode, (line, heads)) in enumerate(SLIDES.items(), 1):
        start = html.index(f'data-screen="{i}"')
        slide = html[start:html.index("</div>\n    <div class=\"in-screen", start)]
        assert f">{mode}</h2>" in slide and f"<p>{line}</p>" in slide and f"{i} of 4" in slide
        assert [h for h in re.findall(r"<b>([^<]+)</b>", slide)] == heads
        assert len(re.findall(r'data-at="\d+,\d+"', slide)) == 4
        img = re.search(r'src="/static/intro/(\w+\.png)"', slide).group(1)
        png = WEB / "intro" / img
        assert png.read_bytes()[16:24] == (1440).to_bytes(4, "big") + (900).to_bytes(4, "big")


def test_approved_note_texts():
    html = intro_html()
    for text in ("the chance it had to be completed.", "The reasons behind each surprising pass's",
                 "Every pass the player has made so far in the match.", "Every surprise, and the line that went on air.</div>",
                 "4 agents working together to write a commentary line for each surprising pass.",
                 "Each team's completed passes against what the model expected.",
                 "Where each team beat or fell short of the model, easy passes to hard."):
        assert text in html


def test_loaded_before_paint_and_mounted_by_app():
    head = HTML[:HTML.index("</head>")]
    assert '<script src="/static/intro.js"></script>' in head
    js = (WEB / "app.js").read_text(encoding="utf-8")
    assert "XplainIntro.mount(" in js and "if (state.intro) return;" in js


def test_intro_styles_use_app_tokens_only():
    block = CSS[CSS.index("/* --- the welcome screen and tour"):]
    assert "--home" not in block and "--away" not in block
    assert "localStorage" not in INTRO_JS.read_text(encoding="utf-8")
