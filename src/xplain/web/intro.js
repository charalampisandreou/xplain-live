// xPlain Live: the welcome screen and the four-slide tour. A layer inside #stage (the same 1440 x 900 canvas, the
// same tokens), shown on every plain visit: welcome -> slides 1..4 -> Start match -> Studio from kick-off. Nothing
// is remembered; ?tour=0 and the deep links skip it. Every text lives in index.html; this file only moves between
// the screens and draws the slides' arrows from each note's data attributes.
//
// Loaded in <head>: it decides before the first paint whether the layer shows (the "intro-on" class on <html>),
// so a deep link never flashes it. app.js mounts it and starts playback when it closes.
(function (root) {
  'use strict';

  // --- pure functions (no DOM; also loaded by node in tests/test_intro_js.py) --------------------

  const DEEP_LINKS = ['t', 'mode', 'replay', 'recap', 'moment', 'player'];
  const ORDER = ['welcome', '1', '2', '3', '4', 'end'];

  // whether a visit with this query string opens on the welcome screen: not with ?tour=0 or a deep link
  function shouldShow(search) {
    const params = new URLSearchParams(search || '');
    if (params.get('tour') === '0') return false;
    return !DEEP_LINKS.some((k) => params.has(k));
  }

  // The screen after an action on a screen: 'next' (Quick tour, Next, Enter / Right), 'back' (Back, Left), 'skip'
  // (Skip, Skip to match, Esc), 'start' (Start match). 'app' closes the layer.
  function nextScreen(screen, action) {
    const i = ORDER.indexOf(screen);
    if (i < 0) return 'app';
    if (action === 'start') return 'app';
    if (action === 'skip') return screen === 'welcome' || screen === 'end' ? 'app' : 'end';
    if (action === 'back') return i >= 2 && i <= 4 ? ORDER[i - 1] : screen;   // slide 1 has no Back
    if (action === 'next') return screen === 'end' ? 'app' : ORDER[i + 1];
    return screen;
  }

  // A note's arrow on the slide: from the note's heading (its side's inner edge, 12px below its top) to the point it
  // marks, given in the 1440 x 900 screenshot, which the slide shows at 840 x 525 from (300, 220). One S curve; a
  // bent note turns at once and runs straight to its point.
  const r1 = (v) => Math.round(v * 10) / 10;
  function arrowPath(note) {
    const left = note.side === 'L';
    const nx = left ? 262 : 1178, ny = note.top + 12;
    const ex = r1(300 + note.x * 840 / 1440), ey = r1(220 + note.y * 525 / 900);
    let d;
    if (note.bend) d = `M${nx} ${ny} C ${nx + 22} ${ny}, ${nx + 16} ${ey}, ${nx + 40} ${ey} L ${ex} ${ey}`;
    else {
      const cx = left ? nx + 50 : nx - 50;
      d = `M${nx} ${ny} C ${cx} ${ny}, ${cx} ${ey}, ${ex} ${ey}`;
    }
    return { d, ex, ey };
  }

  // --- the layer (browser only) -------------------------------------------------------------------

  const SVG = 'http://www.w3.org/2000/svg';
  const SLIDE_MS = 220, GLIDE_MS = 450, FADE_MS = 150, SHIFT_PX = 12;
  const EASE_OUT = 'cubic-bezier(0.2, 0, 0, 1)', EASE_IN_OUT = 'cubic-bezier(0.65, 0, 0.35, 1)';
  const $ = (id) => document.getElementById(id);

  let open = false;
  let screen = 'welcome';
  let busy = null;      // the running transition's promise; a new action finishes it first (running: its animations)
  let running = [];
  let opts = null;

  function isOpen() { return open; }

  function reduced() {
    return root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function screenEl(name) { return $('intro').querySelector(`.in-screen[data-screen="${name}"]`); }

  // the stage's scale (app.js sets --scale): client px per stage px
  function scale() { return $('stage').getBoundingClientRect().width / 1440 || 1; }

  // an element's box in stage px, relative to the stage's top-left corner
  function stageBox(el) {
    const s = scale(), st = $('stage').getBoundingClientRect(), b = el.getBoundingClientRect();
    return { x: (b.left - st.left) / s, y: (b.top - st.top) / s, w: b.width / s, h: b.height / s };
  }

  // the slides' small mark sits exactly on the app header's mark, so nothing jumps between the tour and the app
  function placeMark() {
    const app = document.querySelector('.topright .mark');
    if (!app) return;
    const b = stageBox(app), mark = $('in-mark');
    mark.style.left = b.x + 'px';
    mark.style.top = b.y + 'px';
    mark.style.right = 'auto';
  }

  // each slide's arrows, built once at mount from the notes' data (pure math, no layout reads): they sit in the
  // slide's own svg, so they move and fade with it
  function drawArrows(el) {
    const svg = el.querySelector('.in-arrows');
    if (!svg || svg.childNodes.length) return;
    for (const n of el.querySelectorAll('.in-note')) {
      const [x, y] = n.dataset.at.split(',').map(Number);
      const a = arrowPath({ side: n.dataset.side, top: Number(n.dataset.top), x, y, bend: n.hasAttribute('data-bend') });
      const path = document.createElementNS(SVG, 'path');
      path.setAttribute('d', a.d);
      const dot = document.createElementNS(SVG, 'circle');
      dot.setAttribute('cx', a.ex);
      dot.setAttribute('cy', a.ey);
      dot.setAttribute('r', '5');
      svg.append(path, dot);
    }
  }

  function setNav(name) {
    const slide = /^[1-4]$/.test(name);
    $('in-nav').hidden = !slide;
    $('in-skip').hidden = !slide;
    $('in-mark').hidden = !slide;
    if (!slide) return;
    $('in-back').style.visibility = name === '1' ? 'hidden' : '';
    for (const d of $('in-dots').children) d.classList.toggle('on', d.dataset.dot === name);
  }

  function show(name) {
    for (const s of $('intro').querySelectorAll('.in-screen')) s.hidden = s.dataset.screen !== name;
    const el = screenEl(name);
    $('intro').setAttribute('aria-labelledby', el.querySelector('[data-label]').id);
    setNav(name);
    screen = name;
    return el;
  }

  function focusPrimary() {
    const el = screen === 'welcome' || screen === 'end' ? screenEl(screen) : $('in-nav');
    const b = el.querySelector('.in-b.go');
    if (b) b.focus({ preventScroll: true, focusVisible: false });   // keyboard users still get the ring as they move
  }

  // the transition's animations: kept so a new action can finish them at once; will-change only while they run
  function finished(anims) {
    running = anims;
    const els = anims.map((a) => a.effect && a.effect.target).filter(Boolean);
    for (const e of els) e.style.willChange = 'opacity, transform';
    return Promise.all(anims.map((a) => a.finished.catch(() => {}))).then(() => {
      for (const e of els) e.style.willChange = '';
      running = [];
    });
  }

  // one slide to another: a cross-fade with a 12px shift in the direction of travel
  async function slideTo(name, dir) {
    const from = screenEl(screen);
    const to = show(name);
    from.hidden = false;
    if (reduced()) {
      await finished([from.animate([{ opacity: 1 }, { opacity: 0 }], { duration: FADE_MS }),
        to.animate([{ opacity: 0 }, { opacity: 1 }], { duration: FADE_MS })]);
    } else {
      await finished([
        from.animate([{ opacity: 1, transform: 'none' }, { opacity: 0, transform: `translateX(${-SHIFT_PX * dir}px)` }],
          { duration: SLIDE_MS, easing: EASE_OUT }),
        to.animate([{ opacity: 0, transform: `translateX(${SHIFT_PX * dir}px)` }, { opacity: 1, transform: 'none' }],
          { duration: SLIDE_MS, easing: EASE_OUT })]);
    }
    from.hidden = true;
  }

  // a slide -> Start match: the small mark grows back into the big logo (the glide reversed) while the slide and
  // its nav fade; the tagline and Start match fade in once the logo has landed
  async function toEnd() {
    const from = screenEl(screen);
    const mark = $('in-mark');
    const markBox = stageBox(mark);
    const fading = [from, $('in-nav'), $('in-skip')];
    const to = show('end');
    from.hidden = false;
    $('in-nav').hidden = false;
    $('in-skip').hidden = false;
    if (reduced()) {
      await finished([...fading.map((f) => f.animate([{ opacity: 1 }, { opacity: 0 }], { duration: FADE_MS })),
        to.animate([{ opacity: 0 }, { opacity: 1 }], { duration: FADE_MS })]);
    } else {
      const logo = to.querySelector('.in-big');
      const rest = [...to.children].filter((c) => c !== logo);
      const a = stageBox(logo), k = markBox.w / a.w;
      await finished([
        logo.animate([
          { transform: `translate(${markBox.x - a.x}px, ${markBox.y - a.y}px) scale(${k})` },
          { transform: 'none' },
        ], { duration: GLIDE_MS, easing: EASE_IN_OUT }),
        ...fading.map((f) => f.animate([{ opacity: 1 }, { opacity: 0 }], { duration: SLIDE_MS, easing: EASE_OUT, fill: 'forwards' })),
        ...rest.map((c) => c.animate([{ opacity: 0 }, { opacity: 1 }], { duration: SLIDE_MS, delay: GLIDE_MS, easing: EASE_OUT, fill: 'backwards' })),
      ]);
      fading.forEach((f) => f.getAnimations().forEach((x) => x.cancel()));
    }
    from.hidden = true;
    setNav('end');
  }

  // The big logo glides into a small mark (the slides' mark, or the app header's): translate + scale from its box to
  // the mark's, while everything else on its screen fades. The mark itself stays hidden until the logo lands.
  function glide(logo, target) {
    const a = stageBox(logo), b = stageBox(target);
    const k = b.w / a.w;
    return logo.animate([
      { transform: 'none' },
      { transform: `translate(${b.x - a.x}px, ${b.y - a.y}px) scale(${k})` },
    ], { duration: GLIDE_MS, easing: EASE_IN_OUT, fill: 'forwards' });
  }

  function fadeOut(els, ms) {
    return els.map((e) => e.animate([{ opacity: 1 }, { opacity: 0 }], { duration: ms, easing: EASE_OUT, fill: 'forwards' }));
  }

  // welcome -> slide 1: the logo becomes the slide's small mark; the slide fades in under it
  async function tour() {
    const from = screenEl('welcome');
    placeMark();
    const to = show('1');
    from.hidden = false;
    const mark = $('in-mark');
    if (reduced()) {
      await finished([from.animate([{ opacity: 1 }, { opacity: 0 }], { duration: FADE_MS }),
        to.animate([{ opacity: 0 }, { opacity: 1 }], { duration: FADE_MS })]);
    } else {
      mark.style.visibility = 'hidden';
      const logo = from.querySelector('.in-big');
      const rest = [...from.children].filter((c) => c !== logo);
      const nav = [$('in-nav'), $('in-skip')];
      await finished([
        glide(logo, mark),
        ...fadeOut(rest, SLIDE_MS),
        to.animate([{ opacity: 0 }, { opacity: 1 }], { duration: SLIDE_MS, delay: GLIDE_MS - SLIDE_MS, easing: EASE_OUT, fill: 'backwards' }),
        ...nav.map((n) => n.animate([{ opacity: 0 }, { opacity: 1 }], { duration: SLIDE_MS, delay: GLIDE_MS - SLIDE_MS, fill: 'backwards' })),
      ]);
      mark.style.visibility = '';
      logo.getAnimations().forEach((x) => x.cancel());
      rest.forEach((c) => c.getAnimations().forEach((x) => x.cancel()));
    }
    from.hidden = true;
  }

  // the layer -> the app: the logo glides into the app header's mark while the layer's background fades and Studio
  // shows through; the header's own mark takes over when the logo lands, then playback starts
  async function leave() {
    const stage = $('stage'), intro = $('intro');
    const el = screenEl(screen);
    const logo = el.querySelector('.in-big');
    const appMark = document.querySelector('.topright .mark');
    for (const c of stage.children) c.inert = false;
    if (reduced() || !logo || !appMark) {
      await finished([intro.animate([{ opacity: 1 }, { opacity: 0 }], { duration: FADE_MS, fill: 'forwards' })]);
    } else {
      stage.classList.add('intro-gliding');   // the header's mark waits for the logo
      const rest = [...el.children].filter((c) => c !== logo);
      await finished([
        glide(logo, appMark),
        ...fadeOut(rest, SLIDE_MS),
        $('in-bg').animate([{ opacity: 1 }, { opacity: 0 }], { duration: GLIDE_MS, easing: EASE_IN_OUT, fill: 'forwards' }),
      ]);
      stage.classList.remove('intro-gliding');
    }
    intro.hidden = true;
    document.documentElement.classList.remove('intro-on');
    open = false;
    if (opts && opts.onDone) opts.onDone();
  }

  // An action while a transition runs finishes that transition at once (its end state, cleanly), then runs: quick
  // presses step through the slides without stacking animations or dropping a press.
  async function go(action) {
    while (busy) {
      running.forEach((a) => a.finish());
      await busy;
    }
    if (!open) return;
    const to = nextScreen(screen, action);
    if (to === screen) return;
    busy = (async () => {
      if (to === 'app') await leave();
      else if (screen === 'welcome' && to === '1') await tour();
      else if (to === 'end') await toEnd();
      else await slideTo(to, ORDER.indexOf(to) > ORDER.indexOf(screen) ? 1 : -1);
    })();
    try { await busy; } finally { busy = null; }
    if (open) focusPrimary();
  }

  // Tab and Shift+Tab stay on the layer's shown buttons
  function trapTab(e) {
    const shown = [...$('intro').querySelectorAll('button')].filter((b) => b.offsetParent !== null
      && getComputedStyle(b).visibility !== 'hidden');
    if (!shown.length) return;
    const i = shown.indexOf(document.activeElement);
    const j = e.shiftKey ? (i <= 0 ? shown.length - 1 : i - 1) : (i + 1) % shown.length;
    shown[j].focus({ preventScroll: true });
    e.preventDefault();
  }

  function onKey(e) {
    if (!open) return;
    const onButton = e.target.closest && e.target.closest('button');
    if (e.key === 'Tab') trapTab(e);
    else if (e.key === 'ArrowRight' || (e.key === 'Enter' && !onButton)) { go('next'); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { go('back'); e.preventDefault(); }
    else if (e.key === 'Escape') { if (screen !== 'end') go('skip'); e.preventDefault(); }
  }

  // shows the layer when this visit should (shouldShow); opts.onDone runs once it has closed into the app
  function mount(o) {
    opts = o || {};
    const intro = $('intro');
    open = document.documentElement.classList.contains('intro-on');
    if (!open) { intro.hidden = true; return false; }
    for (const c of $('stage').children) if (c !== intro) c.inert = true;
    intro.addEventListener('click', (e) => {
      const b = e.target.closest('[data-go]');
      if (b) go(b.dataset.go);
    });
    document.addEventListener('keydown', onKey);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(placeMark);
    root.addEventListener('resize', placeMark);
    // decode the four screenshots now, so no slide decodes a 1440 x 900 image in the middle of its fade
    for (const img of intro.querySelectorAll('.in-shot img')) if (img.decode) img.decode().catch(() => {});
    for (const el of intro.querySelectorAll('.in-slide')) drawArrows(el);
    show('welcome');
    focusPrimary();
    return true;
  }

  const api = { shouldShow, nextScreen, arrowPath, mount, isOpen };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else {
    root.XplainIntro = api;
    // before the first paint: the layer shows only on a plain visit
    if (shouldShow(root.location.search)) document.documentElement.classList.add('intro-on');
  }
})(typeof window !== 'undefined' ? window : this);
