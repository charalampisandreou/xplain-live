// xPlain Live: Studio screen. Plays a match bundle (src/xplain/bundle.py) on the match clock.
// Every number shown comes from the bundle; the browser only formats xP as a percent with the same
// rule as analytics.percent, round(100 * x), and shows rank moves from rank and previous_rank.
(function () {
  'use strict';

  const SVG = 'http://www.w3.org/2000/svg';
  const KINDS = ['surprise_failure', 'surprise_success'];
  const EMPTY = {
    surprise_failure: 'No surprise failures yet this half.',
    surprise_success: 'No surprise successes yet this half.',
  };
  const FADED = [0.8, 0.55, 0.35];   // opacity of the 1st, 2nd and 3rd pass before the selected one

  const $ = (id) => document.getElementById(id);
  const percent = (x) => Math.round(100 * x);
  // PVA (the bundle's delta = outcome - xP) as sign + two decimals, with a real minus sign
  const pva = (x) => (x < 0 ? '−' : '+') + Math.abs(x).toFixed(2);
  const code = (name) => (name || '').slice(0, 3).toUpperCase();

  const state = {
    bundle: null,
    now: 0,            // match t, seconds
    playing: true,
    speed: 12,
    follow: true,      // live: the selected pass is the newest pass with t <= now
    selected: -1,      // index into marks when not following
    lang: 'en',
    lastFrame: null,
    renderedKey: '',
  };
  let idx = null;      // lookups built once per bundle

  // --- loading -------------------------------------------------------------------------------

  function matchName() {
    const name = new URLSearchParams(location.search).get('match') || 'seed42';
    return /^[a-z0-9_]+$/.test(name) ? name : 'seed42';
  }

  function showStatus(text) {
    const el = $('status');
    el.textContent = text;
    el.hidden = false;
  }

  async function load() {
    const name = matchName();
    let res;
    try {
      res = await fetch('/api/bundle/' + name);
    } catch (e) {
      showStatus('Could not reach the server to load match "' + name + '".');
      return;
    }
    if (!res.ok) {
      showStatus('No match bundle named "' + name + '" (HTTP ' + res.status + ').');
      return;
    }
    const bundle = await res.json();
    if (!bundle.meta || bundle.meta.schema_version !== '3') {
      showStatus('The match bundle "' + name + '" has an old schema; rebuild it with python -m xplain.bundle.');
      return;
    }
    start(bundle);
  }

  function start(bundle) {
    state.bundle = bundle;
    const teams = {};
    teams[bundle.meta.home.team_id] = bundle.meta.home.name;
    teams[bundle.meta.away.team_id] = bundle.meta.away.name;
    idx = {
      teams,
      marks: bundle.marks,
      markT: bundle.marks.map((m) => m.t),
      ticks: bundle.clock.ticks,
      tickT: bundle.clock.ticks.map((k) => k.t),
      score: bundle.score,
      scoreT: bundle.score.map((s) => s.t),
      moments: new Map(bundle.moments.map((m) => [m.bundle.moment.moment_id, m])),
      // commentated moments in seq order, for the lower third
      live: bundle.moments.filter((m) => m.bundle.qualifies_live)
        .sort((a, b) => a.bundle.moment.seq - b.bundle.moment.seq),
      rankings: bundle.rankings,
      rankT: bundle.rankings.map((s) => s.t),
    };
    $('home-code').textContent = code(bundle.meta.home.name);
    $('away-code').textContent = code(bundle.meta.away.name);
    $('scorebug').setAttribute('aria-label', bundle.meta.home.name + ' against ' + bundle.meta.away.name);
    $('home-code').title = bundle.meta.home.name;
    $('away-code').title = bundle.meta.away.name;
    // ?t=<match t in seconds> starts playback there (demo and checks); default kick-off
    const startT = Number(new URLSearchParams(location.search).get('t'));
    const { first_event_t: first, last_event_t: last } = bundle.clock;
    state.now = Number.isFinite(startT) && startT > first ? Math.min(startT, last) : first;
    wire();
    const lang = new URLSearchParams(location.search).get('lang');
    if (['en', 'es', 'el'].includes(lang)) setLang(lang);
    requestAnimationFrame(frame);
  }

  // --- time ----------------------------------------------------------------------------------

  // index of the last value <= t in a sorted array, -1 when none
  function lastAtOrBefore(sorted, t) {
    let lo = 0, hi = sorted.length - 1, ans = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid] <= t) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    return ans;
  }

  function frame(ts) {
    if (state.lastFrame !== null && state.playing) {
      const end = state.bundle.clock.last_event_t;
      state.now = Math.min(end, state.now + ((ts - state.lastFrame) / 1000) * state.speed);
      if (state.now >= end) setPlaying(false);
    }
    state.lastFrame = ts;
    render();
    requestAnimationFrame(frame);
  }

  function liveIndex() { return lastAtOrBefore(idx.markT, state.now); }
  function selectedIndex() { return state.follow ? liveIndex() : Math.min(state.selected, liveIndex()); }

  // --- controls ------------------------------------------------------------------------------

  function setPlaying(on) {
    if (on && state.now >= state.bundle.clock.last_event_t) {
      state.now = state.bundle.clock.first_event_t;   // play again from kick-off
      state.follow = true;
    }
    state.playing = on;
    $('play').setAttribute('aria-label', on ? 'Pause' : 'Play');
    $('play-icon').innerHTML = on
      ? '<rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor"></rect><rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor"></rect>'
      : '<path d="M7 5 L19 12 L7 19 z" fill="currentColor"></path>';
  }

  function setSpeed(s) {
    state.speed = s;
    for (const b of $('speed').querySelectorAll('button')) {
      const on = Number(b.dataset.speed) === s;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    }
  }

  function setLang(lang) {
    state.lang = lang;
    for (const b of $('lang').querySelectorAll('button')) {
      const on = b.dataset.lang === lang;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    }
  }

  function step(dir) {
    const live = liveIndex();
    if (live < 0) return;
    const from = selectedIndex();
    state.selected = Math.max(0, Math.min(live, from + dir));
    state.follow = false;
  }

  function backToLive() { state.follow = true; }

  function wire() {
    $('play').addEventListener('click', () => setPlaying(!state.playing));
    for (const b of $('speed').querySelectorAll('button')) {
      b.addEventListener('click', () => setSpeed(Number(b.dataset.speed)));
    }
    for (const b of $('lang').querySelectorAll('button')) {
      b.addEventListener('click', () => setLang(b.dataset.lang));
    }
    $('prev').addEventListener('click', () => step(-1));
    $('next').addEventListener('click', () => step(1));
    $('live').addEventListener('click', backToLive);
    document.addEventListener('keydown', (e) => {
      if (e.target.closest && e.target.closest('button') && (e.key === ' ' || e.key === 'Enter')) return;
      if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
      else if (e.key === 'ArrowRight') { step(1); e.preventDefault(); }
      else if (e.key === ' ') { setPlaying(!state.playing); e.preventDefault(); }
      else if (e.key === 'l' || e.key === 'L') backToLive();
    });
    setPlaying(true);
    setSpeed(state.speed);
  }

  // --- render --------------------------------------------------------------------------------

  function render() {
    const sel = selectedIndex();
    const live = liveIndex();
    const tick = lastAtOrBefore(idx.tickT, state.now);
    const score = lastAtOrBefore(idx.scoreT, state.now);
    const half = live >= 0 ? idx.marks[live].period : 1;
    const snaps = KINDS.map((k) => latestSnapshot(k, half));
    const plate = plateMoment(sel);
    const key = [sel, live, tick, score, half, snaps.map((s) => (s ? s.t : '-')).join('|'),
      plate ? plate.bundle.moment.moment_id : '-', state.lang, state.follow].join('/');
    $('live').classList.toggle('is-live', state.follow);
    $('prev').disabled = sel <= 0;
    $('next').disabled = state.follow || sel >= live;
    if (key === state.renderedKey) return;
    state.renderedKey = key;

    renderScoreBug(tick, score, sel);
    $('subtitle').textContent = 'Every pass scored live · ' + (half === 2 ? 'second' : 'first') + ' half so far';
    renderDirection(sel >= 0 ? idx.marks[sel].period : half);
    renderPitch(sel);
    KINDS.forEach((k, i) => renderTable(k, snaps[i], plate));
    renderPlate(plate);
  }

  function renderScoreBug(tick, score, sel) {
    const s = score >= 0 ? idx.score[score] : null;
    $('score').textContent = s ? s.home + '–' + s.away : '–';
    $('minute').textContent = tick >= 0 ? idx.ticks[tick].minute_label + "'" : '';
    // possession: the team of the selected pass gets the cyan bar under its code
    const ball = sel >= 0 ? idx.marks[sel].team_id : null;
    const meta = state.bundle.meta;
    for (const [id, team] of [['home-code', meta.home], ['away-code', meta.away]]) {
      const has = team.team_id === ball;
      $(id).classList.toggle('has-ball', has);
      $(id).setAttribute('aria-label', team.name + (has ? ', in possession' : ''));
    }
  }

  // "DUS →   ← MAR": who attacks right on the display pitch in this half (meta.attacks_right)
  function renderDirection(period) {
    const meta = state.bundle.meta;
    const right = meta.attacks_right[String(period)];
    const left = right === meta.home.team_id ? meta.away.team_id : meta.home.team_id;
    $('attack-dir').textContent = teamCode(right) + ' → ← ' + teamCode(left);
  }

  // the latest snapshot of a kind with t <= now in the given half, or null
  function latestSnapshot(kind, period) {
    for (let i = lastAtOrBefore(idx.rankT, state.now); i >= 0; i--) {
      const s = idx.rankings[i];
      if (s.period !== period) return null;   // snapshots are in t order: an earlier half is not this one
      if (s.kind === kind) return s;
    }
    return null;
  }

  // the latest commentated moment at or before the selected pass, or null
  function plateMoment(sel) {
    if (sel < 0) return null;
    const seq = idx.marks[sel].seq;
    let found = null;
    for (const m of idx.live) {
      if (m.bundle.moment.seq > seq) break;
      found = m;
    }
    return found;
  }

  function teamCode(teamId, fallbackName) { return code(idx.teams[teamId] || fallbackName); }
  function playerName(m) { return m.player_name || m.player_id || 'Unknown player'; }

  // --- pitch ---

  function el(name, attrs) {
    const node = document.createElementNS(SVG, name);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  // Passes are drawn in the bundle's display coordinates (mark.display): one fixed pitch per half,
  // home attacking right in the first half and left in the second. The page does no mirroring.
  function drawPass(parent, mark, selected, opacity) {
    const d = mark.display;
    const [sx, sy, ex, ey] = [d.start_x, d.start_y, d.end_x, d.end_y];
    const g = el('g', { opacity: String(opacity) });
    const width = selected ? 0.75 : 0.45;
    const colour = mark.completed ? '#3BD16F' : '#FF4B47';
    if (selected) {
      g.appendChild(el('circle', { cx: sx, cy: sy, r: 1.6, fill: '#10294A', stroke: '#3FD0FF', 'stroke-width': 0.45 }));
    }
    if (mark.completed) {
      g.appendChild(el('line', { x1: sx, y1: sy, x2: ex, y2: ey, stroke: colour, 'stroke-width': width,
        fill: 'none', 'marker-end': 'url(#arrow-green)' }));
    } else {
      g.appendChild(el('line', { x1: sx, y1: sy, x2: ex, y2: ey, stroke: colour, 'stroke-width': width,
        'stroke-dasharray': selected ? '1.6 1.1' : '1.2 0.9' }));
      const c = selected ? 1.3 : 0.9;
      const cross = el('g', { stroke: colour, 'stroke-width': selected ? 0.85 : 0.5, 'stroke-linecap': 'round' });
      cross.appendChild(el('line', { x1: ex - c, y1: ey - c, x2: ex + c, y2: ey + c }));
      cross.appendChild(el('line', { x1: ex + c, y1: ey - c, x2: ex - c, y2: ey + c }));
      g.appendChild(cross);
    }
    parent.appendChild(g);
  }

  function renderPitch(sel) {
    const layer = $('passes');
    layer.replaceChildren();
    const svg = $('pitch-svg');
    if (sel < 0) {
      svg.setAttribute('aria-label', 'Pitch, no pass yet');
      return;
    }
    const mark = idx.marks[sel];
    for (let back = 3; back >= 1; back--) {
      const prev = idx.marks[sel - back];
      // only passes of the same half: the display pitch turns round at half-time
      if (prev && prev.period === mark.period) drawPass(layer, prev, false, FADED[back - 1]);
    }
    drawPass(layer, mark, true, 1);
    svg.setAttribute('aria-label', 'Pitch with the selected pass by ' + playerName(mark) + ' drawn as ' +
      (mark.completed ? 'a green solid arrow, because it was completed'
        : 'a red dashed line ending in a cross, because it was not completed') +
      ', and the three passes before it faded');
  }

  // --- tables ---

  function moveText(entry) {
    if (entry.previous_rank === null || entry.previous_rank === undefined) return { text: 'NEW', cls: 'new' };
    const d = entry.previous_rank - entry.rank;
    if (d > 0) return { text: '▲ ' + d, cls: '' };
    if (d < 0) return { text: '▼ ' + (-d), cls: '' };
    return { text: '–', cls: '' };
  }

  function div(cls, text) {
    const d = document.createElement('div');
    d.className = cls;
    if (text !== undefined) d.textContent = text;
    return d;
  }

  function renderTable(kind, snap, plate) {
    const rows = $('rows-' + kind);
    rows.replaceChildren();
    if (!snap || snap.entries.length === 0) {
      rows.appendChild(div('empty', EMPTY[kind]));
      return;
    }
    const selId = plate ? plate.bundle.moment.moment_id : null;
    for (const e of snap.entries) {
      const m = idx.moments.get(e.moment_id).bundle.moment;
      const row = div('row' + (e.moment_id === selId ? ' sel' : ''));
      row.appendChild(div('rank', String(e.rank)));
      const who = div('who');
      who.appendChild(div('player', playerName(m)));
      who.appendChild(div('meta', teamCode(m.team_id, m.team_name) + ' · ' + m.minute_label + "'"));
      row.appendChild(who);
      const track = div('track');
      const bar = div('bar ' + (m.delta < 0 ? 'neg' : 'pos'));
      bar.style.width = (100 * Math.abs(m.delta)) + '%';   // |PVA| on a 0..1 track
      track.appendChild(bar);
      row.appendChild(track);
      row.appendChild(div('value', pva(m.delta)));
      const mv = moveText(e);
      row.appendChild(div('move' + (mv.cls ? ' ' + mv.cls : ''), mv.text));
      rows.appendChild(row);
    }
  }

  // --- lower third ---

  function renderPlate(entry) {
    const lower = $('lower');
    lower.classList.remove('completed', 'failed', 'waiting');
    const line = $('lower-line');
    if (!entry) {
      lower.classList.add('waiting');
      $('lower-name').textContent = 'No commentated moment yet';
      $('lower-meta').textContent = '';
      line.textContent = '';
      $('lower-xp').textContent = '–';
      return;
    }
    const m = entry.bundle.moment;
    lower.classList.add(m.completed ? 'completed' : 'failed');
    $('lower-name').textContent = playerName(m);
    $('lower-meta').textContent = teamCode(m.team_id, m.team_name) + ' · ' + m.minute_label + "' · " +
      entry.tier.en + ' pass, ' + (m.completed ? 'completed' : 'not completed') + ' · xP ' + percent(m.xp) + '%';
    const text = entry.line ? entry.line['line_' + state.lang] : '';
    line.textContent = text || '';
    line.setAttribute('lang', state.lang);
    $('lower-xp').textContent = pva(m.delta);
  }

  // --- stage scaling -------------------------------------------------------------------------

  function fit() {
    const s = Math.min(window.innerWidth / 1440, window.innerHeight / 900);
    $('stage').style.setProperty('--scale', String(s));
  }
  window.addEventListener('resize', fit);
  fit();

  load();
})();
