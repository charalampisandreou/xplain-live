// xPlain Live: Studio screen. Plays a match bundle (src/xplain/bundle.py) on the match clock.
// Every number shown comes from the bundle; the browser only formats xP as a percent with the same
// rule as analytics.percent, round(100 * x), PVA in points the same way, and shows rank moves from
// rank and previous_rank. Rankings come ready-made per scope (all / 1 / 2) from the bundle.
(function () {
  'use strict';

  const SVG = 'http://www.w3.org/2000/svg';
  const KINDS = ['surprise_failure', 'surprise_success'];
  const EMPTY = {
    surprise_failure: 'No surprise failures yet.',
    surprise_success: 'No surprise successes yet.',
  };
  const NOT_STARTED = 'Starts in the 2nd half';
  const HALF_LABEL = { 1: '1st half', 2: '2nd half' };
  const SCOPES = ['all', '1', '2'];
  const FADED = [0.8, 0.55, 0.35];   // opacity of the 1st, 2nd and 3rd pass before the selected one
  const FADE_MS = 150;               // the one cross-fade for every plate and tab change
  const GOAL_MIN_MS = 4000;          // the goal plate holds at least this long in real time, at any speed
  const PULSE_MS = 830;              // one ring pulse; three of them round the score plate
  const SOFT_PULSE_EVERY = 4000;     // then one softer pulse this often while the goal plate holds
  const LANGS = ['en', 'es', 'el'];
  const LINE_PX = 18, MIN_LINE_PX = 15, TWO_LINES_PX = 48;   // the line's font size, its floor, 2 x 24px
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  const $ = (id) => document.getElementById(id);
  const percent = (x) => Math.round(100 * x);
  // PVA (the bundle's delta = outcome - xP) in points of completion chance: sign, round(100 * |x|), "%"
  const pva = (x) => (x < 0 ? '−' : '+') + Math.round(Math.abs(x) * 100) + '%';
  const code = (name) => (name || '').slice(0, 3).toUpperCase();

  const state = {
    bundle: null,
    now: 0,            // match t, seconds
    playing: true,
    speed: 12,
    follow: true,      // live: the selected pass is the newest pass with t <= now
    selected: -1,      // index into marks when not following
    lang: 'en',
    scope: 'all',      // which ranking the tables show: whole match so far, or one half
    replay: null,      // moment_id being replayed: pitch, plate and tab show it; clock and tables run on
    mode: 'studio',    // 'studio' or 'player' (player.js: one player's match); both share the state above
    player: null,      // Player mode's player_id
    tableSig: {},      // per table, what it last drew (rows are rebuilt only when this changes)
    lastFrame: null,
    renderedKey: '',
    plateId: null,
    goal: null,        // while the plate shows a goal: { score index, restartT, minUntil, nextPulse }
    fit: new Map(),    // "moment_id/lang" -> a smaller font size for a line that needs it (fitLines)
    fades: new Map(),  // element -> { anim, at }: its running cross-fade and when it started
  };
  let idx = null;      // lookups built once per bundle

  // --- loading -------------------------------------------------------------------------------

  function matchName() {
    const name = new URLSearchParams(location.search).get('match') || 'seed3001';
    return /^[a-z0-9_]+$/.test(name) ? name : 'seed3001';
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
    if (!bundle.meta || (bundle.meta.schema_version !== '4' && bundle.meta.schema_version !== '5')) {
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
      markOf: new Map(bundle.marks.map((m, i) => [m.moment_id, i]).filter(([id]) => id)),
    };
    // narrated moments (commentated, with a stored line), for the plate's prev / next control
    idx.narr = idx.live.filter((m) => m.line);
    idx.narrT = idx.narr.map((m) => m.bundle.moment.t);
    idx.narrSeq = idx.narr.map((m) => m.bundle.moment.seq);
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
    const params = new URLSearchParams(location.search);
    if (LANGS.includes(params.get('lang'))) setLang(params.get('lang'));
    if (SCOPES.includes(params.get('scope'))) setScope(params.get('scope'));
    // ?replay=<moment_id> opens in replay; an unknown or uncommentated id is ignored. A moment later
    // than the start time moves the clock to it, so the tables never trail the replayed moment.
    const r = idx.moments.get(params.get('replay'));
    if (r && r.bundle.qualifies_live && idx.markOf.has(r.bundle.moment.moment_id)) {
      state.now = Math.max(state.now, r.bundle.moment.t);
      enterReplay(r.bundle.moment.moment_id);
    }
    // ?mode=player opens Player mode, with ?player=<player_id> when it names one of the 22
    XplainPlayer.mount({ teamName: (id) => idx.teams[id] || '', percent, pva, enterReplay });
    if (bundle.meta.players.some((p) => p.player_id === params.get('player'))) state.player = params.get('player');
    if (params.get('mode') === 'player') setMode('player');
    requestAnimationFrame(frame);
    fitLines();
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
      const before = lastAtOrBefore(idx.scoreT, state.now);
      state.now = Math.min(end, state.now + ((ts - state.lastFrame) / 1000) * state.speed);
      if (state.now >= end) setPlaying(false);
      // a goal flashes only when live playback crosses it: never on ?t, replay, step mode or language
      const after = lastAtOrBefore(idx.scoreT, state.now);
      if (after > before && idx.score[after].goal) goalFlash(after, ts);
    }
    if (state.goal) {   // held until play restarts after the goal, and at least GOAL_MIN_MS
      if (state.now >= state.goal.restartT && ts >= state.goal.minUntil) state.goal = null;
      else if (ts >= state.goal.nextPulse) { pulse(1, 0.45); state.goal.nextPulse += SOFT_PULSE_EVERY; }
    }
    state.lastFrame = ts;
    render();
    requestAnimationFrame(frame);
  }

  function isLive() { return state.follow && !state.replay; }

  function liveIndex() { return lastAtOrBefore(idx.markT, state.now); }
  function selectedIndex() {
    if (state.replay) return idx.markOf.get(state.replay);
    return state.follow ? liveIndex() : Math.min(state.selected, liveIndex());
  }

  // --- controls ------------------------------------------------------------------------------

  function setPlaying(on) {
    if (on && state.now >= state.bundle.clock.last_event_t) {
      state.now = state.bundle.clock.first_event_t;   // play again from kick-off
      state.goal = null;
      exitReplay();
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

  function setScope(scope) {
    state.scope = scope;
    for (const b of $('scope').querySelectorAll('button')) {
      const on = b.dataset.scope === scope;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    }
  }

  // Studio or Player. Entering Player mode without a player picks one (defaultPlayer). The URL keeps
  // ?mode and ?player for demo links, the way it drops ?replay on leaving replay.
  function setMode(mode) {
    state.mode = mode;
    if (mode === 'player' && !state.player) state.player = defaultPlayer();
    $('stage').classList.toggle('mode-player', mode === 'player');
    for (const b of $('mode').querySelectorAll('button[data-mode]')) {
      const on = b.dataset.mode === mode;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    }
    const url = new URL(location.href);
    if (mode === 'player') {
      url.searchParams.set('mode', 'player');
      if (state.player) url.searchParams.set('player', state.player);
    } else {
      url.searchParams.delete('mode');
      url.searchParams.delete('player');
    }
    history.replaceState(null, '', url);
  }

  // the player of the replayed moment, else of the plate's moment, else of the live pass, else the first listed
  function defaultPlayer() {
    const sel = selectedIndex();
    const plate = plateMoment(sel);
    const from = state.replay ? idx.moments.get(state.replay).bundle.moment
      : plate ? plate.bundle.moment : sel >= 0 ? idx.marks[sel] : null;
    return (from && from.player_id) || state.bundle.meta.players[0].player_id;
  }

  function step(dir) {
    const live = liveIndex();
    if (live < 0) return;
    const from = selectedIndex();
    exitReplay();   // stepping from a replayed pass leaves replay and steps on from it
    state.selected = Math.max(0, Math.min(live, from + dir));
    state.follow = false;
    state.goal = null;
  }

  // Narrated moments the live clock has reached, and where the selected pass sits among them:
  // pos = how many are at or before it (the plate shows narr[pos - 1]), 0 when none yet.
  function narration(sel) {
    const reached = lastAtOrBefore(idx.narrT, state.now) + 1;
    const pos = sel < 0 ? 0 : Math.min(reached, lastAtOrBefore(idx.narrSeq, idx.marks[sel].seq) + 1);
    return {
      reached,
      pos,
      prev: pos >= 2 ? idx.narr[pos - 2] : null,
      next: pos < reached ? idx.narr[pos] : null,   // never past the latest reached: no spoilers
    };
  }

  // chevrons and [ ]: replay the previous / next narrated moment (the match keeps running)
  function narrStep(dir) {
    const n = narration(selectedIndex());
    const target = dir < 0 ? n.prev : n.next;
    if (target) enterReplay(target.bundle.moment.moment_id);
  }

  function backToLive() { exitReplay(); state.follow = true; }

  // --- replay: a table row puts its moment on the pitch, the plate and the tab ---

  function enterReplay(id) {
    if (id === state.replay) return;
    state.replay = id;
    state.goal = null;
  }

  function exitReplay() {
    if (!state.replay) return;
    state.replay = null;
    const url = new URL(location.href);
    if (url.searchParams.has('replay')) {
      url.searchParams.delete('replay');
      history.replaceState(null, '', url);
    }
  }

  function leaveReplay() { exitReplay(); state.follow = true; }   // back to the live pass

  function wire() {
    $('play').addEventListener('click', () => setPlaying(!state.playing));
    for (const b of $('speed').querySelectorAll('button')) {
      b.addEventListener('click', () => setSpeed(Number(b.dataset.speed)));
    }
    for (const b of $('lang').querySelectorAll('button')) {
      b.addEventListener('click', () => setLang(b.dataset.lang));
    }
    for (const b of $('scope').querySelectorAll('button')) {
      b.addEventListener('click', () => setScope(b.dataset.scope));
    }
    for (const b of $('mode').querySelectorAll('button[data-mode]')) {
      b.addEventListener('click', () => setMode(b.dataset.mode));
    }
    $('prev').addEventListener('click', () => step(-1));
    $('next').addEventListener('click', () => step(1));
    $('live').addEventListener('click', backToLive);
    $('narr-prev').addEventListener('click', () => narrStep(-1));
    $('narr-next').addEventListener('click', () => narrStep(1));
    for (const kind of KINDS) {
      const rows = $('rows-' + kind);
      rows.addEventListener('click', (e) => {
        const row = e.target.closest('.row[data-moment]');
        if (row) enterReplay(row.dataset.moment);
      });
      rows.addEventListener('keydown', (e) => {
        const row = e.target.closest('.row[data-moment]');
        if (row && (e.key === 'Enter' || e.key === ' ')) { enterReplay(row.dataset.moment); e.preventDefault(); }
      });
    }
    document.addEventListener('keydown', (e) => {
      const control = e.target.closest && e.target.closest('button, [role="button"]');
      if (control && (e.key === ' ' || e.key === 'Enter')) return;
      if (e.key === 'Escape') {   // leaves replay and step mode
        if (state.replay || !state.follow) { leaveReplay(); e.preventDefault(); }
      }
      else if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
      else if (e.key === 'ArrowRight') { step(1); e.preventDefault(); }
      else if (e.key === '[') { narrStep(-1); e.preventDefault(); }
      else if (e.key === ']') { narrStep(1); e.preventDefault(); }
      else if (e.key === ' ') { setPlaying(!state.playing); e.preventDefault(); }
      else if (e.key === 'l' || e.key === 'L') backToLive();
    });
    setPlaying(state.now < state.bundle.clock.last_event_t);   // ?t at full time opens paused, not restarted
    setSpeed(state.speed);
  }

  // --- render --------------------------------------------------------------------------------

  function render() {
    const sel = selectedIndex();
    const live = liveIndex();
    const tick = lastAtOrBefore(idx.tickT, state.now);
    const score = lastAtOrBefore(idx.scoreT, state.now);
    const half = live >= 0 ? idx.marks[live].period : 1;
    const snaps = KINDS.map((k) => latestSnapshot(k, state.scope));
    const plate = plateMoment(sel);
    const narr = narration(sel);
    const goal = state.goal && isLive() ? state.goal.score : -1;
    const key = [sel, live, tick, score, half, state.scope, snaps.map((s) => (s ? s.t : '-')).join('|'),
      plate ? plate.bundle.moment.moment_id : '-', state.lang, state.follow, state.replay,
      narr.reached, narr.pos, goal, state.mode, state.player].join('/');
    if (key === state.renderedKey) return;   // every DOM write below runs in this one animation frame
    state.renderedKey = key;

    $('live').classList.toggle('is-live', isLive());
    $('prev').disabled = sel <= 0;
    $('next').disabled = isLive() || sel >= live;
    renderBadge(sel);
    renderScoreBug(tick, score, live >= 0 ? half : null);
    renderPossession(sel);
    renderPitch(sel);
    const notStarted = state.scope === '2' && half !== 2;
    KINDS.forEach((k, i) => renderTable(k, snaps[i], plate, notStarted));
    renderNarrNav(narr);
    if (goal >= 0) renderGoalPlate(goal); else renderPlate(plate);
    if (state.mode === 'player') {
      XplainPlayer.render({ bundle: state.bundle, now: state.now, scope: state.scope, playerId: state.player,
        selected: plate ? plate.bundle.moment.moment_id : null, replay: state.replay, notStarted });
    }
  }

  // --- transitions ---------------------------------------------------------------------------

  // The one cross-fade (opacity, FADE_MS) for a content change. Latest wins: a change that comes
  // while a fade is still running cancels it and shows the newest content at once, so fast
  // stepping never queues fades, shows an older line or blinks the plate empty.
  // A change within FADE_MS of the previous one (holding ] or an arrow) is a plain swap.
  function crossFade(node) {
    const t = performance.now();
    const last = state.fades.get(node);
    state.fades.set(node, { anim: null, at: t });
    if (last && last.anim) last.anim.cancel();
    if ((last && t - last.at < FADE_MS) || reducedMotion.matches || !node.animate) return;
    state.fades.get(node).anim = node.animate([{ opacity: 0 }, { opacity: 1 }], { duration: FADE_MS, easing: 'ease-out' });
  }

  // The plate's content fades; its body and the narration language switch stay still.
  function fadePlate() {
    for (const node of [$('nametab'), $('lower-line'), $('lower-facts'), $('lower').querySelector('.body > .right')]) {
      crossFade(node);
    }
  }

  // --- goal ---

  // The score bug always shows the live match, so it pulses on every goal crossed in playback (three
  // rings, the scoring team's colour when the bundle has one, else the accent). The plate shows the
  // goal only when live: until play restarts (the first pass after the goal; none: until full time),
  // and at least GOAL_MIN_MS, with a softer pulse every SOFT_PULSE_EVERY meanwhile.
  function goalFlash(i, ts) {
    const ring = $('goal-ring');
    const team = [state.bundle.meta.home, state.bundle.meta.away].find((t) => t.team_id === idx.score[i].goal.team_id);
    const colour = team && (team.colour || team.color);
    if (colour) ring.style.setProperty('--ring', colour); else ring.style.removeProperty('--ring');
    pulse(3, 0.9);
    if (!isLive()) return;
    const next = lastAtOrBefore(idx.markT, idx.score[i].t) + 1;
    state.goal = {
      score: i,
      restartT: next < idx.markT.length ? idx.markT[next] : state.bundle.clock.last_event_t,
      minUntil: ts + GOAL_MIN_MS,
      nextPulse: ts + SOFT_PULSE_EVERY,
    };
  }

  // ring pulses round the score plate (opacity and scale only); none with reduced motion
  function pulse(times, peak) {
    const ring = $('goal-ring');
    if (reducedMotion.matches || !ring.animate) return;
    ring.getAnimations().forEach((a) => a.cancel());
    ring.animate([
      { opacity: 0, transform: 'scale(1)' },
      { opacity: peak, transform: 'scale(1.02)', offset: 0.35 },
      { opacity: 0, transform: 'scale(1.05)' },
    ], { duration: PULSE_MS, iterations: times, easing: 'ease-out' });
  }

  // "GOAL · DSK · 90+4'": the minute is the score track's minute_label, never computed here
  function goalTabText(s) {
    return 'Goal · ' + teamCode(s.goal.team_id, '') + ' · ' + s.minute_label + "'";
  }

  function renderGoalPlate(i) {
    const id = 'goal/' + i;
    if (id === state.plateId) return;
    state.plateId = id;
    const s = idx.score[i];
    const lower = $('lower');
    lower.classList.remove('completed', 'failed', 'waiting');
    lower.classList.add('goal');
    $('lower-name').textContent = goalTabText(s);
    $('lower-when').textContent = '';
    const line = $('lower-line');
    line.textContent = s.goal.player_name || idx.teams[s.goal.team_id] || '';
    line.removeAttribute('lang');
    line.style.fontSize = '';
    $('lower-meta').textContent = idx.teams[s.goal.team_id] || '';
    $('lower-xp').hidden = true;
    $('lower-pva').textContent = 'GOAL';   // the PVA column: "GOAL" over the score track's minute_label
    $('lower-pva-label').textContent = s.minute_label + "'";
    fadePlate();
  }

  // --- narration nav: "7 / 12" between two chevrons; hidden until a narrated moment is reached ---

  function renderNarrNav(n) {
    $('narrnav').hidden = n.reached === 0;
    $('narr-count').textContent = (n.pos > 0 ? n.pos : '–') + ' / ' + n.reached;
    $('narr-prev').disabled = !n.prev;
    $('narr-next').disabled = !n.next;
  }

  // "REPLAY · 41'": shown whenever the view is not live (a replayed moment, or step mode)
  function renderBadge(sel) {
    const off = state.replay || !state.follow;
    $('replaybar').classList.toggle('on', Boolean(off) && sel >= 0);
    $('replay-badge').textContent = off && sel >= 0 ? 'Replay · ' + idx.marks[sel].minute_label + "'" : '';
  }

  function renderScoreBug(tick, score, half) {
    const s = score >= 0 ? idx.score[score] : null;
    $('score').textContent = s ? s.home + '–' + s.away : '–';
    $('minute').textContent = tick >= 0 ? idx.ticks[tick].minute_label + "'" : '';
    $('half').textContent = half ? HALF_LABEL[half] : '';
  }

  // Possession tab: the team of the selected pass, on the side of the pitch it attacks from, with
  // chevrons toward the goal it attacks. Who attacks right in each half is meta.attacks_right.
  function renderPossession(sel) {
    const box = $('possession');
    if (sel < 0) { box.hidden = true; return; }
    const mark = idx.marks[sel];
    const right = state.bundle.meta.attacks_right[String(mark.period)] === mark.team_id;
    const teamCodeText = teamCode(mark.team_id, mark.team_name);
    $('tab-code').textContent = teamCodeText;
    const label = teamCodeText + ' in possession, attacking ' + (right ? 'right' : 'left');
    if (label !== box.getAttribute('aria-label')) crossFade($('tab'));   // the same cross-fade as the plate
    box.classList.toggle('left', !right);
    box.setAttribute('aria-label', label);
    box.hidden = false;
  }

  // the latest snapshot of a kind in a scope with t <= now, or null
  function latestSnapshot(kind, scope) {
    for (let i = lastAtOrBefore(idx.rankT, state.now); i >= 0; i--) {
      const s = idx.rankings[i];
      if (s.scope === scope && s.kind === kind) return s;
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

  function renderTable(kind, snap, plate, notStarted) {
    const rows = $('rows-' + kind);
    const selIdNow = plate ? plate.bundle.moment.moment_id : null;
    const sig = [notStarted, snap ? snap.scope + snap.t : '-', selIdNow, state.replay].join('/');
    if (state.tableSig[kind] === sig) return;   // unchanged: keep the rows (and keyboard focus)
    state.tableSig[kind] = sig;
    const focused = document.activeElement && rows.contains(document.activeElement)
      ? document.activeElement.dataset.moment : null;
    rows.replaceChildren();
    if (notStarted) {
      rows.appendChild(div('empty', NOT_STARTED));
      return;
    }
    if (!snap || snap.entries.length === 0) {
      rows.appendChild(div('empty', EMPTY[kind]));
      return;
    }
    const selId = plate ? plate.bundle.moment.moment_id : null;
    for (const e of snap.entries) {
      const m = idx.moments.get(e.moment_id).bundle.moment;
      const row = div('row' + (e.moment_id === selId ? ' sel' : ''));
      row.dataset.moment = e.moment_id;
      row.setAttribute('role', 'button');
      row.tabIndex = 0;
      row.setAttribute('aria-pressed', String(e.moment_id === state.replay));
      row.setAttribute('aria-label', 'Replay ' + playerName(m) + ', ' + teamCode(m.team_id, m.team_name) + ', ' +
        m.minute_label + "', PVA " + pva(m.delta));
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
      if (e.moment_id === focused) row.focus();
    }
  }

  // --- lower third ---

  // Text is set in place on the existing nodes; nothing is written when the moment and language are
  // unchanged, so stepping through passes that keep the same moment never re-animates the plate.
  function renderPlate(entry) {
    const id = entry ? entry.bundle.moment.moment_id + '/' + state.lang : '-';
    if (id === state.plateId) return;
    state.plateId = id;
    const lower = $('lower');
    lower.classList.remove('completed', 'failed', 'waiting', 'goal');
    const line = $('lower-line');
    fadePlate();
    if (!entry) {   // before the first commentated moment: the body only
      lower.classList.add('waiting');
      line.textContent = 'No commentated moment yet';
      line.removeAttribute('lang');
      line.style.fontSize = '';
      $('lower-meta').textContent = '';
      $('lower-xp').hidden = true;
      $('lower-pva').textContent = '–';   // the PVA column: a dash over "PVA"
      $('lower-pva-label').textContent = 'PVA';
      return;
    }
    const m = entry.bundle.moment;
    lower.classList.add(m.completed ? 'completed' : 'failed');
    $('lower-name').textContent = playerName(m);
    $('lower-when').textContent = teamCode(m.team_id, m.team_name) + ' · ' + m.minute_label + "'";
    $('lower-meta').textContent = entry.tier.en + ' pass, ' + (m.completed ? 'completed' : 'not completed');
    $('lower-xp-val').textContent = percent(m.xp) + '%';   // " · xP 92%", xP in its own span
    $('lower-xp').hidden = false;
    // no stored line: the line row is empty and hidden (:empty), so no gap is left
    line.textContent = (entry.line && entry.line['line_' + state.lang]) || '';
    line.setAttribute('lang', state.lang);
    line.style.fontSize = state.fit.get(id) || '';
    $('lower-pva').textContent = pva(m.delta);
    $('lower-pva-label').textContent = 'PVA';
  }

  // Once at load, after the fonts: the font size each narrated line needs to fit in two lines of the
  // fixed plate (LINE_PX, down in 1px steps to MIN_LINE_PX). Only lines that need less than LINE_PX
  // are stored; the plate then sets the size with no layout read. The count of shrunk lines is left
  // on <html data-line-fit="shrunk/total"> for checks.
  async function fitLines() {
    try {
      await Promise.all(['500 18px Archivo', '500 18px "Noto Sans"'].map((f) => document.fonts.load(f, 'Aa Αα Ññ')));
      await document.fonts.ready;
    } catch (e) { /* fonts unavailable: measure with the fallback */ }
    const probe = div('line measure');
    $('lower').querySelector('.left').appendChild(probe);
    let shrunk = 0, total = 0;
    for (const m of idx.narr) {
      for (const lang of LANGS) {
        const text = m.line['line_' + lang];
        if (!text) continue;
        total++;
        probe.textContent = text;
        probe.setAttribute('lang', lang);
        let size = LINE_PX;
        probe.style.fontSize = size + 'px';
        while (size > MIN_LINE_PX && probe.offsetHeight > TWO_LINES_PX) probe.style.fontSize = --size + 'px';
        if (size < LINE_PX) { state.fit.set(m.bundle.moment.moment_id + '/' + lang, size + 'px'); shrunk++; }
      }
    }
    probe.remove();
    document.documentElement.dataset.lineFit = shrunk + '/' + total;
    if (state.fit.has(state.plateId)) $('lower-line').style.fontSize = state.fit.get(state.plateId);
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
