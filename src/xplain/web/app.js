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
  const NOT_STARTED = 'Starts in the 2nd half.';
  const NOT_COMMENTATED = "Not commentated: when it happened, it was not among the half's top 3 surprises of its kind.";
  // recaps: the bundle's scope names, their ?recap values, and the plate's words in the recap state
  const RECAP_KEYS = { ht: 'half_time', ft: 'full_time' };
  const RECAP_TAB = { half_time: 'Half-time recap', full_time: 'Full-time recap' };
  const RECAP_LINE = { half_time: 'Press Play to continue.', full_time: 'Full time.' };
  const RECAP_COL = { half_time: 'HT', full_time: 'FT' };
  const RECAP_PAUSE_MS = 10000;      // the half-time recap holds this long in real time, then play resumes
  const RECAP_TEXT_PX = 19, RECAP_MIN_PX = 16;   // the recap text's size, and the least fitRecapText goes to
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
    mode: 'studio',    // 'studio', 'player' (player.js: one player's match), 'producer' (producer.js: the control
                       // room) or 'analyst' (analyst.js: both teams against the model); all share the state above
    analystPick: null, // Analyst mode: the surprise picked on the swing, null for the latest one
    producerRecap: null,   // Producer mode: the recap scope picked in the log (Producer only; Studio is unchanged)
    player: null,      // Player mode's player_id
    playerNarr: null,  // Player mode: the moment_id the plate's chevrons picked, null for the player's latest
    tableSig: {},      // per table, what it last drew (rows are rebuilt only when this changes)
    lastFrame: null,
    renderedKey: '',
    plateId: null,
    goal: null,        // while the plate shows a goal: { score index, restartT, minUntil, nextPulse }
    recap: null,       // while a recap is open: { scope, resumeAt, view } (resumeAt null: no auto-resume;
                       // view 'recap': the card in the pitch's slot, 'map': the pitch and the narration)
    playerPicked: false,   // a player was picked since the recap opened (else Player opens on its star)
    pendingRecap: null,   // a recap crossed while the goal plate still holds: shown when it ends
    unseenRecap: null,    // a recap crossed by live playback in Producer or Player: reached, not yet seen
                          // (showUnseenRecap on switching to Studio)
    fit: new Map(),    // "moment_id/lang" -> a smaller font size for a line that needs it (fitLines)
    aired: new Map(),  // moment_id -> a Live retake put on air in Producer ({ line: {en, es, el}, steps, summary}):
                       // the plate shows it for that moment in every tab, in this browser tab only (nothing is stored)
    airedRev: 0,
    fades: new Map(),  // element -> { anim, at }: its running cross-fade and when it started
  };
  let idx = null;      // lookups built once per bundle
  let drain = null;    // the half-time drain animation on the plate (stopDrain cancels it)

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
      playerOf: new Map(bundle.meta.players.map((p) => [p.player_id, p])),
    };
    // narrated moments (commentated, with a stored line), for the plate's prev / next control
    idx.narr = idx.live.filter((m) => m.line);
    idx.narrT = idx.narr.map((m) => m.bundle.moment.t);
    idx.narrSeq = idx.narr.map((m) => m.bundle.moment.seq);
    // where playback opens each recap: the last first-half pass (the bug still reads the first half's
    // last minute) and the last event; only scopes the bundle has a recap card for
    const h1 = bundle.marks.filter((m) => m.period === 1);
    idx.h1End = h1.length ? h1[h1.length - 1].t : null;   // from here to the second half's kick-off the bug reads HT
    idx.recapT = {};
    if (bundle.recaps && bundle.recaps.half_time && h1.length) idx.recapT.half_time = h1[h1.length - 1].t;
    if (bundle.recaps && bundle.recaps.full_time) idx.recapT.full_time = bundle.clock.last_event_t;
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
    XplainPlayer.mount({ teamName: (id) => idx.teams[id] || '', teamCode: (id) => teamCode(id, ''), percent, pva,
      bundle, current: () => ({ bundle, now: state.now, player: state.player }),
      setPlayer: (id) => { state.player = id; state.playerPicked = true; setMode('player'); } });
    if (bundle.meta.players.some((p) => p.player_id === params.get('player'))) state.player = params.get('player');
    // Producer mode: a log row replays its moment, as a Studio table row does; a recap row is Producer's own pick
    XplainProducer.mount({
      selectMoment: (id) => { if (idx.markOf.has(id)) enterReplay(id); },
      selectRecap: (scope) => { state.producerRecap = scope; },
      teamCode: (id) => teamCode(id, ''),
      aired: (id) => state.aired.get(id) || null,
      air: (id, aired) => {   // a retake on air (fitted like the stored lines), or null: back to the stored line
        if (aired) { state.aired.set(id, aired); fitAired(id, aired.line); } else state.aired.delete(id);
        state.airedRev++;
        state.plateId = null;
      },
      redraw: () => { state.renderedKey = ''; render(); } });
    // Analyst mode: a dot picks its surprise; Replay in Studio replays a commentated one there
    XplainAnalyst.mount({ fixed2: XplainPlayer.fixed2, percent, pva,
      pick: (id) => { state.analystPick = id; },
      replay: (id) => { if (idx.markOf.has(id)) { setMode('studio'); enterReplay(id); } } });
    if (params.get('mode') === 'player') setMode('player');
    else if (params.get('mode') === 'producer') setMode('producer');
    else if (params.get('mode') === 'analyst') {
      // ?moment=<moment_id> picks that surprise (demo links); one the clock has not reached waits until it is
      if (idx.moments.has(params.get('moment'))) state.analystPick = params.get('moment');
      setMode('analyst');
    }
    // ?recap=ht|ft opens that recap at its time, paused, with no auto-resume (demo links, screenshots)
    const rs = RECAP_KEYS[params.get('recap')];
    if (rs && idx.recapT[rs] !== undefined && state.mode === 'studio') {
      state.now = idx.recapT[rs];
      exitReplay();
      state.follow = true;
      setPlaying(false);
      openRecap(rs, null);
    }
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
      const prevNow = state.now;
      state.now = Math.min(end, state.now + ((ts - state.lastFrame) / 1000) * state.speed);
      if (state.now >= end) setPlaying(false);
      // a goal flashes only when live playback crosses it: never on ?t, replay, step mode or language
      const after = lastAtOrBefore(idx.scoreT, state.now);
      if (after > before && idx.score[after].goal) goalFlash(after, ts);
      // a recap opens the same way: only when live playback crosses its time. In Studio it opens (paused); in
      // Producer or Player nothing pauses or shows, it is marked reached-unseen (showUnseenRecap)
      for (const scope of ['half_time', 'full_time']) {
        const at = idx.recapT[scope];
        if (at === undefined || !(prevNow < at && state.now >= at) || !isLive()) continue;
        if (state.mode !== 'studio') {
          if (state.recap && state.recap.scope !== scope) closeRecap();   // a newer recap supersedes an open one
          state.unseenRecap = scope;
          continue;
        }
        state.now = at;
        setPlaying(false);
        state.pendingRecap = scope;
      }
    }
    // after a goal crossed just before it, the recap waits until the goal plate has had its GOAL_MIN_MS
    if (state.pendingRecap && (!state.goal || ts >= state.goal.minUntil)) {
      const scope = state.pendingRecap;
      state.pendingRecap = null;
      openRecap(scope, scope === 'half_time' ? ts + RECAP_PAUSE_MS : null);
    }
    if (state.recap && state.recap.resumeAt !== null && ts >= state.recap.resumeAt) setPlaying(true);
    if (state.goal) {   // held until play restarts after the goal, and at least GOAL_MIN_MS
      if (state.now >= state.goal.restartT && ts >= state.goal.minUntil) state.goal = null;
      else if (ts >= state.goal.nextPulse) { pulse(1, 0.45); state.goal.nextPulse += SOFT_PULSE_EVERY; }
    }
    state.lastFrame = ts;
    render();
    requestAnimationFrame(frame);
  }

  function isLive() { return state.follow && !state.replay; }

  // A recap crossed live in Producer or Player, on switching to Studio: while live, full time opens; half time
  // opens only if no pass after its time has been played yet; otherwise it is dropped. Opening marks it seen (it
  // opens through pendingRecap, paused, half time resuming by itself as when crossed in Studio).
  function showUnseenRecap() {
    const scope = state.unseenRecap;
    state.unseenRecap = null;
    if (!scope || !isLive() || state.recap) return;
    const live = liveIndex();
    if (scope === 'half_time' && live >= 0 && idx.markT[live] > idx.recapT.half_time) return;
    setPlaying(false);
    state.pendingRecap = scope;
  }

  function liveIndex() { return lastAtOrBefore(idx.markT, state.now); }
  function selectedIndex() {
    if (state.replay) return idx.markOf.get(state.replay);
    return state.follow ? liveIndex() : Math.min(state.selected, liveIndex());
  }

  // --- controls ------------------------------------------------------------------------------

  function setPlaying(on) {
    if (on) { closeRecap(); state.pendingRecap = null; }   // Play leaves a recap (at full time it restarts below)
    if (on && state.now >= state.bundle.clock.last_event_t) {
      state.now = state.bundle.clock.first_event_t;   // play again from kick-off
      state.goal = null;
      state.unseenRecap = null;
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
    // Player and Producer show no recap, but an open one stays open: back in Studio it is there again. Leaving
    // Studio turns the half-time auto-resume off (as Map does), else play resumed unseen and closed the recap,
    // taking the Recap / Map switch with it. From a recap, Player opens on its star until a player is picked.
    if (mode !== 'studio' && state.recap) state.recap.resumeAt = null;
    if (mode === 'player' && state.recap && !state.playerPicked) {
      const star = state.bundle.recaps[state.recap.scope].card.star;
      if (star) state.player = star.player_id;
    }
    if (mode === 'player' && !state.player) state.player = defaultPlayer();
    state.playerNarr = null;
    XplainPlayer.invalidate();   // the pitch and its tab are drawn again by whichever mode is shown
    XplainProducer.invalidate();
    XplainAnalyst.invalidate();
    $('stage').classList.toggle('mode-player', mode === 'player');
    $('stage').classList.toggle('mode-producer', mode === 'producer');
    $('stage').classList.toggle('mode-analyst', mode === 'analyst');
    for (const b of $('mode').querySelectorAll('button[data-mode]')) {
      const on = b.dataset.mode === mode;
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', String(on));
    }
    const url = new URL(location.href);
    if (mode === 'player') {
      url.searchParams.set('mode', 'player');
      if (state.player) url.searchParams.set('player', state.player);
    } else if (mode === 'producer' || mode === 'analyst') {
      url.searchParams.set('mode', mode);
      url.searchParams.delete('player');
    } else {
      url.searchParams.delete('mode');
      url.searchParams.delete('player');
    }
    if (mode !== 'analyst') url.searchParams.delete('moment');
    history.replaceState(null, '', url);
    if (mode === 'studio') showUnseenRecap();
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
    recapToMap();   // a recap stays open: the map shows the pass, the Recap button brings the card back
    exitReplay();   // stepping from a replayed pass leaves replay and steps on from it
    state.producerRecap = null;
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
    if (state.mode === 'analyst') {   // Analyst mode: the plate pages through the surprises; the newest follows live
      const n = analystNarration();
      const target = dir < 0 ? n.prev : n.next;
      if (target) state.analystPick = target === n.list[n.list.length - 1] ? null : target.bundle.moment.moment_id;
      return;
    }
    if (state.mode === 'player') {   // Player mode: the plate pages through the player's lines; the newest follows live
      const n = playerNarration();
      const target = dir < 0 ? n.prev : n.next;
      if (target) state.playerNarr = target === n.list[n.list.length - 1] ? null : target.bundle.moment.moment_id;
      return;
    }
    const r = state.recap;
    if (r && r.view === 'recap') {   // the recap sits after the latest narrated moment: ‹ goes to that moment
      if (dir < 0 && narration(liveIndex()).reached > 0) { recapToMap(); backToLive(); }
      return;
    }
    const n = narration(selectedIndex());
    const target = dir < 0 ? n.prev : n.next;
    if (target) enterReplay(target.bundle.moment.moment_id);
    else if (dir > 0 && r) setRecapView('recap');   // › past the latest narrated moment: back to the recap
  }

  // Player mode's plate: the player's narrated moments with t <= now (no spoilers), the latest unless
  // the chevrons picked an earlier one; the same shape as narration() for renderNarrNav.
  function playerNarration() {
    const list = XplainPlayer.narratedFor(state.bundle.moments, state.player, state.now, null);
    const picked = state.playerNarr ? list.findIndex((m) => m.bundle.moment.moment_id === state.playerNarr) : -1;
    const pos = picked >= 0 ? picked + 1 : list.length;
    return { list, reached: list.length, pos, prev: pos >= 2 ? list[pos - 2] : null, next: pos < list.length ? list[pos] : null };
  }

  // Analyst mode: the swing, the panels and the plate's surprise, drawn by analyst.js; the plate's chevrons page
  // through the surprises the clock has reached (the dots on the swing), the same shape as narration()
  function analystNarration() {
    const half = liveIndex() >= 0 ? idx.marks[liveIndex()].period : 1;
    const r = XplainAnalyst.render({ bundle: state.bundle, now: state.now, scope: state.scope,
      period: XplainPlayer.scopePeriod(state.scope), notStarted: state.scope === '2' && half !== 2,
      pick: state.analystPick, h1End: idx.h1End, teamCode: (id) => teamCode(id, '') });
    const pos = r.shown ? r.list.indexOf(r.shown) + 1 : 0;
    return { list: r.list, shown: r.shown, reached: r.list.length, pos,
      prev: pos >= 2 ? r.list[pos - 2] : null, next: pos > 0 && pos < r.list.length ? r.list[pos] : null };
  }

  function backToLive() { exitReplay(); state.follow = true; state.producerRecap = null; }

  // --- replay: a table row puts its moment on the pitch, the plate and the tab ---

  function enterReplay(id) {
    recapToMap();
    state.producerRecap = null;
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

  function leaveReplay() { exitReplay(); state.follow = true; state.producerRecap = null; }   // back to the live pass

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
    for (const b of $('recap-view').querySelectorAll('button')) {
      b.addEventListener('click', () => setRecapView(b.dataset.view));
    }
    $('rc-star').addEventListener('click', openStar);
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
      // Player mode: space plays or pauses, [ ] page the plate; nothing steps or replays
      if (state.mode === 'player' && ![' ', '[', ']'].includes(e.key)) return;
      // Analyst mode: space plays or pauses, [ ] page the plate's surprises; Esc or L drops the pick (the swing
      // shows the latest surprise again)
      if (state.mode === 'analyst') {
        if (e.key === ' ') { setPlaying(!state.playing); e.preventDefault(); }
        else if (e.key === '[' || e.key === ']') { narrStep(e.key === '[' ? -1 : 1); e.preventDefault(); }
        else if (e.key === 'Escape' || e.key === 'l' || e.key === 'L') { state.analystPick = null; e.preventDefault(); }
        return;
      }
      if (e.key === 'Escape') {   // leaves replay and step mode; when live, closes a recap (play stays paused)
        if (state.replay || !state.follow) { leaveReplay(); e.preventDefault(); }
        else if (state.producerRecap) { state.producerRecap = null; e.preventDefault(); }
        else if (state.recap) { closeRecap(); e.preventDefault(); }
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
    const brk = matchBreak();
    const key = [sel, live, tick, score, half, brk, state.scope, snaps.map((s) => (s ? s.t : '-')).join('|'),
      plate ? plate.bundle.moment.moment_id : '-', state.lang, state.follow, state.replay,
      narr.reached, narr.pos, goal, state.mode, state.player, state.playerNarr,
      state.recap ? state.recap.scope + (state.recap.resumeAt !== null) + state.recap.view : '-', state.playerPicked,
      state.producerRecap, state.mode === 'producer' ? XplainProducer.items(state.bundle, state.now, idx.recapT).length : '',
      state.analystPick, state.airedRev].join('/');
    if (key === state.renderedKey) return;   // every DOM write below runs in this one animation frame
    state.renderedKey = key;
    if (state.mode === 'player') {   // Player mode: the score bug stays live; player.js draws the rest
      renderRecap();
      renderScoreBug(tick, score, live >= 0 ? half : null, brk);
      XplainPlayer.render({ bundle: state.bundle, now: state.now, scope: state.scope, playerId: state.player,
        notStarted: state.scope === '2' && half !== 2 });
      const pn = playerNarration();
      const who = state.bundle.meta.players.find((p) => p.player_id === state.player);
      renderNarrNav(pn);
      renderPlate(pn.pos > 0 ? pn.list[pn.pos - 1] : null,
        'No commentated moment for ' + (who ? who.name : 'this player') + ' yet.');
      return;
    }
    if (state.mode === 'analyst') {   // Analyst mode: the score bug stays live; analyst.js draws the rest; the plate
      renderRecap();                  // shows the surprise the swing shows (its line when it was commentated)
      renderScoreBug(tick, score, live >= 0 ? half : null, brk);
      const an = analystNarration();
      renderNarrNav(an);
      renderPlate(an.shown, 'No surprise yet.');
      const shown = an.shown;
      // a surprise that was not commentated has no line: the plate says so instead of an empty line (unless
      // Producer put a Live retake of it on air: the plate shows that)
      if (shown && !shown.line && !state.aired.has(shown.bundle.moment.moment_id)) { $('lower-line').textContent = NOT_COMMENTATED; $('lower-line').removeAttribute('lang'); }
      return;
    }

    $('live').classList.toggle('is-live', isLive());
    $('prev').disabled = sel <= 0;
    $('next').disabled = isLive() || sel >= live;
    renderBadge(sel);
    renderScoreBug(tick, score, live >= 0 ? half : null, brk);
    renderPossession(sel);
    renderPitch(sel);
    const notStarted = state.scope === '2' && half !== 2;
    KINDS.forEach((k, i) => renderTable(k, snaps[i], plate, notStarted));
    renderRecap();
    const r = state.recap;
    if (r && r.view === 'recap') {   // the recap plate; the left chevron goes to the latest narrated moment
      renderNarrNav({ reached: narr.reached, pos: 0, prev: narr.reached > 0, next: null });
      renderRecapPlate();
      if (state.mode === 'producer') renderProducer(plate);
      return;
    }
    // while a recap is open, the right chevron at the latest narrated moment leads back to the recap
    renderNarrNav(r && !narr.next && narr.reached > 0 ? { ...narr, next: true } : narr);
    if (goal >= 0) renderGoalPlate(goal); else renderPlate(plate);
    if (state.mode === 'producer') renderProducer(plate);
  }

  // Producer mode: the log and the selected line. The selection is Studio's: the replayed moment, else the open
  // recap card, else the plate's moment; a recap row picked in the log (Producer only) comes first.
  function producerSelected(plate) {
    if (state.producerRecap) return { type: 'recap', id: state.producerRecap };
    if (state.replay) return { type: 'moment', id: state.replay };
    if (state.recap && state.recap.view === 'recap') return { type: 'recap', id: state.recap.scope };
    return plate ? { type: 'moment', id: plate.bundle.moment.moment_id } : null;
  }

  function renderProducer(plate) {
    XplainProducer.render({ bundle: state.bundle, now: state.now, recapT: idx.recapT, lang: state.lang,
      selected: producerSelected(plate) });
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
    stopDrain();
    lower.classList.remove('completed', 'failed', 'waiting', 'recap-plate');
    lower.classList.add('goal');
    $('lower-num').textContent = '';
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

  // --- recap: the card in the pitch's slot, the plate in its recap state ---

  // view: what the pitch's slot shows while the recap is open, the card ('recap') or the pass map ('map')
  function openRecap(scope, resumeAt) {
    state.recap = { scope, resumeAt, view: 'recap' };
    state.goal = null;
    state.playerPicked = false;
  }

  function closeRecap() {
    if (!state.recap) return;
    state.recap = null;
    const url = new URL(location.href);
    if (url.searchParams.has('recap')) {
      url.searchParams.delete('recap');
      history.replaceState(null, '', url);
    }
  }

  // A card value to two decimals (XplainPlayer.fixed2: "393.55", "+19.45", "−19.11"; formatting only), the
  // decimals in a smaller span (.dec); a signed value is coloured by its sign
  function decimal(cls, x, signed) {
    const text = XplainPlayer.fixed2(x, signed);
    const tone = !signed ? '' : text[0] === '+' ? ' pos' : text[0] === '−' ? ' neg' : '';
    const node = span(cls + tone, '');
    const dot = text.indexOf('.');
    node.append(text.slice(0, dot), span('dec', text.slice(dot)));
    return node;
  }
  function span(cls, text) {
    const s = document.createElement('span');
    s.className = cls;
    s.textContent = text;
    return s;
  }

  // Every number is the card's (bundle.recaps[scope].card, computed in recap.py); the position comes
  // from meta.players. The text is the recap's filled text in the narration language.
  function renderRecap() {
    const open = Boolean(state.recap) && state.mode === 'studio';
    const on = open && state.recap.view === 'recap';
    $('stage').classList.toggle('recap-on', on);
    $('recap').hidden = !on;
    $('recap-view').hidden = !open;
    if (open) {
      for (const b of $('recap-view').querySelectorAll('button')) {
        const sel = b.dataset.view === state.recap.view;
        b.classList.toggle('on', sel);
        b.setAttribute('aria-pressed', String(sel));
      }
    }
    const title = $('title');
    title.textContent = state.mode === 'producer' ? title.dataset.producer
      : state.mode === 'analyst' ? title.dataset.analyst
      : open ? title.dataset[state.recap.scope === 'half_time' ? 'ht' : 'ft'] : title.dataset.live;
    if (!on) return;
    const entry = state.bundle.recaps[state.recap.scope];
    const c = entry.card;
    $('rc-score').textContent = c.score_home + '–' + c.score_away;
    $('rc-home-name').textContent = c.home.name;
    $('rc-away-name').textContent = c.away.name;
    const goals = $('rc-goal-rows');
    goals.replaceChildren();
    if (c.goals.length === 0) goals.appendChild(div('empty', 'No goals.'));
    for (const g of c.goals) {
      const row = div('rc-row');
      row.append(span('rc-min', g.minute_label + "'"), span('rc-who', g.player_name || g.team_name),
        span('rc-tag', teamCode(g.team_id, g.team_name)), span('rc-after', g.home + '–' + g.away));
      goals.appendChild(row);
    }
    const teams = $('rc-team-rows');
    teams.replaceChildren();
    for (const t of [c.home, c.away]) {
      const row = div('rc-row');
      row.append(span('rc-team', teamCode(t.team_id, t.name)), span('rc-n', String(t.completed)),
        decimal('rc-n', t.expected_xp ?? t.expected, false), decimal('rc-n', t.vs_expected_xp ?? t.vs_expected, true));
      teams.appendChild(row);
    }
    $('rc-starblock').hidden = !c.star;
    if (c.star) renderStar(c);
    const text = $('rc-text');
    text.textContent = (entry.text && entry.text['text_' + state.lang]) || '';
    text.setAttribute('lang', state.lang);
    fitRecapText();
  }

  // The star's row: the shirt number and position from meta.players; the name, team, passes, completed,
  // expected and vs expected from the card; the minutes of the card's goals the star scored.
  function renderStar(c) {
    const s = c.star;
    const who = state.bundle.meta.players.find((p) => p.player_id === s.player_id);
    const row = $('rc-star');
    row.replaceChildren();
    const num = span('st-num', '');
    if (who && who.shirt_number != null) num.append(span('st-hash', '#'), String(who.shirt_number));
    const meta = span('st-meta', '');
    if (who && who.position) meta.appendChild(span('st-pos', who.position));
    meta.appendChild(span('st-team', s.team_name));
    const scored = c.goals.filter((g) => g.player_name === s.name).map((g) => g.minute_label + "'");
    if (scored.length) meta.appendChild(span('st-goal', (scored.length > 1 ? 'Goals ' : 'Goal ') + scored.join(', ')));
    // the name with a chevron: the row is a button that opens the star in Player
    const name = span('st-name', '');
    const go = el('svg', { class: 'st-go', width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor',
      'stroke-width': 2.6, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true' });
    go.appendChild(el('path', { d: 'M9 5l7 7-7 7' }));
    name.append(span('st-name-text', s.name), go);
    const ident = span('st-who', '');
    ident.append(name, meta);
    const stats = span('st-stats', '');
    for (const label of ['Passes', 'Completed', 'Expected', 'vs expected']) stats.appendChild(span('pv-label', label));
    stats.append(span('st-val', String(s.passes)), span('st-val', String(s.completed)),
      decimal('st-val', s.expected_xp ?? s.expected, false), decimal('st-val', s.vs_expected_xp ?? s.pva, true));
    row.title = 'Open ' + s.name + ' in Player';
    row.append(num, ident, stats);
  }

  // A long recap (many goals, a long Greek text) steps its text down 1px at a time, to RECAP_MIN_PX, until
  // the card's content fits its 520px. Formatting only.
  function fitRecapText() {
    const card = $('recap'), text = $('rc-text');
    let size = RECAP_TEXT_PX;
    text.style.fontSize = '';
    while (size > RECAP_MIN_PX && card.scrollHeight > card.clientHeight) text.style.fontSize = --size + 'px';
  }

  // Recap: the card, back at the live moment (the recap's time); Map: the pitch and the narration
  function setRecapView(view) {
    if (!state.recap) return;
    if (view === 'recap') { backToLive(); state.recap.view = 'recap'; } else recapToMap();
    render();
  }

  // Stepping, a replay, the plate's arrows or the Map button while a recap is open: the slot shows the map and
  // the switch stays; at half time the automatic restart is off (Play resumes)
  function recapToMap() {
    if (!state.recap) return;
    state.recap.view = 'map';
    state.recap.resumeAt = null;
  }

  // The star's row opens the star in Player
  function openStar() {
    const star = state.recap && state.bundle.recaps[state.recap.scope].card.star;
    if (!star) return;
    state.player = star.player_id;
    state.playerPicked = true;
    setMode('player');
  }

  function renderRecapPlate() {
    const r = state.recap;
    const id = 'recap/' + r.scope + '/' + (r.resumeAt !== null);
    if (id === state.plateId) return;
    state.plateId = id;
    const lower = $('lower');
    lower.classList.remove('completed', 'failed', 'waiting', 'goal');
    lower.classList.add('recap-plate');
    $('lower-num').textContent = '';
    $('lower-name').textContent = RECAP_TAB[r.scope];
    $('lower-when').textContent = '';
    const line = $('lower-line');
    line.textContent = RECAP_LINE[r.scope];
    line.removeAttribute('lang');
    line.style.fontSize = '';
    $('lower-meta').textContent = '';
    $('lower-xp').hidden = true;
    $('lower-pva').textContent = RECAP_COL[r.scope];
    $('lower-pva-label').textContent = '';
    stopDrain();
    const drainLine = $('lower-drain');
    // the drain line runs over the pause, from full to empty; with reduced motion it stays hidden
    if (r.resumeAt !== null && !reducedMotion.matches && drainLine.animate) {
      lower.classList.add('ticking');
      // from what is left of the pause (the full pause unless a frame ran late)
      const left = Math.max(0, r.resumeAt - performance.now());
      drain = drainLine.animate([{ transform: 'scaleX(' + left / RECAP_PAUSE_MS + ')' }, { transform: 'scaleX(0)' }],
        { duration: left, easing: 'linear', fill: 'forwards' });
    }
    fadePlate();
  }

  // The half-time drain (a Web Animation on #lower-drain); cancelled when the recap closes or any
  // other plate renders.
  function stopDrain() {
    if (drain) { drain.cancel(); drain = null; }
    $('lower').classList.remove('ticking');
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

  // brk: "HT" or "FT" (matchBreak) takes the minute's place; the half label and the live dot go meanwhile
  function renderScoreBug(tick, score, half, brk) {
    const s = score >= 0 ? idx.score[score] : null;
    $('score').textContent = s ? s.home + '–' + s.away : '–';
    $('minute').textContent = brk || (tick >= 0 ? idx.ticks[tick].minute_label + "'" : '');
    $('half').textContent = !brk && half ? HALF_LABEL[half] : '';
    $('scorebug').classList.toggle('break', Boolean(brk));
  }

  // "HT" from the last first-half pass (where the half-time recap opens) to the second half's kick-off, "FT"
  // from the last event on, else null. A state of the match, not a minute: minutes only come from clock.match_clock.
  function matchBreak() {
    const c = state.bundle.clock;
    if (state.now >= c.last_event_t) return 'FT';
    const h2 = c.second_half_start;
    if (idx.h1End !== null && h2 !== null && h2 !== undefined && state.now >= idx.h1End && state.now < h2) return 'HT';
    return null;
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

  // the latest commentated moment at or before the selected pass, or null; a replayed uncommentated moment whose
  // Live retake Producer put on air shows itself
  function plateMoment(sel) {
    if (sel < 0) return null;
    if (state.replay && state.aired.has(state.replay)) return idx.moments.get(state.replay);
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
  // "#27" from meta.players, '' when the player or the number is unknown (the tab then hides that cell)
  function shirt(id) {
    const p = idx.playerOf.get(id);
    return p && p.shirt_number !== null && p.shirt_number !== undefined ? '#' + p.shirt_number : '';
  }

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
  function renderPlate(entry, emptyText = 'No commentated moment yet.') {
    const aired = entry ? state.aired.get(entry.bundle.moment.moment_id) : null;
    const id = entry ? entry.bundle.moment.moment_id + '/' + state.lang + (aired ? '/retake' : '') : '-/' + emptyText;
    if (id === state.plateId) return;
    state.plateId = id;
    const lower = $('lower');
    stopDrain();
    lower.classList.remove('completed', 'failed', 'waiting', 'goal', 'recap-plate');
    const line = $('lower-line');
    fadePlate();
    if (!entry) {   // before the first commentated moment: the body only
      lower.classList.add('waiting');
      line.textContent = emptyText;
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
    $('lower-num').textContent = shirt(m.player_id);
    $('lower-name').textContent = playerName(m);
    $('lower-when').textContent = teamCode(m.team_id, m.team_name) + ' · ' + m.minute_label + "'";
    $('lower-meta').textContent = entry.tier.en + ' pass, ' + (m.completed ? 'completed' : 'not completed');
    $('lower-xp-val').textContent = percent(m.xp) + '%';   // " · xP 92%", xP in its own span
    $('lower-xp').hidden = false;
    // no stored line: the line row is empty and hidden (:empty), so no gap is left
    line.textContent = aired ? aired.line[state.lang] : (entry.line && entry.line['line_' + state.lang]) || '';
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
        if (fitOne(probe, m.bundle.moment.moment_id + '/' + lang, text, lang)) shrunk++;
      }
    }
    probe.remove();
    document.documentElement.dataset.lineFit = shrunk + '/' + total;
    if (state.fit.has(state.plateId)) $('lower-line').style.fontSize = state.fit.get(state.plateId);
    if (state.recap && !$('recap').hidden) fitRecapText();   // measured again with the web fonts in
  }

  // The font size one line needs in the plate (LINE_PX down to MIN_LINE_PX), stored under key when smaller; true then
  function fitOne(probe, key, text, lang) {
    probe.textContent = text;
    probe.setAttribute('lang', lang);
    let size = LINE_PX;
    probe.style.fontSize = size + 'px';
    while (size > MIN_LINE_PX && probe.offsetHeight > TWO_LINES_PX) probe.style.fontSize = --size + 'px';
    if (size < LINE_PX) state.fit.set(key, size + 'px'); else state.fit.delete(key);
    return size < LINE_PX;
  }

  // A retake put on air, fitted as the stored lines are (its plate id ends in "/retake")
  function fitAired(id, line) {
    const probe = div('line measure');
    $('lower').querySelector('.left').appendChild(probe);
    for (const lang of LANGS) if (line[lang]) fitOne(probe, id + '/' + lang + '/retake', line[lang], lang);
    probe.remove();
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
