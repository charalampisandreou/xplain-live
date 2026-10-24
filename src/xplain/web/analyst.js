// xPlain Live: Analyst mode. Both teams' passing against the model up to the match clock: the passing swing (each
// team's running vs expected), the teams' passing and their passes per difficulty tier, and for a picked surprise
// the model's reasons (its SHAP drivers, as the bundle stores them).
//
// Browser computation exception (CLAUDE.md): the numbers here are counts and sums over fields already in the
// bundle's marks, at the current clock (bundle.recaps holds only half and full time, which would give the result
// away). Each function that yields a number mirrors a Python reference and is checked against it in
// tests/test_analyst_js.py:
//   teamPasses   ->  swing.team_marks: the team's passes, t <= now, in the period, in seq order
//   teamSwing    ->  swing.team_swing: per pass, the running completed count, xP sum and vs expected
//   teamTiers    ->  swing.team_tiers: passes, completed, xP sum and vs expected per narration tier
//   surprises    ->  the outliers (bundle.moments) with t <= now, in the period, in seq order
// The SHAP panel shows the bundle's explanation values as stored (to two decimals, formatting only); the chart's
// scale and bar lengths are drawing, not numbers. Nothing computed here is ever sent to a model.
(function (root) {
  'use strict';

  // --- pure functions (no DOM; also loaded by node in the parity test) ---------------------------

  // the narration's difficulty tiers (narration.pass_tier), easiest first, as in Player
  const TIERS = ['very easy', 'easy', 'tricky', 'difficult', 'very difficult'];

  // The team's passes with t <= now (all when now is null) and in the period (any when null), in seq order.
  function teamPasses(marks, teamId, now, period) {
    const limit = now === null || now === undefined ? Infinity : now;
    return marks
      .filter((m) => m.team_id === teamId && m.t <= limit && (period === null || period === undefined || m.period === period))
      .sort((a, b) => a.seq - b.seq);
  }

  // One point per pass of the team: the running completed count, the xP added one pass at a time from 0 (left
  // to right, as analytics.seq_sum adds), and completed minus that sum (the recap card's vs expected, unrounded).
  function teamSwing(marks, teamId, now, period) {
    const out = [];
    let done = 0, xp = 0;
    teamPasses(marks, teamId, now, period).forEach((m, i) => {
      done += m.completed ? 1 : 0;
      xp += m.xp;
      out.push({ t: m.t, seq: m.seq, passes: i + 1, completed: done, expectedXp: xp, vsExpectedXp: done - xp });
    });
    return out;
  }

  // The team's passes per tier (mark.tier.en), in TIERS order: passes, completed, the xP sum and completed minus
  // it. A mark without a tier (a bundle written before tiers) is not counted.
  function teamTiers(marks, teamId, now, period) {
    const by = {};
    for (const t of TIERS) by[t] = [];
    for (const m of teamPasses(marks, teamId, now, period)) if (m.tier && m.tier.en in by) by[m.tier.en].push(m);
    return TIERS.map((tier) => {
      let done = 0, xp = 0;
      for (const m of by[tier]) { done += m.completed ? 1 : 0; xp += m.xp; }
      return { tier, passes: by[tier].length, completed: done, expectedXp: xp, vsExpectedXp: done - xp };
    });
  }

  // The surprises (every outlier in bundle.moments) with t <= now and in the period, in seq order.
  function surprises(moments, now, period) {
    const limit = now === null || now === undefined ? Infinity : now;
    return moments
      .filter((e) => {
        const m = e.bundle.moment;
        return m.t <= limit && (period === null || period === undefined || m.period === period);
      })
      .sort((a, b) => a.bundle.moment.seq - b.bundle.moment.seq);
  }

  // --- view ----------------------------------------------------------------------------------------

  const SVG = 'http://www.w3.org/2000/svg';
  const NOT_STARTED = 'Starts in the 2nd half.';
  const NO_PASSES = 'No passes yet.';
  const NO_PICK = ['No surprise yet', 'Each surprise becomes a dot on the swing. Pick one to see how the model rated the pass.'];
  const KIND_LABEL = { surprise_failure: 'Surprise failure', surprise_success: 'Surprise success' };
  const TEAM_CLASS = ['home', 'away'];
  // the chart's drawing area in the svg's own units (= px: the svg is drawn at 780 x 270)
  const W = 780, H = 270, PAD_L = 44, PAD_R = 92, PAD_T = 22, PAD_B = 28;
  const MIN_RANGE = 10;         // the value axis shows at least -10..+10, and ends on a multiple of 10
  const GOAL_LABEL_PX = 76;     // two goal labels closer than this would overlap when centred on their lines
  const AXIS_MINUTES = ['15', '30', '45', '60', '75', '90'];   // clock.ticks labels marked on the time axis

  let view = null;   // what app.js handed over in mount(), and what was last drawn

  const $ = (id) => document.getElementById(id);

  function mount(api) {
    view = { api, sig: '', shapSig: '', frameSig: '', goalSig: '', dotSig: '' };
    const chart = $('an-chart');
    chart.addEventListener('click', (e) => {
      const dot = e.target.closest && e.target.closest('[data-moment]');
      if (dot) api.pick(dot.dataset.moment);
    });
    chart.addEventListener('keydown', (e) => {
      const dot = e.target.closest && e.target.closest('[data-moment]');
      if (dot && (e.key === 'Enter' || e.key === ' ')) { api.pick(dot.dataset.moment); e.preventDefault(); e.stopPropagation(); }
    });
    $('an-replay').addEventListener('click', () => { if (view.shown) api.replay(view.shown); });
  }

  // draw everything again on the next render (entering Analyst mode)
  function invalidate() {
    if (!view) return;
    view.sig = '';
    view.shapSig = '';
    view.frameSig = '';
  }

  function el(name, attrs) {
    const node = document.createElementNS(SVG, name);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  function div(cls, text) {
    const d = document.createElement('div');
    d.className = cls;
    if (text !== undefined) d.textContent = text;
    return d;
  }

  function span(cls, text) {
    const s = document.createElement('span');
    s.className = cls;
    if (text !== undefined) s.textContent = text;
    return s;
  }

  // "+19.45" with the decimals in a smaller span (.dec), coloured by its sign; app.js's decimal() in Studio
  function decimal(cls, x, signed) {
    const text = view.api.fixed2(x, signed);
    const tone = !signed ? '' : text[0] === '+' ? ' pos' : text[0] === '−' ? ' neg' : '';
    const node = span(cls + tone);
    const dot = text.indexOf('.');
    node.append(text.slice(0, dot), span('dec', text.slice(dot)));
    return node;
  }

  // The time span the chart covers in the scope: the whole match, or one half (from the clock, never computed)
  function timeSpan(bundle, period, h1End) {
    const c = bundle.clock;
    if (period === 1) return [c.first_event_t, h1End];
    if (period === 2) return [c.second_half_start, c.last_event_t];
    return [c.first_event_t, c.last_event_t];
  }

  // Analyst mode at the clock and in the scope. ctx: bundle, now, scope, period, notStarted, pick (a moment_id or
  // null: the latest surprise), h1End. Returns { shown, list }: the moment entry the SHAP panel shows (null: none)
  // and the surprises reached, in seq order, for the plate and its chevrons.
  function render(ctx) {
    const { bundle } = ctx;
    const teams = [bundle.meta.home, bundle.meta.away];
    const found = ctx.notStarted ? [] : surprises(bundle.moments, ctx.now, ctx.period);
    const picked = ctx.pick ? found.find((e) => e.bundle.moment.moment_id === ctx.pick) : null;
    const shown = picked || (found.length ? found[found.length - 1] : null);
    view.shown = shown ? shown.bundle.moment.moment_id : null;
    const swings = teams.map((t) => (ctx.notStarted ? [] : teamSwing(bundle.marks, t.team_id, ctx.now, ctx.period)));
    const sig = [ctx.scope, ctx.notStarted, swings[0].length, swings[1].length, found.length].join('/');
    if (sig !== view.sig) {
      view.sig = sig;
      renderChart(ctx, teams, swings, found);
      renderTeams(ctx, teams, swings);
    }
    markShown();
    renderShap(shown, ctx);
    return { shown, list: found };
  }

  // --- the passing swing: one line per team, the surprises as dots on their team's line ---------------
  // The chart is drawn in layers that stay in place: the frame (grid, minutes, half time) and the dots are rebuilt
  // only when the scale, the span or the number of surprises changes; the lines and their end labels are updated in
  // place as passes come in; picking a dot only moves the ring. So a dot under the pointer is not replaced while the
  // match plays, and a click lands.

  function layer(name) {
    const svg = $('an-chart');
    let g = svg.querySelector('g.' + name);
    if (!g) { g = el('g', { class: name }); svg.appendChild(g); }
    return g;
  }

  function renderChart(ctx, teams, swings, found) {
    const svg = $('an-chart');
    const { bundle } = ctx;
    if (ctx.notStarted || (swings[0].length + swings[1].length) === 0) {
      svg.replaceChildren();
      view.frameSig = '';
      $('an-chart-empty').textContent = ctx.notStarted ? NOT_STARTED : NO_PASSES;
      $('an-chart-empty').hidden = false;
      svg.setAttribute('aria-label', 'Passing Swing, ' + (ctx.notStarted ? 'second half not started' : 'no passes yet'));
      return;
    }
    $('an-chart-empty').hidden = true;
    const [t0, t1] = timeSpan(bundle, ctx.period, ctx.h1End);
    let top = MIN_RANGE;
    for (const s of swings) for (const p of s) top = Math.max(top, Math.abs(p.vsExpectedXp));
    top = Math.ceil(top / 10) * 10;   // the axis ends on a multiple of 10, so its half-way lines are whole (drawing only)
    const x = (t) => PAD_L + ((t - t0) / Math.max(t1 - t0, 1)) * (W - PAD_L - PAD_R);
    const y = (v) => PAD_T + ((top - v) / (2 * top)) * (H - PAD_T - PAD_B);

    const frameSig = [ctx.scope, top, t0, t1].join('/');
    if (frameSig !== view.frameSig) {   // a new scale or span: every layer is drawn again, in this order
      view.frameSig = frameSig;
      view.goalSig = '';
      view.dotSig = '';
      svg.replaceChildren();
      for (const name of ['an-grid', 'an-goals', 'an-lines', 'an-ends', 'an-dots']) layer(name);
      for (const i of [0, 1]) layer('an-lines').appendChild(el('path', { class: 'an-line ' + TEAM_CLASS[i], d: '' }));
      drawGrid(ctx, x, y, top, t0, t1);
    }

    // goals reached so far in the span: a dashed line in the scorer's team colour, labelled with the minute
    // (two goals close together: the earlier label ends at its line, the later one starts at its own)
    const goalList = bundle.score.filter((s) => s.goal && s.t <= ctx.now && s.t >= t0 && s.t <= t1);
    if (String(goalList.length) !== view.goalSig) {
      view.goalSig = String(goalList.length);
      const goals = layer('an-goals');
      goals.replaceChildren();
      let prev = null;
      for (const s of goalList) {
        const side = TEAM_CLASS[teams.findIndex((t) => t.team_id === s.goal.team_id)] || 'home';
        const gx = x(s.t);
        goals.appendChild(el('line', { x1: gx, x2: gx, y1: PAD_T, y2: H - PAD_B, class: 'goal ' + side }));
        const label = el('text', { x: gx, y: PAD_T - 8, 'text-anchor': 'middle', class: 'an-goal ' + side });
        label.textContent = 'Goal ' + s.minute_label + "'";
        if (prev && gx - prev.x < GOAL_LABEL_PX) {
          prev.label.setAttribute('text-anchor', 'end');
          prev.label.setAttribute('x', prev.x - 3);
          label.setAttribute('text-anchor', 'start');
          label.setAttribute('x', gx + 3);
        }
        goals.appendChild(label);
        prev = { x: gx, label };
      }
    }

    // the lines (a step at each pass: the value holds until the team's next pass), updated in place
    const paths = layer('an-lines').querySelectorAll('path');
    const ends = [];
    swings.forEach((s, i) => {
      if (!s.length) { paths[i].setAttribute('d', ''); return; }
      let d = 'M' + x(Math.max(t0, s[0].t)).toFixed(1) + ' ' + y(0).toFixed(1);
      for (const p of s) d += 'H' + x(p.t).toFixed(1) + 'V' + y(p.vsExpectedXp).toFixed(1);
      paths[i].setAttribute('d', d);
      const last = s[s.length - 1];
      ends.push({ i, y: y(last.vsExpectedXp), v: last.vsExpectedXp });
    });
    // end labels: the team code over its value, pushed apart when the two lines end close together
    if (ends.length === 2 && Math.abs(ends[0].y - ends[1].y) < 34) {
      const [hi, lo] = ends[0].y <= ends[1].y ? ends : [ends[1], ends[0]];
      const mid = (hi.y + lo.y) / 2;
      hi.ly = mid - 17; lo.ly = mid + 17;
    }
    const endLayer = layer('an-ends');
    endLayer.replaceChildren();
    for (const e of ends) {
      const ly = e.ly !== undefined ? e.ly : e.y;
      const g = el('g', { class: 'an-end ' + TEAM_CLASS[e.i] });
      const code = el('text', { x: W - PAD_R + 10, y: ly - 2, class: 'an-code' });
      code.textContent = ctx.teamCode(teams[e.i].team_id);
      const val = el('text', { x: W - PAD_R + 10, y: ly + 15, class: 'an-val' });
      val.textContent = view.api.fixed2(e.v, true);
      g.append(code, val);
      endLayer.appendChild(g);
    }

    // the surprises: on their team's line at the value just after that pass. A dot never moves once drawn, so the
    // layer is rebuilt only when a surprise is added (keyboard focus stays on the same dot)
    if (String(found.length) !== view.dotSig) {
      view.dotSig = String(found.length);
      const dots = layer('an-dots');
      const focused = document.activeElement && dots.contains(document.activeElement)
        ? document.activeElement.getAttribute('data-moment') : null;
      dots.replaceChildren();
      for (const e of found) {
        const m = e.bundle.moment;
        const i = teams.findIndex((t) => t.team_id === m.team_id);
        const p = i < 0 ? null : swings[i].find((q) => q.seq === m.seq);
        if (!p) continue;
        const g = el('g', { 'data-moment': m.moment_id, class: 'an-dot ' + m.kind, tabindex: '0', role: 'button',
          'aria-pressed': 'false',
          'aria-label': KIND_LABEL[m.kind] + ', ' + (m.player_name || 'unknown player') + ', '
            + ctx.teamCode(m.team_id) + ', ' + m.minute_label + "', PVA " + view.api.pva(m.delta) });
        const cx = x(p.t), cy = y(p.vsExpectedXp);
        g.append(el('circle', { cx, cy, r: 12, class: 'hit' }), el('circle', { cx, cy, r: 9, class: 'ring' }),
          el('circle', { cx, cy, r: 4.5, class: 'dot' }));
        dots.appendChild(g);
        if (m.moment_id === focused) g.focus();
      }
    }

    const words = ends.map((e) => teams[e.i].name + ' ' + view.api.fixed2(e.v, true)).join(', ');
    svg.setAttribute('aria-label', 'Passing Swing: each team\'s completed passes minus expected, running. Now ' + words
      + '. ' + found.length + (found.length === 1 ? ' surprise' : ' surprises') + ' marked as dots.');
  }

  // grid: zero, the axis ends, the minutes; half-time in the whole-match view
  function drawGrid(ctx, x, y, top, t0, t1) {
    const { bundle } = ctx;
    const grid = layer('an-grid');
    for (const v of [top, top / 2, 0, -top / 2, -top]) {
      grid.appendChild(el('line', { x1: PAD_L, x2: W - PAD_R, y1: y(v), y2: y(v), class: v === 0 ? 'zero' : '' }));
      const label = el('text', { x: PAD_L - 8, y: y(v) + 4, 'text-anchor': 'end', class: 'an-axis' });
      label.textContent = v === 0 ? '0' : (v > 0 ? '+' : '−') + Math.abs(v);
      grid.appendChild(label);
    }
    const seen = new Set();
    for (const k of bundle.clock.ticks) {
      if (!AXIS_MINUTES.includes(k.minute_label) || seen.has(k.minute_label) || k.t < t0 || k.t > t1) continue;
      seen.add(k.minute_label);
      const label = el('text', { x: x(k.t), y: H - 8, 'text-anchor': 'middle', class: 'an-axis' });
      label.textContent = k.minute_label + "'";
      grid.appendChild(label);
    }
    if (ctx.period === null && bundle.clock.second_half_start !== null) {
      const hx = x(bundle.clock.second_half_start);
      grid.appendChild(el('line', { x1: hx, x2: hx, y1: PAD_T - 6, y2: H - PAD_B, class: 'half' }));
      const ht = el('text', { x: hx + 6, y: PAD_T - 8, class: 'an-axis' });
      ht.textContent = 'HT';
      grid.appendChild(ht);
    }
  }

  // the shown surprise's dot carries the ring (class "on"); nothing is redrawn
  function markShown() {
    for (const g of $('an-chart').querySelectorAll('.an-dot')) {
      const on = g.getAttribute('data-moment') === view.shown;
      if (g.classList.contains('on') === on) continue;
      g.classList.toggle('on', on);
      g.setAttribute('aria-pressed', String(on));
    }
  }

  // --- the teams: passing so far, and per difficulty tier ------------------------------------------

  function renderTeams(ctx, teams, swings) {
    const rows = $('an-team-rows');
    rows.replaceChildren();
    teams.forEach((t, i) => {
      const last = swings[i].length ? swings[i][swings[i].length - 1] : null;
      const row = div('rc-row');
      row.append(span('rc-team', ''), span('rc-n', last ? String(last.passes) : '–'),
        span('rc-n', last ? String(last.completed) : '–'));
      row.firstChild.append(span('an-key ' + TEAM_CLASS[i]), ctx.teamCode(t.team_id));
      if (last) row.append(decimal('rc-n', last.expectedXp, false), decimal('rc-n', last.vsExpectedXp, true));
      else row.append(span('rc-n', '–'), span('rc-n', '–'));
      rows.appendChild(row);
    });
    const tiers = teams.map((t) => (ctx.notStarted ? null : teamTiers(ctx.bundle.marks, t.team_id, ctx.now, ctx.period)));
    $('an-tier-home').textContent = ctx.teamCode(teams[0].team_id);
    $('an-tier-away').textContent = ctx.teamCode(teams[1].team_id);
    const body = $('an-tier-rows');
    body.replaceChildren();
    TIERS.forEach((tier, k) => {
      const row = div('rc-row an-tier');
      const name = span('rc-team', tier[0].toUpperCase() + tier.slice(1));
      row.appendChild(name);
      for (const ts of tiers) {
        const c = ts ? ts[k] : null;
        const cell = span('an-cell');
        cell.appendChild(span('an-frac', c ? c.completed + '/' + c.passes : '–'));
        if (c && c.passes) cell.appendChild(decimal('an-vs', c.vsExpectedXp, true));
        else cell.appendChild(span('an-vs', ''));
        row.appendChild(cell);
      }
      body.appendChild(row);
    });
  }

  // --- the model's reasons for the shown surprise (bundle explanation, as stored) --------------------

  function renderShap(entry, ctx) {
    const sig = entry ? entry.bundle.moment.moment_id : '-';
    if (sig === view.shapSig) return;
    view.shapSig = sig;
    const box = $('an-shap-rows');
    box.replaceChildren();
    $('an-replay').hidden = !(entry && entry.bundle.qualifies_live);
    if (!entry) {
      $('an-shap-title').textContent = "The model's reasons";
      $('an-shap-sub').textContent = '';
      $('an-shap-foot').textContent = '';
      const wait = div('empty waiting');
      wait.append(div('waiting-title', NO_PICK[0]), div('waiting-sub', NO_PICK[1]));
      box.appendChild(wait);
      return;
    }
    const m = entry.bundle.moment;
    const x = entry.bundle.explanation;
    $('an-shap-title').textContent = 'Why the model rated it ' + entry.tier.en;
    $('an-shap-sub').textContent = (m.player_name || 'Unknown player') + ' · ' + ctx.teamCode(m.team_id) + ' · '
      + m.minute_label + "' · " + (m.completed ? 'completed' : 'not completed');
    // the bars share one scale: the largest |shap| among the drivers and the rest fills the track (drawing only)
    const rows = x.drivers.map((d) => ({ label: d.label, topic: d.topic, shap: d.shap, raises: d.direction === 'raises' }));
    rows.push({ label: 'All other inputs', topic: '', shap: x.rest, raises: x.rest > 0, rest: true });
    const scale = Math.max(...rows.map((r) => Math.abs(r.shap)), 1e-9);
    for (const r of rows) {
      const row = div('an-shap-row' + (r.rest ? ' rest' : ''));
      const words = div('an-shap-words');
      words.append(span('an-shap-label', r.label));
      if (r.topic) words.append(span('an-shap-topic', r.topic));
      const track = div('an-shap-track');
      const bar = div('an-shap-bar ' + (r.raises ? 'up' : 'down'));
      bar.style.width = (50 * Math.abs(r.shap) / scale) + '%';   // half the track each way from the centre
      track.append(div('an-shap-mid'), bar);
      row.append(words, track, decimal('an-shap-val', r.shap, true));
      row.setAttribute('aria-label', r.label + ', ' + (r.raises ? 'raises' : 'lowers') + ' the model\'s score by '
        + view.api.fixed2(Math.abs(r.shap), false));
      box.appendChild(row);
    }
    // the key under the tracks: left of the centre lowered the odds, right of it raised them (not the outcome)
    const key = div('an-shap-row an-shap-key');
    const halves = div('an-shap-keys');
    const harder = div('an-shap-keyside down');
    harder.append(span('an-shap-keytext', 'made it harder'), span('an-shap-swatch down'));
    const easier = div('an-shap-keyside up');
    easier.append(span('an-shap-swatch up'), span('an-shap-keytext', 'made it easier'));
    halves.append(harder, easier);
    key.append(div('an-shap-words'), halves, span('an-shap-val'));
    key.setAttribute('aria-hidden', 'true');
    box.appendChild(key);
    $('an-shap-foot').textContent = 'Typical pass ' + view.api.fixed2(x.baseline, false) + ' → this pass '
      + view.api.fixed2(x.raw_score, false) + ' (model scale, not %) · xP ' + view.api.percent(m.xp) + '%';
  }

  const api = { TIERS, teamPasses, teamSwing, teamTiers, surprises, mount, invalidate, render };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.XplainAnalyst = api;
})(typeof window !== 'undefined' ? window : this);
