// xPlain Live: Player mode ("Player focus"). One player's match up to the match clock.
//
// Browser computation exception (CLAUDE.md): the numbers here are sums, counts and means over
// fields already in the bundle's marks, at the current clock (bundle.players is full time only,
// so it would give the result away). Each function that yields a number mirrors a Python
// reference and is checked against it in tests/test_player_js.py:
//   playerPasses + playerStats  ->  pipeline.player_pass_map / player_summary(marks, id, as_of_t)
//   hardestCompleted            ->  the completed pass with the lowest xp, ties to the earlier seq
//   narratedFor                 ->  the player's moments with a line, t <= now, in seq order
//   bandCounts                  ->  how many of the passes fall in each difficulty band (mark.band)
//   momentsFor                  ->  the player's commentated moments, t <= now, newest first
// Nothing computed here is ever sent to a model.
(function (root) {
  'use strict';

  // --- pure functions (no DOM; also loaded by node in the parity test) ---------------------------

  // the scope selector's value as a period filter: 'all' -> null, '1' -> 1, '2' -> 2
  function scopePeriod(scope) {
    return scope === '1' ? 1 : scope === '2' ? 2 : null;
  }

  // The player's passes with t <= now (all when now is null) and in the period (any when null), in
  // seq order: pipeline.player_pass_map. Passes without a player never match.
  function playerPasses(marks, playerId, now, period) {
    if (playerId === null || playerId === undefined) return [];
    const limit = now === null || now === undefined ? Infinity : now;
    return marks
      .filter((m) => m.player_id === playerId && m.t <= limit && (period === null || period === undefined || m.period === period))
      .sort((a, b) => a.seq - b.seq);
  }

  // Counts, completion rate, mean xP and PVA of the passes: pipeline.player_summary. Sums run in
  // seq order from 0, as Python's sum does, so the floats match. null when there are no passes.
  function playerStats(passes) {
    const n = passes.length;
    if (n === 0) return null;
    let done = 0, xp = 0, pva = 0;
    for (const m of passes) {
      done += m.completed ? 1 : 0;
      xp += m.xp;
      pva += m.delta;
    }
    return {
      passes: n,
      completed: done,
      completionRate: done / n,
      meanXp: xp / n,
      pvaTotal: pva,
      pvaPerPass: pva / n,
    };
  }

  // The completed pass with the lowest xP (the hardest one completed), ties to the earlier seq; null when none.
  function hardestCompleted(passes) {
    let best = null;
    for (const m of passes) {
      if (!m.completed) continue;
      if (best === null || m.xp < best.xp || (m.xp === best.xp && m.seq < best.seq)) best = m;
    }
    return best;
  }

  // The player's narrated moments (a stored line) with t <= now and in the period, in seq order.
  function narratedFor(moments, playerId, now, period) {
    const limit = now === null || now === undefined ? Infinity : now;
    return moments
      .filter((e) => {
        const m = e.bundle.moment;
        return e.line && m.player_id === playerId && m.t <= limit
          && (period === null || period === undefined || m.period === period);
      })
      .sort((a, b) => a.bundle.moment.seq - b.bundle.moment.seq);
  }

  const BANDS = ['routine', 'moderate', 'hard', 'exceptional'];

  // How many of the passes fall in each difficulty band (the bundle's mark.band), in BANDS order.
  function bandCounts(passes) {
    const out = { routine: 0, moderate: 0, hard: 0, exceptional: 0 };
    for (const m of passes) out[m.band] += 1;
    return out;
  }

  // The player's commentated moments (qualifies_live) with t <= now and in the period, newest first.
  function momentsFor(moments, playerId, now, period) {
    const limit = now === null || now === undefined ? Infinity : now;
    return moments
      .filter((e) => {
        const m = e.bundle.moment;
        return e.bundle.qualifies_live && m.player_id === playerId && m.t <= limit
          && (period === null || period === undefined || m.period === period);
      })
      .sort((a, b) => b.bundle.moment.seq - a.bundle.moment.seq);
  }

  // --- view ----------------------------------------------------------------------------------------

  const SVG = 'http://www.w3.org/2000/svg';
  const KIND_LABEL = { surprise_failure: 'Surprise failure', surprise_success: 'Surprise success' };
  const NOT_STARTED = 'Starts in the 2nd half';
  const NO_MOMENTS = 'No surprises yet.';

  let view = null;   // what app.js handed over in mount(), and what was last drawn

  const $ = (id) => document.getElementById(id);

  function mount(api) {
    view = { api, sig: '' };
    // a moment row lights its pass on the map (hover or keyboard focus); nothing else happens
    const rows = $('pv-rows');
    const light = (e) => {
      const row = e.target.closest && e.target.closest('.row[data-moment]');
      spotlight(row ? row.dataset.moment : null);
    };
    rows.addEventListener('mouseover', light);
    rows.addEventListener('focusin', light);
    rows.addEventListener('mouseleave', () => spotlight(null));
    rows.addEventListener('focusout', () => spotlight(null));
  }

  // draw everything again on the next render (entering Player mode, after Studio used the pitch)
  function invalidate() {
    if (view) view.sig = '';
  }

  function spotlight(momentId) {
    const layer = $('passes');
    layer.classList.toggle('spot', Boolean(momentId));
    for (const g of layer.querySelectorAll('g.hot')) g.classList.remove('hot');
    if (momentId) {
      const g = layer.querySelector('g[data-moment="' + momentId + '"]');
      if (g) g.classList.add('hot');
    }
  }

  function setText(id, text) {
    const node = $(id);
    if (node.textContent !== text) node.textContent = text;
  }

  function div(cls, text) {
    const d = document.createElement('div');
    d.className = cls;
    if (text !== undefined) d.textContent = text;
    return d;
  }

  function el(name, attrs) {
    const node = document.createElementNS(SVG, name);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  // "vs expected": completed passes minus the completions xP expected (pvaTotal), to one decimal
  function signed1(x) {
    const r = Math.round(Math.abs(x) * 10) / 10;
    return r === 0 ? '0.0' : (x < 0 ? '−' : '+') + r.toFixed(1);
  }

  // Player mode at the clock and in the scope: the name, the pass map, the numbers, the bands and
  // the moments. Redrawn only when the player, the scope or the player's passes change.
  // ctx: bundle, now, scope, playerId, notStarted.
  function render(ctx) {
    const player = ctx.bundle.meta.players.find((p) => p.player_id === ctx.playerId);
    const period = scopePeriod(ctx.scope);
    const passes = ctx.notStarted ? [] : playerPasses(ctx.bundle.marks, ctx.playerId, ctx.now, period);
    const moments = ctx.notStarted ? [] : momentsFor(ctx.bundle.moments, ctx.playerId, ctx.now, period);
    const sig = [ctx.playerId, ctx.scope, ctx.notStarted, passes.length, moments.length].join('/');
    if (sig === view.sig) return;
    view.sig = sig;
    const stats = playerStats(passes);
    setText('player-name', player ? player.name : '');
    setText('player-team', player ? view.api.teamName(player.team_id) : '');
    renderTab(player);
    renderMap(passes, player, stats, moments.length);
    renderNumbers(stats);
    renderBands(bandCounts(passes));
    renderMoments(moments, ctx.notStarted);
  }

  // the pitch tab: the player's team, attacking right (the map is drawn in the passer's own frame)
  function renderTab(player) {
    const box = $('possession');
    if (!player) { box.hidden = true; return; }
    const teamCode = view.api.teamCode(player.team_id);
    setText('tab-code', teamCode);
    box.classList.remove('left');
    box.setAttribute('aria-label', teamCode + ', ' + player.name + "'s passes attack right");
    box.hidden = false;
  }

  // One pass in the passer's own frame (mark.pass_geometry: the player always attacks +x, in both
  // halves). Studio's marks, thinner: green arrow = completed, red dashed line ending in a cross =
  // not completed. A surprise moment is thicker with a cyan ring at its start; the newest pass has a
  // small white dot at its end.
  function drawPass(parent, mark, newest) {
    const p = mark.pass_geometry;
    const surprise = Boolean(mark.moment_id);
    const g = el('g', surprise ? { 'data-moment': mark.moment_id, class: 'surprise' } : {});
    const width = surprise ? 0.6 : 0.32;
    const colour = mark.completed ? '#3BD16F' : '#FF4B47';
    if (newest) g.appendChild(el('circle', { cx: p.end_x, cy: p.end_y, r: 0.8, fill: '#FFFFFF', class: 'newest' }));
    if (mark.completed) {
      g.appendChild(el('line', { x1: p.start_x, y1: p.start_y, x2: p.end_x, y2: p.end_y, stroke: colour,
        'stroke-width': width, 'marker-end': 'url(#arrow-green)' }));
    } else {
      g.appendChild(el('line', { x1: p.start_x, y1: p.start_y, x2: p.end_x, y2: p.end_y, stroke: colour,
        'stroke-width': width, 'stroke-dasharray': surprise ? '1.4 1' : '1 0.8' }));
      const c = surprise ? 1.1 : 0.7;
      const cross = el('g', { stroke: colour, 'stroke-width': surprise ? 0.7 : 0.4, 'stroke-linecap': 'round' });
      cross.appendChild(el('line', { x1: p.end_x - c, y1: p.end_y - c, x2: p.end_x + c, y2: p.end_y + c }));
      cross.appendChild(el('line', { x1: p.end_x + c, y1: p.end_y - c, x2: p.end_x - c, y2: p.end_y + c }));
      g.appendChild(cross);
    }
    if (surprise) {
      g.appendChild(el('circle', { cx: p.start_x, cy: p.start_y, r: 1.5, fill: '#10294A', stroke: '#3FD0FF',
        'stroke-width': 0.45 }));
    }
    parent.appendChild(g);
  }

  function renderMap(passes, player, stats, surprises) {
    const layer = $('passes');
    layer.replaceChildren();
    layer.classList.remove('spot');
    const newest = passes.length ? passes[passes.length - 1] : null;
    // surprises on top of the rest, so their rings are never covered
    for (const m of passes) if (!m.moment_id) drawPass(layer, m, m === newest);
    for (const m of passes) if (m.moment_id) drawPass(layer, m, m === newest);
    const who = player ? player.name : 'No player';
    $('pitch-svg').setAttribute('aria-label', stats
      ? 'Pass map of ' + who + ', attacking right: ' + stats.passes + ' passes, ' + stats.completed + ' completed as green '
        + 'arrows, ' + (stats.passes - stats.completed) + ' not completed as red dashed lines, ' + surprises
        + ' surprise moments ringed in cyan'
      : 'Pass map of ' + who + ', no passes yet');
  }

  function renderNumbers(s) {
    const { api } = view;
    setText('pv-passes', s ? String(s.passes) : '–');
    setText('pv-completed', s ? api.percent(s.completionRate) + '%' : '–');
    setText('pv-expected', s ? api.percent(s.meanXp) + '%' : '–');
    const text = s ? signed1(s.pvaTotal) : '–';
    setText('pv-vs', text);
    $('pv-vs').classList.toggle('pos', text[0] === '+');
    $('pv-vs').classList.toggle('neg', text[0] === '−');
  }

  // one bar, a segment per band as wide as its count; the four counts below it
  function renderBands(counts) {
    const bar = $('pv-bandbar');
    bar.replaceChildren();
    for (const band of BANDS) {
      setText('pv-band-' + band, String(counts[band]));
      if (counts[band] === 0) continue;
      const seg = div('band ' + band);
      seg.style.flexGrow = String(counts[band]);
      bar.appendChild(seg);
    }
  }

  // Studio's table rows: minute, kind over the tier words, |PVA| bar, PVA. Not buttons: hover or
  // focus lights the moment's pass on the map.
  function renderMoments(moments, notStarted) {
    const { api } = view;
    const rows = $('pv-rows');
    rows.replaceChildren();
    if (notStarted || moments.length === 0) {
      rows.appendChild(div('empty', notStarted ? NOT_STARTED : NO_MOMENTS));
      return;
    }
    for (const e of moments) {
      const m = e.bundle.moment;
      const tier = e.tier.en + ' pass, ' + (m.completed ? 'completed' : 'not completed');
      const row = div('row');
      row.dataset.moment = m.moment_id;
      row.setAttribute('role', 'listitem');
      row.tabIndex = 0;
      row.setAttribute('aria-label', KIND_LABEL[m.kind] + ', ' + m.minute_label + "', " + tier + ', PVA ' + api.pva(m.delta));
      row.appendChild(div('when', m.minute_label + "'"));
      const who = div('who');
      who.appendChild(div('player', KIND_LABEL[m.kind]));
      who.appendChild(div('meta', tier));
      row.appendChild(who);
      const track = div('track');
      const bar = div('bar ' + (m.delta < 0 ? 'neg' : 'pos'));
      bar.style.width = (100 * Math.abs(m.delta)) + '%';   // |PVA| on a 0..1 track, as in Studio
      track.appendChild(bar);
      row.appendChild(track);
      row.appendChild(div('value', api.pva(m.delta)));
      rows.appendChild(row);
    }
  }

  const api = { scopePeriod, playerPasses, playerStats, hardestCompleted, narratedFor, bandCounts, momentsFor,
    mount, invalidate, render };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.XplainPlayer = api;
})(typeof window !== 'undefined' ? window : this);
