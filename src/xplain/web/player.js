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

  const KIND_LABEL = { surprise_failure: 'Surprise failure', surprise_success: 'Surprise success' };
  const NOT_STARTED = 'Starts in the 2nd half';
  const NO_MOMENTS = 'No surprises yet.';

  let view = null;   // what app.js handed over in mount(), and what the column last drew

  const $ = (id) => document.getElementById(id);

  function mount(api) {
    view = { api, sig: '' };
    const rows = $('pv-rows');
    rows.addEventListener('click', (e) => {
      const row = e.target.closest('.row[data-moment]');
      if (row) api.enterReplay(row.dataset.moment);
    });
    rows.addEventListener('keydown', (e) => {
      const row = e.target.closest('.row[data-moment]');
      if (row && (e.key === 'Enter' || e.key === ' ')) { api.enterReplay(row.dataset.moment); e.preventDefault(); }
    });
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

  // "vs expected": completed passes minus the completions xP expected (pvaTotal), to one decimal
  function signed1(x) {
    const r = Math.round(Math.abs(x) * 10) / 10;
    return r === 0 ? '0.0' : (x < 0 ? '−' : '+') + r.toFixed(1);
  }

  // The player's column at the clock and in the scope: the name, the numbers, the bands and the moments.
  // ctx: bundle, now, scope, playerId, selected (the plate's moment_id), replay, notStarted.
  function render(ctx) {
    const players = ctx.bundle.meta.players;
    const player = players.find((p) => p.player_id === ctx.playerId);
    const period = scopePeriod(ctx.scope);
    const passes = ctx.notStarted ? [] : playerPasses(ctx.bundle.marks, ctx.playerId, ctx.now, period);
    const moments = ctx.notStarted ? [] : momentsFor(ctx.bundle.moments, ctx.playerId, ctx.now, period);
    const sig = [ctx.playerId, ctx.scope, ctx.notStarted, passes.length, moments.length, ctx.selected, ctx.replay].join('/');
    if (sig === view.sig) return;
    view.sig = sig;
    setText('player-name', player ? player.name : '');
    setText('player-team', player ? view.api.teamName(player.team_id) : '');
    renderNumbers(playerStats(passes));
    renderBands(bandCounts(passes));
    renderMoments(moments, ctx);
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

  // Studio's table rows: minute, kind over the tier words, |PVA| bar, PVA; a click replays the moment
  function renderMoments(moments, ctx) {
    const { api } = view;
    const rows = $('pv-rows');
    const focused = document.activeElement && rows.contains(document.activeElement)
      ? document.activeElement.dataset.moment : null;
    rows.replaceChildren();
    if (ctx.notStarted || moments.length === 0) {
      rows.appendChild(div('empty', ctx.notStarted ? NOT_STARTED : NO_MOMENTS));
      return;
    }
    for (const e of moments) {
      const m = e.bundle.moment;
      const row = div('row' + (m.moment_id === ctx.selected ? ' sel' : ''));
      row.dataset.moment = m.moment_id;
      row.setAttribute('role', 'button');
      row.tabIndex = 0;
      row.setAttribute('aria-pressed', String(m.moment_id === ctx.replay));
      row.setAttribute('aria-label', 'Replay ' + KIND_LABEL[m.kind] + ', ' + m.minute_label + "', PVA " + api.pva(m.delta));
      row.appendChild(div('when', m.minute_label + "'"));
      const who = div('who');
      who.appendChild(div('player', KIND_LABEL[m.kind]));
      who.appendChild(div('meta', e.tier.en + ' pass, ' + (m.completed ? 'completed' : 'not completed')));
      row.appendChild(who);
      const track = div('track');
      const bar = div('bar ' + (m.delta < 0 ? 'neg' : 'pos'));
      bar.style.width = (100 * Math.abs(m.delta)) + '%';   // |PVA| on a 0..1 track, as in Studio
      track.appendChild(bar);
      row.appendChild(track);
      row.appendChild(div('value', api.pva(m.delta)));
      rows.appendChild(row);
      if (m.moment_id === focused) row.focus();
    }
  }

  const api = { scopePeriod, playerPasses, playerStats, hardestCompleted, narratedFor, bandCounts, momentsFor,
    mount, render };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.XplainPlayer = api;
})(typeof window !== 'undefined' ? window : this);
