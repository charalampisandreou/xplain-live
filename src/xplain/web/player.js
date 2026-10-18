// xPlain Live: Player mode ("Player focus"). One player's match up to the match clock.
//
// Browser computation exception (CLAUDE.md): the numbers here are sums, counts and means over
// fields already in the bundle's marks, at the current clock (bundle.players is full time only,
// so it would give the result away). Each function that yields a number mirrors a Python
// reference and is checked against it in tests/test_player_js.py:
//   playerPasses + playerStats  ->  pipeline.player_pass_map / player_summary(marks, id, as_of_t)
//   hardestCompleted            ->  the completed pass with the lowest xp, ties to the earlier seq
//   narratedFor                 ->  the player's moments with a line, t <= now, in seq order
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

  // --- view ----------------------------------------------------------------------------------------

  let view = null;   // what app.js handed over in mount(), and what the column last drew

  function mount(api) {
    view = { api, sig: '' };
  }

  function setText(id, text) {
    const node = document.getElementById(id);
    if (node.textContent !== text) node.textContent = text;
  }

  // The picker trigger and the numbers, for the player at the clock and in the scope.
  function render(ctx) {
    const { api } = view;
    const players = ctx.bundle.meta.players;
    const player = players.find((p) => p.player_id === ctx.playerId);
    const passes = playerPasses(ctx.bundle.marks, ctx.playerId, ctx.now, scopePeriod(ctx.scope));
    const sig = [ctx.playerId, ctx.scope, passes.length].join('/');
    if (sig === view.sig) return;
    view.sig = sig;
    const name = player ? player.name : 'Pick a player';
    setText('player-pick-name', name);
    setText('pv-team', player ? api.teamName(player.team_id) : '');
    const s = playerStats(passes);
    setText('pv-passes', s ? String(s.passes) : '–');
    setText('pv-completed', s ? s.completed + ' · ' + api.percent(s.completionRate) + '%' : '–');
    setText('pv-expected', s ? api.percent(s.meanXp) + '%' : '–');
    setText('pv-pva', s ? api.pva(s.pvaPerPass) : '–');
  }

  const api = { scopePeriod, playerPasses, playerStats, hardestCompleted, narratedFor, mount, render };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.XplainPlayer = api;
})(typeof window !== 'undefined' ? window : this);
