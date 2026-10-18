// xPlain Live: Player mode ("Player focus"). One player's match up to the match clock.
//
// Browser computation exception (CLAUDE.md): the numbers here are sums, counts and means over
// fields already in the bundle's marks, at the current clock (bundle.players is full time only,
// so it would give the result away). Each function that yields a number mirrors a Python
// reference and is checked against it in tests/test_player_js.py:
//   playerPasses + playerStats  ->  pipeline.player_pass_map / player_summary(marks, id, as_of_t);
//                                  expectedXp, vsExpectedXp -> the recap card's expected_xp / vs_expected_xp
//                                  before rounding: the sum of xP, completed minus that sum
//   fixed2                      ->  recap.round2: two decimals, as toFixed(2) (formatting only)
//   hardestCompleted            ->  the completed pass with the lowest xp, ties to the earlier seq
//   narratedFor                 ->  the player's moments with a line, t <= now, in seq order
//   tierCounts                  ->  how many of the passes have each difficulty tier (mark.tier.en)
//   completedTierCounts         ->  the same over the completed passes only (the column's bar)
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
  // expectedXp and vsExpectedXp are what the recap card shows as Expected and vs expected (recap.star_player):
  // the sum of xP, and completed minus that sum; fixed2 shows them to two decimals.
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
      expectedXp: xp,
      vsExpectedXp: done - xp,
    };
  }

  // Two decimals for Expected and vs expected, in Studio and Players: toFixed(2) rounds the exact float, halves
  // away from zero (recap.round2 is the Python twin; the card's values are already rounded so). signed: "+5.48",
  // "−19.11" with the Studio's minus sign, and "0.00" with no sign.
  function fixed2(x, signed) {
    const a = Math.abs(x).toFixed(2);
    if (a === '0.00') return a;
    if (signed) return (x > 0 ? '+' : '−') + a;
    return x < 0 ? '−' + a : a;
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

  // the narration's difficulty tiers (narration.pass_tier), easiest first
  const TIERS = ['very easy', 'easy', 'tricky', 'difficult', 'very difficult'];

  // How many of the passes have each difficulty tier (the bundle's mark.tier.en), in TIERS order. A
  // mark without a tier (a bundle written before tiers) is not counted.
  function tierCounts(passes) {
    const out = {};
    for (const t of TIERS) out[t] = 0;
    for (const m of passes) if (m.tier && m.tier.en in out) out[m.tier.en] += 1;
    return out;
  }

  // tierCounts over the completed passes only: how hard the passes were that the player did complete.
  function completedTierCounts(passes) {
    return tierCounts(passes.filter((m) => m.completed));
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
  const NOT_STARTED = 'Starts in the 2nd half.';
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
    buildPicker();
  }

  // draw everything again on the next render (entering Player mode, after Studio used the pitch)
  function invalidate() {
    if (!view) return;
    view.sig = '';
    $('passes').classList.remove('spot');
    closePicker(false);
  }

  // --- the picker: a pop-up list of both teams, the current player marked by the cyan bar ----------------

  // Two columns, home then away, each player in meta.players order; built once per bundle. A row: the shirt
  // number ("#27"), the name, the position, and on the right the passes so far (the column head says so).
  function buildPicker() {
    const { bundle } = view.api;
    const panel = $('player-picker');
    const cols = [bundle.meta.home, bundle.meta.away].map((team) => {
      const col = div('pk-col');
      col.setAttribute('role', 'group');
      const head = div('pk-head');
      head.id = 'pk-head-' + team.team_id;
      head.appendChild(div('pk-team', team.name));
      head.appendChild(div('pk-sub', 'Passes'));   // the column head of the count on the right
      col.setAttribute('aria-labelledby', head.id);
      col.appendChild(head);
      for (const pl of bundle.meta.players) {
        if (pl.team_id !== team.team_id) continue;
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'pk-opt';
        b.setAttribute('role', 'option');
        b.dataset.player = pl.player_id;
        b.tabIndex = -1;
        const num = document.createElement('span');   // the shirt number, "#27", muted (meta.players)
        num.className = 'pk-num';
        num.textContent = pl.shirt_number != null ? '#' + pl.shirt_number : '';
        const name = document.createElement('span');
        name.className = 'pk-name';
        name.textContent = pl.name;
        const pos = document.createElement('span');   // GK / DEF / MID / FWD from meta.players, muted
        pos.className = 'pk-pos';
        pos.textContent = pl.position || '';
        const n = document.createElement('span');
        n.className = 'pk-n';
        b.append(num, name, pos, n);
        col.appendChild(b);
      }
      return col;
    });
    panel.replaceChildren(...cols);
    const button = $('player-pick');
    button.addEventListener('click', () => (panel.hidden ? openPicker() : closePicker(true)));
    panel.addEventListener('click', (e) => {
      const opt = e.target.closest('.pk-opt');
      if (opt) choose(opt.dataset.player);
    });
    panel.addEventListener('keydown', pickerKey);
    // a press anywhere outside the list and its button closes it
    document.addEventListener('pointerdown', (e) => {
      if (!panel.hidden && !panel.contains(e.target) && !button.contains(e.target)) closePicker(false);
    }, true);
  }

  // passes so far (playerPasses, t <= now, whole match) and the current player, marked
  function refreshPicker() {
    const { bundle, now, player } = view.api.current();
    for (const b of $('player-picker').querySelectorAll('.pk-opt')) {
      const n = playerPasses(bundle.marks, b.dataset.player, now, null).length;
      const nText = String(n);
      const nNode = b.querySelector('.pk-n');
      if (nNode.textContent !== nText) nNode.textContent = nText;
      b.classList.toggle('none', n === 0);
      const on = b.dataset.player === player;
      b.setAttribute('aria-selected', String(on));
      const num = b.querySelector('.pk-num').textContent;
      b.setAttribute('aria-label', (num ? num + ' ' : '') + b.querySelector('.pk-name').textContent + ', '
        + n + (n === 1 ? ' pass' : ' passes') + ' so far');
    }
  }

  function openPicker() {
    const panel = $('player-picker');
    refreshPicker();
    panel.hidden = false;
    $('player-pick').setAttribute('aria-expanded', 'true');
    const current = panel.querySelector('.pk-opt[aria-selected="true"]') || panel.querySelector('.pk-opt');
    focusOption(current);
  }

  function closePicker(returnFocus) {
    const panel = $('player-picker');
    if (!panel || panel.hidden) return;
    panel.hidden = true;
    $('player-pick').setAttribute('aria-expanded', 'false');
    if (returnFocus) $('player-pick').focus();
  }

  function focusOption(opt) {
    if (!opt) return;
    for (const b of $('player-picker').querySelectorAll('.pk-opt')) b.tabIndex = b === opt ? 0 : -1;
    opt.focus();
  }

  function choose(playerId) {
    closePicker(true);   // first: setting the player redraws the mode, which also closes the list, without focus
    view.api.setPlayer(playerId);
  }

  // arrows move (up / down within a team, left / right across at the same row), Home / End, Enter or
  // space picks, Esc closes back to the button, Tab closes and moves on
  function pickerKey(e) {
    const opt = e.target.closest('.pk-opt');
    if (e.key === 'Escape') { closePicker(true); e.preventDefault(); e.stopPropagation(); return; }
    if (e.key === 'Tab') { closePicker(false); return; }
    if (!opt) return;
    const col = opt.parentElement;
    const opts = [...col.querySelectorAll('.pk-opt')];
    const i = opts.indexOf(opt);
    const cols = [...$('player-picker').querySelectorAll('.pk-col')];
    let target = null;
    if (e.key === 'ArrowDown') target = opts[Math.min(i + 1, opts.length - 1)];
    else if (e.key === 'ArrowUp') target = opts[Math.max(i - 1, 0)];
    else if (e.key === 'Home') target = opts[0];
    else if (e.key === 'End') target = opts[opts.length - 1];
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      const other = cols[(cols.indexOf(col) + 1) % cols.length].querySelectorAll('.pk-opt');
      target = other[Math.min(i, other.length - 1)];
    } else if (e.key === 'Enter' || e.key === ' ') {
      choose(opt.dataset.player);
      e.preventDefault();
      e.stopPropagation();
      return;
    } else return;
    focusOption(target);
    e.preventDefault();
    e.stopPropagation();
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

  // "43.52": the whole part at the value's size, ".52" in a smaller span (.dec), so the decimals stay quiet
  function setDecimal(id, text) {
    const node = $(id);
    if (node.textContent === text) return;
    const dot = text.indexOf('.');
    if (dot < 0) { node.textContent = text; return; }
    const dec = document.createElement('span');
    dec.className = 'dec';
    dec.textContent = text.slice(dot);
    node.replaceChildren(text.slice(0, dot), dec);
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

  // Player mode at the clock and in the scope: the name, the pass map, the numbers, the bands and
  // the moments. Redrawn only when the player, the scope or the player's passes change.
  // ctx: bundle, now, scope, playerId, notStarted.
  function render(ctx) {
    if (!$('player-picker').hidden) refreshPicker();
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
    setText('player-num-val', player && player.shirt_number != null ? String(player.shirt_number) : '');
    $('player-num').hidden = !(player && player.shirt_number != null);
    renderTab(player);
    renderMap(passes, player, stats, moments.length);
    renderNumbers(stats);
    renderTiers(completedTierCounts(passes));
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
  // small white dot at its start (inside the ring when it is also a surprise).
  function drawPass(parent, mark, newest) {
    const p = mark.pass_geometry;
    const surprise = Boolean(mark.moment_id);
    const g = el('g', surprise ? { 'data-moment': mark.moment_id, class: 'surprise' } : {});
    const width = surprise ? 0.6 : 0.32;
    const colour = mark.completed ? '#3BD16F' : '#FF4B47';
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
    if (newest) g.appendChild(el('circle', { cx: p.start_x, cy: p.start_y, r: 0.8, fill: '#FFFFFF', class: 'newest' }));
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
    setText('pv-completed', s ? String(s.completed) : '–');
    setText('pv-completed-pct', s ? api.percent(s.completionRate) + '%' : '');
    // Completed: the count and its rate over the passes; Expected: the sum of xP and its rate (the mean xP);
    // both Expected and vs expected to two decimals, as on the recap card
    setDecimal('pv-expected', s ? fixed2(s.expectedXp, false) : '–');
    setText('pv-expected-pct', s ? api.percent(s.meanXp) + '%' : '');
    const text = s ? fixed2(s.vsExpectedXp, true) : '–';
    setDecimal('pv-vs', text);
    $('pv-vs').classList.toggle('pos', text[0] === '+');
    $('pv-vs').classList.toggle('neg', text[0] === '−');
  }

  // one bar of the completed passes, a segment per tier as wide as its count; the five counts below it
  function renderTiers(counts) {
    const bar = $('pv-bandbar');
    bar.replaceChildren();
    TIERS.forEach((tier, i) => {
      setText('pv-tier-' + i, String(counts[tier]));
      if (counts[tier] === 0) return;
      const seg = div('band t' + i);
      seg.style.flexGrow = String(counts[tier]);
      bar.appendChild(seg);
    });
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

  const api = { scopePeriod, playerPasses, playerStats, fixed2, hardestCompleted, narratedFor, tierCounts,
    completedTierCounts, momentsFor,
    mount, invalidate, render };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.XplainPlayer = api;
})(typeof window !== 'undefined' ? window : this);
