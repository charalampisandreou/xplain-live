// xPlain Live: Producer mode (the control room). For the person who puts AI lines on air: why a line went
// out, what was checked, which model, how long. It reads bundle.trace (one step per agent per attempt) and
// shows bundle fields as stored: no numbers are computed here (latency, attempts and minutes are the bundle's;
// a check count is the length of the step's stored list).
//
// No spoilers: the log holds only the commentated moments with t <= now and the recaps the clock has reached
// (their times come from app.js, the same ones that open them in Studio). items() is pure and is checked
// against a Python twin in tests/test_producer_js.py.
//
// The checks a Verifier step lists are the ones it recorded when the bundle was built (step.checks, labels in
// trace.check_labels); a step without them says so and never guesses.
(function (root) {
  'use strict';

  const KIND_WORDS = { surprise_failure: 'Surprise failure', surprise_success: 'Surprise success' };
  const RECAP_WORDS = { half_time: 'Half-time recap', full_time: 'Full-time recap' };
  const RECAP_MINUTE = { half_time: 'HT', full_time: 'FT' };
  const AGENT_WORDS = {
    analyst: 'Analyst', explainer: 'Explainer', narrator: 'Narrator', verifier: 'Verifier',
    recap_editor: 'Recap Editor', recap_verifier: 'Recap Verifier', recap_filler: 'Numbers filled by code',
  };
  const GATE_WORDS = { ranking: 'top 3 of its kind so far this half', threshold: 'rare enough to always get a line' };
  const ON_AIR = ['accepted', 'retried'];
  const LANGS = ['en', 'es', 'el'];
  const EMPTY = 'This match was built without a record of the AI steps.';
  const NOT_RECORDED = 'Checks not recorded in this bundle.';
  const REFILL_NOTE = "filled again by refresh_recap_cards with the card's two-decimal values";
  const REFILLED = "First filled with whole numbers; refilled with the card's two-decimal values.";
  const FILLED_AGAIN = 'Filled again; the last fill is shown.';

  // --- pure functions (no DOM; also loaded by node in the test) -----------------------------------

  // a bundle with at least one traced moment or recap
  function hasTrace(bundle) {
    const t = bundle && bundle.trace;
    return Boolean(t) && (Object.keys(t.moments || {}).length + Object.keys(t.recaps || {}).length) > 0;
  }

  // The log, newest first: every commentated moment with t <= now, and each recap whose time (recapT[scope],
  // from app.js) the clock has reached. A recap sorts after a moment at the same t (it opens after that pass).
  function items(bundle, now, recapT) {
    const out = [];
    for (const m of bundle.moments) {
      const mo = m.bundle.moment;
      if (!m.bundle.qualifies_live || mo.t > now) continue;
      const n = m.narration || {};
      out.push({ type: 'moment', id: mo.moment_id, t: mo.t, order: 0, minute: mo.minute_label,
        who: mo.player_name || mo.player_id || 'Unknown player', what: KIND_WORDS[mo.kind] || mo.kind,
        onAir: ON_AIR.includes(n.status), attempts: n.attempts || 0 });
    }
    for (const scope of ['half_time', 'full_time']) {
      const entry = bundle.recaps && bundle.recaps[scope];
      const at = recapT ? recapT[scope] : undefined;
      if (!entry || at === undefined || at === null || at > now) continue;
      out.push({ type: 'recap', id: scope, t: at, order: 1, minute: RECAP_MINUTE[scope], who: RECAP_WORDS[scope],
        what: '', onAir: ON_AIR.includes(entry.status), attempts: entry.attempts || 0 });
    }
    return out.sort((a, b) => b.t - a.t || b.order - a.order);
  }

  // the steps of one log item, as stored
  function stepsOf(bundle, item) {
    const t = bundle.trace || {};
    const table = item.type === 'recap' ? t.recaps : t.moments;
    return (table && table[item.id]) || [];
  }

  // "name: detail" (a Verifier reason) as [label, detail] with the step's labels; other reasons as [reason, '']
  function splitReason(reason, labels) {
    const i = reason.indexOf(': ');
    const name = i > 0 ? reason.slice(0, i) : '';
    if (name && labels && labels[name]) return [labels[name], reason.slice(i + 2)];
    return [reason, ''];
  }

  // The recap's numbers in plain words, from its LAST recap_filler step (the text on air): each note
  // "{slot} = value" read as stored (no maths), the teams by their codes. A second line when an earlier filler ran:
  // the refresh's own words when its note says so. { text, again } or null when there is no filler step.
  function fillerSummary(steps, homeCode, awayCode) {
    const fillers = steps.filter((s) => s.agent === 'recap_filler');
    if (fillers.length === 0) return null;
    const last = fillers[fillers.length - 1];
    const slot = {};
    for (const n of last.notes) {
      const m = /^\{([a-z_]+)\} = (.*)$/.exec(n);
      if (m) slot[m[1]] = m[2];
    }
    const parts = [];
    if ('home_vs_expected' in slot) parts.push(homeCode + ' vs expected ' + slot.home_vs_expected);
    if ('away_vs_expected' in slot) parts.push(awayCode + ' vs expected ' + slot.away_vs_expected);
    if ('star_name' in slot) parts.push('Star ' + slot.star_name + ('star_pva' in slot ? ', ' + slot.star_pva : ''));
    const again = fillers.length < 2 ? '' : last.notes.includes(REFILL_NOTE) ? REFILLED : FILLED_AGAIN;
    return { text: parts.length ? parts.join(' · ') : 'No numbers in this text.', again };
  }

  // --- DOM ------------------------------------------------------------------------------------

  const $ = (id) => document.getElementById(id);
  let ctx = null;
  let drawn = { list: '', detail: '' };
  let openChecks = new Set();   // the open check lists, by "item id/step index"; reset when the item changes
  let openFor = null;

  function node(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function mount(options) {
    ctx = options;
    const list = $('pr-rows');
    const pick = (row) => {
      if (!row) return;
      if (row.dataset.type === 'recap') ctx.selectRecap(row.dataset.id); else ctx.selectMoment(row.dataset.id);
    };
    list.addEventListener('click', (e) => pick(e.target.closest('.row[data-id]')));
    list.addEventListener('keydown', (e) => {
      const row = e.target.closest('.row[data-id]');
      if (row && (e.key === 'Enter' || e.key === ' ')) { pick(row); e.preventDefault(); }
    });
    $('pr-detail').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-checks]');
      if (!b) return;
      const key = b.dataset.checks;
      if (openChecks.has(key)) openChecks.delete(key); else openChecks.add(key);
      drawn.detail = '';
      ctx.redraw();
    });
  }

  function invalidate() { drawn = { list: '', detail: '' }; }

  // args: { bundle, now, recapT, lang (the plate's NARRATION language), selected: { type, id } | null }
  function render(args) {
    const { bundle } = args;
    const empty = !hasTrace(bundle);
    $('pr-empty').hidden = !empty;
    $('pr-rows').hidden = empty;
    $('pr-detail').hidden = empty;
    if (empty) return;
    const log = items(bundle, args.now, args.recapT);
    const sel = args.selected;
    const selKey = sel ? sel.type + ':' + sel.id : '';
    const listKey = log.map((i) => i.id).join('|') + '/' + selKey;
    if (listKey !== drawn.list) {
      drawn.list = listKey;
      renderList(log, sel);
    }
    const item = sel && log.find((i) => i.type === sel.type && i.id === sel.id);
    if (selKey !== openFor) { openFor = selKey; openChecks = new Set(); }
    const detailKey = (item ? selKey : '-') + '/' + args.lang + '/' + [...openChecks].join(',');
    if (detailKey !== drawn.detail) {
      drawn.detail = detailKey;
      renderDetail(bundle, item || null, args.lang);
    }
  }

  function renderList(log, sel) {
    const rows = $('pr-rows');
    const scroll = rows.scrollTop;
    rows.replaceChildren();
    if (log.length === 0) rows.appendChild(node('div', 'empty', 'No commentated moment yet.'));
    for (const it of log) {
      const isSel = Boolean(sel) && sel.type === it.type && sel.id === it.id;
      const row = node('div', 'row pr-row' + (isSel ? ' sel' : ''));
      row.setAttribute('role', 'button');
      row.tabIndex = 0;
      row.dataset.type = it.type;
      row.dataset.id = it.id;
      row.setAttribute('aria-pressed', String(isSel));
      const who = node('div', 'who');
      who.append(node('div', 'player', it.who));
      if (it.what) who.append(node('div', 'meta', it.what));
      const status = node('span', 'pr-status' + (it.onAir ? ' on' : ''), it.onAir ? 'On air' : 'No line');
      const tries = node('span', 'pr-tries', it.attempts ? it.attempts + (it.attempts === 1 ? ' attempt' : ' attempts') : '');
      const retry = node('span', 'pr-retry', it.attempts > 1 ? 'Retried' : '');
      row.append(node('span', 'pr-min', it.type === 'recap' ? it.minute : it.minute + "'"), who, status, tries, retry);
      rows.appendChild(row);
    }
    rows.scrollTop = scroll;
  }

  function stepName(s) {
    if (s.agent === 'verifier' && s.final) return 'Final check';
    return AGENT_WORDS[s.agent] || s.agent;
  }

  function stepInfo(s, attempts) {
    const parts = [];
    if (s.kind === 'llm') {
      if (s.model) parts.push(s.model);
      if (s.latency_s !== null && s.latency_s !== undefined) parts.push(s.latency_s + ' s');
    }
    if (attempts > 1) parts.push('attempt ' + s.attempt);
    if (s.agent === 'analyst' && s.ok) parts.push(s.notes.map((n) => GATE_WORDS[n] || n).join(', '));
    if (s.agent === 'explainer' && s.ok) parts.push('drivers computed');
    if (s.agent === 'narrator' && s.ok && s.kind === 'llm') parts.push('line written');
    if (s.agent === 'recap_editor' && s.ok) parts.push('text written with blanks for code to fill the numbers');
    if (s.reused) parts.push('from an earlier build');
    return parts.filter(Boolean).join(' · ');
  }

  function renderDetail(bundle, item, lang) {
    const box = $('pr-detail');
    box.replaceChildren();
    if (!item) {
      box.appendChild(node('div', 'empty', 'Pick a line in the log.'));
      return;
    }
    const steps = stepsOf(bundle, item);
    const labels = (bundle.trace && bundle.trace.check_labels) || {};
    const head = node('div', 'thead');
    head.append(node('h2', '', 'Steps'), node('div', 'tsub', item.type === 'recap' ? item.who : item.minute + "' · " + item.who));
    box.appendChild(head);
    const list = node('div', 'pr-steps');
    if (steps.length === 0) list.appendChild(node('div', 'empty', 'No steps stored for this line.'));
    const lastFiller = steps.map((s) => s.agent).lastIndexOf('recap_filler');
    steps.forEach((s, i) => {
      if (s.agent === 'recap_filler' && i !== lastFiller) return;   // one step: the last fill, the text on air
      const row = node('div', 'pr-step' + (s.ok ? '' : ' fail'));
      const top = node('div', 'pr-step-top');
      top.append(node('span', 'pr-mark', s.ok ? '✓' : '✕'), node('span', 'pr-agent', stepName(s)),
        node('span', 'pr-kind', s.kind === 'llm' ? 'LLM' : 'Code'), node('span', 'pr-info', stepInfo(s, item.attempts)));
      row.appendChild(top);
      if (s.agent === 'verifier' || s.agent === 'recap_verifier') row.appendChild(checksBlock(s, labels[s.agent], item.id + '/' + i));
      else if (!s.ok && s.reasons.length) row.appendChild(reasonsBlock(s.reasons, null));
      else if (s.agent === 'recap_filler') {
        const f = fillerSummary(steps, ctx.teamCode(bundle.meta.home.team_id), ctx.teamCode(bundle.meta.away.team_id));
        const wrap = node('div', 'pr-checks');
        wrap.appendChild(node('div', 'pr-sum', f.text));
        if (f.again) wrap.appendChild(node('div', 'pr-note', f.again));
        row.appendChild(wrap);
      }
      list.appendChild(row);
    });
    box.appendChild(list);
    box.appendChild(onAir(bundle, item, lang));
  }

  // A Verifier step: its recorded checks (collapsed: "18 checks passed" and a button), or the note that they were
  // not recorded; a failed step always shows its reasons.
  function checksBlock(s, labels, key) {
    const wrap = node('div', 'pr-checks');
    if (!s.ok && s.reasons.length) wrap.appendChild(reasonsBlock(s.reasons, labels));
    if (!s.checks) {
      wrap.appendChild(node('div', 'pr-note', NOT_RECORDED));
      return wrap;
    }
    const open = openChecks.has(key);
    const line = node('div', 'pr-sum');
    const n = s.checks.length;
    line.append(node('span', '', s.ok ? n + (n === 1 ? ' check passed' : ' checks passed') : n + (n === 1 ? ' check ran' : ' checks ran')));
    const b = node('button', 'pr-toggle', open ? 'Hide checks' : 'Show checks');
    b.type = 'button';
    b.dataset.checks = key;
    b.setAttribute('aria-expanded', String(open));
    line.appendChild(b);
    wrap.appendChild(line);
    if (open) {
      const ul = node('ul', 'pr-checklist');
      for (const name of s.checks) ul.appendChild(node('li', '', (labels && labels[name]) || name));
      wrap.appendChild(ul);
    }
    return wrap;
  }

  function reasonsBlock(reasons, labels) {
    const ul = node('ul', 'pr-reasons');
    for (const r of reasons) {
      const [label, detail] = splitReason(r, labels);
      const li = node('li', '');
      li.append(node('b', '', label));
      if (detail) li.append(node('span', '', ' ' + detail));
      ul.appendChild(li);
    }
    return ul;
  }

  // the final line in the plate's NARRATION language (a recap: the filled text), or why none went out
  function onAir(bundle, item, lang) {
    const wrap = node('div', 'pr-onair');
    const code = LANGS.includes(lang) ? lang : 'en';
    const head = node('div', 'thead');
    head.append(node('h2', '', 'On air'), node('div', 'tsub', code.toUpperCase()));
    wrap.appendChild(head);
    let text = null, reasons = [];
    if (item.type === 'recap') {
      const e = bundle.recaps[item.id];
      if (e.text) text = e.text['text_' + code];
      reasons = e.reasons || [];
    } else {
      const m = bundle.moments.find((x) => x.bundle.moment.moment_id === item.id);
      if (m && m.line) text = m.line['line_' + code];
      reasons = (m && m.narration && m.narration.reasons) || [];
    }
    if (!text) {
      wrap.appendChild(node('div', 'empty', 'No line went on air.'));
      if (reasons.length) wrap.appendChild(reasonsBlock(reasons, null));
      return wrap;
    }
    const p = node('p', 'pr-text', text);
    p.setAttribute('lang', code);
    wrap.appendChild(p);
    return wrap;
  }

  const api = { hasTrace, items, stepsOf, splitReason, fillerSummary, mount, render, invalidate, EMPTY, NOT_RECORDED,
    REFILLED };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.XplainProducer = api;
})(typeof window !== 'undefined' ? window : this);
