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
//
// The log also lists the surprises that were not commentated (status "Not commentated"): no steps are stored for
// them, and Live retake writes them a line for the first time.
//
// Live retake (a moment line only): POST /api/live/retake narrates the moment again on Foundry and checks it with the
// Verifier on the server (src/xplain/retake.py). The steps then show the stored Analyst and Explainer steps ("from
// the match": not run again) and the server's Narrator and Verifier steps; the box under them shows the new line.
// A line that passed the Verifier waits for the producer's choice: "Put on air" (ctx.air: app.js shows it in the
// plate for that moment, in every tab, in this browser tab only, until the page reloads; nothing is stored) or "Keep
// stored line" ("Keep it off air" for an uncommentated moment). An aired retake shows in On air with its steps, and
// "Back to stored line" ("Take off air") undoes it. Model, latency,
// attempts and network retries are the server's, shown as sent.
//
// Edit line ("Write line" on a moment with no line): the producer edits the line on screen (the stored line, a
// retake's new line or a line already put on air) in EN / ES / EL, and Check sends it to POST /api/line/check, the
// Verifier's fact checks on the server (src/xplain/line_check.py; code only, no model). A blocked line shows its
// failing checks and stays off air; a line that passes waits for "Put on air" (ctx.air, as a retake, with edited:
// true: "On air · edited" in the log, a Producer step before its final Verifier step) or "Keep current line". Edit and
// Live retake close each other. The word count beside each box counts the text typed; it is not match data.
(function (root) {
  'use strict';

  const KIND_WORDS = { surprise_failure: 'Surprise failure', surprise_success: 'Surprise success' };
  const RECAP_WORDS = { half_time: 'Half-time recap', full_time: 'Full-time recap' };
  const RECAP_MINUTE = { half_time: 'HT', full_time: 'FT' };
  const AGENT_WORDS = {
    analyst: 'Analyst', explainer: 'Explainer', narrator: 'Narrator', verifier: 'Verifier',
    recap_editor: 'Recap Editor', recap_verifier: 'Recap Verifier', recap_filler: 'Numbers filled by code',
    producer: 'Producer',
  };
  const KIND_LABELS = { llm: 'LLM', code: 'Code', human: 'Human' };
  const GATE_WORDS = { ranking: 'top 3 of its kind so far this half', threshold: 'rare enough to always get a line' };
  const ON_AIR = ['accepted', 'retried'];
  const LANGS = ['en', 'es', 'el'];
  const EMPTY = 'This match was built without a record of the AI agents.';
  const NOT_RECORDED = 'Checks not recorded in this bundle.';
  const REFILL_NOTE = "filled again by refresh_recap_cards with the card's two-decimal values";
  const REFILLED = "First filled with whole numbers; refilled with the card's two-decimal values.";
  const FILLED_AGAIN = 'Filled again; the last fill is shown.';
  const RETAKE_OFF = 'Live retake is off on this server.';
  const RETAKE_UNREACHABLE = 'Could not reach the server.';
  const NOT_ON_AIR = 'Not on air yet: put it on air, or keep the stored line.';
  const NOT_ON_AIR_NC = 'Not on air yet: put it on air, or keep it off air.';
  const AIRED = 'Live retake, put on air in this browser tab; the match file keeps the stored line.';
  const AIRED_NC = 'Live retake, put on air in this browser tab; the match file has no line for this moment.';
  const NOT_COMMENTATED = "Not commentated: when it happened, it was not among the half's top 3 surprises of its kind.";
  const NO_STEPS_NC = 'No agent records: no line was written when it happened.';
  const FROM_MATCH = 'from the match';
  const CHECKED_AGAIN = 'checked again before air';
  const WORD_LIMIT = 24;   // verifier.WORD_LIMIT: past it the line may not fit the plate (a warning, never a block)
  const PASSED = 'Facts checked: all correct';
  const BLOCKED = 'Blocked: not aired';
  const NO_CHANGE = 'The same as the line on screen: nothing new to put on air.';
  const AIRED_EDIT = 'Edited line, put on air in this browser tab; the match file keeps the stored line.';
  const AIRED_EDIT_NC = 'Written line, put on air in this browser tab; the match file has no line for this moment.';
  const LANG_NAMES = { en: 'English', es: 'Spanish', el: 'Greek' };

  // --- pure functions (no DOM; also loaded by node in the test) -----------------------------------

  // a bundle with at least one traced moment or recap
  function hasTrace(bundle) {
    const t = bundle && bundle.trace;
    return Boolean(t) && (Object.keys(t.moments || {}).length + Object.keys(t.recaps || {}).length) > 0;
  }

  // The log, newest first: every moment with t <= now (commentated: false for a surprise that was not commentated),
  // and each recap whose time (recapT[scope], from app.js) the clock has reached. A recap sorts after a moment at the
  // same t (it opens after that pass).
  function items(bundle, now, recapT) {
    const out = [];
    for (const m of bundle.moments) {
      const mo = m.bundle.moment;
      if (mo.t > now) continue;
      const n = m.narration || {};
      out.push({ type: 'moment', id: mo.moment_id, t: mo.t, order: 0, minute: mo.minute_label,
        who: mo.player_name || mo.player_id || 'Unknown player', what: KIND_WORDS[mo.kind] || mo.kind,
        onAir: ON_AIR.includes(n.status), attempts: n.attempts || 0, commentated: Boolean(m.bundle.qualifies_live) });
    }
    for (const scope of ['half_time', 'full_time']) {
      const entry = bundle.recaps && bundle.recaps[scope];
      const at = recapT ? recapT[scope] : undefined;
      if (!entry || at === undefined || at === null || at > now) continue;
      out.push({ type: 'recap', id: scope, t: at, order: 1, minute: RECAP_MINUTE[scope], who: RECAP_WORDS[scope],
        what: '', onAir: ON_AIR.includes(entry.status), attempts: entry.attempts || 0, commentated: true });
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

  // The checks of a Verifier step that failed: the names its reasons start with ("name: detail"), among its checks.
  function failedChecks(step) {
    const ran = new Set(step.checks || []);
    const out = new Set();
    for (const r of step.reasons || []) {
      const i = r.indexOf(': ');
      if (i > 0 && ran.has(r.slice(0, i))) out.add(r.slice(0, i));
    }
    return [...out];
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

  // The final check folded into the Verifier step before it when both say the same (same result, same checks, no
  // reasons): one row, its check line saying CHECKED_AGAIN (again: true). Two rows whenever they differ, or when no
  // Verifier step precedes the final check (a reused line). The steps are not changed; a new list comes back.
  function foldFinal(steps) {
    const out = [];
    for (const s of steps) {
      const prev = out[out.length - 1];
      if (s.agent === 'verifier' && s.final && prev && prev.agent === 'verifier' && !prev.final &&
          prev.ok === s.ok && s.reasons.length === 0 && prev.reasons.length === 0 &&
          JSON.stringify(prev.checks) === JSON.stringify(s.checks)) {
        out[out.length - 1] = Object.assign({}, prev, { again: true });
        continue;
      }
      out.push(s);
    }
    return out;
  }

  // A retake's steps: the stored Analyst and Explainer steps (fromMatch: not run again), then the server's steps.
  function retakeSteps(stored, result) {
    const before = stored.filter((s) => s.agent === 'analyst' || s.agent === 'explainer')
      .map((s) => Object.assign({}, s, { fromMatch: true }));
    return before.concat(result.steps || []);
  }

  // The retake's summary line, the server's values as sent: model, latency, attempts, network retries (when any).
  function retakeSummary(result) {
    const parts = [result.model];
    if (result.latency_s !== null && result.latency_s !== undefined) parts.push(result.latency_s + ' s');
    if (result.attempts) parts.push(result.attempts + (result.attempts === 1 ? ' attempt' : ' attempts'));
    if (result.network_retries) {
      parts.push(result.network_retries + (result.network_retries === 1 ? ' network retry' : ' network retries'));
    }
    return parts.filter(Boolean).join(' · ');
  }

  // What the retake box says: { text, note, reasons }; text is the new line in lang (null when there is none).
  // commentated: false for a moment that has no stored line to keep.
  function retakeOutcome(result, lang, commentated = true) {
    const code = LANGS.includes(lang) ? lang : 'en';
    if (result.line) return { text: result.line[code], note: commentated ? NOT_ON_AIR : NOT_ON_AIR_NC, reasons: [] };
    if (result.status === 'rejected') {
      const n = result.attempts;
      return { text: null, note: 'Not aired: the Verifier rejected ' + (n === 1 ? 'the line.' : 'all ' + n + ' attempts.'),
        reasons: result.reasons || [] };
    }
    return { text: null, note: 'Retake failed at the ' + (result.stage || 'request') + ' stage.',
      reasons: result.message ? [result.message] : [] };
  }

  // The number of words in a box, as the Verifier counts them (Python's str.split(): runs of whitespace).
  function wordCount(text) {
    const t = (text || '').trim();
    return t ? t.split(/\s+/).length : 0;
  }

  const squash = (text) => (text || '').trim().split(/\s+/).join(' ');

  // The languages whose line differs from the one on screen (base, or null when there was none), whitespace aside.
  function changedLangs(base, lines) {
    return LANGS.filter((l) => squash(base ? base[l] : '') !== squash(lines[l]));
  }

  // The steps of an aired edit: the steps on screen when the edit began without their final check, then the server's
  // Producer step (with the edited languages) and its final Verifier step (with the labels of the checks it ran).
  function editSteps(base, result, edited, written) {
    const out = base.filter((s) => !(s.agent === 'verifier' && s.final));
    for (const s of result.steps || []) {
      if (s.agent === 'producer') out.push(Object.assign({}, s, { edited, written: Boolean(written) }));
      else if (s.agent === 'verifier') out.push(Object.assign({}, s, { labels: result.labels || {} }));
      else out.push(s);
    }
    return out;
  }

  // The aired edit's summary line: who wrote which languages.
  function editSummary(edited, written) {
    return (written ? 'Written by the producer' : 'Edited by the producer') +
      (edited.length ? ': ' + edited.map((l) => l.toUpperCase()).join(', ') : '') + ' · facts checked by code';
  }

  // What the edit box says after Check: { ok, title, fails: [{ name, label, details }], warnings }. The failing
  // checks in the order they ran, each with its reasons' details; warnings are the length notes (never a block).
  function editOutcome(result) {
    const labels = result.labels || {};
    const fails = [];
    for (const r of result.reasons || []) {
      const i = r.indexOf(': ');
      const name = i > 0 ? r.slice(0, i) : '';
      let f = fails.find((x) => x.name === name);
      if (!f) { f = { name, label: labels[name] || name || r, details: [] }; fails.push(f); }
      if (name) f.details.push(r.slice(i + 2));
    }
    const order = result.checks || [];
    fails.sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));
    const warnings = (result.notes || []).filter((n) => n.startsWith('length: ')).map((n) => n.slice(8));
    return { ok: Boolean(result.ok), title: result.ok ? PASSED : BLOCKED, fails, warnings };
  }

  // --- DOM ------------------------------------------------------------------------------------

  const $ = (id) => document.getElementById(id);
  let ctx = null;
  let drawn = { list: '', detail: '' };
  let openChecks = new Set();   // the open check lists, by "item id/step index"; reset when the item changes
  let openFor = null;
  // the Live retake of one moment: state idle | running | done | error; result (the server's answer) or error text
  let retake = { id: null, state: 'idle', result: null, error: '' };
  // the Edit line of one moment: state idle | editing | checking | done | error; lines (the boxes), base (the line on
  // screen when it began, or null), baseSteps (the steps then), written (no line before), result (the server's answer)
  const NO_EDIT = { id: null, state: 'idle', lines: null, base: null, baseSteps: [], written: false, result: null, error: '' };
  let edit = NO_EDIT;
  let lastBundle = null;

  function matchName() {   // as app.js picks the bundle
    const name = new URLSearchParams(location.search).get('match') || 'seed3001';
    return /^[a-z0-9_]+$/.test(name) ? name : 'seed3001';
  }

  async function runRetake(id) {
    retake = { id, state: 'running', result: null, error: '' };
    drawn.detail = '';
    ctx.redraw();
    let next;
    try {
      const res = await fetch('/api/live/retake', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ match: matchName(), moment_id: id }) });
      const body = await res.json().catch(() => ({}));
      if (res.ok) next = { id, state: 'done', result: body, error: '' };
      else if (res.status === 404 && !body.refused) next = { id, state: 'error', result: null, error: RETAKE_OFF };
      else next = { id, state: 'error', result: null, error: body.detail || 'Retake refused (HTTP ' + res.status + ').' };
    } catch (e) {
      next = { id, state: 'error', result: null, error: RETAKE_UNREACHABLE };
    }
    if (retake.id === id) retake = next;   // a reset meanwhile wins
    drawn.detail = '';
    ctx.redraw();
  }

  // The line on screen for a moment: an aired line, else a retake's new line, else the stored line; null when none.
  function currentLine(bundle, id) {
    const aired = ctx.aired(id);
    if (aired) return aired.line;
    if (retake.id === id && retake.state === 'done' && retake.result.line) return retake.result.line;
    const m = bundle.moments.find((x) => x.bundle.moment.moment_id === id);
    if (m && m.line) return { en: m.line.line_en, es: m.line.line_es, el: m.line.line_el };
    return null;
  }

  // The steps on screen for a moment, as renderDetail shows them (before folding).
  function currentSteps(bundle, item) {
    const stored = stepsOf(bundle, item);
    const aired = ctx.aired(item.id);
    if (aired && aired.edited) return aired.steps;
    if (aired) return retakeSteps(stored, aired);
    if (retake.id === item.id && retake.state === 'done' && retake.result.line) return retakeSteps(stored, retake.result);
    return stored;
  }

  function openEdit(id) {
    const base = currentLine(lastBundle, id);
    edit = { id, state: 'editing', lines: base ? Object.assign({}, base) : { en: '', es: '', el: '' }, base,
      baseSteps: currentSteps(lastBundle, { type: 'moment', id }), written: !base, result: null, error: '' };
    retake = { id: null, state: 'idle', result: null, error: '' };   // Edit closes a retake
  }

  async function runCheck() {
    const id = edit.id;
    edit = Object.assign({}, edit, { state: 'checking', result: null, error: '' });
    drawn.detail = '';
    ctx.redraw();
    let next;
    try {
      const res = await fetch('/api/line/check', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ match: matchName(), moment_id: id, line: edit.lines }) });
      const body = await res.json().catch(() => ({}));
      if (res.ok) next = { state: 'done', result: body, error: '' };
      else if (res.status === 422) next = { state: 'error', result: null, error: 'Each line holds at most 400 characters.' };
      else next = { state: 'error', result: null, error: body.detail || 'Check refused (HTTP ' + res.status + ').' };
    } catch (e) {
      next = { state: 'error', result: null, error: RETAKE_UNREACHABLE };
    }
    if (edit.id === id && edit.state === 'checking') edit = Object.assign({}, edit, next);   // a cancel meanwhile wins
    drawn.detail = '';
    ctx.redraw();
  }

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
      const a = e.target.closest('button[data-air]');
      if (a && a.dataset.air.startsWith('edit-')) {
        const id = a.dataset.id;
        if (a.dataset.air === 'edit-on' && edit.id === id && edit.result && edit.result.ok) {
          const edited = changedLangs(edit.base, edit.result.line);
          ctx.air(id, { line: edit.result.line, steps: editSteps(edit.baseSteps, edit.result, edited, edit.written),
            summary: editSummary(edited, edit.written), edited: true });
        }
        edit = NO_EDIT;   // "Put on air" and "Keep current line" both close the edit
        drawn = { list: '', detail: '' };
        ctx.redraw();
        return;
      }
      if (a) {
        const id = a.dataset.id;
        if (a.dataset.air === 'on' && retake.id === id && retake.result && retake.result.line) {
          ctx.air(id, { line: retake.result.line, steps: retake.result.steps || [], summary: retakeSummary(retake.result) });
        } else if (a.dataset.air === 'back') ctx.air(id, null);
        retake = { id: null, state: 'idle', result: null, error: '' };   // "on", "keep" and "back" all close the retake
        drawn = { list: '', detail: '' };
        ctx.redraw();
        return;
      }
      const ed = e.target.closest('button[data-edit]');
      if (ed) {
        const what = ed.dataset.edit;
        if (what === 'check') { if (edit.state !== 'checking') runCheck(); return; }
        if (what === 'cancel') edit = NO_EDIT;
        else if (retake.state !== 'running' && edit.state !== 'checking') openEdit(what);
        drawn.detail = '';
        ctx.redraw();
        return;
      }
      const r = e.target.closest('button[data-retake]');
      if (r) {
        if (r.dataset.retake === 'back') retake = { id: null, state: 'idle', result: null, error: '' };
        else if (retake.state !== 'running' && edit.state !== 'checking') {
          edit = NO_EDIT;   // Live retake closes an edit
          runRetake(r.dataset.retake);
          return;
        }
        drawn.detail = '';
        ctx.redraw();
        return;
      }
      const b = e.target.closest('button[data-checks]');
      if (!b) return;
      const key = b.dataset.checks;
      if (openChecks.has(key)) openChecks.delete(key); else openChecks.add(key);
      drawn.detail = '';
      ctx.redraw();
    });
    // typing in an edit box: no redraw (the box keeps its focus); the count updates, and a result shown is dropped
    $('pr-detail').addEventListener('input', (e) => {
      const box = e.target.closest('input[data-lang]');
      if (!box || !edit.lines) return;
      edit.lines[box.dataset.lang] = box.value;
      const count = box.parentNode.querySelector('.pr-edit-count');
      if (count) setCount(count, box.value);
      if (edit.state === 'done' || edit.state === 'error') {
        edit = Object.assign({}, edit, { state: 'editing', result: null, error: '' });
        const shown = $('pr-detail').querySelector('.pr-edit-result');
        if (shown) shown.remove();
        drawn.detail = detailKey(drawn.at);   // the panel on screen already matches this state
      }
    });
    $('pr-detail').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.target.closest('input[data-lang]') && edit.state !== 'checking') {
        e.preventDefault();
        runCheck();
      }
    });
  }

  function setCount(el, text) {
    const n = wordCount(text);
    el.textContent = n + ' / ' + WORD_LIMIT;
    el.classList.toggle('over', n > WORD_LIMIT);
  }

  function invalidate() { drawn = { list: '', detail: '' }; }

  // args: { bundle, now, recapT, lang (the plate's NARRATION language), selected: { type, id } | null }
  function render(args) {
    const { bundle } = args;
    lastBundle = bundle;
    const empty = !hasTrace(bundle);
    $('pr-empty').hidden = !empty;
    $('pr-rows').hidden = empty;
    $('pr-detail').hidden = empty;
    if (empty) return;
    const log = items(bundle, args.now, args.recapT);
    const sel = args.selected;
    const selKey = sel ? sel.type + ':' + sel.id : '';
    const listKey = log.map((i) => i.id + airedMark(i.id)).join('|') + '/' + selKey;
    if (listKey !== drawn.list) {
      drawn.list = listKey;
      renderList(log, sel);
    }
    const item = sel && log.find((i) => i.type === sel.type && i.id === sel.id);
    if (selKey !== openFor) { openFor = selKey; openChecks = new Set(); }
    const at = { selKey: item ? selKey : '-', lang: args.lang, id: item ? item.id : null };
    const key = detailKey(at);
    if (key !== drawn.detail) {
      drawn.detail = key;
      drawn.at = at;
      renderDetail(bundle, item || null, args.lang);
    }
  }

  function airedMark(id) {
    const a = ctx.aired(id);
    return a ? (a.edited ? '*e' : '*') : '';
  }

  // what the detail panel shows: the item, the language, the open check lists, the retake, the edit, an aired line
  function detailKey(at) {
    if (!at) return '';
    return at.selKey + '/' + at.lang + '/' + [...openChecks].join(',') + '/' + (retake.id || '') + ':' + retake.state +
      '/' + (edit.id || '') + ':' + edit.state + '/' + (at.id ? airedMark(at.id) : '');
  }

  function renderList(log, sel) {
    const rows = $('pr-rows');
    const scroll = rows.scrollTop;
    rows.replaceChildren();
    if (log.length === 0) rows.appendChild(node('div', 'empty', 'No surprise yet.'));
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
      const aired = it.type === 'moment' && ctx.aired(it.id);
      const status = node('span', 'pr-status' + (it.onAir || aired ? ' on' : it.commentated ? '' : ' nc'),
        aired ? (aired.edited ? 'On air · edited' : 'On air · retake') : it.onAir ? 'On air' : it.commentated ? 'No line' : 'Not commentated');
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
    if (s.agent === 'producer' && s.edited) {
      parts.push((s.written ? 'wrote ' : 'edited ') + s.edited.map((l) => l.toUpperCase()).join(', '));
    }
    if (s.reused) parts.push('from an earlier build');
    if (s.fromMatch) parts.push(FROM_MATCH);
    return parts.filter(Boolean).join(' · ');
  }

  function renderDetail(bundle, item, lang) {
    const box = $('pr-detail');
    box.replaceChildren();
    if (!item) {
      box.appendChild(node('div', 'empty', 'Pick a line in the log.'));
      return;
    }
    const mine = item.type === 'moment' && retake.id === item.id ? retake : null;
    const editing = item.type === 'moment' && edit.id === item.id && edit.state !== 'idle' ? edit : null;
    const stored = stepsOf(bundle, item);
    const aired = item.type === 'moment' ? ctx.aired(item.id) : null;
    const shown = mine && mine.state === 'done' ? mine.result : aired;
    // while editing: the steps on screen when the edit began; an aired edit: its own steps (the retake's included)
    const steps = foldFinal(editing ? editing.baseSteps : shown && shown === aired && aired.edited ? aired.steps
      : shown ? retakeSteps(stored, shown) : stored);
    const labels = (bundle.trace && bundle.trace.check_labels) || {};
    const head = node('div', 'thead');
    head.append(node('h2', '', mine && mine.state === 'done' ? 'Live retake' : 'Agents'),
      node('div', 'tsub', item.type === 'recap' ? item.who : item.minute + "' · " + item.who));
    if (item.type === 'moment') {
      head.appendChild(retakeButtons(item.id, mine, item.commentated, Boolean(currentLine(bundle, item.id)), editing));
    }
    box.appendChild(head);
    const list = node('div', 'pr-steps');
    if (steps.length === 0) {
      list.appendChild(node('div', 'empty', item.commentated ? 'No agent records stored for this line.' : NO_STEPS_NC));
    }
    const lastFiller = steps.map((s) => s.agent).lastIndexOf('recap_filler');
    steps.forEach((s, i) => {
      if (s.agent === 'recap_filler' && i !== lastFiller) return;   // one step: the last fill, the text on air
      const row = node('div', 'pr-step' + (s.ok ? '' : ' fail'));
      const top = node('div', 'pr-step-top');
      top.append(node('span', 'pr-mark', s.ok ? '✓' : '✕'), node('span', 'pr-agent', stepName(s)),
        node('span', 'pr-kind', KIND_LABELS[s.kind] || s.kind), node('span', 'pr-info', stepInfo(s, item.attempts)));
      row.appendChild(top);
      if (s.agent === 'verifier' || s.agent === 'recap_verifier') {
        row.appendChild(checksBlock(s, s.labels || labels[s.agent], item.id + '/' + i));   // an edit's carry their own
      }
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
    if (mine && mine.state === 'running') {
      const row = node('div', 'pr-step pending');
      const top = node('div', 'pr-step-top');
      top.append(node('span', 'pr-mark', '…'), node('span', 'pr-agent', 'Narrator and Verifier'),
        node('span', 'pr-kind', 'LLM'), node('span', 'pr-info', 'writing and checking on Foundry'));
      row.appendChild(top);
      list.appendChild(row);
    }
    box.appendChild(list);
    if (editing) {
      box.appendChild(editBox(item.id, editing));
      const said = box.querySelector('.pr-edit-result');   // the verdict scrolls into the slot, under the steps
      if (said) said.scrollIntoView({ block: 'nearest' });
    } else if (mine && mine.state === 'done') box.appendChild(retakeBox(item.id, mine.result, lang, item.commentated));
    else {
      if (mine && mine.state === 'error') box.appendChild(node('div', 'pr-note pr-retake-error', mine.error));
      box.appendChild(onAir(bundle, item, lang));
    }
  }

  // "Edit line" ("Write line" with no line on screen; hidden while editing), "Live retake" (again while a result is
  // shown, disabled while one runs or a check runs) and, with a result that has no line, "Stored line" ("Close" for
  // an uncommentated moment)
  function retakeButtons(id, mine, commentated, hasLine, editing) {
    const wrap = node('div', 'pr-retake-actions');
    if (!editing) {
      const e = node('button', 'pr-retake', hasLine ? 'Edit line' : 'Write line');
      e.type = 'button';
      e.dataset.edit = id;
      e.disabled = retake.state === 'running';
      wrap.appendChild(e);
    }
    if (mine && mine.state === 'done' && !mine.result.line) {   // with a new line, "Keep stored line" says it
      const back = node('button', 'pr-toggle', commentated ? 'Stored line' : 'Close');
      back.type = 'button';
      back.dataset.retake = 'back';
      wrap.appendChild(back);
    }
    const running = retake.state === 'running';
    const b = node('button', 'pr-retake', running && retake.id === id ? 'Retaking…' : 'Live retake');
    b.type = 'button';
    b.dataset.retake = id;
    b.disabled = running || edit.state === 'checking';
    wrap.appendChild(b);
    return wrap;
  }

  // the retake's line in the plate's NARRATION language (never on air), or why there is none
  function retakeBox(id, result, lang, commentated) {
    const wrap = node('div', 'pr-onair');
    const code = LANGS.includes(lang) ? lang : 'en';
    const head = node('div', 'thead');
    head.append(node('h2', '', 'New line'), node('div', 'tsub', code.toUpperCase()));
    wrap.appendChild(head);
    wrap.appendChild(node('div', 'pr-note', retakeSummary(result)));
    const o = retakeOutcome(result, lang, commentated);
    if (o.text) {
      const p = node('p', 'pr-text', o.text);
      p.setAttribute('lang', code);
      wrap.appendChild(p);
    }
    wrap.appendChild(node('div', 'pr-note', o.note));
    if (o.reasons.length) wrap.appendChild(reasonsBlock(o.reasons, null));
    if (o.text) {
      wrap.appendChild(airButtons(id, [['on', 'Put on air', 'pr-retake'],
        ['keep', commentated ? 'Keep stored line' : 'Keep it off air', 'pr-toggle']]));
    }
    return wrap;
  }

  // The edit panel: a box per language (prefilled with the line on screen) with its word count, Check / Cancel, and
  // after a check what the Verifier said: the failing checks (blocked) or Put on air / Keep current line (passed).
  function editBox(id, ed) {
    const wrap = node('div', 'pr-onair pr-edit');
    const head = node('div', 'thead');
    head.append(node('h2', '', ed.written ? 'Write line' : 'Edit line'), node('div', 'tsub', 'Checked by code before air'));
    wrap.appendChild(head);
    for (const lang of LANGS) {
      const row = node('label', 'pr-edit-row');
      const input = node('input', 'pr-edit-input');
      input.type = 'text';
      input.value = ed.lines[lang] || '';
      input.maxLength = 400;
      input.spellcheck = false;
      input.dataset.lang = lang;
      input.setAttribute('lang', lang);
      input.setAttribute('aria-label', 'Line in ' + LANG_NAMES[lang]);
      input.disabled = ed.state === 'checking';
      const count = node('span', 'pr-edit-count');
      setCount(count, input.value);
      row.append(node('span', 'pr-edit-lang', lang.toUpperCase()), input, count);
      wrap.appendChild(row);
    }
    const actions = node('div', 'pr-air');
    const check = node('button', 'pr-retake', ed.state === 'checking' ? 'Checking…' : 'Check');
    check.type = 'button';
    check.dataset.edit = 'check';
    check.disabled = ed.state === 'checking';
    const cancel = node('button', 'pr-toggle', 'Cancel');
    cancel.type = 'button';
    cancel.dataset.edit = 'cancel';
    actions.append(check, cancel);
    wrap.appendChild(actions);
    if (ed.state === 'error') wrap.appendChild(node('div', 'pr-note pr-retake-error pr-edit-result', ed.error));
    if (ed.state === 'done') wrap.appendChild(editResult(id, ed));
    return wrap;
  }

  function editResult(id, ed) {
    const o = editOutcome(ed.result);
    const wrap = node('div', 'pr-edit-result');
    wrap.appendChild(node('div', 'pr-edit-verdict ' + (o.ok ? 'pass' : 'blocked'), o.title));
    if (o.fails.length) {
      const ul = node('ul', 'pr-edit-fails');
      for (const f of o.fails) {
        const li = node('li', '');
        li.appendChild(node('b', '', f.label));
        for (const d of f.details) li.appendChild(node('div', 'pr-edit-detail', d));
        ul.appendChild(li);
      }
      wrap.appendChild(ul);
    }
    for (const w of o.warnings) wrap.appendChild(node('div', 'pr-note', 'Warning: ' + w));
    if (o.ok) {
      if (changedLangs(ed.base, ed.result.line).length) {
        wrap.appendChild(airButtons(id, [['edit-on', 'Put on air', 'pr-retake'], ['edit-keep', 'Keep current line', 'pr-toggle']]));
      } else {
        wrap.appendChild(node('div', 'pr-note', NO_CHANGE));
      }
    }
    return wrap;
  }

  // the producer's choice under a line: buttons [action, words, class] with data-air and the moment's id
  function airButtons(id, buttons) {
    const row = node('div', 'pr-air');
    for (const [action, words, cls] of buttons) {
      const b = node('button', cls, words);
      b.type = 'button';
      b.dataset.air = action;
      b.dataset.id = id;
      row.appendChild(b);
    }
    return row;
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
    line.append(node('span', '', (s.ok ? n + (n === 1 ? ' check passed' : ' checks passed') : n + (n === 1 ? ' check ran' : ' checks ran')) +
      (s.again ? ', ' + CHECKED_AGAIN : '')));
    const b = node('button', 'pr-toggle', open ? 'Hide checks' : 'Show checks');
    b.type = 'button';
    b.dataset.checks = key;
    b.setAttribute('aria-expanded', String(open));
    line.appendChild(b);
    wrap.appendChild(line);
    if (open) {
      const ul = node('ul', 'pr-checklist');
      const failed = new Set(failedChecks(s));
      for (const name of s.checks) ul.appendChild(node('li', failed.has(name) ? 'fail' : '', (labels && labels[name]) || name));
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
      const aired = ctx.aired(item.id);
      if (aired) {
        const p = node('p', 'pr-text', aired.line[code]);
        p.setAttribute('lang', code);
        const said = aired.edited ? (item.commentated ? AIRED_EDIT : AIRED_EDIT_NC) : (item.commentated ? AIRED : AIRED_NC);
        wrap.append(node('div', 'pr-note', aired.summary), p,
          node('div', 'pr-note', said), airButtons(item.id, [['back', item.commentated ? 'Back to stored line' : 'Take off air', 'pr-toggle']]));
        return wrap;
      }
      const m = bundle.moments.find((x) => x.bundle.moment.moment_id === item.id);
      if (m && m.line) text = m.line['line_' + code];
      reasons = (m && m.narration && m.narration.reasons) || [];
    }
    if (!text) {
      wrap.appendChild(node('div', 'empty', item.commentated ? 'No line went on air.' : NOT_COMMENTATED));
      if (reasons.length) wrap.appendChild(reasonsBlock(reasons, null));
      return wrap;
    }
    const p = node('p', 'pr-text', text);
    p.setAttribute('lang', code);
    wrap.appendChild(p);
    return wrap;
  }

  const api = { hasTrace, items, stepsOf, splitReason, fillerSummary, foldFinal, failedChecks, retakeSteps, retakeSummary,
    retakeOutcome, wordCount, changedLangs, editSteps, editSummary, editOutcome, mount, render, invalidate, EMPTY,
    NOT_RECORDED, REFILLED, NOT_ON_AIR, NOT_ON_AIR_NC, AIRED, AIRED_NC, AIRED_EDIT, AIRED_EDIT_NC, NOT_COMMENTATED,
    FROM_MATCH, WORD_LIMIT, PASSED, BLOCKED };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.XplainProducer = api;
})(typeof window !== 'undefined' ? window : this);
