"""Small operator console for the real persisted MVP records.

This is intentionally not a marketing page or a demo simulator. It is a thin
read-only view over the authenticated API: queue, order reasoning, gate inputs,
messages, outcomes, and the live ledger report. It contains no fabricated data.
"""

from __future__ import annotations

CONSOLE_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PreCourier Operations</title>
  <style>
    :root { color-scheme: light; --ink:#10233f; --muted:#61738d; --line:#dbe4ef;
      --blue:#1464f4; --green:#0b9b72; --red:#c74444; --paper:#f5f8fc; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--paper); color:var(--ink); font:14px/1.5 Inter,
      ui-sans-serif, system-ui, -apple-system, sans-serif; }
    header { background:#10233f; color:#fff; padding:20px max(20px, calc((100vw - 1240px)/2));
      display:flex; justify-content:space-between; gap:20px; align-items:center; }
    header h1 { font-size:18px; letter-spacing:.02em; margin:0; }
    header p { color:#b7c7db; margin:3px 0 0; font-size:12px; }
    .auth { display:flex; gap:8px; max-width:430px; width:100%; }
    input, button { border:1px solid var(--line); border-radius:7px; padding:9px 11px;
      font:inherit; }
    input { min-width:0; flex:1; }
    button { background:var(--blue); border-color:var(--blue); color:#fff; cursor:pointer;
      font-weight:650; }
    button.secondary { background:#fff; color:var(--ink); border-color:var(--line); }
    main { max-width:1240px; margin:24px auto; padding:0 20px; }
    .status { color:var(--muted); min-height:24px; }
    .metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:12px 0 20px; }
    .metric, .panel { background:#fff; border:1px solid var(--line); border-radius:10px; }
    .metric { padding:16px; }
    .metric small, .eyebrow { color:var(--muted); text-transform:uppercase; letter-spacing:.09em;
      font-size:10px; font-weight:750; }
    .metric strong { display:block; font-size:25px; margin-top:4px; }
    .workspace { display:grid; grid-template-columns:340px 1fr; gap:16px; align-items:start; }
    .panel { overflow:hidden; }
    .panel-head { border-bottom:1px solid var(--line); padding:14px 16px; display:flex;
      justify-content:space-between; align-items:center; }
    .panel-head h2 { font-size:14px; margin:0; }
    .queue { max-height:640px; overflow:auto; }
    .queue button { width:100%; border:0; border-bottom:1px solid var(--line); border-radius:0;
      color:var(--ink); background:#fff; text-align:left; font-weight:500; padding:13px 16px; }
    .queue button:hover, .queue button.active { background:#edf4ff; }
    .queue .id { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }
    .queue .sub { display:flex; justify-content:space-between; color:var(--muted); margin-top:3px;
      font-size:12px; }
    .detail { padding:18px; }
    .detail h2 { margin:0 0 2px; font-size:20px; }
    .detail .meta { color:var(--muted); margin-bottom:18px; }
    .badges { display:flex; flex-wrap:wrap; gap:7px; margin:10px 0 18px; }
    .badge { background:#eef3f9; border-radius:999px; padding:4px 9px; font-size:12px; }
    .badge.green { color:var(--green); background:#e7f8f1; }
    .badge.red { color:var(--red); background:#fff0f0; }
    .grid { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }
    .card { background:#f8fafc; border:1px solid var(--line); border-radius:8px; padding:13px; }
    .card h3 { font-size:12px; margin:0 0 9px; text-transform:uppercase; letter-spacing:.07em; }
    dl { margin:0; display:grid; grid-template-columns:1fr auto; gap:5px 10px; }
    dt { color:var(--muted); } dd { margin:0; text-align:right; font-family:ui-monospace,
      SFMono-Regular, Menlo, monospace; }
    pre { white-space:pre-wrap; overflow:auto; margin:0; color:#30445e; font:12px/1.55 ui-monospace,
      SFMono-Regular, Menlo, monospace; }
    .timeline { list-style:none; padding:0; margin:0; }
    .timeline li { border-left:2px solid #cbd8e8; padding:0 0 12px 12px; margin-left:4px; }
    .timeline b { font-size:12px; } .timeline small { color:var(--muted); display:block; }
    .empty { color:var(--muted); padding:24px 16px; text-align:center; }
    @media (max-width:800px) { header { display:block; } .auth { margin-top:14px; }
      .metrics { grid-template-columns:repeat(2,1fr); } .workspace { grid-template-columns:1fr; }
      .queue { max-height:260px; } .grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <div><h1>PreCourier / Operations</h1><p>Pre-dispatch decision record for merchant operations</p></div>
    <form class="auth" id="auth-form"><input id="api-key" type="password" autocomplete="off"
      placeholder="Bearer API key"><button type="submit">Connect</button></form>
  </header>
  <main>
    <div class="status" id="status">Enter an API key to load this merchant's queue.</div>
    <section class="metrics" aria-label="Merchant metrics">
      <div class="metric"><small>Orders in queue</small><strong id="queue-count">-</strong></div>
      <div class="metric"><small>Realised ledger</small><strong id="ledger-net">-</strong></div>
      <div class="metric"><small>Action spend</small><strong id="action-spend">-</strong></div>
      <div class="metric"><small>Audit chain</small><strong id="audit-status">-</strong></div>
    </section>
    <section class="workspace">
      <div class="panel"><div class="panel-head"><h2>Decision queue</h2>
        <button class="secondary" id="refresh" type="button">Refresh</button></div>
        <div class="queue" id="queue"><div class="empty">No merchant connected.</div></div>
      </div>
      <div class="panel"><div id="detail" class="detail"><div class="empty">Select an order to inspect its evidence.</div></div></div>
    </section>
  </main>
  <script>
    (() => {
      const state = { key: sessionStorage.getItem('precourier_api_key') || '', selected: '' };
      const $ = (id) => document.getElementById(id);
      const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) =>
        ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
      const money = (paise) => `₹${(Number(paise || 0) / 100).toLocaleString('en-IN',
        {minimumFractionDigits:2, maximumFractionDigits:2})}`;
      const headers = () => ({'Authorization': `Bearer ${state.key}`});
      async function get(path) {
        const response = await fetch(path, {headers: headers()});
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || body.error || `HTTP ${response.status}`);
        return body;
      }
      function setStatus(message, bad = false) { $('status').textContent = message;
        $('status').style.color = bad ? '#c74444' : ''; }
      function renderQueue(body) {
        const items = body.orders || body.awaiting || [];
        $('queue-count').textContent = items.length;
        $('queue').innerHTML = items.length ? items.map((item) =>
          `<button type="button" data-id="${esc(item.order_id)}" class="${item.order_id === state.selected ? 'active' : ''}">
            <span class="id">${esc(item.order_id)}</span><span class="sub"><span>${esc(item.state)}</span><span>${money(item.value_paise)}</span></span></button>`).join('')
          : '<div class="empty">No held orders.</div>';
        $('queue').querySelectorAll('button[data-id]').forEach((button) =>
          button.addEventListener('click', () => loadOrder(button.dataset.id)));
      }
      function renderDetail(order) {
        const latest = (order.decisions || []).slice(-1)[0] || {};
        const score = (order.risk_scores || []).slice(-1)[0] || {};
        const diagnosis = (order.diagnoses || []).slice(-1)[0] || {};
        const gate = latest.gate_inputs || {};
        const messages = order.messages || [];
        const transitions = order.transitions || [];
        $('detail').innerHTML = `<h2>${esc(order.order_id)}</h2>
          <div class="meta">${esc(order.external_id || 'No merchant reference')} · ${esc(order.placed_at)}</div>
          <div class="badges"><span class="badge">${esc(order.state)}</span><span class="badge">${esc(order.mode)}</span>
            <span class="badge ${score.p_rto_ppm >= 500000 ? 'red' : 'green'}">RTO ${esc(score.p_rto_ppm ? (score.p_rto_ppm / 10000).toFixed(1) + '%' : 'not scored')}</span>
            <span class="badge">Cause ${esc(diagnosis.cause || 'uncertain')}</span></div>
          <div class="grid"><div class="card"><h3>Gate arithmetic</h3><dl>
            <dt>Margin M</dt><dd>${money(gate.margin_paise)}</dd><dt>RTO loss L</dt><dd>${money(gate.loss_paise)}</dd>
            <dt>Swing</dt><dd>${money(gate.swing_paise)}</dd><dt>Threshold</dt><dd>${esc(latest.threshold_ppm || 0)} ppm</dd>
            <dt>Expected delta</dt><dd>${esc(latest.expected_delta_micro || 0)} micro</dd><dt>Action</dt><dd>${esc(latest.action || 'no_action')}</dd>
          </dl><p>${esc(latest.reason || 'No decision recorded.')}</p></div>
          <div class="card"><h3>Address evidence</h3><pre>${esc((diagnosis.evidence || []).map((e) => `${e.code}: ${e.detail}`).join('\n') || 'No diagnosis evidence.')}</pre>
            <p><b>Current:</b> ${esc(order.address?.text || '')}<br><b>PIN:</b> ${esc(order.address?.pincode || '')}</p></div>
          <div class="card"><h3>Conversation</h3>${messages.length ? `<pre>${esc(messages.map((m) => `${m.direction} [${m.channel}]\n${m.body}`).join('\n\n'))}</pre>` : '<p>No messages recorded.</p>'}</div>
          <div class="card"><h3>Lifecycle</h3><ol class="timeline">${transitions.map((t) => `<li><b>${esc(t.from)} → ${esc(t.to)}</b><small>${esc(t.at)} · ${esc(t.reason)}</small></li>`).join('') || '<li>No transitions.</li>'}</ol></div></div>`;
      }
      async function loadOrder(id) { state.selected = id; try {
        const order = await get(`/v1/orders/${encodeURIComponent(id)}`); renderDetail(order); await loadQueue();
      } catch (error) { setStatus(error.message, true); } }
      async function loadQueue() { const body = await get('/v1/orders'); renderQueue(body);
        const report = await get('/v1/reports/roi'); $('ledger-net').textContent = money(report.net_paise);
        $('action-spend').textContent = money(report.action_spend_paise);
        const audit = await get('/v1/audit'); $('audit-status').textContent = audit.verification?.ok ? 'verified' : 'broken';
        if (state.selected) { const order = await get(`/v1/orders/${encodeURIComponent(state.selected)}`); renderDetail(order); }
        setStatus(`Connected. ${body.orders?.length || 0} recent order(s) loaded.`);
      }
      $('auth-form').addEventListener('submit', async (event) => { event.preventDefault();
        state.key = $('api-key').value.trim(); if (!state.key) return setStatus('API key is required.', true);
        sessionStorage.setItem('precourier_api_key', state.key); try { await loadQueue(); }
        catch (error) { setStatus(error.message, true); } });
      $('refresh').addEventListener('click', () => loadQueue().catch((error) => setStatus(error.message, true)));
      if (state.key) { $('api-key').value = state.key; loadQueue().catch((error) => setStatus(error.message, true)); }
    })();
  </script>
</body>
</html>'''


__all__ = ["CONSOLE_HTML"]
