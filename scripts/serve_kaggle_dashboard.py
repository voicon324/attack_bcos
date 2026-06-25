#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kaggle CamoPatch Dashboard</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dee7;
      --text: #1b1f24;
      --muted: #59636f;
      --ok: #12805c;
      --run: #0969da;
      --queue: #6e7781;
      --fail: #cf222e;
      --warn: #9a6700;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 18px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { margin: 0 0 8px; font-size: 20px; font-weight: 650; }
    .topline { display: flex; gap: 12px; flex-wrap: wrap; color: var(--muted); }
    main { padding: 18px 24px 32px; max-width: 1500px; margin: 0 auto; }
    .grid { display: grid; gap: 14px; }
    .cards { grid-template-columns: repeat(6, minmax(120px, 1fr)); }
    .card, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .card { padding: 14px; }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    .value { font-size: 26px; font-weight: 700; margin-top: 3px; }
    .bar {
      display: flex;
      height: 12px;
      overflow: hidden;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #eef1f5;
      margin: 16px 0;
    }
    .seg.done { background: var(--ok); }
    .seg.running { background: var(--run); }
    .seg.queued { background: var(--queue); }
    .seg.failed { background: var(--fail); }
    .panel { padding: 14px; margin-top: 14px; }
    .panel h2 { margin: 0 0 10px; font-size: 15px; }
    .toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 10px;
    }
    select, button, a.button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      padding: 7px 10px;
      text-decoration: none;
      font: inherit;
    }
    button { cursor: pointer; }
    a.button { display: inline-block; }
    table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    th, td {
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 8px 7px;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th { color: var(--muted); font-size: 12px; font-weight: 650; }
    .status {
      display: inline-block;
      min-width: 76px;
      padding: 2px 7px;
      border-radius: 999px;
      color: white;
      text-align: center;
      font-size: 12px;
      font-weight: 650;
    }
    .status.done { background: var(--ok); }
    .status.running { background: var(--run); }
    .status.queued { background: var(--queue); }
    .status.failed { background: var(--fail); }
    .status.downloading, .status.submitting { background: var(--warn); }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      color: #24292f;
      background: #f0f3f6;
      border-radius: 6px;
      padding: 10px;
      max-height: 240px;
      overflow: auto;
    }
    .muted { color: var(--muted); }
    .ok { color: var(--ok); }
    .fail { color: var(--fail); }
    .two { grid-template-columns: 1fr 1fr; }
    @media (max-width: 900px) {
      .cards, .two { grid-template-columns: 1fr 1fr; }
      th:nth-child(7), td:nth-child(7), th:nth-child(8), td:nth-child(8) { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Kaggle CamoPatch Dashboard</h1>
    <div class="topline">
      <span id="runRoot"></span>
      <span id="updatedAt"></span>
      <span id="statusText"></span>
    </div>
  </header>
  <main>
    <section class="grid cards" id="cards"></section>
    <div class="bar" id="bar"></div>

    <section class="panel">
      <div class="toolbar">
        <button id="refreshBtn">Refresh</button>
        <select id="statusFilter">
          <option value="active">Active first</option>
          <option value="all">All jobs</option>
          <option value="running">Running</option>
          <option value="done">Done</option>
          <option value="queued">Queued</option>
          <option value="failed">Failed</option>
        </select>
        <a class="button" href="/charts">Charts</a>
        <a class="button" href="/download/results.zip">Download results zip</a>
        <span class="muted" id="manifestText"></span>
      </div>
      <table>
        <thead>
          <tr>
            <th style="width: 28%">Job</th>
            <th style="width: 8%">Status</th>
            <th style="width: 8%">Account</th>
            <th style="width: 8%">Result</th>
            <th style="width: 7%">ASR</th>
            <th style="width: 8%">Elapsed</th>
            <th style="width: 9%">Age</th>
            <th style="width: 12%">Last Check</th>
            <th style="width: 12%">Reason / Link</th>
          </tr>
        </thead>
        <tbody id="jobs"></tbody>
      </table>
    </section>

    <section class="grid two">
      <div class="panel">
        <h2>Accounts</h2>
        <table><tbody id="accounts"></tbody></table>
      </div>
      <div class="panel">
        <h2>Recent Progress</h2>
        <pre id="progressLog"></pre>
      </div>
    </section>
  </main>
  <script>
    const refreshMs = 10000;
    const order = {failed: 0, downloading: 1, submitting: 2, running: 3, queued: 4, done: 5};

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    function fmtAge(minutes) {
      if (minutes === null || minutes === undefined || minutes === '') return '';
      if (minutes < 60) return `${minutes.toFixed(1)}m`;
      return `${(minutes / 60).toFixed(2)}h`;
    }
    function fmtResult(value) {
      return value === null || value === undefined || value === '' ? 'N/A' : String(value);
    }
    function fmtHours(value) {
      if (value === null || value === undefined || value === '') return 'N/A';
      const n = Number(value);
      if (!Number.isFinite(n)) return 'N/A';
      return `${n.toFixed(2)}h`;
    }
    function card(label, value, cls='') {
      return `<div class="card"><div class="label">${esc(label)}</div><div class="value ${cls}">${esc(value)}</div></div>`;
    }
    function render(data) {
      const counts = data.counts || {};
      const total = data.total_jobs || 0;
      document.getElementById('runRoot').textContent = `run root: ${data.run_root}`;
      document.getElementById('updatedAt').textContent = `updated: ${new Date().toLocaleString()}`;
      document.getElementById('statusText').textContent = data.all_done ? 'all done' : 'running';
      document.getElementById('cards').innerHTML = [
        card('Total', total),
        card('Done', counts.done || 0, 'ok'),
        card('Running', counts.running || 0),
        card('Queued', counts.queued || 0),
        card('Failed', counts.failed || 0, (counts.failed || 0) ? 'fail' : ''),
        card('Rows', data.manifest?.summary_rows || 0),
      ].join('');
      document.getElementById('bar').innerHTML = ['done','running','queued','failed'].map(k => {
        const n = counts[k] || 0;
        const w = total ? Math.max(n / total * 100, n ? 1 : 0) : 0;
        return `<div class="seg ${k}" style="width:${w}%;" title="${k}: ${n}"></div>`;
      }).join('');
      const manifest = data.manifest || {};
      document.getElementById('manifestText').textContent =
        `zip: ${manifest.generated_at || 'none'} | success_by_query=${manifest.success_by_query_rows || 0}`;
      renderAccounts(data.by_account || {}, data.account_quota_estimates || {});
      renderJobs(data.jobs || []);
      document.getElementById('progressLog').textContent = (data.progress_tail || []).join('\n');
    }
    function renderAccounts(byAccount, quotaEstimates) {
      const names = Array.from(new Set([...Object.keys(byAccount), ...Object.keys(quotaEstimates)])).sort();
      document.getElementById('accounts').innerHTML = names.map(name => {
        const row = byAccount[name] || {};
        const quota = quotaEstimates[name] || {};
        const bundle = quota.auto_bundle ? '<span class="ok">bundle</span>' : 'single';
        return `<tr>
          <th>${esc(name || '(unassigned)')}</th>
          <td>done ${row.done || 0}</td>
          <td>running ${row.running || 0}</td>
          <td>queued ${row.queued || 0}</td>
          <td>failed ${row.failed || 0}</td>
          <td>used ${esc(fmtHours(quota.used_hours))}</td>
          <td>left ${esc(fmtHours(quota.remaining_hours))}</td>
          <td>avail ${esc(fmtHours(quota.available_hours))}</td>
          <td>reserved ${esc(fmtHours(quota.reserved_hours ?? quota.active_hours))}</td>
          <td>src ${esc(quota.quota_source || 'N/A')}</td>
          <td>${bundle}</td>
          <td>reset ${esc(quota.next_reset || 'N/A')}</td>
        </tr>`;
      }).join('');
    }
    function renderJobs(jobs) {
      const filter = document.getElementById('statusFilter').value;
      let rows = jobs.slice();
      if (filter !== 'all' && filter !== 'active') rows = rows.filter(j => j.status === filter);
      rows.sort((a, b) => (order[a.status] ?? 9) - (order[b.status] ?? 9) || a.job_id.localeCompare(b.job_id));
      if (filter === 'active') rows = rows.filter(j => j.status !== 'done').concat(jobs.filter(j => j.status === 'done').slice(-20));
      document.getElementById('jobs').innerHTML = rows.map(j => {
        const link = j.url ? `<a href="${esc(j.url)}" target="_blank" rel="noreferrer">Kaggle</a>` : '';
        const reason = j.failure_reason || link;
        return `<tr>
          <td>${esc(j.job_id)}</td>
          <td><span class="status ${esc(j.status)}">${esc(j.status)}</span></td>
          <td>${esc(j.account)}</td>
          <td>${esc(fmtResult(j.result_display))}</td>
          <td>${esc(fmtResult(j.asr_display))}</td>
          <td>${esc(fmtResult(j.elapsed_display))}</td>
          <td>${esc(fmtAge(j.age_minutes))}</td>
          <td>${esc(j.last_checked || '')}</td>
          <td>${reason}</td>
        </tr>`;
      }).join('');
    }
    async function refresh() {
      try {
        const res = await fetch('/api/status', {cache: 'no-store'});
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        render(await res.json());
      } catch (err) {
        document.getElementById('statusText').innerHTML = `<span class="fail">${esc(err.message)}</span>`;
      }
    }
    document.getElementById('refreshBtn').addEventListener('click', refresh);
    document.getElementById('statusFilter').addEventListener('change', refresh);
    refresh();
    setInterval(refresh, refreshMs);
  </script>
</body>
</html>
"""


CHARTS_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CamoPatch Result Charts</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dee7;
      --text: #1b1f24;
      --muted: #59636f;
      --accent: #0969da;
      --fail: #cf222e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 18px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { margin: 0 0 8px; font-size: 20px; font-weight: 650; }
    main { padding: 18px 24px 32px; max-width: 1500px; margin: 0 auto; }
    .topline, .toolbar { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; color: var(--muted); }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-top: 14px;
    }
    .panel h2 { margin: 0 0 10px; font-size: 15px; }
    select, button, a.button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--text);
      padding: 7px 10px;
      text-decoration: none;
      font: inherit;
    }
    button { cursor: pointer; }
    a.button { display: inline-block; }
    canvas {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      display: block;
    }
    .muted { color: var(--muted); }
    .fail { color: var(--fail); }
  </style>
</head>
<body>
  <header>
    <h1>CamoPatch Result Charts</h1>
    <div class="topline">
      <a class="button" href="/">Dashboard</a>
      <button id="refreshBtn">Refresh</button>
      <span id="statusText"></span>
    </div>
  </header>
  <main>
    <section class="panel">
      <div class="toolbar">
        <select id="modelFilter"></select>
        <select id="patchFilter"></select>
        <select id="linfFilter"></select>
        <select id="positionFilter"></select>
        <select id="lineLimit">
          <option value="12">12 lines</option>
          <option value="24">24 lines</option>
          <option value="999">All lines</option>
        </select>
        <select id="xScale">
          <option value="log">log query axis</option>
          <option value="linear">linear query axis</option>
        </select>
        <span class="muted" id="filterText"></span>
      </div>
    </section>

    <section class="panel">
      <h2>Attack Success Rate</h2>
      <canvas id="barChart"></canvas>
    </section>

    <section class="panel">
      <h2>Success Rate By Query By Model</h2>
      <canvas id="lineChart"></canvas>
    </section>
  </main>
  <script>
    const refreshMs = 30000;
    const palette = [
      '#0969da', '#12805c', '#cf222e', '#9a6700', '#8250df', '#1f883d',
      '#bf3989', '#bc4c00', '#0550ae', '#57606a', '#2da44e', '#a40e26'
    ];
    let chartData = {jobs: [], filters: {}};

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    function labelFor(job) {
      return `${job.model} s${job.patch_size} ${job.linf} ${job.position}`;
    }
    function sortedValues(values) {
      return Array.from(values).sort((a, b) => String(a).localeCompare(String(b), undefined, {numeric: true}));
    }
    function fillSelect(id, values, allLabel) {
      const select = document.getElementById(id);
      const current = select.value;
      select.innerHTML = `<option value="all">${esc(allLabel)}</option>` +
        sortedValues(values).map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
      if ([...select.options].some(option => option.value === current)) select.value = current;
    }
    function selectedJobs() {
      const model = document.getElementById('modelFilter').value;
      const patch = document.getElementById('patchFilter').value;
      const linf = document.getElementById('linfFilter').value;
      const position = document.getElementById('positionFilter').value;
      return chartData.jobs.filter(job =>
        (model === 'all' || job.model === model) &&
        (patch === 'all' || String(job.patch_size) === patch) &&
        (linf === 'all' || job.linf === linf) &&
        (position === 'all' || job.position === position)
      );
    }
    function lineGroupMode(jobs) {
      const selectedModel = document.getElementById('modelFilter').value;
      const selectedPosition = document.getElementById('positionFilter').value;
      const visibleModels = new Set(jobs.map(job => job.model));
      if (selectedPosition === 'all' && selectedModel !== 'all' && visibleModels.size === 1) {
        return 'position';
      }
      return 'model';
    }
    function aggregateLineCurves(jobs) {
      const mode = lineGroupMode(jobs);
      const groups = new Map();
      for (const job of jobs) {
        const key = mode === 'position'
          ? (job.position || 'unknown')
          : (job.model || 'unknown');
        if (!groups.has(key)) {
          groups.set(key, {
            label: key,
            mode,
            model: job.model || '',
            position: job.position || '',
            rows: 0,
            adversarial: 0,
            jobs: 0,
            queries: 0,
            events: new Map(),
          });
        }
        const group = groups.get(key);
        group.rows += Number(job.rows || 0);
        group.adversarial += Number(job.adversarial || 0);
        group.jobs += 1;
        group.queries = Math.max(group.queries, Number(job.queries || 0));
        for (const event of (job.events || [])) {
          const query = Number(event[0]);
          const count = Number(event[1]);
          if (!Number.isFinite(query) || !Number.isFinite(count) || count <= 0) continue;
          group.events.set(query, (group.events.get(query) || 0) + count);
        }
      }
      return Array.from(groups.values()).map(group => {
        const curve = [];
        let cumulative = 0;
        for (const query of Array.from(group.events.keys()).sort((a, b) => a - b)) {
          cumulative += group.events.get(query) || 0;
          curve.push([query, group.rows ? cumulative / group.rows * 100 : 0]);
        }
        if (!curve.length || curve[curve.length - 1][0] < group.queries) {
          curve.push([group.queries || 10000, group.rows ? group.adversarial / group.rows * 100 : 0]);
        }
        return {
          label: group.label,
          mode: group.mode,
          model: group.model,
          position: group.position,
          rows: group.rows,
          adversarial: group.adversarial,
          jobs: group.jobs,
          queries: group.queries || 10000,
          success_rate: group.rows ? group.adversarial / group.rows : 0,
          curve,
        };
      }).sort((a, b) => b.success_rate - a.success_rate || a.label.localeCompare(b.label));
    }
    function setupCanvas(canvas, cssHeight) {
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(320, Math.floor(rect.width));
      const height = Math.max(260, Math.floor(cssHeight));
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      canvas.style.height = `${height}px`;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      return {ctx, width, height};
    }
    function drawAxes(ctx, width, height, left, top, right, bottom, yTicks) {
      ctx.strokeStyle = '#d8dee7';
      ctx.lineWidth = 1;
      ctx.fillStyle = '#59636f';
      ctx.font = '12px system-ui, sans-serif';
      for (const tick of yTicks) {
        const y = bottom - (tick / 100) * (bottom - top);
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
        ctx.stroke();
        ctx.fillText(`${tick}%`, 8, y + 4);
      }
      ctx.strokeStyle = '#8c959f';
      ctx.beginPath();
      ctx.moveTo(left, top);
      ctx.lineTo(left, bottom);
      ctx.lineTo(right, bottom);
      ctx.stroke();
    }
    function drawBarChart(jobs) {
      const canvas = document.getElementById('barChart');
      const rows = jobs.slice().sort((a, b) =>
        a.model.localeCompare(b.model) ||
        Number(a.patch_size) - Number(b.patch_size) ||
        a.linf.localeCompare(b.linf, undefined, {numeric: true}) ||
        a.position.localeCompare(b.position)
      );
      const {ctx, width, height} = setupCanvas(canvas, Math.max(320, rows.length * 28 + 70));
      const left = Math.min(250, Math.max(150, width * 0.42)), right = width - 58, top = 24, bottom = height - 34;
      ctx.fillStyle = '#1b1f24';
      ctx.font = '12px system-ui, sans-serif';
      drawAxes(ctx, width, height, left, top, right, bottom, [0, 25, 50, 75, 100]);
      if (!rows.length) {
        ctx.fillText('No completed jobs in this filter.', 20, 45);
        return;
      }
      const rowH = Math.max(18, (bottom - top) / rows.length);
      rows.forEach((job, index) => {
        const y = top + index * rowH + rowH * 0.18;
        const h = Math.max(8, rowH * 0.58);
        const w = (right - left) * Math.max(0, Math.min(100, job.success_rate * 100)) / 100;
        ctx.fillStyle = palette[index % palette.length];
        ctx.fillRect(left, y, w, h);
        ctx.fillStyle = '#1b1f24';
        ctx.fillText(labelFor(job), 12, y + h - 1);
        ctx.fillText(`${(job.success_rate * 100).toFixed(1)}%`, Math.min(right - 44, left + w + 6), y + h - 1);
      });
    }
    function queryX(value, maxQuery, left, right, scale) {
      const q = Math.max(0, Number(value) || 0);
      if (scale === 'log') {
        return left + Math.log10(q + 1) / Math.log10(maxQuery + 1) * (right - left);
      }
      return left + q / maxQuery * (right - left);
    }
    function drawLineChart(jobs) {
      const canvas = document.getElementById('lineChart');
      const limit = Number(document.getElementById('lineLimit').value);
      const scale = document.getElementById('xScale').value;
      const rows = aggregateLineCurves(jobs).slice(0, limit);
      const legendRows = Math.ceil(Math.max(1, rows.length) / 3);
      const {ctx, width, height} = setupCanvas(canvas, Math.max(540, 500 + legendRows * 20));
      const left = 56, right = width - 24, top = 22, bottom = height - 62 - legendRows * 20;
      const maxQuery = Math.max(1, ...rows.map(job => Number(job.queries || 10000)));
      drawAxes(ctx, width, height, left, top, right, bottom, [0, 25, 50, 75, 100]);
      ctx.fillStyle = '#59636f';
      ctx.font = '12px system-ui, sans-serif';
      const xTicks = scale === 'log' ? [1, 10, 100, 1000, 10000].filter(v => v <= maxQuery) : [0, 2500, 5000, 7500, 10000].filter(v => v <= maxQuery);
      for (const tick of xTicks) {
        const x = queryX(tick, maxQuery, left, right, scale);
        ctx.strokeStyle = '#eef1f5';
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, bottom);
        ctx.stroke();
        ctx.fillText(String(tick), x - 14, bottom + 18);
      }
      ctx.fillStyle = '#59636f';
      ctx.fillText('Query', right - 34, bottom + 38);
      ctx.save();
      ctx.translate(18, top + 112);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Attack success rate', 0, 0);
      ctx.restore();
      if (!rows.length) {
        ctx.fillStyle = '#1b1f24';
        ctx.fillText('No completed jobs in this filter.', 20, 45);
        return;
      }
      rows.forEach((job, index) => {
        const color = palette[index % palette.length];
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        const points = [[0, 0], ...job.curve];
        if (!points.some(point => Number(point[0]) === Number(job.queries))) {
          points.push([Number(job.queries || maxQuery), job.success_rate * 100]);
        }
        points.forEach((point, pointIndex) => {
          const x = queryX(point[0], maxQuery, left, right, scale);
          const y = bottom - (Math.max(0, Math.min(100, Number(point[1]) || 0)) / 100) * (bottom - top);
          if (pointIndex === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        const legendY = bottom + 40 + Math.floor(index / 3) * 18;
        const legendX = left + (index % 3) * Math.max(230, (right - left) / 3);
        ctx.fillStyle = color;
        ctx.fillRect(legendX, legendY - 9, 10, 10);
        ctx.fillStyle = '#1b1f24';
        ctx.fillText(`${job.label} ${(job.success_rate * 100).toFixed(1)}% (${job.adversarial}/${job.rows}, ${job.jobs} jobs)`, legendX + 14, legendY);
      });
    }
    function renderCharts() {
      const jobs = selectedJobs();
      const lineGroups = aggregateLineCurves(jobs);
      const groupName = lineGroupMode(jobs) === 'position' ? 'position series' : 'model series';
      document.getElementById('filterText').textContent = `${jobs.length} completed jobs | ${lineGroups.length} ${groupName}`;
      drawBarChart(jobs);
      drawLineChart(jobs);
    }
    async function refreshCharts() {
      try {
        const res = await fetch('/api/charts', {cache: 'no-store'});
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        chartData = await res.json();
        fillSelect('modelFilter', new Set(chartData.jobs.map(job => job.model)), 'All models');
        fillSelect('patchFilter', new Set(chartData.jobs.map(job => String(job.patch_size))), 'All sizes');
        fillSelect('linfFilter', new Set(chartData.jobs.map(job => job.linf)), 'All L_inf');
        fillSelect('positionFilter', new Set(chartData.jobs.map(job => job.position)), 'All positions');
        document.getElementById('statusText').textContent = `updated: ${new Date().toLocaleString()} | done: ${chartData.jobs.length}`;
        renderCharts();
      } catch (err) {
        document.getElementById('statusText').innerHTML = `<span class="fail">${esc(err.message)}</span>`;
      }
    }
    for (const id of ['modelFilter','patchFilter','linfFilter','positionFilter','lineLimit','xScale']) {
      document.getElementById(id).addEventListener('change', renderCharts);
    }
    document.getElementById('refreshBtn').addEventListener('click', refreshCharts);
    window.addEventListener('resize', renderCharts);
    refreshCharts();
    setInterval(refreshCharts, refreshMs);
  </script>
</body>
</html>
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def age_minutes(value: str) -> float | None:
    dt = parse_iso(value)
    if dt is None:
        return None
    return round((datetime.now(timezone.utc) - dt).total_seconds() / 60.0, 2)


def tail_lines(path: Path, count: int = 40) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()[-count:]


def read_manifest(run_root: Path) -> dict[str, Any]:
    path = latest_results_zip(run_root)
    if path is None:
        return {}
    try:
        return read_zip_manifest(path)
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {"zip_valid": False, "error": str(exc)}


def latest_results_zip(run_root: Path) -> Path | None:
    aggregate_dir = run_root / "aggregate"
    preferred = aggregate_dir / "camopatch_results_only_latest.zip"
    if preferred.is_file():
        return preferred
    matches = sorted(
        aggregate_dir.glob("*results_only_latest.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def read_zip_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with zipfile.ZipFile(path) as archive:
        bad_file = archive.testzip()
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        manifest["zip_valid"] = bad_file is None
        manifest["bad_zip_entry"] = bad_file
        manifest["zip_name"] = path.name
        return manifest


def find_zip_member(archive: zipfile.ZipFile, suffix: str) -> str | None:
    exact = suffix.lstrip("/")
    if exact in archive.namelist():
        return exact
    matches = sorted(name for name in archive.namelist() if name.endswith(suffix))
    return matches[0] if matches else None


def read_zip_csv(archive: zipfile.ZipFile, suffix: str) -> list[dict[str, str]]:
    name = find_zip_member(archive, suffix)
    if name is None:
        return []
    with archive.open(name) as handle:
        text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
        return list(csv.DictReader(text))


def parse_int(value: Any) -> int | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def parse_job_id_fields(job_id: str) -> dict[str, str]:
    rest = job_id.removeprefix("camopatch-bcos-")
    if rest.startswith("movable-"):
        rest = rest.removeprefix("movable-")
    model = rest.split("-s", 1)[0]
    patch_size = ""
    linf = ""
    position = ""
    if "-s" in rest:
        patch_size = rest.split("-s", 1)[1].split("-", 1)[0]
    if "-linf" in rest:
        linf_tail = rest.split("-linf", 1)[1]
        if "-init-" in linf_tail:
            linf = linf_tail.split("-init-", 1)[0]
        elif "-" in linf_tail:
            linf = linf_tail.rsplit("-", 1)[0]
        else:
            linf = linf_tail
        linf = linf.replace("_", "/")
    if "-init-" in rest:
        position = rest.rsplit("-init-", 1)[1]
    elif job_id.endswith("-bcos_top1"):
        position = "bcos_top1"
    elif job_id.endswith("-gradcam"):
        position = "gradcam"
    elif job_id.endswith("-random"):
        position = "random"
    return {
        "model": model,
        "patch_size": patch_size,
        "linf": linf,
        "position": position,
    }


def parse_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def make_result(row: dict[str, Any]) -> dict[str, Any] | None:
    rows = parse_int(row.get("rows"))
    adversarial = parse_int(row.get("adversarial"))
    if rows is None or rows <= 0 or adversarial is None:
        return None
    elapsed_sec = parse_float(row.get("elapsed_sec"))
    success_rate = adversarial / rows if rows else None
    return {
        "rows": rows,
        "adversarial": adversarial,
        "success_rate": success_rate,
        "elapsed_sec": elapsed_sec,
        "return_code": row.get("return_code", ""),
        "done_at": row.get("done_at", ""),
        "result_display": f"{adversarial}/{rows}",
        "asr_display": f"{success_rate * 100:.1f}%" if success_rate is not None else "N/A",
        "elapsed_display": format_elapsed(elapsed_sec),
    }


def summarize_result_zip(path: Path) -> dict[str, Any] | None:
    try:
        with zipfile.ZipFile(path) as archive:
            if find_zip_member(archive, "outputs/summary.csv") is None:
                return None
            manifest: dict[str, Any] = {}
            manifest_name = find_zip_member(archive, "manifest.json")
            if manifest_name is not None:
                manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
            rows = read_zip_csv(archive, "outputs/summary.csv")
            total = len(rows)
            adversarial = sum(parse_int(row.get("adversarial")) == 1 for row in rows)
            return make_result(
                {
                    "rows": total,
                    "adversarial": adversarial,
                    "elapsed_sec": manifest.get("elapsed_sec"),
                    "return_code": manifest.get("return_code", ""),
                    "done_at": manifest.get("finished_at", ""),
                }
            )
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def chart_job_from_zip(job_id: str, state_job: dict[str, Any], path: Path) -> dict[str, Any] | None:
    try:
        with zipfile.ZipFile(path) as archive:
            summary_rows = read_zip_csv(archive, "outputs/summary.csv")
            by_query_rows = read_zip_csv(archive, "outputs/success_by_query.csv")
            manifest: dict[str, Any] = {}
            manifest_name = find_zip_member(archive, "manifest.json")
            if manifest_name is not None:
                manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not summary_rows:
        return None

    total = len(summary_rows)
    adversarial = sum(parse_int(row.get("adversarial")) == 1 for row in summary_rows)
    if total <= 0:
        return None
    parsed = parse_job_id_fields(job_id)
    first = summary_rows[0]
    manifest_config = manifest.get("job_config", {})
    if not isinstance(manifest_config, dict):
        manifest_config = {}
    model = str(manifest_config.get("model") or parsed["model"] or first.get("model"))
    patch_size = str(manifest_config.get("patch_size") or first.get("patch_size") or parsed["patch_size"])
    position = str(manifest_config.get("position") or first.get("position_rule") or parsed["position"])
    linf = str(manifest_config.get("linf") or parsed["linf"])
    queries = (
        parse_int(manifest_config.get("queries"))
        or parse_int(first.get("queries"))
        or parse_int(manifest.get("queries"))
        or 10000
    )
    curve: list[list[float]] = []
    events: list[list[float]] = []
    previous_cumulative = 0
    for row in sorted(by_query_rows, key=lambda item: parse_int(item.get("first_success_query")) or 0):
        query = parse_int(row.get("first_success_query"))
        cumulative = parse_int(row.get("cumulative_successes"))
        if query is None or cumulative is None:
            continue
        new_successes = parse_int(row.get("new_successes"))
        if new_successes is None:
            new_successes = max(0, cumulative - previous_cumulative)
        previous_cumulative = cumulative
        if new_successes > 0:
            events.append([float(query), float(new_successes)])
        curve.append([float(query), cumulative / total * 100.0])
    if not curve or curve[-1][0] < queries:
        curve.append([float(queries), adversarial / total * 100.0])

    return {
        "job_id": job_id,
        "account": state_job.get("account", ""),
        "url": state_job.get("url", ""),
        "model": model,
        "patch_size": patch_size,
        "linf": linf,
        "position": position,
        "queries": queries,
        "rows": total,
        "adversarial": adversarial,
        "success_rate": adversarial / total,
        "success_percent": adversarial / total * 100.0,
        "curve": curve,
        "events": events,
        "done_at": state_job.get("done_at", ""),
        "elapsed_sec": manifest.get("elapsed_sec", ""),
    }


def build_chart_data(run_root: Path) -> dict[str, Any]:
    state = load_json(run_root / "state.json", {"jobs": {}})
    jobs: list[dict[str, Any]] = []
    for job_id, state_job in state.get("jobs", {}).items():
        if state_job.get("status") != "done":
            continue
        result_zip = str(state_job.get("result_zip", ""))
        if not result_zip:
            continue
        chart_job = chart_job_from_zip(job_id, state_job, Path(result_zip))
        if chart_job is not None:
            jobs.append(chart_job)
    jobs.sort(
        key=lambda job: (
            str(job["model"]),
            int(job["patch_size"]) if str(job["patch_size"]).isdigit() else 0,
            str(job["linf"]),
            str(job["position"]),
        )
    )
    return {
        "generated_at": now_iso(),
        "run_root": str(run_root),
        "done_jobs": len(jobs),
        "jobs": jobs,
    }


def read_job_results(run_root: Path, jobs_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    path = run_root / "aggregate" / "camopatch_job_summary.csv"
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("status") != "done":
                        continue
                    result = make_result(row)
                    if result is not None and row.get("job_id"):
                        results[str(row["job_id"])] = result
        except OSError:
            pass

    # Fill any newly completed jobs before the aggregate monitor refreshes.
    for job_id, job in jobs_state.items():
        if job_id in results or job.get("status") != "done":
            continue
        result_zip = str(job.get("result_zip", ""))
        if not result_zip:
            continue
        result = summarize_result_zip(Path(result_zip))
        if result is not None:
            results[job_id] = result
    return results


def process_alive(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"pid": "", "alive": False, "children": []}
    pid = path.read_text(encoding="utf-8", errors="ignore").strip()
    alive = False
    children: list[str] = []
    try:
        os.kill(int(pid), 0)
        alive = True
    except (OSError, ValueError):
        alive = False
    if alive:
        try:
            output = subprocess.check_output(["pgrep", "-P", pid, "-a"], text=True).strip()
            children = [line for line in output.splitlines() if line]
        except (subprocess.CalledProcessError, FileNotFoundError):
            children = []
    return {"pid": pid, "alive": alive, "children": children}


def build_status(run_root: Path) -> dict[str, Any]:
    state = load_json(run_root / "state.json", {"jobs": {}})
    jobs_state = state.get("jobs", {})
    job_results = read_job_results(run_root, jobs_state)
    jobs: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    by_account: dict[str, dict[str, int]] = {}
    for job_id, job in jobs_state.items():
        status = str(job.get("status", ""))
        account = str(job.get("account", ""))
        counts[status] = counts.get(status, 0) + 1
        by_account.setdefault(account, {})
        by_account[account][status] = by_account[account].get(status, 0) + 1
        result = job_results.get(job_id, {})
        jobs.append(
            {
                "job_id": job_id,
                "status": status,
                "account": account,
                "url": job.get("url", ""),
                "last_checked": job.get("last_checked", ""),
                "submitted_at": job.get("submitted_at", ""),
                "age_minutes": age_minutes(str(job.get("submitted_at", ""))),
                "tries": job.get("tries", ""),
                "failure_reason": job.get("failure_reason", ""),
                "result_zip": job.get("result_zip", ""),
                "result": result if status == "done" else {},
                "result_display": result.get("result_display", "N/A") if status == "done" else "N/A",
                "asr_display": result.get("asr_display", "N/A") if status == "done" else "N/A",
                "elapsed_display": result.get("elapsed_display", "N/A") if status == "done" else "N/A",
            }
        )
    total_jobs = len(jobs)
    manifest = read_manifest(run_root)
    return {
        "generated_at": now_iso(),
        "run_root": str(run_root),
        "total_jobs": total_jobs,
        "counts": counts,
        "by_account": by_account,
        "account_backoff_until": state.get("account_backoff_until", {}),
        "account_quota_estimates": state.get("account_quota_estimates", {}),
        "all_done": total_jobs > 0 and counts.get("done", 0) == total_jobs,
        "jobs": jobs,
        "manifest": manifest,
        "progress_tail": tail_lines(run_root / "progress.log", 40),
        "scheduler_tail": tail_lines(run_root / "scheduler.out", 20),
        "aggregate_tail": tail_lines(run_root / "aggregate_monitor.out", 20),
        "processes": {
            "scheduler": process_alive(run_root / "scheduler.pid"),
            "aggregate_monitor": process_alive(run_root / "aggregate_monitor.pid"),
        },
    }


class DashboardHandler(BaseHTTPRequestHandler):
    run_root: Path

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/charts":
            self.send_bytes(CHARTS_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            body = json.dumps(build_status(self.run_root), indent=2).encode("utf-8")
            self.send_bytes(body, "application/json; charset=utf-8")
            return
        if parsed.path == "/api/charts":
            body = json.dumps(build_chart_data(self.run_root), indent=2).encode("utf-8")
            self.send_bytes(body, "application/json; charset=utf-8")
            return
        if parsed.path == "/download/results.zip":
            zip_path = self.run_root / "aggregate" / "camopatch_results_only_latest.zip"
            if not zip_path.is_file():
                self.send_bytes(b"results zip not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
                return
            body = zip_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition", 'attachment; filename="camopatch_results_only_latest.zip"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_bytes(b"not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)


def make_server(host: str, port: int, run_root: Path) -> ThreadingHTTPServer:
    handler = type("ConfiguredDashboardHandler", (DashboardHandler,), {"run_root": run_root})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local realtime Kaggle run dashboard.")
    parser.add_argument("--run-root", type=Path, default=Path("kaggle_runs_success_query_full"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--port-attempts", type=int, default=20)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    if not (run_root / "state.json").is_file():
        raise SystemExit(f"Missing {run_root / 'state.json'}")

    last_exc: OSError | None = None
    for port in range(args.port, args.port + max(1, args.port_attempts)):
        try:
            server = make_server(args.host, port, run_root)
            break
        except OSError as exc:
            last_exc = exc
    else:
        raise SystemExit(f"Could not bind dashboard port: {last_exc}")

    print(f"dashboard_url=http://{args.host}:{server.server_port}/", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
