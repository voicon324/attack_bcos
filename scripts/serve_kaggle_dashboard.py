#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
      th:nth-child(4), td:nth-child(4), th:nth-child(7), td:nth-child(7) { display: none; }
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
        <a class="button" href="/download/results.zip">Download results zip</a>
        <span class="muted" id="manifestText"></span>
      </div>
      <table>
        <thead>
          <tr>
            <th style="width: 37%">Job</th>
            <th style="width: 8%">Status</th>
            <th style="width: 8%">Account</th>
            <th style="width: 8%">Age</th>
            <th style="width: 15%">Last Check</th>
            <th style="width: 8%">Tries</th>
            <th style="width: 16%">Reason / Link</th>
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
      renderAccounts(data.by_account || {});
      renderJobs(data.jobs || []);
      document.getElementById('progressLog').textContent = (data.progress_tail || []).join('\n');
    }
    function renderAccounts(byAccount) {
      const names = Object.keys(byAccount).sort();
      document.getElementById('accounts').innerHTML = names.map(name => {
        const row = byAccount[name] || {};
        return `<tr><th>${esc(name || '(unassigned)')}</th><td>done ${row.done || 0}</td><td>running ${row.running || 0}</td><td>queued ${row.queued || 0}</td><td>failed ${row.failed || 0}</td></tr>`;
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
          <td>${esc(fmtAge(j.age_minutes))}</td>
          <td>${esc(j.last_checked || '')}</td>
          <td>${esc(j.tries || '')}</td>
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
    path = run_root / "aggregate" / "camopatch_results_only_latest.zip"
    if not path.is_file():
        return {}
    try:
        with zipfile.ZipFile(path) as archive:
            bad_file = archive.testzip()
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            manifest["zip_valid"] = bad_file is None
            manifest["bad_zip_entry"] = bad_file
            return manifest
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {"zip_valid": False, "error": str(exc)}


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
    jobs: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    by_account: dict[str, dict[str, int]] = {}
    for job_id, job in jobs_state.items():
        status = str(job.get("status", ""))
        account = str(job.get("account", ""))
        counts[status] = counts.get(status, 0) + 1
        by_account.setdefault(account, {})
        by_account[account][status] = by_account[account].get(status, 0) + 1
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
        if parsed.path == "/api/status":
            body = json.dumps(build_status(self.run_root), indent=2).encode("utf-8")
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
