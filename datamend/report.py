"""Report generation — HTML dashboard export for all datamend report types."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datamend.core.contract import ContractReport
from datamend.core.drift import DriftReport
from datamend.core.repair import RepairReport
from datamend.core.trace import TraceReport

# ---------------------------------------------------------------------------
# MendReport — aggregated report container
# ---------------------------------------------------------------------------


class MendReport:
    """Container that holds any combination of pillar reports and can export to HTML.

    Args:
        repair: AutoRepair report.
        contract: DataContract validation report.
        drift: DriftRadar report.
        trace: FailureTrace report.
        title: Title shown in the HTML dashboard.
    """

    def __init__(
        self,
        repair: RepairReport | None = None,
        contract: ContractReport | None = None,
        drift: DriftReport | None = None,
        trace: TraceReport | None = None,
        title: str = "datamend Health Report",
        # also accept the _report-suffixed names used in docs / README
        repair_report: RepairReport | None = None,
        contract_report: ContractReport | None = None,
        drift_report: DriftReport | None = None,
        trace_report: TraceReport | None = None,
    ) -> None:
        self.repair = repair or repair_report
        self.contract = contract or contract_report
        self.drift = drift or drift_report
        self.trace = trace or trace_report
        self.title = title

    def to_dict(self) -> dict[str, Any]:
        """Serialise all available reports to a single dictionary."""
        result: dict[str, Any] = {"title": self.title}
        if self.repair:
            result["repair"] = self.repair.to_dict()
        if self.contract:
            result["contract"] = self.contract.to_dict()
        if self.drift:
            result["drift"] = self.drift.to_dict()
        if self.trace:
            result["trace"] = self.trace.to_dict()
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialise all reports to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_html(self, path: str | None = None) -> str:
        """Generate a self-contained HTML dashboard.

        Args:
            path: If provided, write the HTML to this file path.

        Returns:
            HTML string.
        """
        data = self.to_dict()
        html = _render_html_dashboard(data, title=self.title)
        if path:
            Path(path).write_text(html, encoding="utf-8")
        return html

    def serve(self, port: int = 8899, open_browser: bool = True) -> None:
        """Serve the HTML dashboard on a local HTTP server.

        Args:
            port: Port to listen on.
            open_browser: Automatically open the dashboard in the default browser.
        """
        import http.server
        import threading
        import webbrowser

        html = self.to_html()
        html_bytes = html.encode("utf-8")

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)

            def log_message(self, *args: Any) -> None:
                pass

        server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
        print(f"datamend dashboard running at http://127.0.0.1:{port}/")
        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Dashboard server stopped.")
            server.shutdown()


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _render_html_dashboard(data: dict[str, Any], title: str) -> str:
    """Build a fully self-contained, single-file HTML dashboard."""
    repair = data.get("repair", {})
    contract = data.get("contract", {})
    drift = data.get("drift", {})
    trace = data.get("trace", {})

    def _score_colour(score: float) -> str:
        if score >= 80:
            return "#22c55e"
        if score >= 60:
            return "#f59e0b"
        return "#ef4444"

    repair_score = repair.get("mend_score_after", repair.get("mend_score_before", 100))
    contract_score = contract.get("mend_score", 100) if contract else None
    drift_score = drift.get("mend_score", 0) if drift else None
    trace_score = trace.get("mend_score", 0) if trace else None

    cards_html = ""

    if repair:
        c = _score_colour(repair_score)
        cards_html += f"""
        <div class="card">
          <div class="card-header" style="border-left:4px solid {c}">
            <span class="pill">Pillar 1</span>
            <h2>AutoRepair</h2>
            <div class="score" style="color:{c}">{repair_score:.1f}<span>/100</span></div>
          </div>
          <div class="card-body">
            <div class="stat-row">
              <span>Issues fixed</span>
              <strong>{repair.get("total_issues_found", 0)}</strong>
            </div>
            <div class="stat-row">
              <span>Rows affected</span>
              <strong>{repair.get("total_rows_affected", 0)}</strong>
            </div>
            <div class="stat-row">
              <span>Columns repaired</span>
              <strong>{len(repair.get("columns_repaired", []))}</strong>
            </div>
            <div class="stat-row">
              <span>Score delta</span>
              <strong>{repair.get("mend_score_before", 0):.1f} → {repair_score:.1f}</strong>
            </div>
            {"<h3>Actions</h3>" + _render_actions_table(repair.get("actions", [])) if repair.get("actions") else ""}
          </div>
        </div>"""

    if contract:
        c = _score_colour(contract_score or 0)
        status = "PASSED" if contract.get("passed") else "FAILED"
        cards_html += f"""
        <div class="card">
          <div class="card-header" style="border-left:4px solid {c}">
            <span class="pill">Pillar 2</span>
            <h2>DataContract <span class="badge {'badge-ok' if status=='PASSED' else 'badge-fail'}">{status}</span></h2>
            <div class="score" style="color:{c}">{contract_score:.1f}<span>/100</span></div>
          </div>
          <div class="card-body">
            <div class="stat-row">
              <span>Columns checked</span>
              <strong>{contract.get("columns_checked", 0)}</strong>
            </div>
            <div class="stat-row">
              <span>Violations</span>
              <strong style="color:#ef4444">{len(contract.get("violations", []))}</strong>
            </div>
            <div class="stat-row">
              <span>Warnings</span>
              <strong style="color:#f59e0b">{len(contract.get("warnings", []))}</strong>
            </div>
            {_render_violations_table(contract.get("violations", []) + contract.get("warnings", []))}
          </div>
        </div>"""

    if drift:
        drift_display = 100 - (drift_score or 0)
        c = _score_colour(drift_display)
        d_status = "DRIFT DETECTED" if drift.get("overall_drifted") else "STABLE"
        cards_html += f"""
        <div class="card">
          <div class="card-header" style="border-left:4px solid {c}">
            <span class="pill">Pillar 3</span>
            <h2>DriftRadar <span class="badge {'badge-fail' if drift.get('overall_drifted') else 'badge-ok'}">{d_status}</span></h2>
            <div class="score" style="color:{c}">{drift_score:.1f}<span>/100 drift</span></div>
          </div>
          <div class="card-body">
            <div class="stat-row">
              <span>Columns drifted</span>
              <strong style="color:#ef4444">{len(drift.get("columns_drifted", []))}</strong>
            </div>
            {_render_drift_table(drift.get("column_results", {}))}
          </div>
        </div>"""

    if trace:
        trace_stable = 100 - (trace_score or 0)
        c = _score_colour(trace_stable)
        cards_html += f"""
        <div class="card">
          <div class="card-header" style="border-left:4px solid {c}">
            <span class="pill">Pillar 4</span>
            <h2>FailureTrace</h2>
            <div class="score" style="color:{c}">{trace_score:.1f}<span>/100</span></div>
          </div>
          <div class="card-body">
            <div class="stat-row">
              <span>Total rows</span>
              <strong>{trace.get("total_rows", 0)}</strong>
            </div>
            <div class="stat-row">
              <span>Suspicious rows</span>
              <strong style="color:#ef4444">{trace.get("suspicious_row_count", 0)}</strong>
            </div>
            <div class="stat-row">
              <span>Top failure columns</span>
              <strong>{", ".join(trace.get("top_failure_columns", [])[:5])}</strong>
            </div>
            {_render_attribution_table(trace.get("column_attributions", []))}
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title}</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;padding:2rem}}
    h1{{font-size:2rem;font-weight:800;background:linear-gradient(90deg,#6366f1,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.25rem}}
    .subtitle{{color:#64748b;font-size:0.9rem;margin-bottom:2rem}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1.5rem}}
    .card{{background:#1e293b;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.3)}}
    .card-header{{padding:1.25rem 1.5rem;background:#0f172a;display:flex;flex-direction:column;gap:0.25rem}}
    .card-header h2{{font-size:1.1rem;font-weight:700;color:#f1f5f9;display:flex;align-items:center;gap:0.5rem}}
    .score{{font-size:2.5rem;font-weight:900;margin-top:0.25rem}}
    .score span{{font-size:0.9rem;font-weight:400;color:#94a3b8}}
    .pill{{font-size:0.65rem;font-weight:700;background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:999px;padding:0.15rem 0.5rem;text-transform:uppercase;letter-spacing:0.05em;width:fit-content}}
    .badge{{font-size:0.65rem;font-weight:700;border-radius:4px;padding:0.15rem 0.4rem}}
    .badge-ok{{background:#14532d;color:#86efac}}
    .badge-fail{{background:#7f1d1d;color:#fca5a5}}
    .card-body{{padding:1.25rem 1.5rem}}
    .stat-row{{display:flex;justify-content:space-between;padding:0.4rem 0;border-bottom:1px solid #1e293b;font-size:0.85rem;color:#94a3b8}}
    .stat-row strong{{color:#f1f5f9}}
    h3{{font-size:0.8rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin:1rem 0 0.5rem}}
    table{{width:100%;font-size:0.78rem;border-collapse:collapse;margin-top:0.5rem}}
    th{{text-align:left;padding:0.35rem 0.5rem;color:#64748b;font-weight:600;text-transform:uppercase;font-size:0.68rem;letter-spacing:0.05em;border-bottom:1px solid #334155}}
    td{{padding:0.35rem 0.5rem;border-bottom:1px solid #1e293b;color:#cbd5e1}}
    tr:last-child td{{border-bottom:none}}
    .err{{color:#f87171}} .warn{{color:#fbbf24}} .ok{{color:#4ade80}}
    footer{{text-align:center;color:#334155;font-size:0.75rem;margin-top:2.5rem}}
    a{{color:#6366f1}}
  </style>
</head>
<body>
  <h1>datamend</h1>
  <p class="subtitle">{title} &nbsp;·&nbsp; Generated by <a href="https://github.com/vignesh2027/datamend.py" target="_blank">datamend</a></p>
  <div class="grid">
    {cards_html}
  </div>
  <footer>datamend · MIT License · <a href="https://github.com/vignesh2027/datamend.py" target="_blank">GitHub</a></footer>
</body>
</html>"""
    return html


def _render_actions_table(actions: list) -> str:
    if not actions:
        return ""
    rows = "".join(
        f"<tr><td>{a.get('column','')}</td><td>{a.get('issue_type','')}</td>"
        f"<td>{a.get('rows_affected',0)}</td><td>{a.get('strategy','')}</td></tr>"
        for a in actions[:20]
    )
    return f"""<h3>Repairs applied</h3>
<table><thead><tr><th>Column</th><th>Issue</th><th>Rows</th><th>Strategy</th></tr></thead>
<tbody>{rows}</tbody></table>"""


def _render_violations_table(items: list) -> str:
    if not items:
        return "<p style='color:#4ade80;font-size:0.85rem;margin-top:0.5rem'>All checks passed.</p>"
    rows = "".join(
        f"<tr class=\"{'err' if v.get('severity')=='ERROR' else 'warn'}\">"
        f"<td>{v.get('column','')}</td><td>{v.get('violation_type','')}</td>"
        f"<td>{v.get('message','')}</td></tr>"
        for v in items[:20]
    )
    return f"""<h3>Violations & warnings</h3>
<table><thead><tr><th>Column</th><th>Type</th><th>Message</th></tr></thead>
<tbody>{rows}</tbody></table>"""


def _render_drift_table(col_results: dict) -> str:
    if not col_results:
        return ""
    rows = "".join(
        f"<tr class=\"{'err' if r.get('drifted') else 'ok'}\">"
        f"<td>{col}</td><td>{r.get('severity','')}</td>"
        f"<td>{r.get('drift_score',0):.1f}</td>"
        f"<td>{r.get('psi') or r.get('jsd') or '—'}</td></tr>"
        for col, r in list(col_results.items())[:20]
    )
    return f"""<h3>Column drift scores</h3>
<table><thead><tr><th>Column</th><th>Severity</th><th>Score</th><th>PSI/JSD</th></tr></thead>
<tbody>{rows}</tbody></table>"""


def _render_attribution_table(attributions: list) -> str:
    if not attributions:
        return ""
    rows = "".join(
        f"<tr><td>{a.get('column','')}</td>"
        f"<td>{a.get('importance_score',0):.1f}</td>"
        f"<td>{a.get('data_quality_contribution',0):.1f}</td>"
        f"<td>{a.get('model_contribution',0):.1f}</td>"
        f"<td>{a.get('anomaly_rate',0):.2%}</td></tr>"
        for a in attributions[:10]
    )
    return f"""<h3>Column attributions</h3>
<table><thead><tr><th>Column</th><th>Importance</th><th>DQ contrib</th><th>Model contrib</th><th>Anomaly rate</th></tr></thead>
<tbody>{rows}</tbody></table>"""
