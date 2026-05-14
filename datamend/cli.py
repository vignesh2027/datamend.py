"""datamend command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dataframe(path: str) -> "pd.DataFrame":
    """Load a CSV, Parquet, JSON, or Excel file into a DataFrame."""
    import pandas as pd

    p = Path(path)
    if not p.exists():
        click.echo(f"Error: file not found: {path}", err=True)
        sys.exit(1)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    click.echo(f"Error: unsupported file format '{suffix}'. Use CSV, Parquet, JSON, or Excel.", err=True)
    sys.exit(1)


def _save_dataframe(df: "pd.DataFrame", path: str) -> None:
    """Save DataFrame to the given path (inferred from extension)."""
    import pandas as pd

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif suffix == ".json":
        df.to_json(path, orient="records", indent=2)
    elif suffix in {".xls", ".xlsx"}:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="datamend")
def main() -> None:
    """datamend — Repair. Validate. Detect drift. Trace failures.

    Run datamend repair mydata.csv to get started.
    """


# ---------------------------------------------------------------------------
# repair
# ---------------------------------------------------------------------------


@main.command()
@click.argument("input_file")
@click.option("--output", "-o", default=None, help="Output file path. Default: <input>_repaired.csv")
@click.option(
    "--strategy",
    "-s",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "mean", "median"]),
    help="Null imputation strategy.",
)
@click.option("--fast", is_flag=True, help="Enable fast mode (samples large datasets).")
@click.option("--report", "-r", default=None, help="Save repair report to this JSON file.")
@click.option("--html", default=None, help="Save HTML dashboard to this file.")
@click.option("--confirm", is_flag=True, help="Require confirmation before applying repairs.")
@click.option("--no-verbose", is_flag=True, help="Suppress rich output.")
def repair(
    input_file: str,
    output: Optional[str],
    strategy: str,
    fast: bool,
    report: Optional[str],
    html: Optional[str],
    confirm: bool,
    no_verbose: bool,
) -> None:
    """Automatically repair dirty data in INPUT_FILE.

    Detects and fixes nulls, outliers, type mismatches, duplicates,
    encoding corruption, inconsistent categories, and whitespace issues.

    \b
    Examples:
      datamend repair data.csv
      datamend repair data.csv -o clean.csv --report report.json --html dashboard.html
    """
    import datamend as dm

    df = _load_dataframe(input_file)
    click.echo(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {input_file}")

    repaired, rpt = dm.repair(
        df, strategy=strategy, fast_mode=fast, confirm=confirm, verbose=not no_verbose
    )

    out_path = output or str(Path(input_file).stem) + "_repaired.csv"
    _save_dataframe(repaired, out_path)
    click.echo(f"\nRepaired data saved to: {out_path}")

    if report:
        Path(report).write_text(json.dumps(rpt.to_dict(), indent=2, default=str), encoding="utf-8")
        click.echo(f"Repair report saved to: {report}")

    if html:
        from datamend.report import MendReport
        mr = MendReport(repair=rpt, title=f"Repair Report — {input_file}")
        mr.to_html(path=html)
        click.echo(f"HTML dashboard saved to: {html}")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@main.command()
@click.argument("input_file")
@click.argument("contract_file")
@click.option("--report", "-r", default=None, help="Save validation report to JSON file.")
@click.option("--html", default=None, help="Save HTML dashboard.")
@click.option("--fail-fast", is_flag=True, help="Exit with code 1 if any violations found.")
@click.option("--no-verbose", is_flag=True, help="Suppress output.")
def validate(
    input_file: str,
    contract_file: str,
    report: Optional[str],
    html: Optional[str],
    fail_fast: bool,
    no_verbose: bool,
) -> None:
    """Validate INPUT_FILE against a saved CONTRACT_FILE.

    \b
    Examples:
      datamend validate prod_data.csv my_contract.json
      datamend validate prod_data.csv contract.json --fail-fast --report violations.json
    """
    from datamend.core.contract import DataContract

    df = _load_dataframe(input_file)
    contract = DataContract.load(contract_file)
    click.echo(f"Loaded contract '{contract.name}' ({len(contract._specs)} columns)")

    rpt = contract.validate(df)

    if not no_verbose:
        click.echo(rpt.summary())

    if report:
        Path(report).write_text(rpt.to_json(), encoding="utf-8")
        click.echo(f"Validation report saved to: {report}")

    if html:
        from datamend.report import MendReport
        mr = MendReport(contract=rpt, title=f"Contract Report — {input_file}")
        mr.to_html(path=html)
        click.echo(f"HTML dashboard saved to: {html}")

    if fail_fast and not rpt.passed:
        click.echo("Contract validation FAILED.", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# contract (generate)
# ---------------------------------------------------------------------------


@main.command()
@click.argument("input_file")
@click.option("--output", "-o", default=None, help="Output contract JSON path. Default: <input>_contract.json")
@click.option("--name", default="default", show_default=True, help="Contract name.")
@click.option("--null-threshold", default=0.05, show_default=True, type=float, help="Max null rate.")
@click.option("--no-verbose", is_flag=True, help="Suppress output.")
def contract(
    input_file: str,
    output: Optional[str],
    name: str,
    null_threshold: float,
    no_verbose: bool,
) -> None:
    """Generate a DataContract from INPUT_FILE.

    \b
    Examples:
      datamend contract training_data.csv -o my_contract.json
    """
    import datamend as dm

    df = _load_dataframe(input_file)
    dc = dm.contract(df, name=name, null_threshold=null_threshold)

    out_path = output or str(Path(input_file).stem) + "_contract.json"
    dc.save(out_path)
    if not no_verbose:
        click.echo(f"DataContract '{name}' generated and saved to: {out_path}")
        click.echo(f"  Columns tracked: {len(dc._specs)}")


# ---------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------


@main.command()
@click.argument("train_file")
@click.argument("prod_file")
@click.option("--report", "-r", default=None, help="Save drift report to JSON.")
@click.option("--html", default=None, help="Save HTML dashboard.")
@click.option("--alpha", default=0.05, show_default=True, type=float, help="Significance level.")
@click.option("--no-verbose", is_flag=True, help="Suppress output.")
def drift(
    train_file: str,
    prod_file: str,
    report: Optional[str],
    html: Optional[str],
    alpha: float,
    no_verbose: bool,
) -> None:
    """Detect drift between TRAIN_FILE and PROD_FILE.

    \b
    Examples:
      datamend drift training.csv production.csv
      datamend drift train.csv prod.csv --report drift.json --html drift.html
    """
    import datamend as dm

    train_df = _load_dataframe(train_file)
    prod_df = _load_dataframe(prod_file)

    rpt = dm.drift(train_df, prod_df, alpha=alpha, verbose=not no_verbose)

    if report:
        Path(report).write_text(json.dumps(rpt.to_dict(), indent=2, default=str), encoding="utf-8")
        click.echo(f"Drift report saved to: {report}")

    if html:
        from datamend.report import MendReport
        mr = MendReport(drift=rpt, title=f"Drift Report — {train_file} vs {prod_file}")
        mr.to_html(path=html)
        click.echo(f"HTML dashboard saved to: {html}")


# ---------------------------------------------------------------------------
# score (MendScore only)
# ---------------------------------------------------------------------------


@main.command()
@click.argument("input_file")
def score(input_file: str) -> None:
    """Print the MendScore for INPUT_FILE (0–100, higher is better).

    \b
    Examples:
      datamend score mydata.csv
    """
    from datamend.core.repair import _compute_mend_score

    df = _load_dataframe(input_file)
    s = _compute_mend_score(df)
    colour = "green" if s >= 80 else "yellow" if s >= 60 else "red"
    click.echo(click.style(f"MendScore: {s:.1f}/100", fg=colour, bold=True))


# ---------------------------------------------------------------------------
# dashboard (serve live)
# ---------------------------------------------------------------------------


@main.command()
@click.argument("report_json")
@click.option("--port", default=8899, show_default=True, type=int, help="Port for the dashboard server.")
@click.option("--no-open", is_flag=True, help="Do not open the browser automatically.")
def dashboard(report_json: str, port: int, no_open: bool) -> None:
    """Serve an HTML dashboard from a JSON report file.

    \b
    Examples:
      datamend dashboard repair_report.json
      datamend dashboard drift_report.json --port 9000
    """
    data = json.loads(Path(report_json).read_text(encoding="utf-8"))
    from datamend.report import MendReport, _render_html_dashboard

    html = _render_html_dashboard(data, title=data.get("title", "datamend Report"))
    mr = MendReport(title=data.get("title", "datamend Report"))

    import http.server
    import threading
    import webbrowser

    html_bytes = html.encode("utf-8")

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    click.echo(f"Dashboard running at http://127.0.0.1:{port}/  (Ctrl+C to stop)")
    if not no_open:
        threading.Timer(0.4, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nDashboard stopped.")
        server.shutdown()


# ---------------------------------------------------------------------------
# plugins (list)
# ---------------------------------------------------------------------------


@main.command()
def plugins() -> None:
    """List all registered datamend plugins."""
    from datamend.plugins.base import get_registry
    registry = get_registry()
    registry.auto_discover()
    info = registry.list_plugins()
    click.echo("Registered datamend plugins:")
    for plugin_type, names in info.items():
        click.echo(f"  {plugin_type}: {', '.join(names) if names else '(none)'}")
