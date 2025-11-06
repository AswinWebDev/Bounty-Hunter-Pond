#!/usr/bin/env python
"""Simple dashboard generator for viewing Pond prediction results.

Loads the original test CSV (wallet + social handles) and the results.csv file
produced by predict_pond.py, merges them, and renders an interactive HTML
report using DataTables. The generated dashboard includes quick links to BaseScan,
GitHub, and Twitter so you can manually inspect each profile.

Usage:
    python dashboard.py --input pond_test_example.csv --results results.csv

Optional arguments:
    --output dashboard.html  # custom HTML path (default: dashboard.html)
    --open                   # open the dashboard in your default web browser
"""

from __future__ import annotations

import argparse
import html
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate interactive dashboard for Pond predictions")
    parser.add_argument("--input", required=True, help="CSV file passed to predict_pond.py")
    parser.add_argument("--results", default="results.csv", help="Prediction results CSV produced by predict_pond.py")
    parser.add_argument("--output", default="dashboard.html", help="Path for generated HTML dashboard")
    parser.add_argument("--open", action="store_true", help="Open dashboard in default browser after generation")
    return parser.parse_args()


def load_input_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]

    if "wallet_address" not in df.columns:
        raise ValueError("Input CSV must contain a 'wallet_address' column")

    # Normalise column names
    rename_map = {
        "github": "github_username",
        "twitter": "twitter_username",
    }
    df = df.rename(columns=rename_map)
    df["wallet_address"] = df["wallet_address"].str.lower()
    df = df.drop_duplicates(subset=["wallet_address"], keep="first")
    return df


def load_results_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "wallet_address" not in df.columns:
        raise ValueError("Results CSV must contain a 'wallet_address' column")
    df["wallet_address"] = df["wallet_address"].str.lower()
    return df


def bool_badge(value: bool) -> str:
    return "<span class=\"badge badge-yes\">Yes</span>" if bool(value) else "<span class=\"badge badge-no\">No</span>"


def risk_badge(level: str) -> str:
    level = (level or "").upper()
    badge_class = {
        "LOW": "badge-low",
        "MEDIUM": "badge-medium",
        "HIGH": "badge-high",
    }.get(level, "badge-neutral")
    return f"<span class=\"badge {badge_class}\">{html.escape(level or 'UNK')}</span>"


def score_badge(score: float) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "N/A"

    if value < 0.3:
        cls = "pill-low"
    elif value < 0.7:
        cls = "pill-medium"
    else:
        cls = "pill-high"
    return f"<span class=\"pill {cls}\">{value:.2f}</span>"


def make_link(url: str, text: str, title: str | None = None) -> str:
    if not url:
        return ""
    safe_url = html.escape(url)
    safe_text = html.escape(text)
    title_attr = f' title="{html.escape(title)}"' if title else ""
    return f"<a href=\"{safe_url}\" target=\"_blank\" rel=\"noopener\"{title_attr}>{safe_text}</a>"


def build_table_rows(df: pd.DataFrame) -> str:
    rows = []
    for _, row in df.iterrows():
        addr = row.get("wallet_address", "")
        base_link = make_link(f"https://basescan.org/address/{addr}", addr[:10] + "…") if addr else ""

        github_user = row.get("github_username", "") or ""
        twitter_user = (row.get("twitter_username", "") or "").lstrip("@")

        github_link = make_link(f"https://github.com/{github_user}", github_user) if github_user else ""
        twitter_link = make_link(f"https://twitter.com/{twitter_user}", f"@{twitter_user}") if twitter_user else ""

        rows.append(
            "<tr>"
            f"<td class=\"address\">{base_link if base_link else html.escape(addr)}</td>"
            f"<td>{score_badge(row.get('sybil_score'))}</td>"
            f"<td>{html.escape(str(row.get('prediction', '')))}</td>"
            f"<td>{risk_badge(row.get('risk_level', ''))}</td>"
            f"<td>{row.get('confidence', 0):.2f}</td>"
            f"<td>{row.get('onchain_risk', 0):.2f}</td>"
            f"<td>{row.get('social_risk', 0):.2f}</td>"
            f"<td>{bool_badge(row.get('github_verified'))}</td>"
            f"<td>{bool_badge(row.get('twitter_verified'))}</td>"
            f"<td>{github_link}</td>"
            f"<td>{twitter_link}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def build_summary(df: pd.DataFrame) -> str:
    total = len(df)
    sybil_count = int((df["prediction"].str.upper() == "SYBIL").sum()) if total else 0
    genuine_count = total - sybil_count
    avg_score = df["sybil_score"].mean() if "sybil_score" in df.columns and total else 0
    avg_conf = df["confidence"].mean() if "confidence" in df.columns and total else 0

    summary_cards = f"""
    <div class=\"summary-cards\">
        <div class=\"card\">
            <div class=\"card-title\">Total Addresses</div>
            <div class=\"card-value\">{total}</div>
        </div>
        <div class=\"card\">
            <div class=\"card-title\">Genuine</div>
            <div class=\"card-value good\">{genuine_count}</div>
        </div>
        <div class=\"card\">
            <div class=\"card-title\">Sybil / High Risk</div>
            <div class=\"card-value alert\">{sybil_count}</div>
        </div>
        <div class=\"card\">
            <div class=\"card-title\">Avg Sybil Score</div>
            <div class=\"card-value\">{avg_score:.2f}</div>
        </div>
        <div class=\"card\">
            <div class=\"card-title\">Avg Confidence</div>
            <div class=\"card-value\">{avg_conf:.2f}</div>
        </div>
    </div>
    """
    return summary_cards


def build_html(df: pd.DataFrame, input_path: Path, results_path: Path) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    table_rows = build_table_rows(df)
    summary_section = build_summary(df)

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\" />
<title>Pond Sybil Detection Dashboard</title>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<link rel=\"stylesheet\" href=\"https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css\" />
<link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/modern-normalize/2.0.0/modern-normalize.min.css\" />
<style>
body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    margin: 0;
    padding: 2rem 3rem;
}}

h1 {{
    font-size: 2rem;
    margin-bottom: 0.5rem;
}}

h2 {{
    margin-top: 2.5rem;
    font-size: 1.4rem;
}}

p.meta {{
    color: #94a3b8;
    font-size: 0.95rem;
}}

.summary-cards {{
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    margin: 2rem 0;
}}

.card {{
    background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(99,102,241,0.2));
    border: 1px solid rgba(148,163,184,0.2);
    border-radius: 12px;
    padding: 1.2rem;
    backdrop-filter: blur(10px);
}}

.card-title {{
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
}}

.card-value {{
    font-size: 1.8rem;
    font-weight: 600;
    margin-top: 0.3rem;
}}

.card-value.good {{
    color: #34d399;
}}

.card-value.alert {{
    color: #f87171;
}}

table.dataTable thead th {{
    background: rgba(148,163,184,0.1);
    color: #e2e8f0;
}}

.dataTables_wrapper .dataTables_filter input {{
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(148,163,184,0.3);
    color: #e2e8f0;
    border-radius: 6px;
}}

.dataTables_wrapper .dataTables_length select {{
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(148,163,184,0.3);
    color: #e2e8f0;
    border-radius: 6px;
}}

table {{
    width: 100%;
    border-spacing: 0;
    margin-top: 2rem;
}}

table.dataTable tbody tr:hover {{
    background: rgba(59,130,246,0.15);
}}

.badge {{
    display: inline-block;
    padding: 0.3rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}}

.badge-low {{
    background: rgba(52,211,153,0.2);
    color: #34d399;
}}

.badge-medium {{
    background: rgba(251,191,36,0.2);
    color: #fbbf24;
}}

.badge-high {{
    background: rgba(248,113,113,0.2);
    color: #f87171;
}}

.badge-neutral {{
    background: rgba(148,163,184,0.2);
    color: #cbd5f5;
}}

.badge-yes {{
    background: rgba(52,211,153,0.2);
    color: #34d399;
}}

.badge-no {{
    background: rgba(248,113,113,0.2);
    color: #f87171;
}}

.pill {{
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.8rem;
}}

.pill-low {{
    background: rgba(52,211,153,0.2);
    color: #34d399;
}}

.pill-medium {{
    background: rgba(251,191,36,0.2);
    color: #fbbf24;
}}

.pill-high {{
    background: rgba(248,113,113,0.2);
    color: #f87171;
}}

a {{
    color: #60a5fa;
}}

a:hover {{
    color: #93c5fd;
}}

footer {{
    margin-top: 3rem;
    font-size: 0.85rem;
    color: #64748b;
}}

@media (max-width: 768px) {{
    body {{ padding: 1.5rem; }}
    table {{ font-size: 0.85rem; }}
}}
</style>
</head>
<body>
    <h1>Pond Sybil Detection Dashboard</h1>
    <p class=\"meta\">Input: {html.escape(str(input_path))} | Results: {html.escape(str(results_path))} | Generated: {generated_at}</p>

    {summary_section}

    <table id=\"resultsTable\" class=\"display\">
        <thead>
            <tr>
                <th>Wallet</th>
                <th>Sybil Score</th>
                <th>Prediction</th>
                <th>Risk Level</th>
                <th>Confidence</th>
                <th>On-chain Risk</th>
                <th>Social Risk</th>
                <th>GitHub Verified</th>
                <th>Twitter Verified</th>
                <th>GitHub</th>
                <th>Twitter</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>

    <footer>
        Built for Pond Bounty Hunter Competition · Generated with dashboard.py
    </footer>

    <script src=\"https://code.jquery.com/jquery-3.7.1.min.js\"></script>
    <script src=\"https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js\"></script>
    <script>
        $(document).ready(function() {{
            $('#resultsTable').DataTable({{
                pageLength: 25,
                order: [[1, 'desc']]
            }});
        }});
    </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    results_path = Path(args.results)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌ Input CSV not found: {input_path}")
        sys.exit(1)
    if not results_path.exists():
        print(f"❌ Results CSV not found: {results_path}")
        sys.exit(1)

    input_df = load_input_csv(input_path)
    results_df = load_results_csv(results_path)

    merged = results_df.merge(input_df, on="wallet_address", how="left", suffixes=("", "_input"))
    merged = merged.fillna({"github_username": "", "twitter_username": ""})

    html_doc = build_html(merged, input_path, results_path)
    output_path.write_text(html_doc, encoding="utf-8")

    print(f"✅ Dashboard created at {output_path.resolve()}")

    if args.open:
        webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
