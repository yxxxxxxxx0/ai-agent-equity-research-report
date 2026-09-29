"""Command-line entry point.

    python -m eq_report --ticker NVDA --report-date 2026-09-02

Callers provide only a ticker and an as-of date. The ticker is also used as
the initial entity name; provider data may supply a fuller canonical name.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .domain.request import ResearchRequest
from .pipeline.orchestrator import generate_report_sync


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eq_report",
        description="Generate a draft equity research report for a ticker.",
    )
    parser.add_argument("--ticker", help="ticker symbol, e.g. NVDA")
    parser.add_argument("--report-date", help="report as-of date (YYYY-MM-DD)")
    parser.add_argument(
        "--resume", metavar="RUN_ID",
        help="re-run analysis/synthesis/QA/rendering for an existing report_run_id "
             "(from a prior run's output directory), reusing its saved plan and "
             "Evidence Store rows instead of re-running planning and acquisition. "
             "--ticker/--report-date are ignored when this is set.",
    )
    parser.add_argument("--output-dir", type=Path, help="where to write run artefacts")
    parser.add_argument("--technical-appendix", action="store_true",
                        help="append the one-page technical-analysis supplement")
    parser.add_argument("--compact-report", action="store_true",
                        help="also write a two-page investment brief with technical analysis")
    parser.add_argument("--print-request", action="store_true",
                        help="print the normalized request and exit")
    return parser


def build_request(args: argparse.Namespace) -> ResearchRequest:
    if not args.ticker or not args.ticker.strip():
        raise SystemExit("--ticker must not be empty.")
    ticker = args.ticker.strip().upper()
    if not args.report_date:
        raise SystemExit("--report-date is required.")
    try:
        report_date = dt.date.fromisoformat(args.report_date)
    except ValueError as exc:
        raise SystemExit("--report-date must use YYYY-MM-DD format.") from exc

    return ResearchRequest(company=ticker, ticker=ticker, report_date=report_date)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.resume and args.print_request:
        raise SystemExit("--print-request has no effect with --resume.")

    request = None if args.resume else build_request(args)

    if args.print_request:
        print(json.dumps(request.to_dict(), indent=2))
        return 0

    settings = Settings.from_env(output_dir=args.output_dir)
    if args.technical_appendix or args.compact_report:
        settings = replace(
            settings,
            technical_appendix=settings.technical_appendix or args.technical_appendix,
            compact_report=settings.compact_report or args.compact_report,
        )
    result = generate_report_sync(request, settings, resume_from=args.resume)

    print()
    print(result.summary())
    if result.report_json_path:
        print(f"report JSON : {result.report_json_path}")
    if result.pdf_path:
        print(f"PDF         : {result.pdf_path}")
    if result.compact_pdf_path:
        print(f"compact PDF : {result.compact_pdf_path}")
    if result.run_manifest_path:
        print(f"run manifest: {result.run_manifest_path}")

    if result.qa_result and result.qa_result.critical:
        print()
        print("QA blocked the PDF:")
        for finding in result.qa_result.critical:
            print(f"  [{finding.check}] {finding.message}")

    print()
    usage = result.run.llm_usage or {}
    total_cost = usage.get("total_cost_usd", 0.0) or 0.0
    print(f"LLM cost    : ${total_cost:.4f}")
    if usage.get("calls_missing_cost"):
        print(f"  ({usage['calls_missing_cost']} call(s) did not report a cost)")
    duration_ms = result.run.duration_ms
    if duration_ms is not None:
        print(f"runtime     : {duration_ms / 1000:.1f}s")

    return 0 if result.succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
