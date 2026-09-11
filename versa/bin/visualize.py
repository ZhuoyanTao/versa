#!/usr/bin/env python3

"""Generate VERSA result reports and visualizations."""

import argparse
import os
from pathlib import Path

from versa.metric_discovery import create_metric_discovery_registry
from versa.reporting import (
    analyze_records,
    read_result_records,
    write_csv_report,
    write_html_report,
    write_markdown_report,
)


def get_parser() -> argparse.ArgumentParser:
    """Build CLI options for reading results and exporting summary reports."""
    parser = argparse.ArgumentParser(
        description="Create summary tables and visual reports from VERSA results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Input JSONL result file or directory of result files.",
    )
    parser.add_argument(
        "--out",
        default="report.html",
        help="Output report path. Format is inferred from the extension.",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "html", "csv", "md"],
        default="auto",
        help="Report format.",
    )
    parser.add_argument(
        "--group-by",
        default=None,
        help="Optional record field used for per-metric ranking, e.g. _source_file.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional extra CSV summary export path.",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Optional extra Markdown summary export path.",
    )
    parser.add_argument(
        "--outlier-limit",
        type=int,
        default=3,
        help="Maximum outlier examples to keep per metric.",
    )
    return parser


def main() -> None:
    """Read CLI result files and write the requested reports, creating parent folders."""
    parser = get_parser()
    args = parser.parse_args()

    records = read_result_records(args.input)
    registry = create_metric_discovery_registry(include_runtime_imports=False)
    analysis = analyze_records(
        records,
        group_by=args.group_by,
        outlier_limit=args.outlier_limit,
        registry=registry,
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_format = args.format
    if report_format == "auto":
        suffix = output_path.suffix.lower()
        report_format = {
            ".html": "html",
            ".htm": "html",
            ".csv": "csv",
            ".md": "md",
            ".markdown": "md",
        }.get(suffix, "html")

    if report_format == "html":
        write_html_report(analysis, os.fspath(output_path))
    elif report_format == "csv":
        write_csv_report(analysis, os.fspath(output_path))
    else:
        write_markdown_report(analysis, os.fspath(output_path))

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        write_csv_report(analysis, args.csv)
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        write_markdown_report(analysis, args.markdown)

    print(
        "Wrote {format} report for {records} utterances and {metrics} metrics to {path}".format(
            format=report_format,
            records=analysis["record_count"],
            metrics=analysis["metric_count"],
            path=output_path,
        )
    )


if __name__ == "__main__":
    main()
