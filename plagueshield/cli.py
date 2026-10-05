"""Command-line interface.

    plagueshield list
    plagueshield assess PS-SYN-DISC-01
    plagueshield assess --all --publish
    plagueshield fetch
    plagueshield serve
"""

from __future__ import annotations

import argparse
import json
import sys

from .data import available_cases, load_all_cases, load_case
from .models import CaseRecord
from .orchestrator import AssessmentPublisher, Pipeline

AMBER = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _c(text: str, code: str) -> str:
    return text if not sys.stdout.isatty() else f"{code}{text}{RESET}"


def cmd_list(args: argparse.Namespace) -> int:
    rows = available_cases()
    print(_c(f"{len(rows)} bundled case records", BOLD))
    for row in rows:
        origin = row["origin"].replace("_", " ")
        print(
            f"  {_c(row['case_id'].ljust(16), AMBER)}"
            f"{row['label'][:52].ljust(54)}"
            f"{_c(origin, DIM)}"
        )
    return 0


def _assess(cases: list[CaseRecord], args: argparse.Namespace) -> int:
    pipeline = Pipeline(live_evidence=args.live_evidence)
    publisher = AssessmentPublisher(args.url) if args.publish else None
    failures = 0

    for case in cases:
        assessment = pipeline.run(case)

        if args.json:
            print(json.dumps(assessment.model_dump(mode="json"), indent=2))
        elif args.markdown:
            print(assessment.report_markdown)
        else:
            print(_c(assessment.report_ascii, AMBER))

        if publisher:
            ok, message = publisher.publish(assessment)
            status = _c("posted", AMBER) if ok else _c("FAILED", BOLD)
            print(f"  [{status}] {message}", file=sys.stderr)
            if not ok:
                failures += 1
        print()

    return 1 if failures else 0


def cmd_assess(args: argparse.Namespace) -> int:
    if args.all:
        cases = load_all_cases()
    elif args.case_id:
        try:
            cases = [load_case(args.case_id)]
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        print("error: provide a case id or --all", file=sys.stderr)
        return 2
    return _assess(cases, args)


def cmd_fetch(args: argparse.Namespace) -> int:
    from .data.public_sources import fetch_all, save_snapshots

    results = fetch_all()
    for name, result in results.items():
        tag = _c("OK", AMBER) if result.ok else "FAIL"
        print(f"[{tag}] {name}: {result.note}")
    for path in save_snapshots(results):
        print(f"  wrote {path}")
    return 0 if all(r.ok for r in results.values()) else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plagueshield",
        description="Multi-agent decision support for suspected Yersinia pestis "
        "cases. Decision support only — does not prescribe treatment.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List bundled case records").set_defaults(
        func=cmd_list
    )

    a = sub.add_parser("assess", help="Run the agent pipeline over a case")
    a.add_argument("case_id", nargs="?", help="Case id, e.g. PS-SYN-DISC-01")
    a.add_argument("--all", action="store_true", help="Assess every bundled case")
    a.add_argument("--json", action="store_true", help="Emit the full assessment")
    a.add_argument("--markdown", action="store_true", help="Emit the markdown report")
    a.add_argument("--publish", action="store_true", help="POST to the dashboard")
    a.add_argument("--url", default="http://127.0.0.1:8000", help="Dashboard base URL")
    a.add_argument(
        "--live-evidence",
        action="store_true",
        help="Verify guidance URLs live instead of using the bundled snapshot",
    )
    a.set_defaults(func=cmd_assess)

    f = sub.add_parser("fetch", help="Refresh public data snapshots")
    f.set_defaults(func=cmd_fetch)

    s = sub.add_parser("serve", help="Run the dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
