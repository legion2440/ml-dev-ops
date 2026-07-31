"""Generate the repository dependency graph documentation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = REPOSITORY_ROOT / "dependency-graph.json"
OUTPUT_PATH = REPOSITORY_ROOT / "docs/generated/dependency-graph.md"


def _mermaid_id(module_id: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", module_id)
    return f"module_{identifier}" if identifier[:1].isdigit() else identifier


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _node_lines(nodes: list[str]) -> list[str]:
    return [f'    {_mermaid_id(node)}["{node}"]' for node in nodes]


def render_dependency_graph(graph: dict[str, Any]) -> str:
    """Return deterministic Markdown for a validated dependency graph."""
    nodes: list[str] = graph["nodes"]
    allowed_edges: list[dict[str, str]] = graph["allowed_edges"]
    forbidden_edges: list[dict[str, str]] = graph["forbidden_edges"]

    lines = [
        "# Dependency graph",
        "",
        "> Generated from `dependency-graph.json` by "
        "`scripts/generate_dependency_graph.py`. Do not edit manually.",
        "",
        "An arrow points from a consumer to the module it is allowed to depend on.",
        "",
        "## Allowed dependencies",
        "",
        "```mermaid",
        "flowchart LR",
        *_node_lines(nodes),
    ]
    lines.extend(
        f"    {_mermaid_id(edge['from'])} -->|{edge['type']}| {_mermaid_id(edge['to'])}"
        for edge in allowed_edges
    )
    lines.extend(
        [
            "```",
            "",
            "| From | To | Type | Rationale |",
            "| --- | --- | --- | --- |",
        ]
    )
    lines.extend(
        "| {from_} | {to} | {type_} | {rationale} |".format(
            from_=_escape_table_cell(edge["from"]),
            to=_escape_table_cell(edge["to"]),
            type_=_escape_table_cell(edge["type"]),
            rationale=_escape_table_cell(edge["rationale"]),
        )
        for edge in allowed_edges
    )
    lines.extend(
        [
            "",
            "## Forbidden dependencies",
            "",
            "```mermaid",
            "flowchart LR",
            *_node_lines(nodes),
        ]
    )
    lines.extend(
        f"    {_mermaid_id(edge['from'])} -.->|forbidden| {_mermaid_id(edge['to'])}"
        for edge in forbidden_edges
    )
    lines.extend(
        [
            "```",
            "",
            "| From | To | Reason |",
            "| --- | --- | --- |",
        ]
    )
    lines.extend(
        "| {from_} | {to} | {reason} |".format(
            from_=_escape_table_cell(edge["from"]),
            to=_escape_table_cell(edge["to"]),
            reason=_escape_table_cell(edge["reason"]),
        )
        for edge in forbidden_edges
    )
    return "\n".join(lines) + "\n"


def _load_graph() -> dict[str, Any]:
    with GRAPH_PATH.open(encoding="utf-8") as graph_file:
        return json.load(graph_file)


def _write_lf(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the generated document is missing or stale.",
    )
    args = parser.parse_args()

    try:
        expected = render_dependency_graph(_load_graph())
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        print(f"[ERROR] Cannot generate dependency graph: {error}", file=sys.stderr)
        return 1

    if args.check:
        if not OUTPUT_PATH.is_file():
            print(
                "[ERROR] Generated dependency graph is missing. "
                "Run: python scripts/generate_dependency_graph.py",
                file=sys.stderr,
            )
            return 1
        actual = OUTPUT_PATH.read_text(encoding="utf-8")
        if actual != expected:
            print(
                "[ERROR] Generated dependency graph is stale. "
                "Run: python scripts/generate_dependency_graph.py",
                file=sys.stderr,
            )
            return 1
        print("[OK] Generated dependency graph is current.")
        return 0

    _write_lf(OUTPUT_PATH, expected)
    print("[OK] Generated docs/generated/dependency-graph.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
