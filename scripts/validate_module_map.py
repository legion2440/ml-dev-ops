"""Validate module metadata, architectural dependencies, and generated artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_MAP_PATH = REPOSITORY_ROOT / "module-map.json"
DEPENDENCY_GRAPH_PATH = REPOSITORY_ROOT / "dependency-graph.json"
MODULE_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/module-map.schema.json"
GRAPH_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/dependency-graph.schema.json"
REFERENCE_FIELDS = (
    "entrypoints",
    "public_interfaces",
    "tests",
    "docs",
    "generated_artifacts",
    "owned_configuration",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as json_file:
        value = json.load(json_file)
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _format_json_path(parts: Iterable[object]) -> str:
    rendered = "$"
    for part in parts:
        rendered += f"[{part}]" if isinstance(part, int) else f".{part}"
    return rendered


def _validate_schema(
    instance: dict[str, Any],
    schema: dict[str, Any],
    source_name: str,
    errors: list[str],
) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        errors.append(f"Invalid schema for {source_name}: {error.message}")
        return

    validator = Draft202012Validator(schema)
    for validation_error in sorted(
        validator.iter_errors(instance),
        key=lambda item: _format_json_path(item.absolute_path),
    ):
        location = _format_json_path(validation_error.absolute_path)
        errors.append(f"{source_name} {location}: {validation_error.message}")


def _iter_references(module_map: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for module in module_map.get("modules", []):
        module_id = module.get("id", "<unknown>")
        for field in REFERENCE_FIELDS:
            for reference in module.get(field, []):
                yield f"module {module_id}.{field}", reference
    for reference in module_map.get("repository_artifacts", []):
        yield "repository_artifacts", reference


def _check_modules(module_map: dict[str, Any], errors: list[str]) -> set[str]:
    modules = module_map.get("modules", [])
    module_ids = [module.get("id") for module in modules]
    roots = [module.get("root") for module in modules]

    string_module_ids = [module_id for module_id in module_ids if isinstance(module_id, str)]
    string_roots = [root for root in roots if isinstance(root, str)]

    duplicate_ids = sorted(
        module_id
        for module_id in set(string_module_ids)
        if string_module_ids.count(module_id) > 1
    )
    if duplicate_ids:
        errors.append(f"Duplicate module IDs: {', '.join(duplicate_ids)}")

    duplicate_roots = sorted(
        root for root in set(string_roots) if string_roots.count(root) > 1
    )
    if duplicate_roots:
        errors.append(f"Duplicate module roots: {', '.join(duplicate_roots)}")

    for module in modules:
        module_id = module.get("id", "<unknown>")
        root = module.get("root")
        if isinstance(root, str) and not (REPOSITORY_ROOT / root).is_dir():
            errors.append(f"Module {module_id} root does not exist: {root}")

        for field in REFERENCE_FIELDS:
            references = module.get(field, [])
            paths = [
                reference.get("path")
                for reference in references
                if isinstance(reference, dict)
            ]
            duplicate_paths = sorted(path for path in set(paths) if paths.count(path) > 1)
            if duplicate_paths:
                errors.append(
                    f"Module {module_id}.{field} contains duplicate paths: "
                    f"{', '.join(duplicate_paths)}"
                )

    return set(string_module_ids)


def _check_reference_statuses(module_map: dict[str, Any], errors: list[str]) -> set[str]:
    generators: set[str] = set()
    owned_paths: dict[str, str] = {}

    for owner, reference in _iter_references(module_map):
        path_value = reference.get("path")
        status = reference.get("status")
        if not isinstance(path_value, str):
            continue

        path = REPOSITORY_ROOT / path_value
        if status in {"implemented", "generated"} and not path.exists():
            errors.append(f"{owner} marks missing path as {status}: {path_value}")

        if status == "generated":
            generator = reference.get("generator")
            sources = reference.get("sources", [])
            if isinstance(generator, str):
                generator_path = REPOSITORY_ROOT / generator
                if not generator_path.is_file():
                    errors.append(f"Generated path {path_value} has missing generator: {generator}")
                else:
                    generators.add(generator)
            for source in sources:
                if not (REPOSITORY_ROOT / source).exists():
                    errors.append(f"Generated path {path_value} has missing source: {source}")

        if ".owned_configuration" in owner or ".generated_artifacts" in owner:
            previous_owner = owned_paths.get(path_value)
            if previous_owner is not None and previous_owner != owner:
                errors.append(
                    f"Path {path_value} has multiple owners: {previous_owner} and {owner}"
                )
            owned_paths[path_value] = owner

    return generators


def _check_graph(
    graph: dict[str, Any],
    module_ids: set[str],
    errors: list[str],
) -> None:
    nodes = graph.get("nodes", [])
    node_set = {node for node in nodes if isinstance(node, str)}
    missing_nodes = sorted(module_ids - node_set)
    unknown_nodes = sorted(node_set - module_ids)
    if missing_nodes:
        errors.append(f"Dependency graph is missing modules: {', '.join(missing_nodes)}")
    if unknown_nodes:
        errors.append(f"Dependency graph has unknown modules: {', '.join(unknown_nodes)}")

    allowed_keys: set[tuple[str, str, str]] = set()
    allowed_pairs: set[tuple[str, str]] = set()
    for edge in graph.get("allowed_edges", []):
        source = edge.get("from")
        target = edge.get("to")
        edge_type = edge.get("type")
        if source not in node_set or target not in node_set:
            errors.append(f"Allowed edge references an unknown module: {source} -> {target}")
        if source == target:
            errors.append(f"Self-dependency is not allowed: {source} -> {target}")
        key = (source, target, edge_type)
        if key in allowed_keys:
            errors.append(f"Duplicate allowed edge: {source} -> {target} ({edge_type})")
        allowed_keys.add(key)
        allowed_pairs.add((source, target))

    forbidden_keys: set[tuple[str, str]] = set()
    for edge in graph.get("forbidden_edges", []):
        source = edge.get("from")
        target = edge.get("to")
        if source not in node_set or target not in node_set:
            errors.append(f"Forbidden edge references an unknown module: {source} -> {target}")
        if source == target:
            errors.append(f"Forbidden self-edge is not meaningful: {source} -> {target}")
        key = (source, target)
        if key in forbidden_keys:
            errors.append(f"Duplicate forbidden edge: {source} -> {target}")
        if key in allowed_pairs:
            errors.append(f"Dependency is both allowed and forbidden: {source} -> {target}")
        forbidden_keys.add(key)

    _check_acyclic_edges(graph, node_set, errors)


def _check_acyclic_edges(
    graph: dict[str, Any],
    nodes: set[str],
    errors: list[str],
) -> None:
    acyclic_types = set(graph.get("acyclic_edge_types", []))
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for edge in graph.get("allowed_edges", []):
        if edge.get("type") in acyclic_types:
            source = edge.get("from")
            target = edge.get("to")
            if source in nodes and target in nodes:
                adjacency[source].add(target)

    visited: set[str] = set()
    active: set[str] = set()
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        if node in active:
            cycle_start = stack.index(node)
            return [*stack[cycle_start:], node]
        if node in visited:
            return None

        active.add(node)
        stack.append(node)
        for target in sorted(adjacency[node]):
            cycle = visit(target)
            if cycle is not None:
                return cycle
        stack.pop()
        active.remove(node)
        visited.add(node)
        return None

    for node in sorted(nodes):
        cycle = visit(node)
        if cycle is not None:
            errors.append(f"Forbidden dependency cycle: {' -> '.join(cycle)}")
            return


def _check_generated_artifacts(generators: set[str], errors: list[str]) -> None:
    for generator in sorted(generators):
        process = subprocess.run(
            [sys.executable, str(REPOSITORY_ROOT / generator), "--check"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip() or "unknown error"
            errors.append(f"Generated artifact check failed for {generator}: {detail}")


def main() -> int:
    errors: list[str] = []
    try:
        module_map = _load_json(MODULE_MAP_PATH)
        graph = _load_json(DEPENDENCY_GRAPH_PATH)
        module_schema = _load_json(MODULE_SCHEMA_PATH)
        graph_schema = _load_json(GRAPH_SCHEMA_PATH)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        print(f"[ERROR] Cannot load architecture metadata: {error}", file=sys.stderr)
        return 1

    _validate_schema(module_map, module_schema, MODULE_MAP_PATH.name, errors)
    _validate_schema(graph, graph_schema, DEPENDENCY_GRAPH_PATH.name, errors)

    module_ids = _check_modules(module_map, errors)
    generators = _check_reference_statuses(module_map, errors)
    _check_graph(graph, module_ids, errors)
    _check_generated_artifacts(generators, errors)

    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(f"[FAIL] Architecture metadata validation found {len(errors)} issue(s).", file=sys.stderr)
        return 1

    print(
        f"[OK] Validated {len(module_ids)} modules, dependency rules, "
        "path statuses, and generated artifacts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
