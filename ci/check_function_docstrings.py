"""Count actual Python function nodes without class/module coverage heuristics."""

import argparse
import ast
from pathlib import Path


def function_inventory(path):
    """Yield (line, qualified name, documented) for every function in one file.

    Include async functions, constructors, private methods, and nested functions.
    Require a nonempty literal docstring on the definition itself; inherited or
    dynamically assigned documentation does not count. Syntax errors propagate.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def walk(node, parents=()):
        """Track lexical class/function names while visiting every AST child."""
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            parents = parents + (node.name,)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.lineno, ".".join(parents), bool(ast.get_docstring(node))
        for child in ast.iter_child_nodes(node):
            yield from walk(child, parents)

    yield from walk(tree)


def main():
    """Print missing function locations and fail below the requested percentage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("versa")])
    parser.add_argument("--fail-under", type=float, default=80.0)
    args = parser.parse_args()
    if not 0 <= args.fail_under <= 100:
        parser.error("--fail-under must be between 0 and 100")
    files = set()
    for path in args.paths:
        if not path.exists():
            parser.error(f"path does not exist: {path}")
        files.update(path.rglob("*.py") if path.is_dir() else [path])
    total = covered = 0
    for path in sorted(files):
        for line, name, documented in function_inventory(path):
            total += 1
            covered += documented
            if not documented:
                print(f"{path}:{line}: missing {name}")
    if not total:
        parser.error("no Python functions found")
    percent = 100 * covered / total
    print(f"Function docstrings: {covered}/{total} ({percent:.2f}%)")
    return 0 if percent >= args.fail_under else 1


if __name__ == "__main__":
    raise SystemExit(main())
