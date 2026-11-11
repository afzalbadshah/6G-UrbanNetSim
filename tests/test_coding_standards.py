"""Automated readability gate for the new packet/KPI implementation.

This does not judge prose quality. It prevents the easiest regressions: adding a
public API with no explanation, or adding a module without telling researchers
where its supported extension point lives.
"""

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

READABILITY_GATED_MODULES = (
    "core/network_adapter.py",
    "core/network_runtime.py",
    "core/task_execution.py",
    "config/units.py",
    "entities/resources.py",
    "metrics/cost.py",
    "metrics/delay.py",
    "metrics/energy.py",
    "metrics/kpi.py",
    "network/__init__.py",
    "network/accounting.py",
    "network/contracts.py",
    "network/engine.py",
    "network/links.py",
    "network/packetization.py",
    "network/profiles.py",
    "network/reliability.py",
    "network/routing.py",
    "network/topology.py",
    "network/wired.py",
)


def public_definitions(tree: ast.AST):
    """Yield public classes, functions, methods, and properties."""

    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                yield node


class CodingStandardsTests(unittest.TestCase):
    """Require basic researcher-facing guidance on Codex-owned modules."""

    def test_modules_name_their_supported_extension_point(self):
        for relative_path in READABILITY_GATED_MODULES:
            with self.subTest(module=relative_path):
                source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
                module_docstring = ast.get_docstring(ast.parse(source)) or ""
                self.assertIn("How to extend", module_docstring)

    def test_public_definitions_have_docstrings(self):
        for relative_path in READABILITY_GATED_MODULES:
            source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
            tree = ast.parse(source)
            missing = [
                f"line {node.lineno}: {node.name}"
                for node in public_definitions(tree)
                if ast.get_docstring(node) is None
            ]
            with self.subTest(module=relative_path):
                self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
