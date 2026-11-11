"""Readability gate for Claude-owned config/channel modules (CODING_STANDARDS.md).

Parallels Codex's tests/test_coding_standards.py for its modules: it parses each
gated module and fails if the module lacks a docstring, omits a "how to extend"
note, or has a public class/function/method without a docstring. Pure static
analysis — imports nothing, changes no behaviour.
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GATED_MODULES = (
    "config/run.py",
    "config/servers.py",
    "config/workload.py",
    "config/reward.py",
    "config/mobility.py",
    "config/radio.py",
    "config/topology.py",
    "config/energy.py",
    "config/evaluation.py",
    "config/_provenance.py",
    "config/_references.py",
    "metrics/channel.py",
)

# A module must tell a researcher where to change things (CODING_STANDARDS §2/§5).
EXTENSION_PHRASES = ("how to extend", "to extend", "to add", "to change",
                     "one place", "add an entry", "add a ")


def _tree(relative_path):
    return ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))


class TestConfigReadabilityGate(unittest.TestCase):
    def test_module_docstring_with_extension_note(self):
        for rel in GATED_MODULES:
            doc = ast.get_docstring(_tree(rel))
            with self.subTest(module=rel):
                self.assertTrue(doc, f"{rel} is missing a module docstring")
                self.assertTrue(
                    any(p in doc.lower() for p in EXTENSION_PHRASES),
                    f"{rel} module docstring lacks a 'how to extend' note",
                )

    def test_public_api_has_docstrings(self):
        for rel in GATED_MODULES:
            for node in _tree(rel).body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                        and not node.name.startswith("_"):
                    with self.subTest(module=rel, symbol=node.name):
                        self.assertTrue(ast.get_docstring(node),
                                        f"{rel}:{node.name} lacks a docstring")
                    if isinstance(node, ast.ClassDef):
                        for sub in node.body:
                            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                                    and not sub.name.startswith("_"):
                                with self.subTest(module=rel, symbol=f"{node.name}.{sub.name}"):
                                    self.assertTrue(
                                        ast.get_docstring(sub),
                                        f"{rel}:{node.name}.{sub.name} lacks a docstring")


if __name__ == "__main__":
    unittest.main()
