"""Dependency-light checks for the Ubuntu 20.04 Python 3.8 source contract.

A grammar check on newer Python is not a substitute for running in Python 3.8.
The container smoke job must also execute this test on the target interpreter.
"""
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from navigation_control import legal_action_objects, normalize_action_object


class Python38CompatibilityTests(unittest.TestCase):
    def test_python38_grammar(self):
        for directory in ('src', 'scripts', 'tests'):
            for path in sorted((ROOT / directory).rglob('*.py')):
                if '__pycache__' not in path.parts:
                    with self.subTest(path=str(path.relative_to(ROOT))):
                        ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=8)

    def test_no_post38_string_methods_in_production(self):
        for path in sorted((ROOT / 'src').rglob('*.py')):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, ('removeprefix', 'removesuffix'), str(path))

    def test_experiment_annotations_are_postponed(self):
        tree = ast.parse((ROOT / 'src/run_experiments.py').read_text(encoding='utf-8'))
        self.assertTrue(any(isinstance(n, ast.ImportFrom) and n.module == '__future__'
                            and any(a.name == 'annotations' for a in n.names) for n in tree.body))

    def test_turn_prefix_keeps_exact_semantics(self):
        for raw, expected in [('TURN_LEFT', 'LEFT'), ('left', 'LEFT'), ('turn_right', 'RIGHT')]:
            with self.subTest(raw=raw):
                result = normalize_action_object({'turn': raw})
                self.assertEqual(result['action'], {'type': 'turn', 'direction': expected})
        self.assertNotIn('action', normalize_action_object({'turn': 'TURN_TURN_LEFT'}))
        actions = legal_action_objects({'TURN_BACK': {'type': 'turn'}}, completion=False)
        self.assertEqual(actions, [{'type': 'turn', 'direction': 'BACK'}])


if __name__ == '__main__':
    unittest.main()
