#!/usr/bin/env python3
"""Run deterministic offline unit and integration tests with import guards."""
import importlib.abc
from importlib.machinery import PathFinder
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / 'tests'
SRC_DIR = ROOT / 'src'
AGENT_DIR = SRC_DIR / 'agent'
AGENT_ENTRYPOINT = (AGENT_DIR / 'agent.py').resolve()

FORBIDDEN_ROOT_MODULES = frozenset({
    'torch',
    'ultralytics',
    'open3d',
    'habitat',
    'habitat_sim',
    'transformers',
    'huggingface_hub',
    'mobileclip',
    'mapper',
    'simWrapper',
})
FORBIDDEN_MODULES = frozenset({
    'agent.agent',
})

OFFLINE_TEST_MODULES = [
    'test_v3_boundary_height',
    'test_v3_navigation_control',
    'test_v3_waypoint_reachability',
    'test_v3_memory_cost',
    'test_v3_recovery_lifecycle',
    'test_v3_recovery_cutoff',
    'test_v3_verification_contract',
    'test_v3_retest_metrics',
    'test_v3_terminal_integration',
    'test_v3_stair_sequence',
    'test_v3_semantic_evidence',
    'test_v3_endpoint_grounding',
    'test_v3_endpoint_integration',
    'test_v3_route_progress',
    'test_v3_route_prefix_stability',
    'test_v3_grounding_comparison',
    'test_v3_verifier_evidence_repair',
    'test_v3_initial_arrival',
    'test_v3_landing_observation',
    'test_v3_endpoint_relocalization',
    'test_v3_planner_history_protocol',
    'test_v3_target_identity',
    'test_failure_segments',
    'test_v3_turn_execution',
    'test_habitat_geometry',
    'test_structural_endpoints',
    'test_endpoint_controller',
    'test_recovery_supervisor',
    'test_habitat_control_integration',
    'test_task_visual_perception',
    'test_visual_candidate_mapping',
    'test_visual_candidate_update',
    'test_visual_candidate_binding',
    'test_spatial_relation',
    'test_verification_evidence',
    'test_verification_bev',
    'test_verification_inputs',
    'test_visual_grounding_integration',
    'test_visual_verifier_contract',
    'test_visual_turn_grounding',
    'test_visual_stair_grounding',
    'test_verification_request_integration',
]


class OfflineImportGuard(importlib.abc.MetaPathFinder):
    """Forbid heavyweight model, simulator, and misrouted agent entrypoint imports."""

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition('.')[0]
        if (root in FORBIDDEN_ROOT_MODULES
                or fullname in FORBIDDEN_MODULES
                or any(fullname.startswith(mod + '.') for mod in FORBIDDEN_MODULES)):
            raise ImportError(
                f"Forbidden heavyweight/offline import '{fullname}' in offline test runner."
            )
        if fullname == 'agent' or fullname.startswith('agent.'):
            search_path = path if fullname == 'agent' else None
            spec = PathFinder.find_spec('agent', search_path)
            if spec is not None and spec.origin is not None:
                origin = Path(spec.origin).resolve()
                if origin == AGENT_ENTRYPOINT or origin.name == 'agent.py':
                    raise ImportError(
                        "Forbidden heavyweight/offline import: 'agent' resolved to "
                        f"{spec.origin} instead of the agent package (check sys.path)."
                    )
        return None


def configure_sys_path():
    resolved_agent = AGENT_DIR.resolve()
    resolved_tests = TESTS_DIR.resolve()
    resolved_src = SRC_DIR.resolve()
    cleaned = []
    for entry in sys.path:
        try:
            resolved = Path(entry or '.').resolve()
        except OSError:
            cleaned.append(entry)
            continue
        if resolved in {resolved_agent, resolved_tests, resolved_src}:
            continue
        cleaned.append(entry)
    sys.path[:] = [str(TESTS_DIR), str(SRC_DIR), *cleaned]


def install_import_guard():
    if not any(isinstance(finder, OfflineImportGuard) for finder in sys.meta_path):
        sys.meta_path.insert(0, OfflineImportGuard())


def main():
    configure_sys_path()
    install_import_guard()
    suite = unittest.defaultTestLoader.loadTestsFromNames(OFFLINE_TEST_MODULES)
    empty_modules = [
        name for name, module_suite in zip(OFFLINE_TEST_MODULES, suite)
        if module_suite.countTestCases() <= 0
    ]
    if empty_modules:
        raise RuntimeError(
            'Offline test module(s) loaded 0 unittest cases: '
            + ', '.join(empty_modules)
        )
    if suite.countTestCases() <= 0:
        raise RuntimeError('No offline tests were loaded.')
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    if not result.wasSuccessful() or result.testsRun <= 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

