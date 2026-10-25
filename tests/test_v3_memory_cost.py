"""Offline v3 memory regressions; no Habitat, chat API, or result-tree access."""

import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from navigation_memory import (
    MemoryBackendError,
    NavigationMemory,
    clear_shared_navigation_memories,
    close_all_shared_navigation_memories,
    evict_shared_navigation_memory,
    get_memory_cache_key,
    get_shared_navigation_memory,
)


def verified_record(episode="ep-1", subtask="P1", scene="scene-a"):
    record = {
        "memory_schema_version": 2,
        "scene_id": scene,
        "subtask_id": subtask,
        "subtask": "walk past the stairs",
        "task_instruction": "walk past the stairs to the door",
        "recognized_objects": ["stairs", "door"],
        "outcome": True,
        "verified": True,
        "retrieval_eligible": True,
        "completion_evidence_reliable": True,
        "evidence_status": "subtask_verified",
        "verifier_history": [{"available": True, "outcome": True, "contract_conflict": False}],
    }
    if episode is not None:
        record["episode_id"] = episode
    return record


class FakeMem0:
    def __init__(self, results=None, add_error=None):
        self.results = results or []
        self.add_error = add_error
        self.add_calls = []
        self.search_calls = []

    def add(self, *args, **kwargs):
        self.add_calls.append((args, kwargs))
        if self.add_error:
            raise self.add_error
        return [{"id": "fake-vector-id"}]

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return {"results": self.results}


class MemoryCostTests(unittest.TestCase):
    def setUp(self):
        clear_shared_navigation_memories()
        self.tmp = tempfile.TemporaryDirectory(prefix="hsgm-memory-test-")
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(clear_shared_navigation_memories)
        self.root = Path(self.tmp.name)
        old_env_backend = os.environ.pop("HSGM_MEMORY_BACKEND", None)
        if old_env_backend is not None:
            self.addCleanup(os.environ.__setitem__, "HSGM_MEMORY_BACKEND", old_env_backend)

    def memory(self, user="user-a", agent="agent-a"):
        return NavigationMemory(
            storage_path=str(self.root), user_id=user, agent_id=agent, backend="local"
        )

    @staticmethod
    def attach_fake(memory, fake):
        # Exercise the required-Mem0 path without loading embeddings or a chat service.
        memory.requested_backend = "mem0"
        memory.backend = "mem0"
        memory._mem0 = fake

    def test_audit_cache_first_read_append_partial_line_and_replace(self):
        memory = self.memory()
        initial = {
            "id": "first", "role": "episode_summary", "user_id": "user-a",
            "agent_id": "agent-a", "record": {"episode_id": "one"},
        }
        first_line = (json.dumps(initial) + "\n").encode()
        memory.audit_path.write_bytes(first_line)
        self.assertEqual([r["id"] for r in memory._read_local()], ["first"])
        first_bytes = memory.status()["audit_bytes_read"]
        self.assertEqual(first_bytes, len(first_line))
        self.assertEqual([r["id"] for r in memory._read_local()], ["first"])
        self.assertEqual(memory.status()["audit_bytes_read"], first_bytes)

        second = dict(initial, id="second")
        second_line = (json.dumps(second) + "\n").encode()
        with memory.audit_path.open("ab") as handle:
            handle.write(second_line[:12])
        self.assertEqual([r["id"] for r in memory._read_local()], ["first"])
        with memory.audit_path.open("ab") as handle:
            handle.write(second_line[12:])
        self.assertEqual([r["id"] for r in memory._read_local()], ["first", "second"])
        self.assertEqual(memory._audit_offset, len(first_line) + len(second_line))
        self.assertEqual(memory.status()["audit_full_loads"], 1)

        replacement = self.root / "replacement.jsonl"
        replacement.write_text(json.dumps(dict(initial, id="replacement")) + "\n")
        os.replace(replacement, memory.audit_path)
        self.assertEqual([r["id"] for r in memory._read_local()], ["replacement"])
        self.assertEqual(memory.status()["audit_full_loads"], 2)

    def test_index_once_per_episode_and_subtask_but_not_without_id(self):
        memory = self.memory()
        fake = FakeMem0()
        self.attach_fake(memory, fake)
        first = memory.add(verified_record())
        duplicate = memory.add(verified_record())
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(duplicate["id"], first["id"])
        self.assertEqual(len(fake.add_calls), 1)
        self.assertEqual(memory.status()["local_records"], 1)
        self.assertEqual(memory.status()["duplicate_add_count"], 1)

        memory.add(verified_record(episode="ep-2"))
        memory.add(verified_record(subtask="P2"))
        memory.add(verified_record(episode=None))
        memory.add(verified_record(episode=None))
        self.assertEqual(len(fake.add_calls), 5)
        self.assertEqual(memory.status()["local_records"], 5)
        self.assertEqual(memory.status()["mem0_write_attempt_count"], 5)
        self.assertEqual(memory.status()["mem0_write_count"], 5)
        self.assertTrue(all(kwargs["infer"] is False for _, kwargs in fake.add_calls))

    def test_typeerror_never_retries_with_llm_and_preserves_audit(self):
        memory = self.memory()
        fake = FakeMem0(add_error=TypeError("infer is unsupported"))
        self.attach_fake(memory, fake)
        with self.assertRaisesRegex(MemoryBackendError, "LLM fallback forbidden"):
            memory.add(verified_record())
        self.assertEqual(len(fake.add_calls), 1)
        self.assertIs(fake.add_calls[0][1]["infer"], False)
        lines = [json.loads(line) for line in memory.audit_path.read_text().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["index_status"], "pending")
        self.assertEqual(lines[1]["event"], "memory_index_result")
        self.assertEqual(lines[1]["audit_id"], lines[0]["id"])
        self.assertEqual(lines[1]["index_status"], "failed")
        self.assertIn("LLM fallback forbidden", lines[1]["index_error"])
        reopened = self.memory()
        self.assertEqual(reopened._read_local()[0]["index_status"], "failed")
        self.assertEqual(memory.status()["mem0_write_attempt_count"], 1)
        self.assertEqual(memory.status()["mem0_write_count"], 0)
        self.assertEqual(memory.status()["operation_costs"]["add"]["errors"], 1)

    def test_search_requires_matching_scope_scene_and_local_audit(self):
        current = self.memory()
        good = current.add(verified_record())
        other_scene = current.add(verified_record(episode="ep-2", scene="scene-b"))
        other_user = self.memory(user="user-b").add(verified_record(episode="ep-3"))
        other_agent = self.memory(agent="agent-b").add(verified_record(episode="ep-4"))
        fake = FakeMem0(results=[
            {"memory": "missing audit", "metadata": {"hsgm_audit_id": "unknown"}},
            *({"memory": row["id"], "metadata": {"hsgm_audit_id": row["id"]}}
              for row in (other_user, other_agent, other_scene, good)),
        ])
        self.attach_fake(current, fake)
        found = current.search("stairs door", limit=10, scene_id="scene-a")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["record"]["episode_id"], "ep-1")
        self.assertEqual(found[0]["metadata"]["hsgm_audit_id"], good["id"])
        self.assertEqual(len(fake.search_calls), 1)
        self.assertEqual(fake.search_calls[0]["user_id"], "user-a")
        self.assertEqual(fake.search_calls[0]["filters"], {
            "hsgm_agent_id": "agent-a", "hsgm_verified": True,
            "hsgm_role": "verified_subtask", "scene_id": "scene-a",
        })

    def test_unverified_audit_cannot_become_advice(self):
        memory = self.memory()
        unverified = verified_record()
        unverified["retrieval_eligible"] = False
        row = memory.add(unverified)
        fake = FakeMem0(results=[{
            "memory": "looks useful",
            "metadata": {"hsgm_audit_id": row["id"], "hsgm_verified": True},
        }])
        self.attach_fake(memory, fake)
        self.assertEqual(memory.search("stairs", scene_id="scene-a"), [])
        self.assertEqual(len(fake.add_calls), 0)
        self.assertEqual(row["index_status"], "audit_only")

    def test_status_counters_and_agent_configuration_wiring(self):
        memory = self.memory()
        memory.add(verified_record())
        memory.search("stairs", scene_id="scene-a")
        status = memory.status()
        self.assertEqual(status["local_records"], 1)
        self.assertEqual(status["audit_append_count"], 1)
        self.assertEqual(status["operation_costs"]["add"]["calls"], 1)
        self.assertEqual(status["operation_costs"]["search"]["calls"], 1)
        self.assertGreaterEqual(status["operation_costs"]["add"]["total_seconds"], 0)
        self.assertGreaterEqual(status["operation_costs"]["search"]["total_seconds"], 0)

        agent_tree = ast.parse((SRC / "agent/agent.py").read_text())
        calls = [node for node in ast.walk(agent_tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "get_shared_navigation_memory"]
        self.assertEqual(len(calls), 1)
        self.assertEqual({keyword.arg for keyword in calls[0].keywords},
                         {"storage_path", "user_id", "agent_id", "backend"})
        raw_nm_calls = [node for node in ast.walk(agent_tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name) and node.func.id == "NavigationMemory"]
        self.assertEqual(len(raw_nm_calls), 0)

        runner_tree = ast.parse((SRC / "run_experiments.py").read_text())
        episode_assignments = [node for node in ast.walk(runner_tree)
                               if isinstance(node, ast.Assign)
                               and any(isinstance(target, ast.Attribute)
                                       and target.attr == "_current_episode_id"
                                       for target in node.targets)]
        self.assertEqual(len(episode_assignments), 1)
        self.assertIsInstance(episode_assignments[0].value, ast.Name)
        self.assertEqual(episode_assignments[0].value.id, "episode_ndx")

    def test_process_shared_memory_same_config_and_audit_retention(self):
        mem1 = get_shared_navigation_memory(storage_path=str(self.root), backend="local")
        mem2 = get_shared_navigation_memory(storage_path=str(self.root), backend="local")
        self.assertIs(mem1, mem2)
        self.assertTrue(mem1.status()["is_shared"])
        self.assertIs(NavigationMemory.get_shared(storage_path=str(self.root), backend="local"), mem1)

        row1 = mem1.add(verified_record(episode="ep-1", subtask="P1"))
        self.assertEqual(mem1.status()["local_records"], 1)
        self.assertEqual(mem1.status()["audit_full_loads"], 1)

        # Episode completion must not close shared memory
        mem1.close()
        self.assertFalse(mem1._closed)
        self.assertFalse(mem1.status()["closed"])

        # Subsequent episode in same process gets same instance and retains audit index
        mem_ep2 = get_shared_navigation_memory(storage_path=str(self.root), backend="local")
        self.assertIs(mem_ep2, mem1)

        found = mem_ep2.search("stairs door", scene_id="scene-a")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["id"], row1["id"])
        self.assertEqual(found[0]["record"]["episode_id"], "ep-1")
        self.assertEqual(mem_ep2.status()["audit_full_loads"], 1)

        mem_ep2.add(verified_record(episode="ep-2", subtask="P2"))
        self.assertEqual(mem_ep2.status()["local_records"], 2)
        self.assertEqual(mem_ep2.status()["audit_full_loads"], 1)

    def test_process_shared_memory_isolation_and_secret_protection(self):
        mem_base = get_shared_navigation_memory(storage_path=str(self.root), backend="local")

        # Different storage_path is isolated
        alt_root = Path(self.tmp.name) / "alt_storage"
        mem_diff_path = get_shared_navigation_memory(storage_path=str(alt_root), backend="local")
        self.assertIsNot(mem_base, mem_diff_path)

        # Different user_id is isolated
        mem_diff_user = get_shared_navigation_memory(
            storage_path=str(self.root), user_id="user_diff", backend="local"
        )
        self.assertIsNot(mem_base, mem_diff_user)

        # Different agent_id is isolated
        mem_diff_agent = get_shared_navigation_memory(
            storage_path=str(self.root), agent_id="agent_diff", backend="local"
        )
        self.assertIsNot(mem_base, mem_diff_agent)

        # Different requested backend is isolated (temporarily mock _try_init_mem0 to prevent real Mem0/HF/API init)
        with unittest.mock.patch.object(NavigationMemory, "_try_init_mem0"):
            mem_diff_backend = get_shared_navigation_memory(
                storage_path=str(self.root), backend="auto"
            )
            self.assertIsNot(mem_base, mem_diff_backend)

        # Non-secret embedding env vars affect cache key and create isolated instances
        old_device = os.environ.get("HSGM_MEM0_EMBEDDING_DEVICE")
        try:
            os.environ["HSGM_MEM0_EMBEDDING_DEVICE"] = "cuda:0"
            mem_diff_device = get_shared_navigation_memory(
                storage_path=str(self.root), backend="local"
            )
            self.assertIsNot(mem_base, mem_diff_device)
        finally:
            if old_device is not None:
                os.environ["HSGM_MEM0_EMBEDDING_DEVICE"] = old_device
            else:
                os.environ.pop("HSGM_MEM0_EMBEDDING_DEVICE", None)

        # Secret protection: API key changes must NEVER alter cache key or appear in keys/repr/status
        old_key = os.environ.get("OPENAI_API_KEY")
        secret1 = "test-only-not-an-api-key-11111111"
        secret2 = "test-only-not-an-api-key-22222222"
        try:
            os.environ["OPENAI_API_KEY"] = secret1
            key1 = get_memory_cache_key(storage_path=str(self.root), backend="local")
            os.environ["OPENAI_API_KEY"] = secret2
            key2 = get_memory_cache_key(storage_path=str(self.root), backend="local")

            self.assertEqual(key1, key2)
            self.assertNotIn(secret1, str(key1))
            self.assertNotIn(secret2, str(key2))
            self.assertNotIn(secret1, repr(mem_base))
            self.assertNotIn(secret2, repr(mem_base))
            self.assertNotIn(secret1, json.dumps(mem_base.status()))
            self.assertNotIn(secret2, json.dumps(mem_base.status()))
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    def test_process_shared_memory_eviction_and_cleanup(self):
        mem = get_shared_navigation_memory(storage_path=str(self.root), backend="local")
        evicted = evict_shared_navigation_memory(mem)
        self.assertTrue(evicted)
        mem_new = get_shared_navigation_memory(storage_path=str(self.root), backend="local")
        self.assertIsNot(mem, mem_new)

        fake_failing = FakeMem0(add_error=RuntimeError("backend connection failed"))
        self.attach_fake(mem_new, fake_failing)
        with self.assertRaises(MemoryBackendError):
            mem_new.add(verified_record())
        self.assertTrue(mem_new._closed)
        mem_recovered = get_shared_navigation_memory(storage_path=str(self.root), backend="local")
        self.assertIsNot(mem_new, mem_recovered)

        mem_active = get_shared_navigation_memory(storage_path=str(self.root), backend="local")
        clear_shared_navigation_memories()
        self.assertTrue(mem_active._closed)
        mem_after_clear = get_shared_navigation_memory(storage_path=str(self.root), backend="local")
        self.assertIsNot(mem_active, mem_after_clear)

    def test_agent_memory_wiring_enabled_guard(self):
        agent_tree = ast.parse((SRC / "agent/agent.py").read_text())
        guard_found = False
        for node in ast.walk(agent_tree):
            if isinstance(node, ast.If):
                call_nodes = [
                    n for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "get_shared_navigation_memory"
                ]
                if call_nodes:
                    guard_found = True
                    assign_nones = [
                        n for n in ast.walk(node)
                        if isinstance(n, ast.Assign) and any(
                            isinstance(t, ast.Attribute) and t.attr == "navigation_memory"
                            for t in n.targets
                        ) and (isinstance(n.value, ast.Constant) and n.value.value is None)
                    ]
                    self.assertGreaterEqual(len(assign_nones), 1)
        self.assertTrue(guard_found, "get_shared_navigation_memory must be guarded by memory.enabled")


if __name__ == "__main__":
    unittest.main()

