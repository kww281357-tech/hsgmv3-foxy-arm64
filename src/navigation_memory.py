#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent navigation memory backed by Mem0.

V3 comparison runs require Mem0 and stop if its backend fails. A local JSONL
audit is always kept. Optional ``auto`` and ``local`` modes remain available
for explicit diagnostics, but are not the V3 comparison configuration.

The public interface deliberately mirrors the two Mem0 operations used by the
agent: ``search`` before planning and ``add`` after an episode. Only verified
subtask experiences enter the retrieval index; other records remain auditable.
"""

from __future__ import annotations

import atexit
import json
import uuid
from functools import wraps
import logging
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from navigation_policy import ModelServiceError

logger = logging.getLogger(__name__)
_MEMORY_SCHEMA_VERSION = 2


class MemoryBackendError(ModelServiceError):
    """A required memory backend failed; stop instead of changing the framework."""


# JSON field names are part of the persisted API and must stay ASCII/English.
# The aliases keep old memory files readable while all new writes use the
# canonical English names.
_FIELD_ALIASES = {
    "选点理由": "waypoint_reason",
    "移动过程的轨迹描述": "trajectory_description",
    "环境中识别到的物体": "recognized_objects",
}


def _json_safe(value: Any) -> Any:
    """Convert common NumPy/scalar values without importing NumPy."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _englishize_fields(value: Any) -> Any:
    """Recursively canonicalize persisted JSON keys to English names.

    Mem0 returns the original record as a JSON string in some versions, so
    JSON-valued strings are normalized as well. Natural-language strings are
    left unchanged; only object keys are part of this naming contract.
    """
    if isinstance(value, dict):
        return {
            _FIELD_ALIASES.get(str(key), str(key)): _englishize_fields(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_englishize_fields(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return value
            if isinstance(parsed, (dict, list)):
                return json.dumps(
                    _englishize_fields(parsed), ensure_ascii=False, separators=(",", ":")
                )
    return value


def _tokens(text: str) -> List[str]:
    # ``\w`` is Unicode-aware in Python and works for both English and Chinese
    # task descriptions.  Keep short tokens because object names can be short.
    return [t.lower() for t in re.findall(r"[\w]+", str(text), flags=re.UNICODE)]


def _one_line(value: Any, limit: int) -> str:
    """Bound a piece of evidence before it is placed in a planner prompt."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit].rstrip() + ("..." if len(text) > limit else "")


def _measure(operation):
    """Measure synchronous work, including failures; this is not a timeout."""
    def decorate(fn):
        @wraps(fn)
        def measured(self, *args, **kwargs):
            started = time.perf_counter()
            stats = self._costs[operation]
            stats['calls'] += 1
            try:
                return fn(self, *args, **kwargs)
            except Exception:
                stats['errors'] += 1
                raise
            finally:
                elapsed = time.perf_counter() - started
                stats['last_seconds'] = elapsed
                stats['total_seconds'] += elapsed
                stats['max_seconds'] = max(stats['max_seconds'], elapsed)
        return measured
    return decorate


class NavigationMemory:
    """Store and retrieve verified navigation experience.

    Args:
        storage_path: Directory for Mem0's local state/audit files.
        user_id: Mem0 scope.  One scope is shared across episodes so that a
            later task can retrieve similar subtasks from earlier episodes.
        agent_id: Optional second Mem0 scope for separating agent versions.

    Environment variables:
        ``HSGM_MEMORY_BACKEND=local`` disables Mem0 explicitly.
        ``HSGM_MEMORY_BACKEND=mem0`` requires Mem0 to initialize successfully.
        The default ``auto`` uses Mem0 when available and otherwise falls back
        to the local JSONL index.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        user_id: str = "hsgm_navigation",
        agent_id: str = "hsgm",
        backend: Optional[str] = None,
    ) -> None:
        default_path = os.environ.get("HSGM_MEMORY_PATH", "../results/navigation_memory")
        self.root = Path(storage_path or default_path).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.root / "memories.jsonl"
        self.user_id = str(user_id or "hsgm_navigation").replace(" ", "_")
        self.agent_id = str(agent_id or "hsgm").replace(" ", "_")
        self.requested_backend = str(
            os.environ.get("HSGM_MEMORY_BACKEND", backend or "auto")
        ).lower()
        if self.requested_backend not in {"auto", "mem0", "local"}:
            raise ValueError(f"Unknown navigation memory backend: {self.requested_backend}")
        self._mem0 = None
        self.backend = "local_lexical"
        self.last_error = ""
        self._closed = False
        self._mem0_search_count = 0
        self._mem0_write_count = 0
        self._indexed_subtasks = {}
        self._audit_rows = []
        self._audit_by_id = {}
        self._audit_signature = None
        self._audit_offset = 0
        self._audit_anchor = b''
        self._audit_mtime = None
        self._audit_bytes_read = 0
        self._audit_loads = 0
        self._audit_append_count = 0
        self._mem0_write_attempt_count = 0
        self._duplicate_add_count = 0
        self._is_shared = False
        self._costs = {op: dict(calls=0, errors=0, last_seconds=0.0,
                               total_seconds=0.0, max_seconds=0.0)
                       for op in ('search', 'add')}

        if self.requested_backend != "local":
            self._try_init_mem0()

    def _try_init_mem0(self) -> None:
        try:
            # Mem0 reads MEM0_DIR while constructing its default config.  Do
            # not overwrite a user-provided value; use this run directory as
            # the default only.
            os.environ.setdefault("MEM0_DIR", str(self.root))
            # Mem0 data remains on this machine; avoid an unrelated analytics
            # request delaying or blocking a simulator episode.
            os.environ.setdefault("MEM0_TELEMETRY", "False")
            from mem0 import Memory  # type: ignore

            # A llama.cpp generation server intentionally does not expose an
            # embeddings endpoint. When requested, use Mem0's local
            # sentence-transformers embedder and a persistent local Qdrant
            # store instead of depending on /v1/embeddings.
            # Use local embeddings independently of the chat provider; a chat
            # service need not expose a compatible /v1/embeddings endpoint.
            embedder_provider = os.environ.get("HSGM_MEM0_EMBEDDER", "local").lower()
            if embedder_provider in {"huggingface", "local"}:
                embedder_model = os.environ.get(
                    "HSGM_MEM0_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                )
                model_name = os.environ.get("OPENAI_MODEL", "Qwen/Qwen3.8-27B-FP8")
                base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
                api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
                mem0_config = {
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "collection_name": "hsgm_navigation",
                            "embedding_model_dims": 384,
                            "path": str(self.root / "qdrant"),
                            # Mem0's Qdrant provider deletes an existing local
                            # directory on initialization unless on_disk=True.
                            "on_disk": True,
                        },
                    },
                    "embedder": {
                        "provider": "huggingface",
                        "config": {
                            "model": embedder_model,
                            "model_kwargs": {
                                "device": os.environ.get("HSGM_MEM0_EMBEDDING_DEVICE", "cpu"),
                                "local_files_only": os.environ.get("HF_HUB_OFFLINE", "").lower()
                                in {"1", "true", "yes"},
                            },
                        },
                    },
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": model_name,
                            "openai_base_url": base_url,
                            "api_key": api_key,
                            "temperature": 0.1,
                            "max_tokens": int(os.environ.get("HSGM_MAX_OUTPUT_TOKENS", 512)),
                        },
                    },
                    "history_db_path": str(self.root / "history.db"),
                }
                self._mem0 = Memory.from_config(mem0_config)
            else:
                self._mem0 = Memory()
            self.backend = "mem0"
            logger.info("Navigation memory backend: Mem0 (%s)", self.root)
        except Exception as exc:  # optional dependency/service failure
            self._mem0 = None
            self.last_error = f"Mem0 unavailable: {exc}"
            if self.requested_backend == "mem0":
                self.backend = "mem0_error"
                raise MemoryBackendError(
                    f"Mem0 is required but could not initialize: {exc}"
                ) from exc
            else:
                logger.info("Mem0 unavailable; using local navigation memory: %s", exc)

    @property
    def is_mem0(self) -> bool:
        return self._mem0 is not None and self.backend == "mem0"

    def _disable_mem0(self, message: str) -> None:
        """A required backend must not silently become keyword retrieval."""
        self.last_error = message
        self.close(force=True)
        evict_shared_navigation_memory(self)
        if self.requested_backend == "mem0":
            self.backend = "mem0_error"
            raise MemoryBackendError(message)
        self.backend = "local_lexical"

    def close(self, force: bool = False) -> None:
        """Release Qdrant locks and history resources between episodes.

        Shared process-level instances ignore ordinary episode close calls
        unless force=True (used on process exit, backend eviction, or explicit teardown).
        """
        if getattr(self, "_is_shared", False) and not force:
            return
        memory = self._mem0
        self._mem0 = None
        if memory is not None:
            for attr in ("vector_store", "_telemetry_vector_store"):
                client = getattr(getattr(memory, attr, None), "client", None)
                if client is not None and callable(getattr(client, "close", None)):
                    try:
                        client.close()
                    except Exception as exc:
                        logger.warning("Could not close Mem0 %s: %s", attr, exc)
            database = getattr(memory, "db", None)
            if database is not None and callable(getattr(database, "close", None)):
                try:
                    database.close()
                except Exception as exc:
                    logger.warning("Could not close Mem0 history database: %s", exc)
        self._closed = True

    def _dedup_key(self, record, role):
        episode = record.get('episode_id')
        subtask = record.get('subtask_id')
        if (role != 'verified_subtask' or episode is None or str(episode).strip() == ''
                or subtask is None or str(subtask).strip() == ''
                or self.verification_status(record) == 'unverified'):
            return None  # No-ID callers must not collapse unrelated episodes.
        return (self.user_id, self.agent_id, str(record.get('scene_id', '')),
                str(episode), str(subtask))

    def _reset_audit_index(self):
        self._audit_rows = []
        self._audit_by_id = {}
        self._indexed_subtasks = {}
        self._audit_offset = 0
        self._audit_anchor = b''
        self._audit_mtime = None

    def _ingest_audit_row(self, row):
        if row.get('event') == 'memory_index_result':
            original = self._audit_by_id.get(row.get('audit_id'))
            if original is not None:
                original['index_status'] = row.get('index_status')
                original['index_error'] = row.get('index_error')
            return
        self._audit_rows.append(row)
        if row.get('id'):
            self._audit_by_id[row['id']] = row
        if row.get('user_id') == self.user_id and row.get('agent_id') == self.agent_id:
            key = self._dedup_key(self._record_from_row(row), row.get('role'))
            if key is not None:
                self._indexed_subtasks.setdefault(key, row)

    def _refresh_audit_index(self):
        """Read complete new JSONL lines only; invalidate rotated/truncated files."""
        try:
            with self.audit_path.open('rb') as handle:
                stat = os.fstat(handle.fileno())
                signature = (stat.st_dev, stat.st_ino)
                reset = (signature != self._audit_signature or stat.st_size < self._audit_offset)
                if not reset and self._audit_offset and stat.st_mtime_ns != self._audit_mtime:
                    # Detect truncate+regrow at the old boundary; same-size
                    # rewrites always invalidate. No full-file hashing per search.
                    handle.seek(self._audit_offset - len(self._audit_anchor))
                    anchor = handle.read(len(self._audit_anchor))
                    self._audit_bytes_read += len(anchor)
                    reset = (anchor != self._audit_anchor or stat.st_size == self._audit_offset)
                if reset:
                    self._reset_audit_index()
                    self._audit_loads += 1
                self._audit_signature = signature
                handle.seek(self._audit_offset)
                while True:
                    line = handle.readline()
                    self._audit_bytes_read += len(line)
                    if not line or not line.endswith(b'\n'):
                        break  # Retry a partially appended line on next access.
                    self._audit_offset = handle.tell()
                    self._audit_anchor = line[-64:]
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(row, dict):
                        self._ingest_audit_row(row)
                self._audit_mtime = stat.st_mtime_ns
        except FileNotFoundError:
            self._reset_audit_index()
            self._audit_signature = None
        except OSError as exc:
            self._reset_audit_index()
            self._audit_signature = None
            self.last_error = f'Cannot read local memory: {exc}'
            raise MemoryBackendError(self.last_error) from exc

    def _read_local(self) -> List[Dict[str, Any]]:
        self._refresh_audit_index()
        return list(self._audit_rows)

    def _append_local(self, record: Dict[str, Any]) -> None:
        try:
            with self.audit_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(_englishize_fields(_json_safe(record)), ensure_ascii=False) + '\n')
            self._audit_append_count += 1
            self._refresh_audit_index()
        except OSError as exc:
            self.last_error = f'Cannot write local memory: {exc}'
            raise MemoryBackendError(self.last_error) from exc

    @staticmethod
    def _normalize_mem0_result(result: Any) -> List[Dict[str, Any]]:
        if isinstance(result, dict):
            result = result.get("results", result.get("memories", []))
        if not isinstance(result, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in result:
            if isinstance(item, dict):
                row = _englishize_fields(dict(item))
            else:
                row = {"memory": str(item)}
            row["memory"] = str(row.get("memory", row.get("text", "")))
            normalized.append(row)
        return normalized

    @staticmethod
    def verification_status(record: Dict[str, Any]) -> str:
        """Admit only benchmark-supported successes or physically confirmed bad branches."""
        if not isinstance(record, dict) or record.get('memory_schema_version') != _MEMORY_SCHEMA_VERSION:
            # Legacy memories used model acceptance as ground truth. Keep them
            # in audit storage, but do not retrieve them as trusted advice.
            return 'unverified'
        history = record.get("verifier_history") or []
        if not history and isinstance(record.get("verifier"), dict):
            history = [record["verifier"]]
        valid_history = [
            value for value in history
            if isinstance(value, dict) and value.get('available') is True
            and not value.get('contract_conflict')
        ]
        latest_valid = valid_history[-1] if valid_history else {}
        if record.get('retrieval_eligible') is not True or record.get('verified') is False:
            return 'unverified'
        if (record.get('outcome') is True
                and record.get('completion_evidence_reliable', True) is True
                and record.get('evidence_status') in {'subtask_verified', 'episode_success_supported'}
                and latest_valid.get('outcome') is True):
            return 'accepted'
        rollback_subtask = record.get('subtask_key') or record.get('subtask_id')
        rollback_reason = record.get('outcome_reason')
        if (record.get('outcome') is False
                and record.get('failure_type') == 'wrong_path'
                and record.get('evidence_status') == 'rollback_confirmed'
                and record.get('rollback_confirmed') is True
                and latest_valid.get('failure_type') == 'wrong_path'
                and latest_valid.get('rollback_required') is True
                and latest_valid.get('confidence') == 'high'
                and latest_valid.get('scene_consistent') is False
                and bool(latest_valid.get('contradiction_evidence'))
                and any(
                    isinstance(event, dict)
                    and event.get('event') == 'rollback_completed'
                    and (not rollback_subtask or event.get('subtask_id') == rollback_subtask)
                    and (not rollback_reason or event.get('reason') == rollback_reason)
                    for event in record.get('rollback_events', [])
                )):
            return 'wrong_path'
        return "unverified"

    @staticmethod
    def _record_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
        record = row.get("record")
        if isinstance(record, dict):
            return record
        for field in ("memory", "text"):
            try:
                parsed = json.loads(str(row.get(field, "")))
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}

    @staticmethod
    def _observed_condition(record: Dict[str, Any]) -> str:
        """Prefer current-episode observations over a model's route rationale."""
        observations = []
        for condition in record.get("completion_conditions") or []:
            if not isinstance(condition, dict) or condition.get("status") != "satisfied":
                continue
            detail = _one_line(condition.get("evidence") or condition.get("text"), 130)
            if detail and detail not in observations:
                observations.append(detail)
            if len(observations) == 2:
                break
        if not observations:
            verifier = record.get("verifier")
            if isinstance(verifier, dict):
                detail = _one_line(verifier.get("evidence"), 180)
                if detail:
                    observations.append(detail)
        return "; ".join(observations)

    @staticmethod
    def _historical_planner_cue(record: Dict[str, Any]) -> str:
        """Keep a past proposed action useful without claiming it was executed."""
        descriptions = record.get("trajectory_description") or []
        if not isinstance(descriptions, (list, tuple)):
            descriptions = [descriptions]
        movement = next(
            (_one_line(item, 75) for item in reversed(descriptions) if _one_line(item, 75)),
            "",
        )
        reason = _one_line(record.get("waypoint_reason"), 75)
        if movement and reason and reason.lower() not in movement.lower():
            return f"movement: {movement}; reason: {reason}"
        return movement or reason

    @staticmethod
    def _detected_labels(record: Dict[str, Any], field: str = "recognized_objects") -> str:
        labels = record.get(field) or []
        if not isinstance(labels, (list, tuple)):
            return ""
        return _one_line(", ".join(_one_line(label, 20) for label in labels[:4]), 65)

    @classmethod
    def _agent_text(cls, record: Dict[str, Any], status: str) -> str:
        """Render a verified episode as scoped, readable navigation advice."""
        if status not in {"accepted", "wrong_path"}:
            return ""
        scene = _one_line(Path(str(record.get("scene_id") or "unknown")).name, 65)
        subtask = _one_line(record.get("subtask") or "navigation segment", 100)
        if status == "accepted":
            observed = _one_line(cls._observed_condition(record), 125)
            if not observed:
                observed = "route conditions accepted; no specific visual cue recorded"
            labels = cls._detected_labels(record)
            label_text = f" Detected labels: {labels}." if labels else ""
            cue = cls._historical_planner_cue(record)
            cue_text = f" Prior planner suggestion: {_one_line(cue, 155)}." if cue else ""
            distance = record.get("measured_path_distance")
            travel = "the subtask verifier accepted the route"
            if isinstance(distance, (int, float)) and math.isfinite(distance) and distance >= 0:
                travel = f"measured travel {distance:.2f} m; subtask verifier accepted"
            goal_status = "; episode goal reached" if record.get("episode_goal_reached") else ""
            return (
                f'Scene {scene}; subtask "{subtask}". Evidence: {observed}.{label_text}'
                f'{cue_text} Outcome: {travel}{goal_status}. '
                "Reuse only for matching scene, route order, and current view/map; "
                "select a currently legal waypoint. Prior waypoint IDs are stale."
            )
        contradiction = _one_line(
            record.get("contradiction_evidence") or record.get("outcome_reason")
            or "the branch contradicted the current subtask", 170
        )
        landmarks = cls._detected_labels(record, "negative_landmarks")
        landmark_text = f" Rejected-branch labels: {landmarks}." if landmarks else ""
        cue = cls._historical_planner_cue(record)
        cue_text = f" Prior branch suggestion: {_one_line(cue, 110)}." if cue else ""
        return (
            f'Scene {scene}; subtask "{subtask}". Contradiction: {contradiction}.{landmark_text}'
            f'{cue_text} Outcome: branch rejected; physical rollback completed. '
            "Avoid that branch only with matching scene and route evidence; "
            "recheck the current view/map and legal waypoints."
        )

    @classmethod
    def _retrieval_key(cls, record: Dict[str, Any]) -> str:
        """Short semantic key, with the full evidence retained in JSONL."""
        objects = record.get("recognized_objects") or []
        if not isinstance(objects, (list, tuple)):
            objects = []
        object_text = ", ".join(_one_line(obj, 30) for obj in objects[:5])
        cue = cls._observed_condition(record) or _one_line(
            record.get("contradiction_evidence"), 150
        )
        return (
            f'Task: {_one_line(record.get("task_instruction"), 160)}; '
            f'subtask: {_one_line(record.get("subtask"), 140)}; '
            f'objects: {object_text}; observed cue: {_one_line(cue, 190)}'
        )

    @classmethod
    def _index_text(cls, record: Dict[str, Any]) -> str:
        """Embed only concise readable text, never the audit JSON."""
        status = record.get("verification_status") or cls.verification_status(record)
        agent_text = record.get("agent_text") or cls._agent_text(record, status)
        retrieval_key = record.get("retrieval_key") or cls._retrieval_key(record)
        return f"{retrieval_key}\n{agent_text}".strip()

    def _unique_verified(self, rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        result = []
        seen = set()
        for row in rows:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            # Mem0's infer=False stores the message role as "user"; our
            # navigation role is kept separately in its metadata.
            role = metadata.get("hsgm_role") or row.get("role")
            agent_id = row.get("agent_id", metadata.get("hsgm_agent_id"))
            if role != "verified_subtask" or (agent_id and agent_id != self.agent_id):
                continue
            record = self._record_from_row(row)
            # Audit provenance is required; native Mem0 metadata alone cannot
            # make legacy or quarantined records eligible for retrieval.
            if not record or self.verification_status(record) == "unverified":
                continue
            if record:
                key = (
                    str(record.get("task_instruction", "")), str(record.get("subtask", "")),
                    str(record.get("outcome", "")), str(record.get("lessons_learned", "")),
                    str(record.get("waypoint_reason", "")), str(record.get("outcome_reason", "")),
                )
                row = {**row, "record": record}
            else:
                key = (str(row.get("memory", "")),)
            if key in seen:
                continue
            seen.add(key)
            result.append(_englishize_fields(_json_safe(row)))
            if len(result) >= limit:
                break
        return result

    def _local_search(self, query: str, limit: int, scene_id=None) -> List[Dict[str, Any]]:
        query_tokens = set(_tokens(query))
        scored = []
        rows = self._read_local()
        for index, row in enumerate(rows):
            if row.get("role") != "verified_subtask" or row.get("agent_id", self.agent_id) != self.agent_id:
                continue
            record = self._record_from_row(row)
            if row.get('user_id') != self.user_id or (scene_id and record.get('scene_id') != str(scene_id)):
                continue
            if self.verification_status(record) == "unverified":
                continue
            searchable = self._index_text(record)
            row_tokens = set(_tokens(searchable))
            overlap = len(query_tokens.intersection(row_tokens))
            # Exact object/subtask words should dominate recency, while a
            # small recency term makes equally relevant memories deterministic.
            score = float(overlap) / max(1, len(query_tokens)) + index / max(1, len(rows)) * 0.01
            if overlap or not query_tokens:
                result = dict(row)
                result["score"] = round(score, 4)
                result["memory"] = str(row.get("text", row.get("memory", "")))
                scored.append(result)
        scored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return self._unique_verified(scored, limit)

    @_measure("search")
    def search(self, query: str, limit: int = 1, scene_id: str = None) -> List[Dict[str, Any]]:
        """Return at most ``limit`` relevant memories in a JSON-safe shape."""
        if self.requested_backend == "mem0" and not self.is_mem0:
            raise MemoryBackendError(self.last_error or "Required Mem0 backend is not open.")
        # Navigation planning consumes at most one episodic memory per retrieval.
        limit = min(1, max(0, int(limit)))
        if not limit:
            return []
        query = _one_line(query, 600)[:600]
        candidates = []
        if self._mem0 is not None:
            try:
                # Mem0 requires the scope as a named argument; placing it
                # only inside filters is insufficient in mem0ai 0.1.x.
                filters = {'hsgm_agent_id': self.agent_id, 'hsgm_verified': True,
                           'hsgm_role': 'verified_subtask'}
                if scene_id:
                    filters['scene_id'] = str(scene_id)
                self._mem0_search_count += 1
                result = self._mem0.search(
                    query=query, user_id=self.user_id, filters=filters, limit=10,
                )
                candidates = self._normalize_mem0_result(result)
            except Exception as exc:
                self._disable_mem0(f"Mem0 search failed: {exc}")
        if candidates:
            self._refresh_audit_index()
            audit_rows = self._audit_by_id
            hydrated = []
            for row in candidates:
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                audit = audit_rows.get(metadata.get("hsgm_audit_id"))
                # A native Mem0 record/metadata alone is not trusted provenance.
                if (audit is None or audit.get('user_id') != self.user_id
                        or audit.get('agent_id') != self.agent_id):
                    continue
                hydrated.append({**row, 'record': audit.get('record'), 'role': audit.get('role'),
                                 'agent_id': audit.get('agent_id')})
            if scene_id:
                hydrated = [row for row in hydrated
                            if self._record_from_row(row).get('scene_id') == str(scene_id)]
            candidates = self._unique_verified(hydrated, limit)
        if self.requested_backend != "mem0" and len(candidates) < limit:
            candidates = self._unique_verified(candidates + self._local_search(query, limit, scene_id), limit)
        if scene_id:
            # Instance-specific route successes/failures do not establish
            # room identity in another scene. Retrieval remains real Mem0.
            candidates = [row for row in candidates
                          if self._record_from_row(row).get('scene_id') == str(scene_id)]
        return candidates

    @_measure("add")
    def add(self, record: Dict[str, Any], role: str = "verified_subtask") -> Dict[str, Any]:
        """Audit once before indexing; never fall back to LLM inference."""
        excluded = {'bev_map', 'trajectory_checkpoint', 'planner_memory', 'metrics'}
        safe_record = _englishize_fields(_json_safe({k: v for k, v in record.items() if k not in excluded}))
        status = self.verification_status(safe_record)
        self._refresh_audit_index()
        key = self._dedup_key(safe_record, role)
        if key is not None and key in self._indexed_subtasks:
            self._duplicate_add_count += 1
            return {**self._indexed_subtasks[key], 'backend': 'local_audit', 'duplicate': True}
        if role == 'verified_subtask' and status != 'unverified':
            safe_record['agent_text'] = self._agent_text(safe_record, status)
            safe_record['retrieval_key'] = self._retrieval_key(safe_record)
            safe_record['verification_status'] = status
        text = json.dumps(safe_record, ensure_ascii=False, separators=(',', ':'))
        local_row = dict(id='hsgm-' + uuid.uuid4().hex,
                         created_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                         role=role, user_id=self.user_id, agent_id=self.agent_id,
                         record=safe_record, text=text, memory=text, index_status='audit_only')
        should_index = role == 'verified_subtask' and status != 'unverified'
        if should_index and self._mem0 is not None:
            local_row['index_status'] = 'pending'
        self._append_local(local_row)
        mem0_result = None
        error = None
        if self._mem0 is not None and should_index:
            metadata = dict(hsgm_role=role, hsgm_verified=True, hsgm_audit_id=local_row['id'],
                            hsgm_agent_id=self.agent_id, episode_id=str(safe_record.get('episode_id', '')),
                            scene_id=str(safe_record.get('scene_id', '')), memory_type='episodic_navigation',
                            memory_schema_version=_MEMORY_SCHEMA_VERSION)
            self._mem0_write_attempt_count += 1
            try:
                mem0_result = self._mem0.add(self._index_text(safe_record), user_id=self.user_id,
                                            metadata=metadata, infer=False)
                self._mem0_write_count += 1
            except TypeError as exc:
                error = f'Mem0 add rejected infer=False or raised TypeError; LLM fallback forbidden: {exc}'
            except Exception as exc:
                error = f'Mem0 add failed (no automatic retry): {exc}'
            self._append_local(dict(event='memory_index_result', audit_id=local_row['id'],
                                    index_status='failed' if error else 'indexed', index_error=error))
        elif self.requested_backend == 'mem0' and not self.is_mem0:
            error = self.last_error or 'Required Mem0 backend is not open.'
        if error:
            self._disable_mem0(error)
        return {**self._audit_by_id.get(local_row['id'], local_row),
                'mem0_result': _englishize_fields(_json_safe(mem0_result)),
                'backend': self.backend if mem0_result is not None else 'local_audit'}

    def add_subtask(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(record)
        record['memory_schema_version'] = _MEMORY_SCHEMA_VERSION
        history = record.get('verifier_history') or []
        valid_history = [
            value for value in history
            if isinstance(value, dict) and value.get('available') is True
            and not value.get('contract_conflict')
        ]
        latest_valid = valid_history[-1] if valid_history else {}
        has_accepted_verification = latest_valid.get('outcome') is True or record.get('subtask_verified') is True
        supported_success = bool(
            record.get('outcome') is True
            and record.get('completion_evidence_reliable', True) is True
            and has_accepted_verification
        )
        record['evidence_status'] = (
            'subtask_verified' if supported_success else 'audit_only'
        )
        record['retrieval_eligible'] = supported_success
        if record.get('outcome') is True and not supported_success:
            record['suspect_fp'] = record['quarantined'] = True
        status = self.verification_status(record)
        record["verification_status"] = status
        record["verified"] = status != "unverified"
        return self.add(record, role="verified_subtask" if record["verified"] else "unverified_subtask")

    def add_negative_event(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Audit wrong_path claims; retrieve them only after physical rollback succeeds."""
        rec = dict(record)
        rec['memory_schema_version'] = _MEMORY_SCHEMA_VERSION
        rec["outcome"] = False
        rec["failure_type"] = "wrong_path"
        events = rec.get('rollback_events') or []
        rollback_completed = any(
            isinstance(event, dict) and event.get('event') == 'rollback_completed'
            and (not rec.get('subtask_key') or event.get('subtask_id') == rec.get('subtask_key'))
            and (not rec.get('outcome_reason') or event.get('reason') == rec.get('outcome_reason'))
            for event in events
        )
        rec['rollback_confirmed'] = bool(rollback_completed)
        rec['retrieval_eligible'] = bool(rollback_completed)
        rec['evidence_status'] = 'rollback_confirmed' if rollback_completed else 'audit_only'
        rec['verified'] = bool(rollback_completed)
        rec['verification_status'] = 'wrong_path' if rollback_completed else 'unverified'
        if not rec.get("lessons_learned"):
            scope = f"in scene {rec.get('scene_id')}" if rec.get('scene_id') else "in this episode"
            rec["lessons_learned"] = (
                f"CONFIRMED REJECTED BRANCH {scope}: "
                f"{rec.get('contradiction_evidence') or rec.get('outcome_reason', 'deviation')}. "
                "Use only as a local clue; recheck current geometry and instruction."
            )
        status = self.verification_status(rec)
        rec['verification_status'] = status
        rec['verified'] = status != 'unverified'
        role = 'verified_subtask' if rec['verified'] else 'unverified_subtask'
        return self.add(rec, role=role)


    def add_episode(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return self.add(record, role="episode_summary")

    def status(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "requested_backend": self.requested_backend,
            "storage_path": str(self.root),
            "audit_path": str(self.audit_path),
            "local_records": len(self._read_local()),
            "last_error": self.last_error or None,
            "required": self.requested_backend == "mem0",
            "mem0_search_count": self._mem0_search_count,
            "mem0_write_count": self._mem0_write_count,
            "mem0_write_attempt_count": self._mem0_write_attempt_count,
            "duplicate_add_count": self._duplicate_add_count,
            "audit_bytes_read": self._audit_bytes_read,
            "audit_full_loads": self._audit_loads,
            "audit_append_count": self._audit_append_count,
            "operation_costs": {name: dict(values) for name, values in self._costs.items()},
            "embedding_device": os.environ.get("HSGM_MEM0_EMBEDDING_DEVICE", "cpu"),
            "closed": self._closed,
            "is_shared": getattr(self, "_is_shared", False),
        }

    def __repr__(self) -> str:
        return (
            f"<NavigationMemory backend={self.backend!r} "
            f"user_id={self.user_id!r} agent_id={self.agent_id!r} "
            f"root={str(self.root)!r} shared={getattr(self, '_is_shared', False)}>"
        )

    @classmethod
    def get_shared(
        cls,
        storage_path: Optional[str] = None,
        user_id: str = "hsgm_navigation",
        agent_id: str = "hsgm",
        backend: Optional[str] = None,
    ) -> NavigationMemory:
        """Classmethod to retrieve or initialize a process-shared NavigationMemory instance."""
        return get_shared_navigation_memory(
            storage_path=storage_path,
            user_id=user_id,
            agent_id=agent_id,
            backend=backend,
        )

    def prompt_context(self, memories: Iterable[Dict[str, Any]], limit: int = 1) -> str:
        """Show at most one verified, scene-scoped experience to the planner."""
        limit = min(1, max(0, int(limit)))
        lines = []
        for row in list(memories)[:limit]:
            if not isinstance(row, dict):
                continue
            record = self._record_from_row(row)
            status = record.get("verification_status")
            if status not in {"accepted", "wrong_path"}:
                continue
            agent_text = record.get("agent_text") or self._agent_text(record, status)
            if not agent_text:
                continue
            score = row.get("score")
            score_text = f" (relevance: {score})" if score is not None else ""
            lines.append(f"{len(lines) + 1}. {agent_text}{score_text}")
        return "\n".join(lines)


def get_memory_cache_key(
    storage_path: Optional[str] = None,
    user_id: str = "hsgm_navigation",
    agent_id: str = "hsgm",
    backend: Optional[str] = None,
) -> Tuple[str, ...]:
    """Construct a process-level cache key for NavigationMemory.

    Includes normalized storage path, user and agent IDs, requested backend,
    and non-secret embedding configurations. API keys and secrets are strictly excluded.
    """
    default_path = os.environ.get("HSGM_MEMORY_PATH", "../results/navigation_memory")
    norm_path = str(Path(storage_path or default_path).expanduser().resolve())
    norm_user = str(user_id or "hsgm_navigation").replace(" ", "_")
    norm_agent = str(agent_id or "hsgm").replace(" ", "_")
    req_backend = str(os.environ.get("HSGM_MEMORY_BACKEND", backend or "auto")).lower()

    embedder_provider = str(os.environ.get("HSGM_MEM0_EMBEDDER", "local")).lower()
    embedder_model = str(
        os.environ.get("HSGM_MEM0_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedder_device = str(os.environ.get("HSGM_MEM0_EMBEDDING_DEVICE", "cpu")).lower()
    hf_hub_offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
    openai_model = str(os.environ.get("OPENAI_MODEL", "Qwen/Qwen3.8-27B-FP8"))
    openai_base_url = str(os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))

    return (
        norm_path,
        norm_user,
        norm_agent,
        req_backend,
        embedder_provider,
        embedder_model,
        embedder_device,
        str(hf_hub_offline),
        openai_model,
        openai_base_url,
    )


_SHARED_MEMORY_CACHE: Dict[Tuple[str, ...], NavigationMemory] = {}
_CACHE_LOCK = threading.Lock()


def get_shared_navigation_memory(
    storage_path: Optional[str] = None,
    user_id: str = "hsgm_navigation",
    agent_id: str = "hsgm",
    backend: Optional[str] = None,
) -> NavigationMemory:
    """Retrieve or initialize a process-shared NavigationMemory instance.

    Reuses the instance matching the normalized configuration across episodes,
    amortizing initial initialization (~9.95s) and avoiding full audit reloads.
    """
    key = get_memory_cache_key(storage_path, user_id, agent_id, backend)
    with _CACHE_LOCK:
        instance = _SHARED_MEMORY_CACHE.get(key)
        if instance is not None:
            if instance._closed or instance.backend == "mem0_error" or (
                instance.requested_backend == "mem0" and not instance.is_mem0
            ):
                _SHARED_MEMORY_CACHE.pop(key, None)
                instance = None
        if instance is None:
            instance = NavigationMemory(
                storage_path=storage_path,
                user_id=user_id,
                agent_id=agent_id,
                backend=backend,
            )
            instance._is_shared = True
            _SHARED_MEMORY_CACHE[key] = instance
        return instance


def evict_shared_navigation_memory(
    target: Optional[Union[NavigationMemory, Tuple[str, ...]]] = None,
    *,
    storage_path: Optional[str] = None,
    user_id: str = "hsgm_navigation",
    agent_id: str = "hsgm",
    backend: Optional[str] = None,
) -> bool:
    """Evict a shared navigation memory instance from the process cache.

    Can evict by object instance, cache key tuple, or config arguments.
    Returns True if an entry was removed, False otherwise.
    """
    with _CACHE_LOCK:
        if isinstance(target, tuple):
            return _SHARED_MEMORY_CACHE.pop(target, None) is not None
        if isinstance(target, NavigationMemory):
            for k, v in list(_SHARED_MEMORY_CACHE.items()):
                if v is target:
                    _SHARED_MEMORY_CACHE.pop(k, None)
                    return True
            return False
        if (
            storage_path is not None
            or backend is not None
            or user_id != "hsgm_navigation"
            or agent_id != "hsgm"
        ):
            key = get_memory_cache_key(storage_path, user_id, agent_id, backend)
            return _SHARED_MEMORY_CACHE.pop(key, None) is not None
        # Prune any closed or errored instances
        removed = False
        for k, v in list(_SHARED_MEMORY_CACHE.items()):
            if v._closed or v.backend == "mem0_error" or (
                v.requested_backend == "mem0" and not v.is_mem0
            ):
                _SHARED_MEMORY_CACHE.pop(k, None)
                removed = True
        return removed


def close_all_shared_navigation_memories() -> None:
    """Close all cached shared memory instances and clear cache on process exit."""
    with _CACHE_LOCK:
        for key, mem in list(_SHARED_MEMORY_CACHE.items()):
            try:
                mem.close(force=True)
            except Exception as exc:
                logger.warning("Could not close shared memory instance: %s", exc)
        _SHARED_MEMORY_CACHE.clear()


def clear_shared_navigation_memories() -> None:
    """Explicitly close and clear all shared memory instances (e.g. for testing)."""
    close_all_shared_navigation_memories()


atexit.register(close_all_shared_navigation_memories)

