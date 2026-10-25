#!/usr/bin/env python3
"""Dedicated runner for v3 arm1_improved (qwen3.8-flash) with 4 parallel persistent workers.

Evaluates v3 on the episodes where v2 baseline ran, giving direct 1-to-1 comparison.
"""

import glob
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time

V3_REPO = Path("/home/w/project/hsgmv3-target-grounding")
HSGM_PUBLIC = Path("/home/w/project/HSGM_public")
ENV_FILE = Path("/home/w/project/hsgmv3/.env")
PYTHON_BIN = "/home/w/miniconda3/envs/hsgm/bin/python"
DATASET_PATH = HSGM_PUBLIC / "data/datasets/r2r_vlnce/val_unseen.json"

OUT_ROOT = V3_REPO / "results/full_1839_remaining_qwen38flash"
V3_OUT = OUT_ROOT / "v3_arm1_improved"
V2_OUT = OUT_ROOT / "v2_baseline"

val_data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["episodes"]

NUM_WORKERS = 4
CHUNK_SIZE = 4


def load_env(model_name="qwen3.8-flash"):
    env_vars = os.environ.copy()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip().strip('"').strip("'")
    env_vars["OPENAI_MODEL"] = model_name
    env_vars["HSGM_ENABLE_THINKING"] = "0"
    env_vars["HSGM_DISABLE_THINKING"] = "1"
    env_vars["HSGM_VERIFIER_STRUCTURED_THINKING"] = "1"
    env_vars["HSGM_MAX_EPISODE_SECONDS"] = "1200"
    env_vars["HSGM_SOFT_BUDGET_MARGIN_S"] = "30"
    env_vars["PYTHONUNBUFFERED"] = "1"
    env_vars["OMP_NUM_THREADS"] = "2"
    env_vars["MKL_NUM_THREADS"] = "2"
    env_vars["OPENBLAS_NUM_THREADS"] = "2"
    env_vars["NUMEXPR_NUM_THREADS"] = "2"
    env_vars["HSGM_MEM0_EMBEDDING_MODEL"] = "/home/w/.cache/models/all-MiniLM-L6-v2"
    env_vars["HF_HUB_OFFLINE"] = "1"
    env_vars["TRANSFORMERS_OFFLINE"] = "1"
    for p in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        env_vars.pop(p, None)
    env_vars["NO_PROXY"] = "*"
    env_vars["no_proxy"] = "*"
    return env_vars


def exact_match_ep(d, json_path):
    m = re.search(r"episode_(\d+)_results\.json$", json_path)
    traj = d.get("agent_trajectory", [])
    if not traj:
        return None
    p0 = traj[0]
    cands = []
    for idx, ep in enumerate(val_data):
        sp = ep["start_position"]
        if abs(sp[0] - p0[0]) < 0.02 and abs(sp[1] - p0[1]) < 0.02 and abs(sp[2] - p0[2]) < 0.02:
            cands.append((idx, ep))
    if not cands:
        return None
    if m:
        file_ndx = int(m.group(1))
        for idx, ep in cands:
            if idx == file_ndx:
                return ep["episode_id"]
    return cands[0][1]["episode_id"]


def get_v2_results_map():
    v2_map = {}
    for p in glob.glob(str(V2_OUT / "**/episode_*_results.json"), recursive=True):
        try:
            d = json.loads(Path(p).read_text())
            eid = exact_match_ep(d, p)
            if eid:
                m = d.get("metrics", {})
                v2_map[eid] = {
                    "distance_to_goal": m.get("distance_to_goal", 999.0),
                    "finish_status": m.get("finish_status", "unknown"),
                    "goal_reached": m.get("goal_reached", 0),
                    "spl": m.get("spl", 0.0),
                    "path": p,
                }
        except Exception:
            pass
    return v2_map


def get_existing_v3_state():
    v3_done = set()
    v3_interrupted = {}
    for p in glob.glob(str(V3_OUT / "**/episode_*_results.json"), recursive=True):
        try:
            d = json.loads(Path(p).read_text())
            m = d.get("metrics", {})
            st = m.get("finish_status", "")
            eid = exact_match_ep(d, p)
            if not eid:
                continue
            # Only count completed non-error non-budget-exhausted runs
            if st in ["success", "fp", "max_steps"]:
                v3_done.add(eid)
                v3_interrupted.pop(eid, None)
            elif eid not in v3_done:
                dist = float(m.get("distance_to_goal", 999.0) or 999.0)
                is_bug_fix_target = 0 if st != "budget_exhausted" else 1
                v3_interrupted[eid] = (is_bug_fix_target, dist, eid)
        except Exception:
            pass
    interrupted_ordered = [eid for _, _, eid in sorted(v3_interrupted.values())]
    return v3_done, interrupted_ordered


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    V3_OUT.mkdir(parents=True, exist_ok=True)
    progress_log = OUT_ROOT / "live_progress.log"
    lock = threading.Lock()

    def log_msg(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with lock:
            with open(progress_log, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    env_vars = load_env("qwen3.8-flash")
    v2_map = get_v2_results_map()
    v3_done, v3_interrupted = get_existing_v3_state()
    interrupted_set = set(v3_interrupted)

    # Prioritize: 1) previously interrupted/bug-aborted v3 episodes, 2) remaining v2-fails, 3) remaining v2-successes
    v2_fails = sorted([eid for eid, r in v2_map.items() if (r["finish_status"] != "success" and r["goal_reached"] != 1) and eid not in v3_done and eid not in interrupted_set])
    v2_succs = sorted([eid for eid, r in v2_map.items() if (r["finish_status"] == "success" or r["goal_reached"] == 1) and eid not in v3_done and eid not in interrupted_set])
    v3_queue_list = v3_interrupted + v2_fails + v2_succs

    log_msg(
        f"=== Launching V3 Arm1-Improved Batch Run (qwen3.8-flash) ===\n"
        f"  Total V2 baseline episodes available: {len(v2_map)}\n"
        f"  V3 already completed (clean): {len(v3_done)}\n"
        f"  V3 interrupted/aborted to resume first: {len(v3_interrupted)}\n"
        f"  V3 queue total: {len(v3_queue_list)} episodes ({len(v3_interrupted)} resumed + {len(v2_fails)} v2-fails + {len(v2_succs)} v2-succs)\n"
        f"  Workers: {NUM_WORKERS} parallel, Chunk size: {CHUNK_SIZE}"
    )

    v3_q = queue.Queue()
    for i in range(0, len(v3_queue_list), CHUNK_SIZE):
        v3_q.put(v3_queue_list[i : i + CHUNK_SIZE])

    def v3_worker(worker_id):
        w_dir = V3_OUT / f"worker_{worker_id}"
        w_raw = w_dir / "raw"
        w_mem = w_dir / "memory"
        w_log = w_dir / "logs"
        w_raw.mkdir(parents=True, exist_ok=True)
        w_mem.mkdir(parents=True, exist_ok=True)
        w_log.mkdir(parents=True, exist_ok=True)

        chunk_idx = 0
        while True:
            try:
                chunk = v3_q.get_nowait()
            except queue.Empty:
                break
            chunk_idx += 1
            ndxs_str = ",".join(str(eid - 1) for eid in chunk)
            comment = f"v3_w{worker_id}_c{chunk_idx}_r2_eps_{chunk[0]}_{chunk[-1]}"
            log_file = w_log / f"{comment}.log"
            log_msg(f"[v3-W{worker_id}] Starting chunk #{chunk_idx}: Episodes {chunk} (remaining chunks in queue: {v3_q.qsize()})")

            cmd = [
                PYTHON_BIN,
                "-u",
                "run_experiments.py",
                "--task",
                "r2r",
                "--config",
                str(V3_REPO / "config/vlnce_test.yaml"),
                "--model_name",
                "qwen3.8-flash",
                "--episode_ids",
                ndxs_str,
                "--max_steps",
                "90",
                "--dataset_path",
                str(DATASET_PATH),
                "--output_root",
                str(w_raw),
                "--memory_storage_path",
                str(w_mem / f"r2_c{chunk_idx}"),
                "--success_threshold",
                "2",
                "--comment",
                comment,
            ]
            ep_env = dict(env_vars)
            with open(log_file, "w", encoding="utf-8") as lf:
                try:
                    subprocess.run(
                        cmd,
                        cwd=str(V3_REPO / "src"),
                        env=ep_env,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        timeout=len(chunk) * 450,
                    )
                except subprocess.TimeoutExpired:
                    lf.write("\n[CHUNK_TIMEOUT]\n")

            for eid in chunk:
                ep_ndx = eid - 1
                found = glob.glob(str(w_raw / f"*{comment}*" / f"episode_{ep_ndx}_results.json"))
                if found:
                    try:
                        d = json.loads(Path(found[0]).read_text())
                        m = d.get("metrics", {})
                        dist = m.get("distance_to_goal", 999.0)
                        status = m.get("finish_status", "unknown")
                        is_v3_succ = (dist <= 3.0 and status == "success") or (m.get("goal_reached", 0) == 1)
                        v2_r = v2_map.get(eid, {})
                        v2_st = v2_r.get("finish_status", "N/A")
                        v2_dist = v2_r.get("distance_to_goal", 999.0)
                        v2_succ = (v2_dist <= 3.0 and v2_st == "success") or (v2_r.get("goal_reached", 0) == 1)

                        if is_v3_succ and not v2_succ:
                            log_msg(f"🎉 [V2_FAIL -> V3_SUCC WIN] Ep {eid}: v2={v2_st} ({v2_dist:.2f}m) -> v3=SUCCESS ({dist:.2f}m)!")
                        elif is_v3_succ:
                            log_msg(f"✅ [v3-W{worker_id}] Ep {eid}: V3_SUCC ({dist:.2f}m, status={status})")
                        else:
                            log_msg(f"❌ [v3-W{worker_id}] Ep {eid}: V3_FAIL ({dist:.2f}m, status={status})")
                    except Exception:
                        pass
            v3_q.task_done()

    threads = []
    for wid in range(NUM_WORKERS):
        t = threading.Thread(target=v3_worker, args=(wid,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(3)

    for t in threads:
        t.join()
    log_msg("=== ALL QUEUED V3 EPISODES COMPLETED ===")


if __name__ == "__main__":
    main()

