#!/usr/bin/env python3
"""Dedicated runner for v2 baseline only (qwen3.8-flash) with 10 parallel persistent workers."""

import glob
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time

V2_REPO = Path("/home/w/project/HSGM_public/edition/HSGM_public_v2")
HSGM_PUBLIC = Path("/home/w/project/HSGM_public")
ENV_FILE = Path("/home/w/project/hsgmv3/.env")
PYTHON_BIN = "/home/w/miniconda3/envs/hsgm/bin/python"
DATASET_PATH = HSGM_PUBLIC / "data/datasets/r2r_vlnce/val_unseen.json"

OUT_ROOT = Path("/home/w/project/hsgmv3-target-grounding/results/full_1839_remaining_qwen38flash")
V2_OUT = OUT_ROOT / "v2_baseline"

val_data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["episodes"]

NUM_WORKERS = 10
CHUNK_SIZE = 8


def load_env(model_name="qwen3.8-flash"):
    env_vars = os.environ.copy()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip().strip('"').strip("'")
    env_vars["OPENAI_MODEL"] = model_name
    env_vars["HSGM_ENABLE_THINKING"] = "1"
    env_vars["HSGM_REASONING_EFFORT"] = "medium"
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
    sub_str = json.dumps(d.get("sub_instruction", "")).lower()
    vlm_str = json.dumps(d.get("vlm_responses", [])[:1]).lower()
    best_ep = cands[0][1]["episode_id"]
    best_score = -1
    for idx, ep in cands:
        inst_words = set(re.findall(r"[a-z]{3,}", ep["instruction"]["instruction_text"].lower()))
        score = sum(1 for w in inst_words if w in sub_str or w in vlm_str) / max(1, len(inst_words))
        if score > best_score:
            best_score = score
            best_ep = ep["episode_id"]
    return best_ep


def get_existing_v2_done():
    v2_done = set()
    for pat in [
        "/home/w/project/HSGM_public/edition/HSGM_public_v2/results/**/*qwen3_8-flash*/episode_*_results.json",
        "/home/w/project/hsgmv3-target-grounding/results/v2_vs_v3_showcase_qwen38flash/v2_sandbox/results/**/episode_*_results.json",
        str(V2_OUT / "**/episode_*_results.json"),
    ]:
        for p in glob.glob(pat, recursive=True):
            try:
                d = json.loads(Path(p).read_text())
                m = d.get("metrics", {})
                status = m.get("finish_status", "")
                dist = m.get("distance_to_goal", 999.0)
                if status != "error" and dist < 100:
                    eid = exact_match_ep(d, p)
                    if eid:
                        v2_done.add(eid)
            except Exception:
                pass
    return v2_done


def setup_v2_worker_sandbox(worker_id):
    w_dir = V2_OUT / f"worker_{worker_id}"
    (w_dir / "results").mkdir(parents=True, exist_ok=True)
    (w_dir / "logs").mkdir(parents=True, exist_ok=True)
    (w_dir / "src").mkdir(parents=True, exist_ok=True)
    cfg_dir = w_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "vlnce_test.yaml").write_text(
        "agent:\n"
        "  radius: 0.1\n"
        "  initial_position: [7.40658, -0.53553, 0.92098]\n"
        "  initial_rotation: [0, 0.53996, 0, 0.84169]\n"
        "camera:\n"
        "  height: 1.2\n"
        "  fov: 135.0\n"
        "  pitch: -30.0\n"
        "  min_depth: 0.5\n"
        "  max_depth: 50.0\n"
        "  res_factor: 2\n"
        "env:\n"
        "  max_steps: 70\n"
        "  success_threshold: 2.0\n"
        "  name: 'r2r'\n",
        encoding="utf-8",
    )
    for name, target in [("data", HSGM_PUBLIC / "data"), ("ckpt", V2_REPO / "ckpt")]:
        link = w_dir / name
        if not link.exists():
            os.symlink(str(target), str(link))
    mc_link = w_dir / "src" / "mobileclip2_b.ts"
    if not mc_link.exists():
        os.symlink(str(HSGM_PUBLIC / "mobileclip2_b.ts"), str(mc_link))
    return w_dir


def stride_order(ep_ids, stride=17):
    ordered = []
    for rem in range(stride):
        ordered.extend([eid for i, eid in enumerate(ep_ids) if i % stride == rem])
    return ordered


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    V2_OUT.mkdir(parents=True, exist_ok=True)
    progress_log = OUT_ROOT / "live_progress.log"
    lock = threading.Lock()

    def log_msg(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with lock:
            with open(progress_log, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    env_vars = load_env("qwen3.8-flash")
    v2_done = get_existing_v2_done()
    all_ids = list(range(1, 1840))
    v2_todo = stride_order([eid for eid in all_ids if eid not in v2_done])

    log_msg(
        f"=== Starting V2-ONLY Dedicated High-Concurrency Run (qwen3.8-flash) ===\n"
        f"  Total already completed in v2: {len(v2_done)} episodes\n"
        f"  Remaining to run: {len(v2_todo)} episodes\n"
        f"  Parallel workers: {NUM_WORKERS} persistent workers (chunk_size={CHUNK_SIZE})\n"
        f"  Est. cost for entire remaining v2: ~35-45 RMB"
    )

    v2_q = queue.Queue()
    for i in range(0, len(v2_todo), CHUNK_SIZE):
        v2_q.put(v2_todo[i : i + CHUNK_SIZE])

    def v2_worker(worker_id):
        w_dir = setup_v2_worker_sandbox(worker_id)
        chunk_idx = 0
        while True:
            try:
                chunk = v2_q.get_nowait()
            except queue.Empty:
                break
            chunk_idx += 1
            ndxs_str = ",".join(str(eid - 1) for eid in chunk)
            comment = f"v2only_w{worker_id}_c{chunk_idx}_eps_{chunk[0]}_{chunk[-1]}"
            log_file = w_dir / "logs" / f"{comment}.log"
            log_msg(f"[v2-W{worker_id}] Starting chunk #{chunk_idx}: Episodes {chunk} (remaining chunks: {v2_q.qsize()})")

            py_code = (
                "import cv2, torch, open3d, habitat_sim, runpy, sys; "
                f"sys.argv = ['run_experiments.py', '--task', 'r2r', '--config', '../config/vlnce_test.yaml', "
                f"'--model_name', 'qwen3.8-flash', '--episode_ids', '{ndxs_str}', '--max_steps', '70', "
                f"'--comment', '{comment}']; "
                f"runpy.run_path('{V2_REPO / 'src/run_experiments.py'}', run_name='__main__')"
            )
            cmd = [PYTHON_BIN, "-u", "-c", py_code]
            ep_env = dict(env_vars)
            ep_env["PYTHONPATH"] = str(V2_REPO / "src")
            with open(log_file, "w", encoding="utf-8") as lf:
                try:
                    subprocess.run(
                        cmd,
                        cwd=str(w_dir / "src"),
                        env=ep_env,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        timeout=len(chunk) * 240,
                    )
                except subprocess.TimeoutExpired:
                    lf.write("\n[CHUNK_TIMEOUT]\n")

            for eid in chunk:
                ep_ndx = eid - 1
                found = glob.glob(str(w_dir / "results" / f"*{comment}*" / f"episode_{ep_ndx}_results.json"))
                if found:
                    try:
                        d = json.loads(Path(found[0]).read_text())
                        m = d.get("metrics", {})
                        dist = m.get("distance_to_goal", 999)
                        status = m.get("finish_status", "")
                        tag = "V2_SUCC" if dist <= 2.0 and status == "success" else "V2_FAIL"
                        log_msg(f"[v2-W{worker_id}] Ep {eid:4d}: {tag} | dist={dist:.2f}m, status={status}")
                    except Exception:
                        pass
            v2_q.task_done()

    threads = []
    for wid in range(NUM_WORKERS):
        t = threading.Thread(target=v2_worker, args=(wid,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(2)

    for t in threads:
        t.join()
    log_msg("=== ALL REMAINING V2 EPISODES COMPLETED ===")


if __name__ == "__main__":
    main()

