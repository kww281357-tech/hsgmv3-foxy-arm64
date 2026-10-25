#!/usr/bin/env python3
"""Runner for 20 Stair Episodes Slice Evaluation in HSGM v3.

Evaluates stair navigation improvements across 20 representative stair episodes.
Supports parallel execution across workers.
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path("/home/w/project/hsgmv3")
HSGM_PUBLIC = Path("/home/w/project/HSGM_public")
PYTHON_BIN = "/home/w/miniconda3/envs/hsgm/bin/python"
SRC_DIR = REPO_ROOT / "src"
CONFIG_FILE = REPO_ROOT / "config/vlnce_test.yaml"
DATASET_PATH = HSGM_PUBLIC / "data/datasets/r2r_vlnce/val_unseen_stair_20_slice.json"
RESULTS_DIR = REPO_ROOT / "results/stair_20_eval"


def load_env(model_name="qwen3.8-flash"):
    env_file = HSGM_PUBLIC / ".env"
    env_vars = os.environ.copy()
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()

    env_vars["OPENAI_MODEL"] = model_name
    env_vars["HSGM_ENABLE_THINKING"] = "1"
    env_vars["HSGM_REASONING_EFFORT"] = "medium"
    env_vars["PYTHONUNBUFFERED"] = "1"
    env_vars["HSGM_MEM0_EMBEDDING_MODEL"] = "/home/w/.cache/models/all-MiniLM-L6-v2"
    env_vars["HF_HUB_OFFLINE"] = "1"
    env_vars["TRANSFORMERS_OFFLINE"] = "1"

    # Strictly isolate from proxy network
    for p in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        env_vars.pop(p, None)
    env_vars["NO_PROXY"] = "*"
    env_vars["no_proxy"] = "*"
    return env_vars


def run_single_episode(
    ep_idx,
    worker_id,
    max_steps,
    timeout_sec,
    env_vars,
    model_name="qwen3.8-flash",
    dataset_path=DATASET_PATH,
    results_dir=RESULTS_DIR,
    force=False,
):
    res_dir = results_dir / "results"
    logs_dir = results_dir / "logs"
    res_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    final_json = res_dir / f"episode_{ep_idx}_results.json"
    log_file = logs_dir / f"episode_{ep_idx}.log"

    if not force and final_json.exists() and final_json.stat().st_size > 100:
        try:
            with open(final_json, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            metric = data.get("metrics", data.get("metric", {}))
            sr = float(metric.get("goal_reached", metric.get("success", 0.0)))
            spl = float(metric.get("spl", 0.0))
            ne = float(metric.get("distance_to_goal", metric.get("oracle_navigation_error", 0.0)))
            traj = data.get("agent_trajectory", [])
            delta_z = 0.0
            if len(traj) >= 2:
                y_coords = [p[1] for p in traj]
                delta_z = round(max(y_coords) - min(y_coords), 2)
            return {
                "episode_index": ep_idx,
                "status": "success" if sr > 0.5 else "failure",
                "sr": sr,
                "spl": spl,
                "ne": ne,
                "delta_z": delta_z,
                "duration_s": 0.0,
                "exit_code": 0,
            }
        except Exception:
            pass

    mem_dir = results_dir / "memory" / f"ep_{ep_idx}"
    mem_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir = results_dir / "raw" / f"worker_{worker_id}"
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_BIN,
        "-u",
        "run_experiments.py",
        "--task",
        "r2r",
        "--config",
        str(CONFIG_FILE),
        "--model_name",
        model_name,
        "--episode_ids",
        str(ep_idx),
        "--max_steps",
        str(max_steps),
        "--dataset_path",
        str(dataset_path),
        "--output_root",
        str(raw_output_dir),
        "--memory_storage_path",
        str(mem_dir),
        "--success_threshold",
        "2",
        "--comment",
        f"slice_w{worker_id}_ep{ep_idx}",
    ]

    start_time = time.time()
    ep_env = dict(env_vars)
    ep_env["HSGM_MAX_EPISODE_SECONDS"] = str(max(240, int(timeout_sec) - 45))

    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"=== Starting episode {ep_idx} on worker {worker_id} at {time.ctime()} ===\n")
        lf.flush()
        try:
            p = subprocess.run(
                cmd,
                cwd=str(SRC_DIR),
                env=ep_env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
            )
            exit_code = p.returncode
        except subprocess.TimeoutExpired:
            lf.write(f"\n[TIMEOUT] Episode {ep_idx} timed out after {timeout_sec}s\n")
            exit_code = 124
        except Exception as exc:
            lf.write(f"\n[ERROR] Process execution failed: {exc}\n")
            exit_code = 1

    duration = time.time() - start_time

    # Find result json in raw directory
    json_candidates = list(raw_output_dir.glob(f"**/episode_{ep_idx}_results.json")) + list(
        raw_output_dir.glob(f"**/results/episode_{ep_idx}_results.json")
    )
    if json_candidates:
        latest_json = max(json_candidates, key=lambda f: f.stat().st_mtime)
        try:
            with open(latest_json, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            shutil.copy2(latest_json, final_json)
            metric = data.get("metrics", data.get("metric", list(data.values())[0] if isinstance(data, dict) else {}))
            sr = float(metric.get("goal_reached", metric.get("success", 0.0)))
            spl = float(metric.get("spl", 0.0))
            ne = float(metric.get("distance_to_goal", metric.get("oracle_navigation_error", 0.0)))
            traj = data.get("agent_trajectory", [])
            delta_z = 0.0
            if len(traj) >= 2:
                y_coords = [p[1] for p in traj]
                delta_z = round(max(y_coords) - min(y_coords), 2)
            return {
                "episode_index": ep_idx,
                "status": "success" if sr > 0.5 else "failure",
                "sr": sr,
                "spl": spl,
                "ne": ne,
                "delta_z": delta_z,
                "duration_s": round(duration, 1),
                "exit_code": exit_code,
            }
        except Exception as e:
            return {
                "episode_index": ep_idx,
                "status": "parse_error",
                "message": str(e),
                "duration_s": round(duration, 1),
                "exit_code": exit_code,
            }

    return {
        "episode_index": ep_idx,
        "status": "no_output",
        "duration_s": round(duration, 1),
        "exit_code": exit_code,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Episodes Slice Benchmark")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--episodes", type=str, default="0-19", help="Episode range, e.g. 0-19 or 0,1,2")
    parser.add_argument("--max_steps", type=int, default=150, help="Max steps per episode")
    parser.add_argument("--timeout", type=int, default=480, help="Timeout in seconds per episode")
    parser.add_argument("--model", type=str, default="qwen3.8-flash", help="Model name")
    parser.add_argument("--dataset", type=str, default="", help="Custom dataset JSON path")
    parser.add_argument("--results_name", type=str, default="stair_20_eval", help="Results directory name under results/")
    parser.add_argument("--force", action="store_true", help="Re-run even if result JSON already exists")
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else DATASET_PATH
    results_dir = REPO_ROOT / "results" / args.results_name
    env_vars = load_env(args.model)

    if "-" in args.episodes and "," not in args.episodes:
        start_idx, end_idx = map(int, args.episodes.split("-"))
        ep_indices = list(range(start_idx, end_idx + 1))
    else:
        ep_indices = [int(x.strip()) for x in args.episodes.split(",") if x.strip()]

    print(f"=== Starting Slice Benchmark ({args.results_name}) ===")
    print(f"Dataset: {dataset_path}")
    print(f"Episodes: {ep_indices} (Total: {len(ep_indices)})")
    print(f"Workers: {args.workers}")
    print(f"Model: {args.model}")
    print(f"Max steps: {args.max_steps}, Timeout: {args.timeout}s")
    print(f"Results dir: {results_dir}\n")

    t_start = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for worker_id, ep_idx in enumerate(ep_indices):
            w_id = worker_id % args.workers
            fut = executor.submit(
                run_single_episode,
                ep_idx=ep_idx,
                worker_id=w_id,
                max_steps=args.max_steps,
                timeout_sec=args.timeout,
                env_vars=env_vars,
                model_name=args.model,
                dataset_path=dataset_path,
                results_dir=results_dir,
                force=args.force,
            )
            futures[fut] = ep_idx

        for fut in concurrent.futures.as_completed(futures):
            ep_idx = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                sr_str = "SUCCESS" if res.get("sr", 0) > 0.5 else "FAIL"
                print(
                    f"[{len(results)}/{len(ep_indices)}] Ep {ep_idx}: {sr_str} | "
                    f"SR={res.get('sr', 0.0):.1f}, SPL={res.get('spl', 0.0):.3f}, NE={res.get('ne', 0.0):.2f}m, "
                    f"ΔZ={res.get('delta_z', 0.0)}m | Time={res.get('duration_s', 0)}s"
                )
            except Exception as e:
                print(f"[{len(results)}/{len(ep_indices)}] Ep {ep_idx}: EXCEPTION {e}")

    total_time = time.time() - t_start
    completed = [r for r in results if "sr" in r]
    if completed:
        avg_sr = sum(r["sr"] for r in completed) / len(completed) * 100
        avg_spl = sum(r["spl"] for r in completed) / len(completed) * 100
        avg_ne = sum(r["ne"] for r in completed) / len(completed)
        print("\n" + "=" * 50)
        print(f"=== SUMMARY: {args.results_name} ===")
        print(f"Total Evaluated: {len(completed)} / {len(ep_indices)}")
        print(f"Success Rate (SR): {avg_sr:.2f}%")
        print(f"SPL: {avg_spl:.2f}%")
        print(f"Navigation Error (NE): {avg_ne:.2f} m")
        print(f"Total Duration: {total_time:.1f} s")
        print("=" * 50 + "\n")

        # Save summary JSON
        summary_file = results_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as sf:
            json.dump(
                {
                    "evaluated": len(completed),
                    "sr": avg_sr,
                    "spl": avg_spl,
                    "ne": avg_ne,
                    "duration_s": total_time,
                    "episodes": results,
                },
                sf,
                indent=2,
            )
        print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()

