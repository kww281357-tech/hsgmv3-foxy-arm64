#!/usr/bin/env python3
"""Watchdog budget monitor for full 1839 remaining qwen3.8-flash run.

Monitors cumulative API token consumption in real-time.
Gracefully shuts down all workers once the cost reaches target budget (270-275 RMB).
"""

import glob
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path("/home/w/project/hsgmv3-target-grounding/results/full_1839_remaining_qwen38flash")
LOG_FILE = ROOT / "live_progress.log"
STATUS_JSON = ROOT / "live_budget_status.json"
REPORT_JSON = ROOT / "final_budget_stop_report.json"

TARGET_BUDGET_RMB = 275.0
SAFETY_STOP_RMB = 268.0  # Stop at 268 RMB (7 RMB remaining buffer) to guarantee we never exceed 275 RMB

# Empirical console-calibrated pricing for qwen3.8-flash:
# - At 06:36 (808 v2 calls + 688 v3 calls): console spent = 18.0 RMB (257.0 RMB left)
# - At 12:15 (8,944 v2 calls + 688 v3 calls): console spent = 141.0 RMB (134.0 RMB left)
# -> Delta: 8,136 v2 calls cost 123.0 RMB -> 0.015118 RMB / v2 HTTP request (~0.224 RMB / v2 episode)
# -> Earlier 688 v3 calls + base overhead cost: 18.0 - (808 * 0.015118) = 5.785 RMB
EMPIRICAL_V2_COST_PER_REQ = 123.0 / 8136.0  # 0.015117994 RMB / v2 HTTP call
EMPIRICAL_V3_COST_MULT = 5.785 / 3.1762     # ~1.821x multiplier over raw text token estimate (exact 141.0 RMB @ 8944 v2 calls)


def append_progress_log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [WATCHDOG] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def calculate_current_usage():
    v3_prompt = 0
    v3_cached = 0
    v3_uncached = 0
    v3_completion = 0
    v3_calls = 0

    v2_calls = 0

    for log_p in ROOT.rglob("*.log"):
        if log_p.name == "live_progress.log":
            continue
        try:
            txt = log_p.read_text(errors="ignore")
        except Exception:
            continue

        if "v3_arm1_improved" in str(log_p):
            matches = re.findall(
                r"\[TokenUsage\]\s+\w+:\s+prompt=(\d+)\s+\(cached=(\d+),\s+hit=[\d.]+%\),\s+completion=(\d+)",
                txt,
            )
            for m in matches:
                p = int(m[0])
                c = int(m[1])
                comp = int(m[2])
                v3_prompt += p
                v3_cached += c
                v3_uncached += p - c
                v3_completion += comp
                v3_calls += 1
        elif "v2_baseline" in str(log_p):
            v2_calls += len(re.findall(r"HTTP Request:\s+POST", txt))

    raw_cost_v3 = (
        (v3_uncached / 1e6) * 0.20
        + (v3_cached / 1e6) * 0.04
        + (v3_completion / 1e6) * 2.00
    )
    cost_v3 = raw_cost_v3 * EMPIRICAL_V3_COST_MULT
    cost_v2 = v2_calls * EMPIRICAL_V2_COST_PER_REQ
    total_cost = cost_v3 + cost_v2
    remaining_rmb = max(0.0, TARGET_BUDGET_RMB - total_cost)

    # Count completed episodes
    v3_ep_done = len(list(ROOT.glob("v3_arm1_improved/**/episode_*_results.json")))
    v2_ep_done = len(list(ROOT.glob("v2_baseline/**/episode_*_results.json")))

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cost_rmb": round(total_cost, 4),
        "remaining_budget_rmb": round(remaining_rmb, 4),
        "target_budget_rmb": TARGET_BUDGET_RMB,
        "safety_stop_rmb": SAFETY_STOP_RMB,
        "budget_used_percent": round(total_cost / TARGET_BUDGET_RMB * 100, 2),
        "v3": {
            "api_calls": v3_calls,
            "uncached_tokens": v3_uncached,
            "cached_tokens": v3_cached,
            "completion_tokens": v3_completion,
            "cost_rmb": round(cost_v3, 4),
            "episodes_completed": v3_ep_done,
        },
        "v2": {
            "api_calls": v2_calls,
            "empirical_cost_per_req_rmb": round(EMPIRICAL_V2_COST_PER_REQ, 6),
            "cost_rmb": round(cost_v2, 4),
            "episodes_completed": v2_ep_done,
        },
        "total_episodes_completed": v3_ep_done + v2_ep_done,
    }


def stop_experiments(usage_data, reason="BUDGET_OR_TIME_REACHED"):
    append_progress_log(
        f"!!! STOP THRESHOLD REACHED ({reason}): {usage_data['total_cost_rmb']} RMB "
        f"(Remaining: {usage_data['remaining_budget_rmb']} RMB, {usage_data['budget_used_percent']}% of {TARGET_BUDGET_RMB} RMB) !!!"
    )
    append_progress_log("Stopping all runner and worker processes gracefully...")

    # Kill runner and child processes (v3+v2, v2-only, and v3-only runners)
    subprocess.run(["pkill", "-f", "[r]un_remaining_1541_v3_and_v2.py"], capture_output=True)
    subprocess.run(["pkill", "-f", "[r]un_v2_only_batch.py"], capture_output=True)
    subprocess.run(["pkill", "-f", "[r]un_v3_only_batch.py"], capture_output=True)
    subprocess.run(["pkill", "-f", "[v]3_w[0-9]_c"], capture_output=True)
    subprocess.run(["pkill", "-f", "[v]2_w[0-9]_c"], capture_output=True)
    subprocess.run(["pkill", "-f", "[v]2only_w[0-9]_c"], capture_output=True)

    usage_data["status"] = f"STOPPED_{reason}"
    usage_data["stop_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        REPORT_JSON.write_text(json.dumps(usage_data, indent=2, ensure_ascii=False), encoding="utf-8")
        STATUS_JSON.write_text(json.dumps(usage_data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"Error writing report: {e}")

    append_progress_log("All processes stopped safely. Final report saved.")


def main():
    max_runtime_sec = 5 * 3600  # 5 hours hard limit per user instruction
    start_ts = time.time()
    deadline_ts = start_ts + max_runtime_sec
    deadline_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(deadline_ts))

    append_progress_log(
        f"Budget & 5h-timer watchdog initialized. Target: {TARGET_BUDGET_RMB} RMB "
        f"(Safety Stop: {SAFETY_STOP_RMB} RMB, Hard Time Stop: {deadline_str})."
    )
    last_print = 0

    while True:
        try:
            now = time.time()
            rem_sec = max(0.0, deadline_ts - now)
            usage = calculate_current_usage()
            usage["status"] = "RUNNING"
            usage["hard_stop_deadline"] = deadline_str
            usage["remaining_minutes_to_time_stop"] = round(rem_sec / 60.0, 1)

            # Write status file for real-time inspection
            STATUS_JSON.write_text(json.dumps(usage, indent=2, ensure_ascii=False), encoding="utf-8")

            # Print every 2 minutes
            if now - last_print >= 120:
                append_progress_log(
                    f"Status: {usage['total_cost_rmb']:.2f}/{TARGET_BUDGET_RMB} RMB "
                    f"(Rem: {usage['remaining_budget_rmb']:.2f} RMB, Time left: {usage['remaining_minutes_to_time_stop']}m) | "
                    f"v2 completed: {usage['v2']['episodes_completed']} | "
                    f"v2 calls: {usage['v2']['api_calls']}"
                )
                last_print = now

            if usage["total_cost_rmb"] >= SAFETY_STOP_RMB:
                stop_experiments(usage, reason="BUDGET_LIMIT_268RMB")
                break

            if now >= deadline_ts:
                stop_experiments(usage, reason="FIVE_HOUR_TIME_LIMIT")
                break

        except Exception as e:
            print(f"[Watchdog Error] {e}", flush=True)

        time.sleep(15)


if __name__ == "__main__":
    main()

