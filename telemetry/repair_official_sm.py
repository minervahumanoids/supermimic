"""Back up and append exact compatibility metrics to existing official SM runs."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import wandb

from official_sm_contract import ENTITY, PROJECT, VERSION, normalize, normalized_definitions


def export(api, run_id, root):
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    path = root / f"{run_id}-source.jsonl.gz"
    if not path.exists():
        temporary = path.with_suffix(".partial")
        with gzip.open(temporary, "wt") as stream:
            for row in run.scan_history(page_size=10000):
                stream.write(json.dumps(row) + "\n")
        temporary.rename(path)
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    rows = [row for row in rows if "Train/iter" in row and ("Perf/collect_s" in row or "Perf/time/collect_s" in row)]
    return run, rows


def repair(run_id, root):
    api = wandb.Api()
    original, rows = export(api, run_id, root)
    initial_state = original.config.get("telemetry_contract", {}).get("original_state", original.state)
    if original.state == "running" or initial_state == "running":
        raise RuntimeError("Finish the native publisher before this historical import")
    if initial_state not in ("killed", "finished", "failed"):
        raise RuntimeError(f"Unsupported archived state: {initial_state}")
    last_iteration = 0
    normalized_rows = []
    for row in rows:
        iteration = int(row["Train/iter"])
        if iteration <= last_iteration:
            continue
        record = normalize(row, last_iteration, aliases_only=True)
        last_iteration = iteration
        if record:
            normalized_rows.append(record)
    if not normalized_rows:
        raise RuntimeError("No source measurements to normalize")
    receipt_path = root / f"{run_id}-repair.json"
    if receipt_path.exists():
        raise RuntimeError("This run already has a repair receipt. Do not duplicate its history.")
    native_definitions = original.config.get("metric_definitions", {})
    definitions = normalized_definitions(native_definitions)
    original.config["telemetry_contract"] = {
        "version": VERSION,
        "canonical_definitions": definitions,
        "historical_keys_preserved": True,
        "episode_window": native_definitions.get("Reward/total", {}).get("sampling_window"),
        "episode_recomputation": "Historical raw episode events are unavailable. Preserve measured historical reductions.",
        "timing_reduction": "Seconds per actual collector iteration.",
        "original_state": initial_state,
    }
    original.update()
    run = wandb.init(entity=ENTITY, project=PROJECT, id=run_id, resume="must", mode="online", settings=wandb.Settings(x_primary=False, x_update_finish_state=True, x_disable_stats=True, console="off", disable_code=True))
    try:
        run.define_metric("Train/iter")
        for key in sorted(set().union(*(record.keys() for record in normalized_rows))):
            if key not in ("Train/iter", "_timestamp"):
                run.define_metric(key, step_metric="Train/iter")
        for record in normalized_rows:
            run.log(record)
        run.finish(exit_code=255 if initial_state == "killed" else 1 if initial_state in ("failed", "crashed") else 0)
    finally:
        if not run._is_finished:
            run.finish(exit_code=255 if initial_state == "killed" else 1)
    deadline = time.monotonic() + 30
    while True:
        remote = wandb.Api().run(f"{ENTITY}/{PROJECT}/{run_id}")
        if remote.state == initial_state or time.monotonic() >= deadline:
            break
        time.sleep(1)
    assert remote.state == initial_state, (remote.state, initial_state)
    receipt = dict(run_id=run_id, contract=VERSION, source_rows=len(rows), appended_rows=len(normalized_rows), first_iteration=normalized_rows[0]["Train/iter"], last_iteration=normalized_rows[-1]["Train/iter"], original_state=initial_state, final_state=remote.state, original_launch_commit=original.config.get("launch_commit"), original_source_commit=original.config.get("source_commit"), no_training_restart=True, timestamp_source="Original server-confirmed source history. No synthetic heartbeat.")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    repair(args.run_id, args.evidence)
