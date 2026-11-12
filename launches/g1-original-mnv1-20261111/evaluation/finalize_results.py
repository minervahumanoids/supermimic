"""Record the completed evaluation without resuming the stopped training."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

import wandb

ROOT = Path(__file__).resolve().parent
API = "https://orion1.taile1c92b.ts.net/api/availability"
RUN = "minervahumanoids/Supermimic/ga216ipt"
UNIT = "supermimic-g1-original-train-20261111.service"


def read_flags():
    with urllib.request.urlopen(API, timeout=20) as response:
        return json.load(response)["flags"]


def main():
    state = subprocess.check_output(
        ["systemctl", "--user", "show", UNIT, "-p", "ActiveState", "-p", "MainPID", "-p", "Restart"],
        text=True,
    )
    assert all(value in state for value in ("ActiveState=inactive", "MainPID=0", "Restart=no"))
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True
    ).strip()
    assert not gpu, "Another compute process is active. Preserve the reservation."
    summary = json.loads((ROOT / "evaluation-summary.json").read_text())
    assert summary["optimizer_steps"] == 0 and not summary["policy_modified"]
    assert summary["iteration"] == 403000
    evidence = {
        "status": "complete",
        "checkpoint_artifact": "minervahumanoids/Supermimic/model-ga216ipt:v403",
        "actor_sha256": "07e064bd4b6500b97a4a6b4af0060a801588abb8f0abdceb38138e2c40c0144b",
        "backend": "standalone CPU MuJoCo",
        "source_clock_hz": 50,
        "route": "motion encoder",
        "reference_window": "Original current frame and nine future frames, stride five",
        "pull_request": "https://github.com/minervahumanoids/supermimic/pull/1",
        "baseline": "Policy A unavailable on G1",
        "smpl_route": "Unavailable in the official G1 actor",
        "metric_reduction": "Equal-weight mean of per-clip means. Palms average both hands relative to the pelvis.",
        "cohorts": summary["cohorts"],
        "report_sha256": {
            cohort: hashlib.sha256((ROOT / "recordings" / cohort / "report.json").read_bytes()).hexdigest()
            for cohort in ("bones", "pico")
        },
        "limitations": "Root drift remains substantial on PICO. Completion does not establish foot-contact quality or smoothness.",
    }
    api = wandb.Api(timeout=60)
    run = api.run(RUN)
    assert run.state == "killed"
    launch_commit = run.config["launch_commit"]
    run.config["stopped_evaluation"] = evidence
    run.update()
    run.summary["stopped_evaluation"] = summary
    run.summary["evaluation_status"] = "complete"
    run.summary.update()
    verified = wandb.Api(timeout=60).run(RUN)
    assert verified.state == "killed"
    assert verified.config["launch_commit"] == launch_commit
    assert verified.config["stopped_evaluation"] == evidence
    assert verified.summary["evaluation_status"] == "complete"
    for cohort, values in summary["cohorts"].items():
        for key, value in values.items():
            assert verified.summary[f"stopped_evaluation.cohorts.{cohort}.{key}"] == value

    before = read_flags().get("mnv1")
    if before:
        assert before["owner"] == "Mattia" and before["kind"] == "reservation"
        assert before["note"] == "PR1 G1 eval: checkpoint 403000"
        assert before["wandb_url"] == verified.url
        request = urllib.request.Request(
            API,
            data=json.dumps({"machine": "mnv1", "operation": "clear", "expectedVersion": before["version"]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
    after = read_flags().get("mnv1")
    assert after is None
    receipt = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "service_state": state,
        "gpu_compute_processes": gpu,
        "wandb_url": verified.url,
        "wandb_state": verified.state,
        "launch_commit": launch_commit,
        "checkpoint_iteration": 403000,
        "evaluation_published": True,
        "orion_before": before,
        "orion_after": after,
        "unrelated_jobs_modified": False,
    }
    (ROOT / "completion-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
