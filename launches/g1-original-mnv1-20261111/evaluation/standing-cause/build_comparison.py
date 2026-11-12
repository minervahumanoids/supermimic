"""Display unchanged milestone and final actors with the approved viewer."""
import copy
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
ORIGINAL = PROJECT / "experiments/g1-stop-review-20261112"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(version=25):
    review = ROOT / f"review-{version*1000}"
    summary = json.loads((review / 'evaluation-summary.json').read_text())
    policy_name = f'G1_{version}'
    manifest = copy.deepcopy(json.loads((ORIGINAL / "review-manifest.json").read_text()))
    old_card = manifest["policies"]["G1"]
    new_card = copy.deepcopy(old_card)
    new_card.update(actor_sha256=summary['actor_sha256'], update=version*1000,
        checkpoint_artifact=summary['checkpoint_artifact'])
    manifest["policies"] = {policy_name: new_card, "G1_403": old_card}
    manifest["display"] = {
        "description": "### G1 standing regression\n\nUnchanged official actor architecture. "
            "Both checkpoints use the same standalone MuJoCo plant and the full approved cohort. "
            "The original human input, G1 reference, and both policies share the saved 50 Hz clock.",
        "provenance": f"The selected {version*1000:,}-iteration checkpoint replaces the regressed 403,000-iteration checkpoint for this review. "
            "Both are unchanged artifacts from the same stopped run. No new training or input filtering occurred.\n\n"
            "The native code, architecture, rewards, original motions, simulation parameters, and metric definitions remain unchanged. "
            "The motion encoder receives both policies' inputs. The official velocity encoder remains untrained. "
            "A SMPL encoder and matched Roger Policy A are unavailable on G1.\n\n"
            "All 30 BONES clips and three PICO sessions remain in the comparison. "
            "The reviewed rise clip remains excluded from training. PICO remains external. "
            "The 25,000 checkpoint passes the standing check at nominal friction 0.6 and diagnostic friction 1.0. "
            "Its hand-orientation error exceeds the final checkpoint's error. "
            "The standing check does not establish perfect gait or eliminate external root drift.",
        "lanes": {
            f"g1_{version}": {"policy": policy_name, "label": f"Selected G1 · {version*1000:,}", "short": f"G1 {version}k",
                "route": "robot", "color": "#356aa0", "dash": ""},
            "g1_403": {"policy": "G1_403", "label": "Regressed G1 · 403,000", "short": "G1 403k",
                "route": "robot", "color": "#bd3c2f", "dash": ""}}}
    manifest["recordings"] = {}
    for cohort in ("bones", "pico"):
        manifest["recordings"][cohort] = {}
        for policy, directory in ((policy_name, review), ("G1_403", ORIGINAL)):
            report_path = directory / "recordings" / cohort / "report.json"
            report = json.loads(report_path.read_text())
            for trial in report["trials"]:
                path = report_path.parent / trial["file"]
                record = dict(trial, file=str(path), sha256=sha(path),
                    report_file=str(report_path), report_sha256=sha(report_path))
                manifest["recordings"][cohort].setdefault(trial["clip"], {})[policy] = {"robot": record}
    (review / "review-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(review / "review-manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=25)
    main(parser.parse_args().version)
