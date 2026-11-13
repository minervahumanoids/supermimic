"""Select one verified G1 motion from the immutable BONES-SEED conversion."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

CLIP = "clip_6659630b5d665786"
NAME = "walk_ff_loop_180_R_normal_pace_001__A446"
MJLAB = "dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd"


def digest(path):
  return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def prepare(parent, output):
  output.mkdir(parents=True, exist_ok=False)
  records = [json.loads(line) for line in (parent / "selection.jsonl").open()]
  matches = [record for record in records if record.get("source_id") == CLIP]
  if len(matches) != 1:
    raise ValueError("The parent must contain exactly one requested source identity")
  selected = matches[0]
  if (selected["source_name"], selected["source_frames"], selected["output_frames"]) != (NAME, 513, 214):
    raise ValueError("The requested source identity or frame interval differs")
  if not selected["retained"] or selected["source_fps"] != 120 or selected["output_fps"] != 50:
    raise ValueError("The requested source conversion differs")
  source = parent / "npz" / f"{NAME}.npz"
  motion = output / f"{CLIP}.npz"
  shutil.copyfile(source, motion)
  if digest(source) != digest(motion):
    raise ValueError("The selected motion differs from the parent")
  with np.load(motion, allow_pickle=False) as data:
    if len(data["joint_pos"]) != 214 or data["joint_pos"].shape[1] != 29:
      raise ValueError("The selected motion is not the requested G1 interval")
    if float(data["fps"].reshape(-1)[0]) != 50:
      raise ValueError("The selected motion has an incorrect clock")
  selection = {
    "clip_id": CLIP,
    "source": selected,
    "motion_file": motion.name,
    "motion_sha256": digest(motion),
    "frames": 214,
    "rate_hz": 50,
    "first_sample_s": 0.0,
    "last_sample_s": 4.26,
    "source_end_s": 512 / 120,
    "mirrored": False,
  }
  exclusions = {
    "excluded_source_names": [
      "faint_stand_up_lying_side_R_001__A493",
      "faint_stand_up_lying_side_R_001__A493_M",
    ],
    "reason": "Reviewed infeasible rise motions. Neither source belongs to this selection.",
  }
  for name, value in (("selection.json", selection), ("exclusions.json", exclusions)):
    (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
  provenance = {
    "clips": 1,
    "frames": 214,
    "rate_hz": 50,
    "clip_ids": [CLIP],
    "mjlab_commit": MJLAB,
    "parent_directory": str(parent),
    "parent_selection_sha256": digest(parent / "selection.jsonl"),
    "selection_sha256": digest(output / "selection.json"),
    "exclusion_sha256": digest(output / "exclusions.json"),
    "motion_sha256": digest(motion),
    "archive": "gs://minerva-datasets/releases/bones-seed-g1/0.1.0/inputs/g1.tar.gz#1792535636281233",
    "archive_sha256": "52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a",
    "conversion": "Parent G1 conversion. Native mjlab interpolation and derivatives. No new retargeting or filtering.",
  }
  (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
  for path in output.iterdir():
    path.chmod(0o444)
  print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--parent", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  prepare(args.parent.resolve(), args.output.resolve())
