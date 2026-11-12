"""Normalize BONES G1 columns and use the original mjlab conversion methods."""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import multiprocessing
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import torch
from mjlab.scripts.csv_to_npz import JOINT_NAMES, MotionLoader
from mjlab.scripts.pack_motions import FIELDS
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = DATA / "npz"
PACKED = DATA / "packed"
EXCLUDED = {
  "faint_stand_up_lying_side_R_001__A493": "1e47c51014b5fbfe",
  "faint_stand_up_lying_side_R_001__A493_M": "4de57019aa40237a",
}
SOURCES_SHA256 = "48153713c9d2ec5cd20c63f87c588ce67370a0ece473701f4693202a86c50382"


class BonesMotion(MotionLoader):
  def __init__(self, name, content, fps):
    self.content = content
    super().__init__(name, input_fps=fps, output_fps=50, device="cpu")

  def _load_motion(self):
    header = self.content.split(b"\n", 1)[0].decode().strip().split(",")
    expected = [
      "Frame",
      "root_translateX",
      "root_translateY",
      "root_translateZ",
      "root_rotateX",
      "root_rotateY",
      "root_rotateZ",
    ]
    expected += [name + "_dof" for name in JOINT_NAMES]
    if header != expected:
      raise ValueError(f"Unexpected BONES G1 columns: {self.motion_file}")
    raw = np.loadtxt(io.BytesIO(self.content), delimiter=",", skiprows=1)
    if raw.ndim != 2 or raw.shape[1] != 36 or not np.isfinite(raw).all():
      raise ValueError(f"Invalid source CSV: {self.motion_file}")
    normalized = np.concatenate(
      [
        raw[:, 1:4] / 100,
        Rotation.from_euler("xyz", raw[:, 4:7], degrees=True).as_quat(),
        np.deg2rad(raw[:, 7:]),
      ],
      axis=1,
    )
    motion = torch.from_numpy(normalized).to(torch.float32)
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]
    self.motion_dof_poss_input = motion[:, 7:]
    self.input_frames = len(motion)
    self.duration = (self.input_frames - 1) * self.input_dt


def initialize():
  torch.set_num_threads(1)


def convert(item):
  member, content, metadata = item
  name = Path(member).stem
  record = {
    "source_name": name,
    "archive_member": member,
    "csv_sha256": hashlib.sha256(content).hexdigest(),
    "source_fps": metadata["fps"],
    "source_id": metadata["clip_id"],
    "human_source_sha256": metadata["source_sha256"],
  }
  if name in EXCLUDED:
    source_frames = len(content.splitlines()) - 1
    remaining = int(np.floor((source_frames - 1) / metadata["fps"] * 50 + 1e-6)) - 571
    if remaining >= 3:
      raise RuntimeError("The reviewed exclusion leaves a usable source interval")
    record.update(
      retained=False,
      exclusion="reviewed rise and mirrored source",
      reviewed_fragment=EXCLUDED[name],
      source_frames=source_frames,
      reviewed_interval_s=[0.0, 11.42],
      reason="The remaining output interval has fewer than three frames",
    )
    return record
  with contextlib.redirect_stdout(io.StringIO()):
    motion = BonesMotion(name, content, metadata["fps"])
  arrays = motion.arrays()
  if any(not np.isfinite(arrays[key]).all() for key in FIELDS):
    raise ValueError(f"Nonfinite native conversion: {name}")
  if motion.output_frames < 3:
    raise ValueError(f"Source is too short for native angular velocity: {name}")
  np.savez(OUTPUT / (name + ".npz"), **arrays)
  record.update(
    retained=True,
    source_frames=motion.input_frames,
    human_source_frames=metadata["frames"],
    output_frames=motion.output_frames,
    output_fps=50,
  )
  return record


def inputs(metadata):
  command = subprocess.Popen(
    ["pigz", "-dc", str(DATA / "g1.tar.gz")], stdout=subprocess.PIPE
  )
  count = 0
  names = set()
  with tarfile.open(fileobj=command.stdout, mode="r|") as archive:
    for member in archive:
      if not member.isfile() or not member.name.endswith(".csv"):
        continue
      name = Path(member.name).stem
      if name in names or name not in metadata:
        raise ValueError(f"Duplicate or unrecognized G1 source: {name}")
      names.add(name)
      count += 1
      yield member.name, archive.extractfile(member).read(), metadata[name]
  if command.wait() != 0 or count != 142220 or names != set(metadata):
    raise RuntimeError("The G1 archive population differs from its source index")


def pack(records):
  retained = sorted(
    (r for r in records if r["retained"]), key=lambda r: r["source_name"]
  )
  lengths = np.array([r["output_frames"] for r in retained], dtype=np.int64)
  frames = int(lengths.sum())
  PACKED.mkdir(exist_ok=False)
  arrays = {
    field: np.lib.format.open_memmap(
      PACKED / (field + ".npy"),
      mode="w+",
      dtype=np.float32,
      shape=(
        frames,
        29 if field.startswith("joint_") else 4 if field == "root_quat_w" else 3,
      ),
    )
    for field in FIELDS
  }
  offset = 0
  for i, record in enumerate(retained):
    end = offset + record["output_frames"]
    with np.load(OUTPUT / (record["source_name"] + ".npz"), allow_pickle=False) as clip:
      for field in FIELDS:
        arrays[field][offset:end] = clip[field]
    offset = end
    if i % 10000 == 0:
      print(json.dumps({"packed_clips": i, "frames": frames}), flush=True)
  for array in arrays.values():
    array.flush()
  np.save(PACKED / "motion_ends.npy", np.cumsum(lengths))
  (PACKED / "joint_names.json").write_text(json.dumps(list(JOINT_NAMES)))
  (PACKED / "motion_names.json").write_text(
    json.dumps([r["source_name"] for r in retained])
  )
  return len(retained), frames


def main():
  receipt = json.loads((ROOT / "source-receipt.json").read_text())
  if not receipt["verified"]:
    raise RuntimeError("Verify the immutable G1 archive first")
  source_path = ROOT / "sources.jsonl.gz"
  if (
    hashlib.file_digest(source_path.open("rb"), "sha256").hexdigest() != SOURCES_SHA256
  ):
    raise RuntimeError("The source index checksum differs")
  with gzip.open(source_path, "rt") as stream:
    metadata = {r["name"]: r for r in map(json.loads, stream)}
  if len(metadata) != 142220 or any(r["fps"] != 120 for r in metadata.values()):
    raise RuntimeError("The per-source frame rates differ")
  for name in EXCLUDED:
    if metadata[name]["frames"] != 1376:
      raise RuntimeError("The reviewed rise interval does not cover the source")
  OUTPUT.mkdir(exist_ok=True)
  records = []
  with multiprocessing.get_context("fork").Pool(20, initializer=initialize) as pool:
    with (DATA / "selection.jsonl").open("w") as selection:
      for record in pool.imap(convert, inputs(metadata), chunksize=4):
        records.append(record)
        selection.write(json.dumps(record) + "\n")
        if len(records) % 1000 == 0:
          print(json.dumps({"converted_sources": len(records)}), flush=True)
  clips, frames = pack(records)
  selection_sha = hashlib.file_digest(
    (DATA / "selection.jsonl").open("rb"), "sha256"
  ).hexdigest()
  exclusions = [r for r in records if not r["retained"]]
  (DATA / "exclusions.json").write_text(json.dumps(exclusions, indent=2) + "\n")
  exclusion_sha = hashlib.file_digest(
    (DATA / "exclusions.json").open("rb"), "sha256"
  ).hexdigest()
  provenance = {
    "source": receipt,
    "source_population": len(records),
    "clips": clips,
    "frames": frames,
    "rate_hz": 50,
    "selection_sha256": selection_sha,
    "exclusion_sha256": exclusion_sha,
    "source_index_sha256": SOURCES_SHA256,
    "conversion": "BONES units and columns normalized, then unchanged mjlab MotionLoader interpolation and derivatives",
    "mjlab_commit": "dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd",
    "upstream_code_changed": False,
  }
  (PACKED / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
  print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
  main()
