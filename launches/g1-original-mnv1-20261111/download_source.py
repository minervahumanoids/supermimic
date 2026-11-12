"""Download the immutable G1 source without storing the access token."""

import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "data" / "g1.tar.gz"
GENERATION = "1792535636281233"
SIZE = 23499973647
SHA256 = "52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a"
URL = (
  "https://storage.googleapis.com/minerva-datasets/"
  "releases/bones-seed-g1/0.1.0/inputs/g1.tar.gz?generation=" + GENERATION
)


def main():
  token = sys.stdin.readline().strip()
  TARGET.parent.mkdir(exist_ok=True)
  temporary = TARGET.with_suffix(".gz.partial")
  start = temporary.stat().st_size if temporary.exists() else 0
  parts = TARGET.parent / "download-parts"
  parts.mkdir(exist_ok=True)

  def fetch(bounds):
    first, last = bounds
    path = parts / f"{first}-{last}"
    if path.exists() and path.stat().st_size == last - first + 1:
      return path
    request = urllib.request.Request(
      URL,
      headers={
        "Authorization": "Bearer " + token,
        "Range": f"bytes={first}-{last}",
      },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
      if response.status != 206:
        raise RuntimeError("The server did not honor the range")
      with path.open("wb") as stream:
        while chunk := response.read(8 << 20):
          stream.write(chunk)
    if path.stat().st_size != last - first + 1:
      raise RuntimeError("Incomplete source range")
    return path

  bounds = [
    (first, min(first + (256 << 20), SIZE) - 1)
    for first in range(start, SIZE, 256 << 20)
  ]
  with ThreadPoolExecutor(max_workers=8) as pool, temporary.open("ab") as stream:
    for path in pool.map(fetch, bounds):
      with path.open("rb") as source:
        while chunk := source.read(8 << 20):
          stream.write(chunk)
          start += len(chunk)
      stream.flush()
      path.unlink()
      print(json.dumps({"download_bytes": start, "total": SIZE}), flush=True)
  if temporary.stat().st_size != SIZE:
    raise RuntimeError("The source archive size differs")
  digest = hashlib.file_digest(temporary.open("rb"), "sha256").hexdigest()
  if digest != SHA256:
    raise RuntimeError("The source archive checksum differs")
  temporary.rename(TARGET)
  (ROOT / "source-receipt.json").write_text(
    json.dumps(
      {
        "gcs_uri": "gs://minerva-datasets/releases/bones-seed-g1/0.1.0/inputs/g1.tar.gz",
        "generation": GENERATION,
        "bytes": SIZE,
        "sha256": digest,
        "verified": True,
      },
      indent=2,
    )
    + "\n"
  )
  print(json.dumps({"source_verified": True, "bytes": SIZE}), flush=True)


if __name__ == "__main__":
  main()
