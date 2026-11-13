"""Associate the original G1 training with its Orion reservation."""

import json
import threading
import time
import urllib.request
from pathlib import Path

import wandb

API = "https://orion1.taile1c92b.ts.net/api/availability"


def read():
  with urllib.request.urlopen(API, timeout=20) as response:
    return json.load(response)


def associate(run, pr_url, directory):
  number = int(pr_url.rstrip("/").rsplit("/", 1)[1])
  batch = run.config["runner_cfg"]["policy"]["batch_size"]
  note = f"PR{number} G1 walk: batch {batch}"
  current = read()["flags"].get("mnv1")
  if current and (
    current["owner"] != "Mattia"
    or current["kind"] != "reservation"
    or not current["note"].startswith(f"PR{number} ")
  ):
    raise RuntimeError("MNV1 has another reservation")
  payload = {
    "machine": "mnv1",
    "operation": "save",
    "owner": "Mattia",
    "kind": "reservation",
    "duration": 1440,
    "note": note,
    "wandb_url": run.url,
    "expectedVersion": current["version"] if current else None,
  }

  def save():
    request = urllib.request.Request(
      API,
      data=json.dumps(payload).encode(),
      headers={"Content-Type": "application/json"},
      method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
      response.read()
    flag = read()["flags"].get("mnv1")
    if not flag or flag["note"] != note or flag["wandb_url"] != run.url:
      raise RuntimeError("Orion did not confirm the training association")
    payload["expectedVersion"] = flag["version"]
    (Path(directory) / "orion-receipt.json").write_text(
      json.dumps(flag, indent=2) + "\n"
    )

  save()

  def renew():
    while wandb.run is run:
      time.sleep(6 * 60 * 60)
      if wandb.run is not run:
        return
      try:
        flag = read()["flags"].get("mnv1")
        if not flag or flag["version"] != payload["expectedVersion"]:
          return
        remote = wandb.Api().run(f"{run.entity}/{run.project}/{run.id}")
        if time.time() - remote.summary.get("_timestamp", 0) < 900:
          save()
      except Exception as error:
        print(json.dumps({"orion_renewal_error": str(error)}), flush=True)

  threading.Thread(target=renew, daemon=True).start()
