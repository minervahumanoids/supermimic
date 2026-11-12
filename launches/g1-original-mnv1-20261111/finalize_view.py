"""Add measured native system panels to this training's dedicated W&B view."""

import json
from pathlib import Path

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ROOT = Path(__file__).resolve().parent
ready = json.loads((ROOT / "telemetry-ready.json").read_text())
run = wandb.Api().run("minervahumanoids/Supermimic/" + ready["run_id"])
rows = run.history(stream="system", samples=1, pandas=False)
if not rows:
  raise RuntimeError("The server has no native system measurements")
metrics = [
  key
  for key in (
    "system.gpu.0.gpu",
    "system.gpu.0.memoryAllocatedBytes",
    "system.gpu.0.powerWatts",
    "system.gpu.0.temp",
    "system.cpu",
    "system.memory_percent",
  )
  if key in rows[-1]
]
workspace = ws.Workspace.from_url(ready["workspace_url"])
workspace.sections = [s for s in workspace.sections if s.name != "System"]
workspace.sections.append(
  ws.Section(
    name="System",
    is_open=True,
    layout_settings=ws.SectionLayoutSettings(columns=3, rows=1),
    panels=[
      wr.LinePlot(
        title=key,
        x="_runtime",
        y=[key],
        title_x="Elapsed time (s)",
        layout=wr.Layout(x=(i % 3) * 8, y=(i // 3) * 6, w=8, h=6),
      )
      for i, key in enumerate(metrics)
    ],
  )
)
workspace.save()
restored = ws.Workspace.from_url(ready["workspace_url"])
assert restored.settings.x_axis == "Train/iter"
assert all(
  s.layout_settings.rows == 1 and s.layout_settings.columns == 3
  for s in restored.sections
)
assert not {"Curriculum", "Media"}.intersection(s.name for s in restored.sections)
receipt = {
  "workspace_url": ready["workspace_url"],
  "verified": True,
  "sections": [s.name for s in restored.sections],
  "native_system_keys": metrics,
  "system_axis": "Native system samples use elapsed time because they have no training iteration field",
  "default_axis": "Train/iter",
  "rows": 1,
  "columns": 3,
}
(ROOT / "workspace-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
