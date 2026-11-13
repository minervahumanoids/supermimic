"""Save the shared official SM layout in the user's existing W&B workspace."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from wandb_workspaces.workspaces import internal

from official_sm_contract import ENTITY, PROJECT, PERSONAL_VIEW_NAME, WORKSPACE_URL, REWARD_TITLES, canonical_key

QUERY = '''query Views($entityName: String, $name: String) {
 project(name: $name, entityName: $entityName) {
  allViews(viewType: "project-view") { edges { node { id name displayName spec } } }
 }
}'''
SAVE = '''mutation SaveShared($id: ID, $entityName: String, $projectName: String, $type: String, $name: String, $displayName: String, $spec: String) {
 upsertView(input: {id: $id, entityName: $entityName, projectName: $projectName, type: $type, name: $name, displayName: $displayName, spec: $spec, createdUsing: WANDB_SDK}) { view { id name } }
}'''


def views():
    data = internal.execute_graphql(wandb.Api(), QUERY, dict(entityName=ENTITY, name=PROJECT))
    return [edge["node"] for edge in data["project"]["allViews"]["edges"]]


def template(reward_names):
    directory = Path(__file__).resolve().parent
    standard = json.loads((directory / 'standard.json').read_text())
    profile = json.loads((directory / 'sm-profile.json').read_text())
    assert standard['workspace']['rows_per_section'] == profile['workspace']['rows_per_section'] == 1
    assert standard['workspace']['columns_per_section'] == profile['workspace']['columns_per_section'] == 3
    assert standard['axes']['default'] == profile['workspace']['x_axis'] == 'Train/iter'
    reward_keys = list(dict.fromkeys(canonical_key("Reward/terms/" + name) for name in reward_names))
    preferred = next(section['panels'] for section in profile['default_panels'] if section['section'] == 'Reward')
    order = {key: index for index, panel in enumerate(preferred) for key in panel.get('keys', [])}
    reward_keys.sort(key=lambda key: order.get(key, len(order)))
    metrics = {
        "Reward": [("Reward/total", "Episode return", "Reward"), ("Reward/step", "Reward per transition", "Reward / transition")]
                  + [(key, REWARD_TITLES.get(key.rsplit("/", 1)[-1], key.rsplit("/", 1)[-1].replace("_", " ")), "Signed episode reward") for key in reward_keys],
        "Loss": [("Loss/wm", "World model loss", "Normalized prediction loss"), ("Loss/policy", "Policy objective", "Reward"), ("Loss/value", "Value loss", "Reward squared"), ("Loss/latent_bc", "Latent imitation loss", "Weighted loss")],
        "Train": [("Train/grad/wm", "World model gradient", "L2 norm after clipping"), ("Train/grad/policy", "Policy gradient", "L2 norm before clipping"), ("Train/steps", "Collected transitions", "Transitions"), ("Train/policy/imag_return", "Imagined return", "Discounted reward"), ("Train/routes/robot_frac", "Robot reference exposure", "Fraction")],
        "Metrics": [("Metrics/robot/root_xy_m", "Robot root XY error", "World XY error (m)"), ("Metrics/wm/root_m", "World model root prediction error", "Root position RMS (m)")],
        "Episode": [("Episode/len_s", "Episode duration", "Seconds"), ("Episode/fail_frac", "Task failure fraction", "Fraction of completed episodes"), ("Episode/timeout_frac", "Clip and time-limit endings", "Fraction of completed episodes")],
        "Perf": [("Perf/fps", "Full throughput", "Actual transitions / wall second"), ("Perf/time/collect_s", "Collection time", "Seconds / collector iteration"), ("Perf/time/learn_s", "Learning time", "Seconds / collector iteration"), ("Perf/time/wm_s", "World model time", "Seconds / collector iteration"), ("Perf/time/policy_s", "Policy time", "Seconds / collector iteration")],
    }
    sections = []
    for name, values in metrics.items():
        panels = [wr.LinePlot(title=title, x="Train/iter", y=[key], title_x="Training iterations", title_y=unit, smoothing_type="none", max_runs_to_show=3,
                             layout=wr.Layout(x=(i % 3) * 8, y=(i // 3) * 6, w=8, h=6)) for i, (key, title, unit) in enumerate(values)]
        section = ws.Section(name=name, panels=panels, is_open=True, layout_settings=ws.SectionLayoutSettings(columns=3, rows=1))
        section._is_panels_auto = False
        sections.append(section)
    # Native SDK statistics have their own wall clock and no collector counter.
    system = [("system/gpu.0.gpu", "GPU use", "%"), ("system/gpu.0.memoryAllocatedBytes", "GPU memory", "Bytes"), ("system/cpu", "CPU use", "%"), ("system/proc.memory.rssMB", "Process RAM", "MiB")]
    panels = [wr.LinePlot(title=title, x="_runtime", y=[key], title_x="Elapsed seconds", title_y=unit, smoothing_type="none", max_runs_to_show=3, layout=wr.Layout(x=(i % 3) * 8, y=(i // 3) * 6, w=8, h=6)) for i, (key, title, unit) in enumerate(system)]
    section = ws.Section(name="System", panels=panels, is_open=True, layout_settings=ws.SectionLayoutSettings(columns=3, rows=1))
    section._is_panels_auto = False
    sections.append(section)
    return ws.Workspace(entity=ENTITY, project=PROJECT, name="Official SuperMimic", sections=sections, settings=ws.WorkspaceSettings(x_axis="Train/iter", group_by_prefix="first", max_runs=3), auto_generate_panels=False)._to_model().spec.model_dump(by_alias=True, exclude_none=True)


def signature(spec):
    return [(section["name"], [(panel.get("config", {}).get("chartTitle"), panel.get("config", {}).get("metrics"), panel.get("config", {}).get("xAxis")) for panel in section["panels"]]) for section in spec["section"]["panelBankConfig"]["sections"]]


def apply_shared_view(run=None, reward_names=None, evidence=None):
    names = reward_names or ["motion_global_root_pos", "motion_global_root_ori", "motion_body_pos", "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel", "action_rate_l2", "joint_limit"]
    before = views()
    targets = [view for view in before if view["name"] == PERSONAL_VIEW_NAME or view["displayName"] in ("g1-SM-walk-20261112-seed69", "g1-SM-original-20261111-seed69", "roger-SM-original-20261112-seed69")]
    assert any(view['name'] == PERSONAL_VIEW_NAME for view in targets)
    generated = template(names)
    for view in targets:
        spec = json.loads(view["spec"])
        section = spec["section"]
        section["panelBankConfig"] = copy.deepcopy(generated["section"]["panelBankConfig"])
        if "settings" in generated["section"]:
            section["settings"] = copy.deepcopy(generated["section"]["settings"])
        section.setdefault("workspaceSettings", {}).update(generated["section"].get("workspaceSettings", {}))
        section["workspaceSettings"]["shouldAutoGeneratePanels"] = False
        # The selection tree contains visible run IDs.
        selection = section["runSets"][0].get("selections", {})
        visible = list(selection.get("tree", []))
        requested = getattr(run, "id", None)
        if requested and requested not in visible:
            visible.append(requested)
        selection["tree"] = visible
        internal.execute_graphql(wandb.Api(), SAVE, dict(id=view["id"], entityName=ENTITY, projectName=PROJECT, type="project-view", name=view["name"], displayName=view["displayName"], spec=json.dumps(spec)))
    after = views()
    expected = signature(generated)
    checked = []
    for original in targets:
        updated = next(v for v in after if v["id"] == original["id"])
        spec = json.loads(updated["spec"])
        assert updated["name"] == original["name"] and signature(spec) == expected
        assert spec["section"]["workspaceSettings"]["shouldAutoGeneratePanels"] is False
        checked.append(dict(id=updated["id"], name=updated["name"], display_name=updated["displayName"]))
    if evidence:
        Path(evidence).write_text(json.dumps(dict(workspace_url=WORKSPACE_URL, verified=True, views=checked, sections=[s[0] for s in expected], same_panel_manifest=True, x_axis="Train/iter", rows=1, columns=3, before=before, after=after), indent=2) + "\n")
    return WORKSPACE_URL


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    print(apply_shared_view(evidence=args.evidence), flush=True)
