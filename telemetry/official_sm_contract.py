"""One telemetry vocabulary for official SuperMimic training and old run views."""

from __future__ import annotations

VERSION = "official-sm-1.0"
ENTITY = "minervahumanoids"
PROJECT = "Supermimic"
PERSONAL_VIEW_NAME = "nw-nwusermattia_mh-w"
WORKSPACE_URL = "https://forge.coreweave.com/wandb/minervahumanoids/Supermimic/workspace?nw=nwusermattia_mh"

ALIASES = {
    "Train/wm_grad_norm": "Train/grad/wm",
    "Train/policy_grad_norm": "Train/grad/policy",
    "Train/imag_return": "Train/policy/imag_return",
    "Metrics/wm_root_m": "Metrics/wm/root_m",
    "Metrics/motion/root_xy_m": "Metrics/robot/root_xy_m",
    "Reward/motion/total": "Reward/robot/total",
    "Episode/motion/len_s": "Episode/robot/len_s",
    "Episode/motion/fail_frac": "Episode/robot/fail_frac",
    "Episode/motion/timeout_frac": "Episode/robot/timeout_frac",
    "Episode/motion/count": "Episode/robot/count",
    "Train/motion/steps": "Train/routes/robot_steps",
    "Train/motion/frac": "Train/routes/robot_frac",
    "Reward/terms/action_rate_l2": "Reward/terms/action_rate",
    "Reward/terms/motion_global_root_pos": "Reward/terms/root_pos",
    "Reward/terms/motion_global_root_ori": "Reward/terms/root_ori",
    "Reward/terms/motion_body_pos": "Reward/terms/body_pos",
    "Reward/terms/motion_body_ori": "Reward/terms/body_ori",
    "Reward/terms/motion_body_lin_vel": "Reward/terms/body_linvel",
    "Reward/terms/motion_body_ang_vel": "Reward/terms/body_angvel",
}
INTERVAL_TIMES = {
    "Perf/collect_s": "Perf/time/collect_s",
    "Perf/learn_s": "Perf/time/learn_s",
    "Perf/wm_update_s": "Perf/time/wm_s",
    "Perf/policy_update_s": "Perf/time/policy_s",
}
REWARD_TITLES = {
    "action_rate": "Action change",
    "joint_limit": "Joint limit",
    "root_pos": "Root position",
    "root_ori": "Root orientation",
    "body_pos": "Body position",
    "body_ori": "Body orientation",
    "body_linvel": "Body linear velocity",
    "body_angvel": "Body angular velocity",
}


def canonical_key(key):
    return ALIASES.get(key, INTERVAL_TIMES.get(key, key))


def normalize(row, previous_iteration, *, aliases_only=False):
    """Convert interval timing totals using the actual collector-counter delta.

    Historical episode reductions remain unchanged. Missing raw episodes cannot
    reconstruct an interval reduction from a rolling completed-episode mean.
    """
    iteration = int(row["Train/iter"])
    count = iteration - previous_iteration
    if count <= 0:
        raise ValueError("Collector iterations must increase within the source stream")
    result = {} if aliases_only else {key: value for key, value in row.items() if value is not None}
    for source, destination in ALIASES.items():
        if row.get(source) is not None:
            if row.get(destination) is not None and row[destination] != row[source]:
                raise ValueError(f"Conflicting metric aliases: {destination}")
            result[destination] = row[source]
            if not aliases_only:
                result.pop(source, None)
    for source, destination in INTERVAL_TIMES.items():
        if row.get(source) is not None:
            value = row[source] / count
            if row.get(destination) is not None and abs(row[destination] - value) > 1e-7:
                raise ValueError(f"Conflicting timing reductions: {destination}")
            result[destination] = value
            if not aliases_only:
                result.pop(source, None)
    if row.get("Train/motion/steps") is not None and row.get("Train/steps", 0):
        # The official runner uses a fixed slot allocation throughout each run.
        result.setdefault("Train/routes/robot_frac", row["Train/motion/steps"] / row["Train/steps"])
    if result:
        result["Train/iter"] = iteration
        if aliases_only and "_timestamp" in row:
            result["_timestamp"] = row["_timestamp"]
        if aliases_only and "_runtime" in row:
            result["_runtime"] = row["_runtime"]
    for key in result:
        if not key.startswith(("_", "system/")) and (len(key) > 64 or len(key.split("/")) > 3 or any(len(s) > 32 for s in key.split("/"))):
            raise ValueError(f"Nonstandard metric name: {key}")
    return result


def normalized_definitions(definitions):
    result = {}
    for source, definition in definitions.items():
        value = dict(definition)
        if source in INTERVAL_TIMES or source in INTERVAL_TIMES.values():
            value.update(unit="s / collector iteration", reduction="measured interval wall seconds / actual collector iteration delta", denominator="actual collector iterations in the logging interval", sampling_window="logging interval")
        result[canonical_key(source)] = value
    if "Train/grad/wm" in result:
        result["Train/grad/wm"].update(reduction="L2 parameter gradient norm after native clipping", denominator="latest completed WM update", sampling_window="latest completed WM update")
    if "Train/grad/policy" in result:
        result["Train/grad/policy"].update(reduction="L2 parameter gradient norm before native clipping", denominator="latest completed policy update", sampling_window="latest completed policy update")
    result.setdefault("Train/routes/robot_frac", dict(unit="fraction", frame="not spatial", reduction="fixed native motion-slot fraction", denominator="all collector environments", population="robot-reference training diagnostics", sampling_window="collector interval"))
    return result
