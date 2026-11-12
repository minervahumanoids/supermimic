# Original G1 evaluation

The training stopped at collector iteration 403,091.
The evaluation uses the preserved checkpoint at iteration 403,000.
The checkpoint matches W&B artifact `model-ga216ipt:v403`.

Open the synchronized [Viser comparison](http://127.0.0.1:8971/).

| Cohort | Clips completed | Root XY error | Palm position error | Palm orientation error |
| --- | ---: | ---: | ---: | ---: |
| BONES | 30 / 30 | 13.7 cm | 3.6 cm | 6.7 degrees |
| PICO | 3 / 3 | 81.6 cm | 3.2 cm | 5.8 degrees |

Each value averages the per-clip mean.
Palm measurements average the left and right hands relative to the pelvis.
Root error measures instantaneous world XY distance from the G1 reference.

Global root drift remains substantial on the external PICO sessions.
The native failure gates check root height and projected gravity.
Completion does not establish foot-contact quality or smoothness.

## Execution and provenance

- The simulator is standalone CPU MuJoCo 3.16.0.
- The evaluator loads the original actor with strict state checks.
- The saved world model confirms the action scale and action offset.
- The evaluator uses the official current-plus-future goal window.
- The evaluator creates no optimizer and does not modify the policy.
- The physical model retains the official G1 actuators, limits, contacts, and simulation parameters.
- The clean evaluation uses nominal parameters without noise, pushes, or random resets.
- All recordings retain the complete approved source intervals and the saved 50 Hz clock.

The BONES cohort contains 29 training members and the excluded rise clip.
The rise clip remains excluded from training.
The three PICO sessions remain external inputs.

BONES references come from the original G1 release.
PICO references use the MDP GMR backend at commit `af90c22c64759568c99e5e39bc1ced50eb7ef25c`.
The retargeter uses the official G1 kinematic model and the pinned GMR mapping.
The MDP support detector supplies one constant floor correction for each PICO session.
The original human files and source clocks remain unchanged.

The official model defines palm sites at the left and right wrists.
The evaluator uses those sites for the existing palm measurements.
The viewer retains the original robot colors and its shared ground grid.

The motion encoder receives the evaluation inputs.
The velocity encoder has no training exposure.
The official actor has no SMPL encoder.
Policy A uses Roger, so a matched G1 baseline is unavailable.

## Viewer validation

Mattia explicitly approved G1 support in this chat on November 12, 2026.
The implementation extends the canonical viewer and preserves its controls and metric implementation.
The original Roger comparisons pass validation with both encoder routes.
The G1 recordings pass validation for all 30 BONES clips and all three PICO sessions.
Playback checks include walking and PICO, with synchronized plot cursors.

The viewer retains the lower drift and palm tabs, manual camera orbit, source-frame controls, and root traces.
The G1 camera framing keeps the robot above the lower plot panel.

## Evidence

- `stop-receipt.json` records the stop, source commits, and checkpoint hashes.
- `plant-contract.json` records the official physical model and action transformation.
- `action-contract-parity.json` verifies the saved action buffers.
- `bones-selection.json` and `pico-selection.json` preserve the authoritative cohort identities.
- `bones-reference-index.json` and `pico-reference-index.json` identify the G1 references and their producers.
- `bones-report.json` and `pico-report.json` record standalone simulator provenance and every trial.
- `evaluation-summary.json` contains the aggregate results.
- `viewer-validation.json` and `roger-regression-validation.json` verify the approved viewer contract.
- `compatibility.json` records the approval, compatibility findings, and original viewer hashes.
- `approved-viewer-g1.patch` preserves the approved G1 extension.
- `completion-receipt.json` confirms W&B results and the released Orion reservation.

The training remains stopped, with automatic restart disabled.
The policy artifacts remain on MNV1 and in W&B.

## Recorded execution

The scripts preserve the executed source with its original path contracts.
The remote evaluation directory is `/home/mattia/workspace/supermimic-official-g1-mnv1-20261111/evaluation-403000`.
The complete local recordings remain in `/Users/mattia/Documents/ChatGPT/Supermimic/experiments/g1-stop-review-20261112`.
The canonical viewer remains in `/Users/mattia/Documents/ChatGPT/Supermimic/experiments/master-baseline-review-20261104`.
The patch applies from the local workspace root against the recorded original viewer hashes.
The repository contains provenance and source files. The recordings, human inputs, meshes, and policy weights remain outside Git.

The original training launch uses commit `ad813ef33ed15201044b8d8c0bf0c808628ae64f`.
The evaluation evidence commit does not change that launch identity.

[W&B run](https://forge.coreweave.com/wandb/minervahumanoids/Supermimic/runs/ga216ipt) · [Training PR](https://github.com/minervahumanoids/supermimic/pull/1)
