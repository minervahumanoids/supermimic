# Standing regression and checkpoint correction

The standing movement comes from the learned controller at checkpoint 403,000.
The defect remains when the native simulator executes the unchanged actor.
The input, action mapping, and source conversion checks do not explain the regression.

The corrected selection uses the original checkpoint at iteration 25,000.
The selection changes no weights, normalizers, inputs, rewards, official source, or architecture.
The external gate checks standing movement and the full approved evaluation cohort.
The final checkpoint remains available with its original provenance.

No new training starts during this investigation.
The training service remains inactive with automatic restart disabled.
The precise training choice that caused the regression remains unproven.
The checkpoint correction does not establish a cure for learning regression in another run.
Keep the next training on hold until its recipe addresses that unresolved issue.

## Standing results

All standing trials use the same original 1,001 frames at 50 Hz.
Standalone MuJoCo executes the original position actuators with four physics steps per control period.

| Measurement | Selected 25,000 | Final 403,000 |
| --- | ---: | ---: |
| Root XY error at 20 seconds | 3.76 cm | 40.49 cm |
| Maximum root XY error | 3.86 cm | 47.07 cm |
| Joint velocity RMS | 0.0201 rad/s | 0.2648 rad/s |
| Action change RMS | 0.00436 | 0.14685 |
| Left ankle XY speed RMS after two seconds | 0.0456 cm/s | 11.4027 cm/s |
| Right ankle XY speed RMS after two seconds | 0.0424 cm/s | 18.7892 cm/s |

Ankle speed measures the movement of the ankle frame in the world XY plane.
Ankle speed does not estimate contact slipping.
The nominal foot friction is 0.6.
At diagnostic friction 1.0, checkpoint 25,000 has a maximum root error of 3.83 cm.
Its left and right ankle speeds remain below 0.051 cm/s.
The native MuJoCo-Warp diagnostic has a maximum root error of 3.86 cm.
Native recordings do not replace the standalone recordings in Viser.

Checkpoint 100,000 initially appeared suitable at nominal friction.
That checkpoint moves 59.46 cm at friction 1.0, so the selection rejects that candidate.
Later retained checkpoints also show substantial standing movement.
`checkpoint-screen.json` preserves all tested results.

## Cause and excluded explanations

The previous audit compared 100 identical generalized states between native physics and standalone MuJoCo.
The observations, goals, actions, and position targets agree within numerical precision.
The standing reference matches the packed training arrays exactly.
The first audit preserves its corrected observation-cache procedure.

The frozen checkpoint comparison changes only the actor artifact.
The earlier checkpoint stands quietly on the same plant and inputs.
Diagnostic normalizer swaps preserve most of the late checkpoint's movement.
The movement therefore follows learned policy parameters rather than normalization statistics alone.
The swaps remain diagnostic files and never become a selected policy.

CUDA inference, TF32, and batch-shaped inference do not remove the final checkpoint's movement.
Held reference poses and removed goal velocities do not provide a reliable correction.
Those altered inputs remain diagnostic copies.
The selected checkpoint receives the unchanged original reference.

The official actor omits root XY position error.
A one-meter root translation leaves its represented state, goal, and action unchanged.
That omission prevents direct correction of accumulated XY error.
That omission alone does not explain why the later checkpoint starts stepping.

The saved dynamics model predicts short standing transitions accurately at the tested starting frames.
The late model has a mean one-step root error below 0.003 cm in those tests.
The 32-step tracking returns also agree closely with physical transitions.
The tests do not support model exploitation as the demonstrated cause.
Local derivative tests do not establish global stability or an exact equilibrium.

## Full fixed cohort

Both checkpoints complete the original 30 BONES clips and three PICO sessions.
Both evaluations preserve source identities, intervals, and clocks.
The reviewed rise clip remains excluded from training and remains present in evaluation.

| Cohort | Checkpoint | Root XY error | Palm position error | Palm orientation error |
| --- | ---: | ---: | ---: | ---: |
| BONES | 25,000 | 10.48 cm | 3.66 cm | 8.01 degrees |
| BONES | 403,000 | 13.71 cm | 3.61 cm | 6.75 degrees |
| PICO | 25,000 | 38.38 cm | 3.42 cm | 6.59 degrees |
| PICO | 403,000 | 81.57 cm | 3.21 cm | 5.81 degrees |

Each result averages the per-clip mean.
The canonical palm metrics average both hands relative to the pelvis.
The earlier checkpoint trades higher hand errors for quieter standing and lower aggregate root error.
Completion counts do not establish gait quality, smoothness, or correct foot contact.
PICO root drift remains substantial.

## Checkpoint gate

`select_checkpoint.py` controls checkpoint selection only.
The gate does not change a training budget or introduce an automatic training stop.
The gate verifies all source intervals, simulator provenance, artifact hashes, and full recordings.
The gate requires both friction trials to satisfy these standing thresholds:

- Maximum root XY error: 10 cm.
- Joint velocity RMS: 0.05 rad/s.
- Action change RMS: 0.02.
- Each ankle XY speed RMS after two seconds: 1 cm/s.

These thresholds define a standing regression check, not a complete locomotion certificate.
Checkpoint 25,000 passes all 33 gate checks.
Checkpoint 403,000 fails the root, joint-velocity, and action-change thresholds as a negative control.
`checkpoint-selection.json` preserves the measured values and report hashes.

## Official source and paper recipe

The official SuperMimic source remains `abc124c1bd5976341034668af3d12c087f2f1f55`.
The mjlab source remains `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`.
Both remote checkouts have no tracked changes.
The fork changes only external launch and evaluation files.
The actor retains both official encoder modules and the original motion-only slot allocation.
The actor has no SMPL encoder.
The original reference includes the current frame and nine future frames, with stride five.

The [paper appendix](https://arxiv.org/abs/2610.28480) specifies Adam and cosine learning-rate decay to 0.03 times the initial rate.
The released source uses Muon for selected weights and Adam for the remaining parameters.
The released source uses constant learning rates.
The stopped run follows that released implementation.
This recipe difference is a candidate contributor, not a proven cause.
The investigation makes no optimizer or schedule patch.

The run also differs from the paper through its policy batch and dataset selection.
The run uses a policy batch of 2,048 because 4,096 exceeds the available GPU memory.
The run uses the full BONES-SEED G1 source after the two authorized exclusions.
The authors' exact filtered clip manifest remains unavailable.
The released quick-start configuration references one LAFAN1 running clip.

## Provenance and review

The selected actor SHA256 is `a0ce1eb1614c6380dc304a0781dc0dc16c9f79e9b05b33d1b0e3ccd06a3c013c`.
Its artifact is `minervahumanoids/Supermimic/model-ga216ipt:v25`.
The previous final artifact remains `model-ga216ipt:v403`.
The original launch commit remains `ad813ef33ed15201044b8d8c0bf0c808628ae64f`.
The W&B run remains stopped with its original identity and history.
The native checkpoints contain no optimizer state.

[W&B run](https://forge.coreweave.com/wandb/minervahumanoids/Supermimic/runs/ga216ipt) · [Training PR](https://github.com/minervahumanoids/supermimic/pull/1)

The unchanged canonical viewer serves the synchronized comparison at `http://127.0.0.1:8972/`.
The comparison shows the original human input, G1 reference, selected checkpoint, and final checkpoint.
The default review uses standalone MuJoCo recordings.
The viewer retains the approved controls, layout, metric definitions, and robot materials.
`completion-receipt.json` records source preservation, stopped training, W&B verification, and release of the owned Orion reservation.
