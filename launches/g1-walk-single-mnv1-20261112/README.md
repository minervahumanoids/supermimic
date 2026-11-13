# G1 training on one walking clip

This run trains fresh G1 weights on `clip_6659630b5d665786` only.
The goal is to overfit the selected motion.
The run preserves the official learner, architecture, rewards, augmentation, optimizer, and sampling behavior.
The run changes the dataset selection and external bookkeeping.

| Item | Selection |
| --- | --- |
| SuperMimic source | `abc124c1bd5976341034668af3d12c087f2f1f55` |
| mjlab source | `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd` |
| Task | `SuperMimic-GenCoord-G1` |
| Host | MNV1, RTX 5090 |
| Clip | `clip_6659630b5d665786` |
| G1 source | `walk_ff_loop_180_R_normal_pace_001__A446` |
| Selection | One motion, 214 frames, 50 Hz |
| Sample interval | 0.00–4.26 seconds |
| Initialization | Fresh weights |
| Environments | 4,096 |
| WM batch | 2,048, ten batches |
| Policy batch | 4,096 |
| Native stop | 1,000,000 collector iterations |
| Wall limit | None |
| W&B project | `minervahumanoids/Supermimic` |

The actor retains both official encoders: motion and velocity.
The official slot allocation uses the motion encoder for all environments.
The task has no SMPL encoder.
The reference contains the current frame and nine future frames, with stride five.

## Data provenance

The selection uses the original G1 conversion from the immutable BONES-SEED archive.
The selection preserves the source identity, 513 source frames, and native conversion at 50 Hz.
The selection includes no mirrored motion or other clip.
The selection excludes the two reviewed infeasible rise motions.
The original dataset and historical checkpoints remain unchanged.

`prepare_data.py` verifies the requested identity and copies the existing G1 motion.
The script writes motion, selection, exclusion, and parent hashes.
`launch.py` verifies these hashes before each execution.
The launcher verifies both official source commits and rejects changes to tracked source files.
The launcher rejects changes to native runner parameters, except the run name and W&B project.

## Validation and telemetry

The local smoke execution tests the complete native environment and learner configurations.
The smoke execution uses no W&B run.
The smoke execution checks finite collection, finite learner losses, and episode reward additivity.
The full training requires confirmed online W&B metrics before the native training loop starts.
The external logger pools all completed episodes within each logging interval.
The logger uses the same episodes for returns, reward components, and episode lengths.
The W&B workspace uses training iterations and one visible row with three columns per section.
Orion stores the PR number and the dedicated W&B link.

The persistent service has no runtime deadline and no automatic restart.
The original native budget was 1,000,000 collector iterations.
The later user request replaced that budget with a stop at checkpoint 15,000.
Training measurements describe fit to this clip. They do not measure generalization.

## Stopped run

[W&B training](https://forge.coreweave.com/wandb/minervahumanoids/Supermimic/runs/45u76a15) · [PR #2](https://github.com/minervahumanoids/supermimic/pull/2)

The launch uses commit `3796d8fe7c4f8cd42e253339ae77d9aa89eb96a5`.
The full native smoke configuration passed five learner updates without W&B initialization.
The online verification confirmed iteration 331 and 1,355,776 actual transitions.
The verification confirmed finite losses, fresh server metrics, the saved workspace, and the Orion association.
The source checkouts remain unchanged.
The [launch evidence](evidence/launch-verification.json) preserves the service configuration and dataset hashes.


## Stop and matched evaluation

The service is inactive, and the Orion reservation is released.
The last logged collector iteration is 15,051, which exceeds the requested stop threshold.
The preserved native checkpoint 15,000 matches W&B artifact `model-45u76a15:v15`.
The original launch commit and checkpoint identities remain unchanged.

Fresh standalone MuJoCo recordings compare this checkpoint with full-dataset G1 checkpoint 25,000 on the same requested clip.
Both policies complete 214/214 samples with the same MuJoCo 3.15.0 plant and Torch 2.15.0+cpu runtime.
The comparison uses zero optimizer steps.

| Mean error | Single clip, 15k | Full dataset, 25k |
| --- | ---: | ---: |
| Root XY | 2.00 cm | 5.72 cm |
| Palm position | 1.92 cm | 2.54 cm |
| Palm orientation | 4.13 degrees | 5.38 degrees |
| Joint RMSE | 3.28 degrees | 3.95 degrees |

The single-clip policy has higher left wrist orientation error: 5.37 degrees versus 4.67 degrees.
These measurements describe this training clip only.
The approved viewer shows synchronized original human motion, the reference, and both policies.
The default cohorts remain unchanged, and all 11 viewer tests pass.
The [evaluation evidence](evidence/comparison-summary.json) preserves the actor and plant hashes.
The [stop receipt](evidence/stop-receipt.json) preserves the exact service and checkpoint state.

## Shared logs

The three requested official runs now share canonical metric names and timing units.
The personal W&B workspace and original named views use the same sections and one-row, three-column layout.
Original metrics, run IDs, launch commits, checkpoint provenance, and stopped states remain intact.
The original full-dataset G1 and Roger runs retain their historical rolling episode reductions.
Raw episode events are unavailable, so those reductions cannot become interval reductions retrospectively.
Each run records its actual definitions in configuration.

Future launches use `telemetry/launch_with_standard_logs.py` and one shared logger.
The shared logger pools all completed episodes in each interval and excludes terminal samples from root diagnostics.
The wrapper leaves the native learner, architecture, rewards, and stop condition unchanged.
The [telemetry verification](evidence/telemetry-final-verification.json) records every checked source row.
