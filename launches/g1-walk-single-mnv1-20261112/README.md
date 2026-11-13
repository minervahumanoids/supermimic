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
The native iteration count supplies the training stop condition.
Training measurements describe fit to this clip. They do not measure generalization.
