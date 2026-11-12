# Original G1 training on MNV1

This launch uses the official SuperMimic learner and G1 robot.
The remote source checkouts remain unchanged.

| Item | Selection |
| --- | --- |
| SuperMimic source | `abc124c1bd5976341034668af3d12c087f2f1f55` |
| mjlab source | `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd` |
| Task | `SuperMimic-GenCoord-G1` |
| Host | MNV1, RTX 5090 |
| Source dataset | Full BONES-SEED G1 archive |
| Source population | 142,220 motions, 120 Hz |
| Training frequency | 50 Hz |
| Training iterations | 1,000,000 |
| Environment count | 4,096 |
| WM batch | 2,048, ten batches |
| Policy batch | 2,048 |
| Wall limit | None |
| W&B project | `minervahumanoids/Supermimic` |

The actor retains the official motion and velocity encoders.
The official slot mix assigns all environments to the motion encoder.
The reference contains the current frame and nine future frames, with stride five.
The launch retains the official rewards, horizons, and other runner parameters.
The default policy batch of 4,096 exhausts MNV1 memory during CUDA graph capture.
The launch reduces the policy batch to 2,048 through the native configuration.
The launch records this hardware adjustment in W&B configuration.

## Dataset provenance

The immutable archive is:
`gs://minerva-datasets/releases/bones-seed-g1/0.1.0/inputs/g1.tar.gz#1792535636281233`.

Its SHA256 is `52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a`.
Its size is 23,499,973,647 bytes.

The adapter converts centimeters to meters and joint degrees to radians.
The adapter converts XYZ Euler degrees to quaternions with SciPy.
The original mjlab methods interpolate poses and compute velocities.
The adapter preserves the original G1 joint order.
The packer preserves every converted float32 sample and clip boundary.

The selection excludes the reviewed `faint_stand_up_lying_side_R_001__A493` source and its mirror.
The excluded interval leaves fewer than three output frames in each source.
The selection preserves all other source motions.
The processed GCS release has gain changes and additional filtering.
This launch uses the immutable source archive.

The conversion writes source identities, CSV hashes, frame counts, selection hashes, and exclusion hashes.
The packed provenance records the source generation and conversion commit.

## Runtime and telemetry

The separate runtime project preserves the original dependency files.
The runtime pins Torch 2.14.1 with CUDA 13.0 and mjviser 0.0.16.
MuJoCo 3.16.0 supports the official asset format.

The smoke run uses local logs and the full selected dataset.
The smoke run tests the complete environment and learner configurations.
The full run initializes online W&B telemetry before the native training loop starts.
The admission check verifies a server metric and the saved workspace layout.
The external logger measures actual completed episodes and actual transitions.
The external logger preserves the native learner, reward, sampler, and optimizer behavior.
Orion records the PR number and the dedicated W&B link.

The persistent service has no runtime deadline and no automatic restart.
The native learner stops after its original iteration count.
The training starts from fresh model parameters.

## Files

- `download_source.py` retrieves and verifies the immutable source archive.
- `convert_bones.py` converts and packs the complete selected G1 dataset.
- `runtime-pyproject.toml` defines the separate runtime environment.
- `launch.py` verifies provenance and admits the original learner.
- `standard_logger.py` adapts training telemetry.
- `orion.py` associates the training with its reservation.

## User-requested stop and evaluation

The training stopped at collector iteration 403,091 on November 12, 2026.
The evaluation uses checkpoint 403,000, W&B artifact `model-ga216ipt:v403`.
The standalone MuJoCo evaluation completed all 30 approved BONES clips and all three PICO sessions.
The [evaluation report](evaluation/README.md) preserves the results, source identities, compatibility approval, and checkpoint hashes.
The service remains inactive with automatic restart disabled.
The original launch commit remains `ad813ef33ed15201044b8d8c0bf0c808628ae64f`.

## Standing regression correction

The final checkpoint develops standing movement that also occurs in native physics.
The [standing investigation](evaluation/standing-cause/README.md) selects the unchanged checkpoint at iteration 25,000.
The selection reduces the standing endpoint root error from 40.49 cm to 3.76 cm.
The earlier checkpoint passes the full approved cohort and the separate standing gate at two friction values.
The earlier checkpoint has higher aggregate hand errors than the final checkpoint.
The official source and architecture remain unchanged.
The precise training cause remains unproven, so the next training remains on hold.
