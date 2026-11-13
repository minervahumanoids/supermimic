# G1 walking comparison

The comparison uses `clip_6659630b5d665786`, with 214 samples at 50 Hz.
The single-clip policy uses native checkpoint 15,000 from run `45u76a15`.
The full-dataset policy uses native checkpoint 25,000 from run `ga216ipt`.
Both policies train on this clip. These results measure fit, not generalization.

| Mean error | Single clip, 15k | Full dataset, 25k |
| --- | ---: | ---: |
| Root XY | 2.00 cm | 5.72 cm |
| Palm position | 1.92 cm | 2.54 cm |
| Palm orientation | 4.13 degrees | 5.38 degrees |
| Joint RMSE | 3.28 degrees | 3.95 degrees |

Both policies complete all 214 samples without a guard stop.
The single-clip policy has higher left wrist orientation error: 5.37 degrees versus 4.67 degrees.
The reports preserve separate left and right measurements.

## Matched simulator

Both recordings use standalone CPU MuJoCo 3.15.0 and Torch 2.15.0+cpu.
Both recordings use the same plant, gains, initial state, solver, source samples, and native reference window.
The actor retains the official motion and velocity encoders.
The native G1 task allocates all slots to the motion encoder.
The reference contains the current frame and nine future frames, with stride five.
Evaluation performs zero optimizer steps and does not initialize W&B.
`comparison-results.json` and `comparison-summary.json` preserve hashes and simulator provenance.

## Viewer

[Open the synchronized Viser comparison](http://127.0.0.1:8932/).
The approved viewer uses an explicit comparison profile for the requested clip.
The original human input, retargeted reference, and both policies share the source clock.
The viewer preserves the default controls, six lower tabs, native materials, and manual camera.
The default cohorts remain unchanged: 21 designated clips and four priority clips, with 75 recordings.
The viewer passes all 11 existing tests.
`validation.json` records browser playback and control checks.

```bash
experiments/checkpoint-tests-20261103/review-venv/bin/python \
  experiments/ali-baseline-prototype-20261112/review.py --port 8932 \
  --comparison-manifest experiments/g1-walk-15000-vs-25000-20261112/viewer-manifest.json
```

## Stop and provenance

The training stopped after the last logged collector iteration 15,051.
The stop exceeded the requested 15,000 threshold.
The comparison uses the exact preserved native checkpoint 15,000, not the later in-memory weights.
`stop-receipt.json` verifies the checkpoint against W&B artifact `model-45u76a15:v15`.
The training service is inactive and has no automatic restart.
The Orion reservation is released.

[Single-clip W&B run](https://forge.coreweave.com/wandb/minervahumanoids/Supermimic/runs/45u76a15) · [PR #2](https://github.com/minervahumanoids/supermimic/pull/2)

[Full-dataset W&B run](https://forge.coreweave.com/wandb/minervahumanoids/Supermimic/runs/ga216ipt) · [PR #1](https://github.com/minervahumanoids/supermimic/pull/1)

The viewer patch uses zero context lines. Apply the patch with `git apply --unidiff-zero` against the saved viewer revision.
