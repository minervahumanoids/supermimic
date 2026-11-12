# Standing drift audit

The preserved checkpoint has poor standing control.
The approved standalone recording ends 40.49 cm from its G1 reference after 20 seconds.
The reference spans only 0.47 cm in X and 1.50 cm in Y.
The policy reaches a maximum XY error of 47.07 cm.

The native MuJoCo-Warp diagnostic also drifts, with a maximum XY error of 33.24 cm.
The native diagnostic does not replace the approved standalone recordings or the Viser comparison.
The simulators produce different closed-loop trajectories, so their endpoint errors do not establish trajectory parity.

## Input and action verification

The audit compares both input paths at 100 identical generalized states.
Both paths use the frozen actor from artifact `model-ga216ipt:v403`.
The audit creates no learner, optimizer, or W&B run.

| Quantity | Maximum absolute difference |
| --- | ---: |
| Full state | 5.96e-8 |
| Actor goal | 2.00e-7 |
| Action | 3.46e-6 |
| Position target | 1.16e-6 rad |

The action target names match the G1 joint order.
The physical model fields agree within the precision of XML serialization.
The standing reference matches the converted source and packed training samples exactly.

The native observation manager caches observations.
The diagnostic explicitly refreshes that cache after each manual reset.
The initial probe omitted the refresh and produced an invalid first-frame mismatch.
The corrected probe passes for all 100 frames.

## Position observability

The official actor receives root height, gravity, body-frame velocities, joint states, and the motion goal.
Its represented state omits global XY position.
Its motion goal omits root translation error.

A controlled probe translates the root by one meter while preserving the reference and other state values.
The represented state, actor goal, and resulting action remain exactly identical.
The actor therefore cannot directly observe accumulated XY position error.

This limitation does not identify the sole cause of the initial stepping.
The audit does not establish that dataset filtering or the smaller policy batch caused the standing behavior.
It also does not establish that the official method performs worse than the Roger policy.
The robot, data selection, controller, and training recipe differ between those experiments.

## Interpretation and preserved state

The earlier 33 completed clips indicate that the native height and gravity gates did not terminate those trials.
The completion count does not establish standing stability, smoothness, or foot-contact quality.

The canonical evaluator, viewer, metrics, cohort, and standalone recordings remain unchanged.
The original SMPL input remains synchronized in Viser.
The checkpoint remains unchanged, and the training remains stopped.
The diagnostic reservation is released after the GPU becomes idle.

The evidence includes `review-findings.json`, `input-parity.json`, `model-parity.json`, `data-parity.json`, and `summary.json`.
The local recording directory also preserves diagnostic trajectories and the original invalid cache probe.
