# Shared training telemetry

Use `standard.json` and `sm-profile.json` for metric names, definitions, and view layout.
Use one logger for every official SuperMimic launch.

```bash
python telemetry/launch_with_standard_logs.py /path/to/external/launch.py [launch arguments]
```

The wrapper supplies `official_sm_logger.py` to existing external launchers.
The wrapper preserves the native learner, architecture, rewards, optimizer, and stop condition.
The wrapper does not initialize W&B during smoke executions.

`official_sm_contract.py` defines canonical names and exact timing conversions.
Timing plots show seconds per actual collector iteration.
The converter uses the recorded iteration difference, including the first one-step interval.
Missing measurements remain absent.

`official_sm_workspace.py` updates the existing personal view and the three original saved views.
Each section has one visible row and three columns.
All custom plots use training iterations.
Native SDK system plots retain their original keys and wall clock.
The view omits unavailable metrics and empty curriculum or media sections.

`repair_official_sm.py` appends compatibility measurements to stopped runs.
The repair preserves original measurements, source timestamps, run IDs, launch commits, checkpoint provenance, and stopped states.
The repair does not restart a trainer.

Historical episode reductions remain documented in each run's `telemetry_contract` configuration.
The original G1 and Roger runs use the last 100 completed episodes.
The single-clip G1 run pools completed episodes within each logging interval.
The archived logs lack raw episode events, so the repair preserves those historical reductions.
Historical root diagnostics retain their recorded sampling and reduction definitions.
Use matched MuJoCo recordings for quantitative policy comparisons.
Every future official launch uses the same interval reduction through the shared logger.

The November 12 correction covers `ga216ipt`, `zx0w9xq2`, and `45u76a15`.
Compact receipts appear in `launches/g1-walk-single-mnv1-20261112/evidence`.
Full local backups remain in the review workspace.

Run `python telemetry/verify_official_sm.py` to verify the shared reductions without W&B.
