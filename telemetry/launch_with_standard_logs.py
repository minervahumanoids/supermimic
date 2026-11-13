"""Run an external official SM launcher with the single shared logger."""

import json
from pathlib import Path
import runpy
import sys


def main():
    directory = Path(__file__).resolve().parent
    routing = json.loads((directory / 'project-routing.json').read_text())
    if (routing['entity'], routing['supermimic_project']) != ('minervahumanoids', 'Supermimic'):
        raise RuntimeError('Unexpected SuperMimic telemetry destination')
    if len(sys.argv) < 2:
        raise SystemExit('Usage: launch_with_standard_logs.py LAUNCHER [LAUNCHER_ARGUMENTS]')
    launcher = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(directory))
    import official_sm_logger
    sys.modules['standard_logger'] = official_sm_logger
    sys.path.insert(0, str(launcher.parent))
    sys.argv = [str(launcher), *sys.argv[2:]]
    runpy.run_path(str(launcher), run_name='__main__')


if __name__ == '__main__':
    main()
