"""IDE entry point for running one Rule or Model experiment.

Edit ``EXPERIMENT_PATH`` below, then run this file from PyCharm, VS Code or
another Python IDE.  MySQL connection settings stay in
``qmt_example/configs/system.yaml``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from etf_backtest.experiment import run_experiment

PROJECT_ROOT = Path(__file__).resolve().parent

# Change only this path when selecting another experiment.
EXPERIMENT_PATH = PROJECT_ROOT / "private_strategy" / "beginner_example" / "experiment.yaml"
SYSTEM_PATH = PROJECT_ROOT / "qmt_example" / "configs" / "system.yaml"


def main() -> dict[str, object]:
    """Run ``EXPERIMENT_PATH`` and print the result location."""

    # Project resources configured with relative paths are resolved from here.
    os.chdir(PROJECT_ROOT)
    result = run_experiment(
        EXPERIMENT_PATH,
        system_path=SYSTEM_PATH,
        project_root=PROJECT_ROOT,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
