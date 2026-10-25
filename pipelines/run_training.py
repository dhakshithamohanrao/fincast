"""Stage 2 runner: aggregate, build features, train every model, export the champion."""
import importlib.util
import subprocess
import sys
import time

from src.utils.helpers import get_project_root

STAGES = [
    ("Monthly aggregates", "src.preprocessing.aggregate_monthly"),
    ("Features", "src.features.build_features"),
    ("Baselines", "src.training.train_baselines"),
    ("Categoriser", "src.training.train_categorizer"),
    ("Forecaster backtest", "src.training.train_lgbm_v2"),
    ("Champion export", "src.training.train_forecaster"),
    ("MLflow registry", "src.training.register_champion"),
    ("Anomaly model", "src.training.train_anomaly"),
]


def run_module(module: str) -> int:
    """Run `python -m module` in the project root and return its exit code."""
    return subprocess.run([sys.executable, "-m", module], cwd=get_project_root()).returncode


def has_main_block(module: str) -> bool:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        return False
    with open(spec.origin, encoding="utf-8") as f:
        return "__main__" in f.read()


def main() -> None:
    started = time.time()
    for number, (title, module) in enumerate(STAGES, start=1):
        print(f"\n=== {number}/{len(STAGES)} {title} ({module}) ===", flush=True)
        if not has_main_block(module):
            raise SystemExit(f"{module} has no __main__ block, so it would do nothing. "
                             "Add one, or remove this stage.")
        if run_module(module) != 0:
            raise SystemExit(f"Stage failed: {title} ({module}). Fix it and run again.")
    print(f"\nTraining done in {(time.time() - started) / 60:.1f} minutes. "
          "Models are in models/.")


if __name__ == "__main__":
    main()