"""Phase 7.9b: register the exported champion models in the MLflow Model Registry."""
import json

import joblib
import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient

from src.training.train_forecaster import MODEL_FILES
from src.utils.helpers import get_project_root

REGISTERED_NAMES = {
    "median": "fincast_forecaster_median",
    "q10": "fincast_forecaster_q10",
    "q90": "fincast_forecaster_q90",
}


def run() -> None:
    models_dir = get_project_root() / "models"
    meta = json.loads((models_dir / "model_meta.json").read_text(encoding="utf-8"))
    if meta["champion"] != "lgbm":
        print("The baseline is champion; nothing to register.")
        return

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("forecaster")
    client = MlflowClient()

    with mlflow.start_run(run_name="register_champion"):
        mlflow.log_params({"model_version": meta["model_version"], "n_rows": meta["n_rows"]})
        mlflow.log_metrics(meta["backtest_metrics"])
        mlflow.set_tag("champion", "lgbm")
        mlflow.log_artifact(str(models_dir / "model_meta.json"))

        for key, file_name in MODEL_FILES.items():
            name = REGISTERED_NAMES[key]
            model = joblib.load(models_dir / file_name)
            mlflow.lightgbm.log_model(model, name=key, registered_model_name=name)
            versions = client.search_model_versions(f"name = '{name}'")
            latest = max(int(v.version) for v in versions)
            client.set_registered_model_alias(name, "champion", str(latest))
            print(f"Registered {name} version {latest} with alias 'champion'")


if __name__ == "__main__":
    run()