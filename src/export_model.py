"""Export the trained CNN from MLflow to models/ and register it in the MLflow Model Registry.

Run from the project root:  python -m src.export_model
"""
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

from src.model import CNN, SIGNAL_LENGTH

ROOT = Path(__file__).resolve().parent.parent
RUN_NAME = "cnn-sqrt"
REGISTERED_NAME = "ecg-cnn"
WEIGHTS_PATH = ROOT / "models" / "cnn_sqrt.pt"

mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")

runs = mlflow.search_runs(
    experiment_names=["ecg-arrhythmia"],
    filter_string=f"tags.mlflow.runName = '{RUN_NAME}' and attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
)
run_id = runs.iloc[0]["run_id"]
print("source run:", run_id)

# Save only the weights (state_dict), not the pickled class, so the app depends on src/model.py.
logged_model = mlflow.pytorch.load_model(f"runs:/{run_id}/model").eval()
WEIGHTS_PATH.parent.mkdir(exist_ok=True)
torch.save(logged_model.state_dict(), WEIGHTS_PATH)
print("saved weights:", WEIGHTS_PATH, f"({WEIGHTS_PATH.stat().st_size / 1024:.0f} KB)")

# Check that the saved weights reproduce the logged model's outputs.
reloaded = CNN().eval()
reloaded.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
x = torch.rand(8, SIGNAL_LENGTH)  # (8, 187)
with torch.no_grad():
    assert torch.allclose(logged_model(x), reloaded(x), atol=1e-6), "reloaded model differs from logged model"
print("check passed: reloaded weights match the logged model")

# Register in the MLflow Model Registry and mark this version as the current champion.
version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_NAME)
mlflow.MlflowClient().set_registered_model_alias(REGISTERED_NAME, "champion", version.version)
print(f"registered '{REGISTERED_NAME}' version {version.version} with alias 'champion'")
