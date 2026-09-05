"""Register the trained run as `fraud-detector` and promote it.

train.py logs a run to the `fraud-detection` experiment. This script
turns that run into the model the serving layer targets: it registers
`runs:/<run_id>/model` as a version of `fraud-detector`, then puts the
`production` alias on that version. serve.py resolves
`models:/fraud-detector@production`, so this alias is the handoff from
training to serving.

The run lookup is written for you. Author the TODO, then run:
    python3 /root/code/register.py
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import yaml

HERE = Path(__file__).resolve().parent
cfg = yaml.safe_load((HERE / "config.yaml").read_text())

os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

os.environ["AWS_ACCESS_KEY_ID"] = cfg["s3"]["access_key"]
os.environ["AWS_SECRET_ACCESS_KEY"] = cfg["s3"]["secret_key"]
os.environ["MLFLOW_S3_ENDPOINT_URL"] = cfg["s3"]["endpoint"]

TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "fraud-detector"
ALIAS = "production"

mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.MlflowClient()


def _latest_run_id() -> str:
    """The most recent run id in the `fraud-detection` experiment."""
    exp = client.get_experiment_by_name("fraud-detection")
    if exp is None:
        raise SystemExit("no `fraud-detection` experiment yet -- run train.py first")
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("no runs in `fraud-detection` -- run train.py first")
    return runs[0].info.run_id


run_id = _latest_run_id()
print(f"[register] latest run_id={run_id}")

model_uri = f"runs:/{run_id}/model"

model_version = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
print(f"[register] Registered version {model_version.version} of {MODEL_NAME}")

client.set_registered_model_alias(
    name=MODEL_NAME, alias=ALIAS, version=model_version.version
)
print(f"[register] Promoted version {model_version.version} to alias '{ALIAS}'")
