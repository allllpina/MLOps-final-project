import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Ті самі енви, що і в лабі
os.environ["AWS_ACCESS_KEY_ID"] = "weedadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "weedadmin123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:8333"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud-detection")

# 1. Тренуємо V1 на reference.csv
df = pd.read_csv("data/reference.csv")
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# 2. Логуємо та реєструємо як fraud-detector
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "model", registered_model_name="fraud-detector")
    run_id = run.info.run_id

# 3. Вішаємо аліас production (робимо її V1)
client = mlflow.tracking.MlflowClient()
model_version = client.get_latest_versions("fraud-detector")[0].version
client.set_registered_model_alias("fraud-detector", "production", model_version)

print(
    f"Pre-staged state ready: fraud-detector Version {model_version} is now 'production'"
)
