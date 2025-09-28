# MLflow
import mlflow

def setup_mlflow(experiment_name: str = "random-experiement", tracking_uri: str = "file:./mlruns"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
