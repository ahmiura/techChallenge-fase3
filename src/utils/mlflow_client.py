import mlflow
import mlflow.sklearn
from typing import Optional


def set_tracking_uri(uri: str):
    try:
        mlflow.set_tracking_uri(uri)
    except Exception:
        pass


def set_experiment(name: str):
    try:
        mlflow.set_experiment(name)
    except Exception:
        pass


def start_run(**kwargs):
    try:
        return mlflow.start_run(**kwargs)
    except Exception:
        # Retorna um context manager vazio para não quebrar 'with start_run()' quando mlflow não estiver disponível
        try:
            from contextlib import nullcontext
            return nullcontext()
        except Exception:
            return None


def active_run():
    try:
        return mlflow.active_run()
    except Exception:
        return None


def log_param(key: str, value):
    try:
        mlflow.log_param(key, value)
    except Exception:
        pass


def log_artifact(path: str, artifact_path: Optional[str] = None):
    try:
        if artifact_path:
            mlflow.log_artifact(path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(path)
    except Exception:
        pass


def log_model(model, artifact_path: str):
    try:
        mlflow.sklearn.log_model(model, artifact_path)
    except Exception:
        pass
