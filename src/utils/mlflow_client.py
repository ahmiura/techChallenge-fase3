import mlflow
import mlflow.sklearn
from typing import Optional
import warnings
import logging
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


def set_tracking_uri(uri: str):
    try:
        mlflow.set_tracking_uri(uri)
    except Exception as e:
        print(f"Error setting tracking URI: {e}")


def set_experiment(name: str):
    try:
        # Lógica centralizada: Verifica se existe e restaura se estiver deletado
        client = MlflowClient()
        experiments = client.search_experiments(view_type=ViewType.ALL, filter_string=f"name = '{name}'")
        if experiments:
            experiment = experiments[0]
            if experiment.lifecycle_stage == 'deleted':
                print(f"♻️ Restaurando experimento deletado: {name}")
                client.restore_experiment(experiment.experiment_id)

        mlflow.set_experiment(name)
    except Exception as e:
        print(f"Error setting experiment {name}: {e}")


def start_run(**kwargs):
    try:
        return mlflow.start_run(**kwargs)
    except Exception as e:
        print(f"Error starting run: {e}")
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
    except Exception as e:
        print(f"Error logging param {key}: {e}")


def log_metric(key: str, value: float):
    try:
        mlflow.log_metric(key, value)
    except Exception as e:
        print(f"Error logging metric {key}: {e}")


def log_artifact(path: str, artifact_path: Optional[str] = None):
    try:
        if artifact_path:
            mlflow.log_artifact(path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(path)
    except Exception as e:
        print(f"Error logging artifact {path}: {e}")


def log_model(model, artifact_path: str):
    try:
        # Filtra warning interno do MLflow 2.10 sobre artifact_path
        # Silencia logger específico do MLflow que emite o warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*artifact_path is deprecated.*")
            mlflow.sklearn.log_model(sk_model=model, artifact_path=artifact_path)
    except Exception as e:
        print(f"Error logging model: {e}")
