import os
import mlflow
from mlflow.tracking import MlflowClient
import warnings


def register_best(experiment_name, registered_model_name, artifact_path='model', promote_threshold: float = None):
    client = MlflowClient(tracking_uri=os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f'Experiment {experiment_name} not found')
        return None
    exp_id = exp.experiment_id
    runs = client.search_runs([exp_id], order_by=['metrics.f1 DESC'], max_results=5)
    if not runs:
        print('No runs found for', experiment_name)
        return None
    best = runs[0]
    run_id = best.info.run_id
    # get f1 metric if present
    f1_val = None
    try:
        f1_val = float(best.data.metrics.get('f1')) if best.data.metrics.get('f1') is not None else None
    except Exception:
        f1_val = None

    model_uri = f'runs:/{run_id}/{artifact_path}'
    print(f'Registering model from run {run_id} as {registered_model_name} (uri={model_uri}) — f1={f1_val}')
    try:
        # create registered model if not exists
        try:
            client.create_registered_model(registered_model_name)
        except Exception:
            pass
        mv = client.create_model_version(name=registered_model_name, source=model_uri, run_id=run_id)
        print('Created model version:', mv.version)
        # Auto-promotion to Staging if meets threshold AND is better than current staging
        threshold = promote_threshold
        if threshold is None:
            try:
                threshold = float(os.getenv('REGISTER_F1_THRESHOLD', '0.0'))
            except Exception:
                threshold = 0.0

        def get_staging_best_f1(name):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    staging_versions = client.get_latest_versions(name, stages=['Staging'])
                best = None
                for sv in staging_versions:
                    try:
                        run_id_s = sv.run_id
                        run_info = client.get_run(run_id_s)
                        f1_s = run_info.data.metrics.get('f1')
                        if f1_s is not None:
                            f1_val_s = float(f1_s)
                            if best is None or f1_val_s > best:
                                best = f1_val_s
                    except Exception:
                        continue
                return best
            except Exception:
                return None

        staging_best = get_staging_best_f1(registered_model_name)
        promote = False
        if f1_val is not None and threshold > 0.0 and f1_val >= threshold:
            # if no staging model, promote; else promote only if strictly better
            if staging_best is None:
                promote = True
            else:
                try:
                    if f1_val > staging_best:
                        promote = True
                except Exception:
                    promote = False

        if promote:
            try:
                print(f'Promoting model {registered_model_name} version {mv.version} to Staging (f1={f1_val} >= {threshold} and better than staging={staging_best})')
                client.transition_model_version_stage(name=registered_model_name, version=mv.version, stage='Staging', archive_existing_versions=True)
                print('Promotion to Staging completed')
            except Exception as e:
                print('Failed to promote model version:', e)
        else:
            print(f'Not promoting: candidate f1={f1_val}, staging best f1={staging_best}, threshold={threshold}')

        return mv
    except Exception as e:
        # fallback to mlflow.register_model API
        try:
            res = mlflow.register_model(model_uri, registered_model_name)
            print('Registered via mlflow.register_model:', res.version)
            # attempt promotion if applicable
            threshold = promote_threshold if promote_threshold is not None else float(os.getenv('REGISTER_F1_THRESHOLD', '0.0'))
            if f1_val is not None and threshold > 0.0 and f1_val >= threshold:
                try:
                    client.transition_model_version_stage(name=registered_model_name, version=res.version, stage='Staging', archive_existing_versions=True)
                    print('Promotion to Staging completed (fallback)')
                except Exception as e2:
                    print('Failed to promote model version (fallback):', e2)
            return res
        except Exception as e2:
            print('Failed to register model:', e, e2)
            return None


def run():
    # Register best from RF tuning and XGBoost tuning experiments
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    # threshold can be set via env var REGISTER_F1_THRESHOLD (e.g., 0.35)
    try:
        threshold = float(os.getenv('REGISTER_F1_THRESHOLD', '0.0'))
    except Exception:
        threshold = 0.0

    pairs = [
        ('flight_delay_training_random_forest_v4', 'flight_delay_rf_simple', 'random_forest_model'),
        ('flight_delay_training_random_forest_tuned_v4', 'flight_delay_rf_best', 'random_forest_tuned'),
        ('flight_delay_tuning_xgboost_v4', 'flight_delay_xgb_best', 'model')
    ]
    for exp_name, reg_name, art_path in pairs:
        print('Processing', exp_name)
        register_best(exp_name, reg_name, artifact_path=art_path, promote_threshold=threshold)


if __name__ == '__main__':
    run()
