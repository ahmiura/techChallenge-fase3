import os
import time
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient


def collect_runs(experiment_name, max_results=20):
    client = MlflowClient(tracking_uri=os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return pd.DataFrame()
    exp_id = exp.experiment_id
    df = mlflow.search_runs([exp_id], order_by=["start_time DESC"], max_results=max_results)
    if df is None or df.empty:
        return pd.DataFrame()
    df['experiment_name'] = experiment_name
    return df


def normalize_row(df_row):
    # extract common fields into a flat dict
    d = {
        'run_id': df_row['run_id'],
        'run_name': df_row.get('tags.mlflow.runName') or df_row.get('tags.runName') or df_row.get('tags.mlflow.runName', ''),
        'experiment_name': df_row.get('experiment_name'),
        'model': df_row.get('params.model') if 'params.model' in df_row else None,
        'class_weight': df_row.get('params.class_weight') if 'params.class_weight' in df_row else None,
        'sampling': df_row.get('params.sampling') if 'params.sampling' in df_row else None,
        'n_estimators': df_row.get('params.n_estimators') if 'params.n_estimators' in df_row else None,
        'accuracy': df_row.get('metrics.accuracy'),
        'precision': df_row.get('metrics.precision'),
        'recall': df_row.get('metrics.recall'),
        'f1': df_row.get('metrics.f1'),
        'roc_auc': df_row.get('metrics.roc_auc'),
        'search_time_sec': df_row.get('metrics.search_time_sec')
    }
    return d


def main():
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    mlflow.set_tracking_uri(tracking_uri)

    exps = ['flight_delay_training_random_forest_v4', 'flight_delay_training_random_forest_tuned_v4', 'flight_delay_tuning_xgboost_v4']
    frames = []
    for e in exps:
        df = collect_runs(e, max_results=10)
        if not df.empty:
            frames.append(df)

    if not frames:
        print('No runs found for experiments:', exps)
        return

    all_df = pd.concat(frames, ignore_index=True)

    # normalize rows
    rows = [normalize_row(all_df.loc[i]) for i in range(len(all_df))]
    comp_df = pd.DataFrame(rows)

    # sort by f1 desc then recall
    comp_df = comp_df.sort_values(by=['f1', 'recall'], ascending=[False, False])

    ts = int(time.time())
    out_path = f'/tmp/model_comparison_{ts}.csv'
    comp_df.to_csv(out_path, index=False)
    print('Saved comparison CSV to', out_path)

    # log artifact in a short mlflow run
    mlflow.set_experiment('model_comparison_v4')
    with mlflow.start_run(run_name='compare_rf_xgb'):
        try:
            mlflow.log_artifact(out_path)
            print('Logged artifact to MLflow')
        except Exception as e:
            print('Could not log artifact to MLflow:', e)

    print('Done')


if __name__ == '__main__':
    main()
