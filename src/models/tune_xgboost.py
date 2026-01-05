import os
import time
import json
import mlflow
import mlflow.sklearn
from src.utils.storage import get_engine
from src.utils.mlflow_client import set_experiment
from sqlalchemy import text
import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


def load_sample(engine, table: str, limit: int = 200_000):
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT * FROM {table} LIMIT {limit}"))
        rows = result.fetchall()
        cols = result.keys()
    df = pd.DataFrame(rows, columns=cols)
    return df


def prepare(df):
    df['IS_DELAYED'] = pd.to_numeric(df['IS_DELAYED'], errors='coerce').fillna(0).astype(int)
    X = df.drop(columns=['IS_DELAYED']).copy()
    # coercing and encoding: numeric where possible, otherwise factorize categorical
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            coerced = pd.to_numeric(X[col], errors='coerce')
            # if most values can be coerced, use numeric
            if coerced.notna().sum() / len(coerced) > 0.8:
                X[col] = coerced.fillna(0).astype(float)
            else:
                X[col] = pd.Categorical(X[col]).codes
        else:
            # ensure numeric
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            except Exception:
                X[col] = pd.Categorical(X[col]).codes
    y = df['IS_DELAYED']
    return X, y


def run(limit=200_000, n_iter=20, random_state=42):
    if XGBClassifier is None:
        raise RuntimeError('XGBoost not installed')

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    set_experiment('flight_delay_tuning_xgboost_v4')

    engine = get_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://airflow:airflow@postgres/airflow'))
    df = load_sample(engine, 'gold_features', limit)
    X, y = prepare(df)

    # simple train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    # search space
    param_dist = {
        'n_estimators': [100, 300, 600],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 1, 5],
        'reg_alpha': [0, 0.1, 1.0],
        'scale_pos_weight': [1, max(1, int((y_train==0).sum()/(y_train==1).sum()))]
    }

    clf = XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_jobs=1, random_state=random_state)

    scorer = make_scorer(f1_score, pos_label=1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter, scoring=scorer, cv=cv, verbose=1, random_state=random_state)

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    best = search.best_estimator_
    preds = best.predict(X_test)

    metrics = {
        'f1': float(f1_score(y_test, preds, zero_division=0)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
    }
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_test, best.predict_proba(X_test)[:, 1]))
    except Exception:
        metrics['roc_auc'] = None

    # log to mlflow
    with mlflow.start_run(run_name='xgboost_random_search'):
        mlflow.log_params({'limit': limit, 'n_iter': n_iter})
        mlflow.log_params(search.best_params_)
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, v)
        mlflow.log_metric('search_time_sec', elapsed)
        # save best params (safe)
        try:
            mlflow.log_dict(search.best_params_, 'best_params.json')
        except Exception as e:
            print('warning: could not log params artifact:', e)
        try:
            mlflow.sklearn.log_model(best, artifact_path='model', registered_model_name='flight_delay_xgboost')
        except Exception as e:
            print('model registration failed or artifact logging not allowed:', e)

    print('Done. Best metrics:', metrics)


if __name__ == '__main__':
    run()
