from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
from airflow.models import Variable
from sklearn.ensemble import RandomForestClassifier

# utils
from src.utils.storage import get_engine, read_table, save_df
from src.utils.mlflow_client import set_tracking_uri, set_experiment, start_run, log_artifact, log_model, log_param
import mlflow

# Adiciona o diretório src ao path para importar os módulos
src_path = '/opt/airflow/dags/src'
if src_path not in sys.path:
    sys.path.append(src_path)
try:
    from src.ingest.loader import FlightDataLoader
    from src.features.engineer import FeatureEngineer
    from src.models.supervised import SupervisedModeler
except Exception:
    # Fallback: tenta caminho alternativo caso a montagem seja diferente
    if '/opt/airflow/dags' not in sys.path:
        sys.path.append('/opt/airflow/dags')
    from src.ingest.loader import FlightDataLoader
    from src.features.engineer import FeatureEngineer
    from src.models.supervised import SupervisedModeler

# Configurações
# Prefer Airflow Variables (mais seguro em produção); fallback para env
CSV_PATH = "/opt/airflow/data/flights.csv"
DB_URL = Variable.get('TARGET_DATABASE_CONN', default_var=os.getenv("TARGET_DATABASE_CONN", "postgresql+psycopg2://airflow:airflow@postgres/airflow"))
MLFLOW_TRACKING_URI = Variable.get('MLFLOW_TRACKING_URI', default_var=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def extract_and_load_raw():
    """Lê do CSV e salva na tabela raw_flights"""
    print("🚀 Iniciando Extração...")
    loader = FlightDataLoader(file_path=CSV_PATH)
    # Carrega uma amostra para teste ou tudo (None)
    loader.load_data(sample_size=50000) 
    loader.save_to_db(table_name="raw_flights", db_url=DB_URL)

def preprocess_data():
    """Lê da raw, aplica limpeza extra se necessário e salva em silver_flights"""
    print("🧹 Iniciando Pré-processamento...")
    engine = get_engine(DB_URL)
    df = read_table(engine, 'raw_flights')

    # Aqui poderiam entrar mais limpezas específicas
    # Como o DataLoader já limpa um pouco, vamos apenas repassar para fins de arquitetura

    save_df(engine, df, "silver_flights", if_exists='replace')
    print("✅ Dados pré-processados salvos em 'silver_flights'")

def feature_engineering():
    """Lê da silver, cria features e registra no MLflow"""
    print("⚙️ Iniciando Engenharia de Features...")
    
    # Configura MLflow via util
    set_tracking_uri(MLFLOW_TRACKING_URI)
    set_experiment("flight_delay_features")

    engine = get_engine(DB_URL)
    df = read_table(engine, 'silver_flights')

    with start_run():
        engineer = FeatureEngineer(df)
        
        # 1. Cria Target
        engineer.create_target_classification(threshold=15)
        log_param("target_threshold", 15)
        
        # 2. Prepara Features (Isso loga as features usadas internamente na classe agora)
        X, y = engineer.prepare_features_supervised()
        
        # Salva tabela Gold (pronta para treino)
        df_gold = pd.concat([X, y], axis=1)
        save_df(engine, df_gold, "gold_features", if_exists='replace')
        print("✅ Features salvas em 'gold_features'")

        # Log sample of gold table as artifact for traceability
        try:
            sample_path = '/tmp/gold_sample.csv'
            df_gold.head(100).to_csv(sample_path, index=False)
            log_artifact(sample_path, artifact_path='gold_samples')
        except Exception:
            pass

def train_model():
    """Lê a tabela gold, treina o modelo e loga no MLflow"""
    print("🤖 Iniciando Treinamento do Modelo...")
    set_tracking_uri(MLFLOW_TRACKING_URI)
    set_experiment("flight_delay_training")

    engine = get_engine(DB_URL)
    df = read_table(engine, 'gold_features')

    X = df.drop(columns=['IS_DELAYED'])
    y = df['IS_DELAYED']

    with start_run():
        trainer = SupervisedModeler(X, y)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        # Treina e loga artefato (modelo)
        trainer.train_evaluate("RandomForest", model)

        # Log de métricas simples e artefatos
        try:
            if hasattr(model, 'feature_importances_'):
                import pandas as _pd
                fi = _pd.DataFrame({
                    'feature': X.columns.tolist(),
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                fi_path = '/tmp/feature_importances.csv'
                fi.to_csv(fi_path, index=False)
                log_artifact(fi_path, artifact_path='feature_importance')
        except Exception:
            pass

        log_model(model, "random_forest_model")

with DAG(
    'flight_mlops_pipeline',
    default_args=default_args,
    description='Pipeline de Voos: CSV -> Postgres -> MLflow',
    schedule_interval=timedelta(days=1),
    catchup=False
) as dag:

    t1 = PythonOperator(
        task_id='extract_data',
        python_callable=extract_and_load_raw
    )

    t2 = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )

    t3 = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering
    )

    t4 = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    t1 >> t2 >> t3 >> t4