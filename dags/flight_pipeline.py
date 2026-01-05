from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
from airflow.models import Variable
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine
import matplotlib
matplotlib.use('Agg') # Define backend não-interativo para evitar erros no Docker
import matplotlib.pyplot as plt
import seaborn as sns

# utils
from src.utils.storage import get_engine, read_table, save_df
from src.utils.mlflow_client import set_tracking_uri, set_experiment, start_run, log_artifact, log_model, log_param, log_metric
import mlflow
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
import numpy as np

# Adiciona o diretório src ao path para importar os módulos
src_path = '/opt/airflow/dags/src'
if src_path not in sys.path:
    sys.path.append(src_path)
try:
    from src.ingest.loader import FlightDataLoader
    from src.features.engineer import FeatureEngineer
    from src.models.supervised import SupervisedModeler
    from src.models.unsupervised import UnsupervisedModeler
except Exception:
    # Fallback: tenta caminho alternativo caso a montagem seja diferente
    if '/opt/airflow/dags' not in sys.path:
        sys.path.append('/opt/airflow/dags')
    from src.ingest.loader import FlightDataLoader
    from src.features.engineer import FeatureEngineer
    from src.models.supervised import SupervisedModeler
    from src.models.unsupervised import UnsupervisedModeler

# Configurações
# Prefer Airflow Variables (mais seguro em produção); fallback para env
CSV_PATH = "/opt/airflow/data/flights.csv"
DB_URL = Variable.get('TARGET_DATABASE_CONN', default_var=os.getenv("TARGET_DATABASE_CONN", "postgresql+psycopg2://airflow:airflow@postgres/airflow"))
MLFLOW_TRACKING_URI = Variable.get('MLFLOW_TRACKING_URI', default_var=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

# Controle de amostragem: se `SAMPLE_SIZE` não estiver definido, usa None (dataset completo)
SAMPLE_SIZE_ENV = Variable.get('SAMPLE_SIZE', default_var=os.getenv('SAMPLE_SIZE', None))
SAMPLE_SIZE = int(SAMPLE_SIZE_ENV) if SAMPLE_SIZE_ENV not in (None, 'None', '') else None

# Proteções para treino (evitar OOM em ambientes limitados)
MAX_TRAIN_ROWS = int(Variable.get('MAX_TRAIN_ROWS', default_var=os.getenv('MAX_TRAIN_ROWS', '200000')))

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

def extract_and_load_raw():
    """Lê do CSV e salva na tabela raw_flights"""
    print(f"🚀 Iniciando Extração com SAMPLE_SIZE={SAMPLE_SIZE}...")
    loader = FlightDataLoader(file_path=CSV_PATH)
    # Carrega com SAMPLE_SIZE (None = arquivo inteiro)
    loader.load_data(sample_size=SAMPLE_SIZE)
    loader.save_to_db(table_name="raw_flights", db_url=DB_URL)

def preprocess_data():
    """Lê da raw, aplica limpeza extra se necessário e salva em silver_flights"""
    print("🧹 Iniciando Pré-processamento...")
    engine = create_engine(DB_URL)
    
    # Com Pandas 2.1.4 e SQLAlchemy 1.4, a compatibilidade é nativa.
    # Usamos chunksize para evitar OOM, e o Pandas gerencia a iteração.
    chunk_size = 50000
    
    # stream_results=True na engine ajuda a não carregar tudo na memória no lado do driver
    with engine.connect().execution_options(stream_results=True) as conn:
        for i, chunk in enumerate(pd.read_sql("SELECT * FROM raw_flights", conn, chunksize=chunk_size)):
            mode = 'replace' if i == 0 else 'append'
            chunk.to_sql("silver_flights", engine, if_exists=mode, index=False)
            print(f"📦 Chunk {i+1} processado e salvo em 'silver_flights'")
        
    print("✅ Dados pré-processados salvos em 'silver_flights'")

def run_automated_eda():
    """Gera estatísticas e visualizações obrigatórias (EDA) e loga no MLflow."""
    print("📊 Iniciando EDA Automatizada...")
    set_tracking_uri(MLFLOW_TRACKING_URI)
    set_experiment("flight_data_eda_v4")
    
    engine = get_engine(DB_URL)
    # Lê uma amostra para não estourar memória na geração de gráficos
    df = pd.read_sql(f"SELECT * FROM silver_flights LIMIT {MAX_TRAIN_ROWS}", engine)
    
    # Garante que colunas numéricas sejam tratadas como tal para evitar erro no heatmap
    cols_to_numeric = ['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'DISTANCE', 'AIR_TIME', 'TAXI_IN', 'TAXI_OUT', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'SCHEDULED_ARRIVAL']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Cria target temporário para EDA se não existir na silver (pois é criado na feature engineering)
    if 'IS_DELAYED' not in df.columns and 'ARRIVAL_DELAY' in df.columns:
        df['IS_DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

    with start_run(run_name="Exploratory_Data_Analysis"):
        # Loga um parâmetro para confirmar que a run processou dados e aparecer na UI
        log_param("rows_analyzed", len(df))

        # 1. Estatísticas Descritivas
        desc = df.describe()
        desc_path = "/tmp/descriptive_stats.csv"
        desc.to_csv(desc_path)
        log_artifact(desc_path, artifact_path="eda")
        
        # 2. Distribuição do Target (se existir)
        if 'IS_DELAYED' in df.columns:
            plt.figure(figsize=(6, 4))
            sns.countplot(x='IS_DELAYED', data=df)
            plt.title("Distribuição de Atrasos (Target)")
            target_path = "/tmp/target_dist.png"
            plt.savefig(target_path)
            plt.close()
            log_artifact(target_path, artifact_path="eda")
            
        # 3. Matriz de Correlação (apenas numéricos)
        plt.figure(figsize=(10, 8))
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
            plt.title("Matriz de Correlação")
            corr_path = "/tmp/correlation_matrix.png"
            plt.savefig(corr_path)
            plt.close()
            log_artifact(corr_path, artifact_path="eda")
        else:
            print("⚠️ Nenhuma coluna numérica encontrada para gerar matriz de correlação.")

def feature_engineering():
    """Lê da silver, cria features e registra no MLflow"""
    print("⚙️ Iniciando Engenharia de Features...")
    
    # Configura MLflow via util
    set_tracking_uri(MLFLOW_TRACKING_URI)
    set_experiment("flight_delay_features_v4")

    engine = get_engine(DB_URL)
    df = read_table(engine, 'silver_flights')

    with start_run(run_name="Feature_Engineering"):
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
    set_experiment("flight_delay_training_random_forest_v4")

    engine = get_engine(DB_URL)
    df = read_table(engine, 'gold_features')

    X = df.drop(columns=['IS_DELAYED'])
    y = df['IS_DELAYED'].astype(int)

    # Se o dataset para treino for muito grande, amostramos para evitar exaustão
    if len(df) > MAX_TRAIN_ROWS:
        print(f"⚠️ Dataset de treino grande ({len(df)} linhas). Amostrando {MAX_TRAIN_ROWS} linhas para treino.")
        df = df.sample(n=MAX_TRAIN_ROWS, random_state=42)
        X = df.drop(columns=['IS_DELAYED'])
        y = df['IS_DELAYED'].astype(int)
    # Treino simples (compatibilidade): RandomForest básico
    with start_run(run_name="RandomForest_Simple"):
        trainer = SupervisedModeler(X, y)
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        trainer.train_evaluate("RandomForest_simple", model)
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


def train_random_forest_tuned():
    """Tuning leve do RandomForest com RandomizedSearchCV e log no MLflow."""
    print("🔎 Tunando RandomForest...")
    set_tracking_uri(MLFLOW_TRACKING_URI)
    set_experiment("flight_delay_training_random_forest_tuned_v4")

    engine = get_engine(DB_URL)
    df = read_table(engine, 'gold_features')

    if len(df) > MAX_TRAIN_ROWS:
        df = df.sample(n=MAX_TRAIN_ROWS, random_state=42)

    X = df.drop(columns=['IS_DELAYED'])
    y = df['IS_DELAYED'].astype(int)

    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample'],
        'criterion': ['gini', 'entropy']
    }

    clf = RandomForestClassifier(random_state=42, n_jobs=1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, scoring='f1', cv=cv, random_state=42, verbose=1, n_jobs=-1)

    with start_run(run_name="RandomForest_Tuned"):
        search.fit(X, y)
        best = search.best_estimator_
        # Evaluate on a hold-out split using SupervisedModeler for convenience
        from src.models.supervised import SupervisedModeler
        trainer = SupervisedModeler(X, y)
        trainer.train_evaluate('RandomForest_tuned', best)
        # Log params/metrics
        try:
            log_param('tuning_method', 'RandomizedSearchCV')
            log_param('n_iter', 10)
            for k, v in search.best_params_.items():
                log_param(f'best_{k}', v)
        except Exception:
            pass
        try:
            log_model(best, 'random_forest_tuned')
        except Exception:
            pass


def tune_xgboost_task():
    """Chama o script de tuning XGBoost (implementado em src/models/tune_xgboost.py)."""
    print("🔎 Tunando XGBoost via script...")
    
    # Import e execução direta do módulo
    try:
        from src.models.tune_xgboost import run as xgb_run
        xgb_run(limit=MAX_TRAIN_ROWS, n_iter=30, random_state=42)
    except Exception as e:
        print('Erro ao executar tuning XGBoost:', e)


def unsupervised_analysis():
    """Roda análise não-supervisionada: agrega por aeroporto, encontra K ótimo e clusteriza."""
    print("🔬 Iniciando análise não-supervisionada (clustering)...")

    set_tracking_uri(MLFLOW_TRACKING_URI)
    set_experiment("flight_delay_unsupervised_v4")

    engine = get_engine(DB_URL)
    df = read_table(engine, 'silver_flights')

    # Garante conversão para numérico para evitar erro de agregação (TypeError: agg function failed)
    # Isso corrige colunas que podem ter sido lidas como strings/object do banco
    cols_to_numeric = ['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'DISTANCE', 'AIR_TIME', 'TAXI_IN', 'TAXI_OUT']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Prepara agregação por aeroporto (FeatureEngineer oferece utilitário)
    engineer = FeatureEngineer(df)
    df_agg = engineer.prepare_data_unsupervised()

    if df_agg is None or df_agg.shape[0] == 0:
        print("⚠️ Nenhum dado agregado disponível para clusterização. Pulando tarefa.")
        return

    # Inicializa o modelador não-supervisionado
    modeler = UnsupervisedModeler(df_agg)
    if modeler.X_scaled is None or modeler.X_scaled.shape[0] < 2:
        print("⚠️ Dados insuficientes para clusterização.")
        return

    # Define caminhos para plots
    elbow_path = '/tmp/unsup_elbow.png'
    sil_path = '/tmp/unsup_silhouette.png'

    # Encontra K ótimo e gera plots
    try:
        modeler.find_optimal_k(max_k=8, plot_path=elbow_path)
        modeler.plot_silhouette(plot_path=sil_path)
    except Exception as e:
        print('Erro ao salvar plots de clusterização:', e)
        elbow_path = None
        sil_path = None

    # Obtém melhor K e treina
    best_k = modeler.get_best_k()
    df_out = modeler.train_clustering(k=best_k)

    # Persiste clusters em uma tabela dedicada
    try:
        save_df(engine, df_out.reset_index().rename(columns={'ORIGIN_AIRPORT': 'ORIGIN_AIRPORT'}), 'unsupervised_airport_clusters', if_exists='replace')
        print("✅ Clusters salvos em 'unsupervised_airport_clusters'")
    except Exception as e:
        print('Erro ao salvar clusters no banco:', e)

    # Tenta logar artefatos e parâmetros no MLflow (não deve quebrar se falhar)
    try:
        with start_run(run_name="Unsupervised_Analysis"):
            log_param('unsupervised_best_k', best_k)
            
            # Loga métricas e o modelo (agora disponíveis na classe)
            if modeler.score is not None:
                log_metric("silhouette_score", modeler.score)
            
            if modeler.model is not None:
                log_model(modeler.model, "kmeans_model")

            if elbow_path:
                log_artifact(elbow_path, artifact_path='unsupervised')
            if sil_path:
                log_artifact(sil_path, artifact_path='unsupervised')
            csv_path = '/tmp/unsupervised_airport_clusters.csv'
            df_out.to_csv(csv_path)
            log_artifact(csv_path, artifact_path='unsupervised')
    except Exception as e:
        print('Aviso: falha ao logar artefatos de clusterização no MLflow:', e)

def compare_models_task():
    """Executa o script de comparação de modelos para gerar relatório final."""
    print("⚖️ Comparando modelos treinados...")
    from src.models import compare_and_log
    compare_and_log.main()

def register_best_models_task():
    """Importa e executa o registro do melhor modelo."""
    print("🏆 Registrando o melhor modelo...")
    from src.models import register_best_models
    register_best_models.run()

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

    t_eda = PythonOperator(
        task_id='run_eda',
        python_callable=run_automated_eda
    )

    t3 = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering
    )

    t4 = PythonOperator(
        task_id='train_model_simple',
        python_callable=train_model
    )

    t_unsup = PythonOperator(
        task_id='unsupervised_analysis',
        python_callable=unsupervised_analysis
    )

    t5 = PythonOperator(
        task_id='tune_random_forest',
        python_callable=train_random_forest_tuned
    )

    t6 = PythonOperator(
        task_id='tune_xgboost',
        python_callable=tune_xgboost_task
    )

    t_compare = PythonOperator(
        task_id='compare_models',
        python_callable=compare_models_task
    )

    t7 = PythonOperator(
        task_id='register_best_models',
        python_callable=register_best_models_task
    )

    # Orquestra: features -> (treino simples + análise não-supervisionada) -> tuning paralelos
    t1 >> t2
    t2 >> [t3, t_eda] # EDA roda em paralelo com Feature Engineering
    t3 >> [t4, t_unsup]
    t4 >> [t5, t6]
    [t5, t6] >> t_compare >> t7 # Compara antes de registrar o melhor