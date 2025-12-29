import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Importações dos módulos locais (pasta src)
from src.ingest.loader import FlightDataLoader
from src.features.engineer import FeatureEngineer
from src.models.supervised import SupervisedModeler
from src.models.unsupervised import UnsupervisedModeler

# Configurações Globais de Estilo
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)



# ==========================================
# EXECUÇÃO DO PIPELINE (Exemplo de uso)
# ==========================================

if __name__ == "__main__":
    # 1. Carregamento
    # Define o caminho do arquivo relativo à localização deste script (main.py)
    # Isso evita erros se você rodar o script de outro diretório
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "flights.csv")
    
    loader = FlightDataLoader(file_path=csv_path)
    
    # Carrega os dados. Defina sample_size=None para carregar tudo.
    # Isso permitirá verificar o volume total processado.
    df_raw = loader.load_data(sample_size=None) 

    if not df_raw.empty:
        # 2. Engenharia de Features
        engineer = FeatureEngineer(df_raw)
        engineer.create_target_classification(threshold=15)
        
        # 3. Modelagem Supervisionada
        print("\n--- INICIANDO FASE SUPERVISIONADA ---")
        X, y = engineer.prepare_features_supervised()
        
        trainer = SupervisedModeler(X, y)
        
        # Modelo 1: Baseline (Regressão Logística)
        trainer.train_evaluate("Logistic Regression", LogisticRegression(max_iter=1000))
        
        # Modelo 2: Avançado (Random Forest)
        rf_model = trainer.train_evaluate("Random Forest", RandomForestClassifier(n_estimators=50, random_state=42))
        trainer.plot_feature_importance("Random Forest", X.columns)

        # 4. Modelagem Não Supervisionada
        print("\n--- INICIANDO FASE NÃO SUPERVISIONADA ---")
        # Prepara dados agregados por aeroporto
        df_airports = engineer.prepare_data_unsupervised()
        
        cluster_modeler = UnsupervisedModeler(df_airports)
        
        # Análise do cotovelo para decidir K
        cluster_modeler.find_optimal_k(max_k=8)
        
        # Supondo que K=3 seja bom após olhar o gráfico
        df_clustered = cluster_modeler.train_clustering(k=3)
        
        # Interpretação dos Clusters
        print("\n📊 Perfil dos Clusters (Médias):")
        print(df_clustered.groupby('CLUSTER').mean())