import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class UnsupervisedModeler:
    """
    Gerencia a clusterização (K-Means) e análise de K ótimo.
    """
    def __init__(self, df_aggregated: pd.DataFrame):
        self.raw_df = df_aggregated
        
        # Validação para debug: verifica se a coluna esperada existe no input
        if 'TOTAL_FLIGHTS' not in self.raw_df.columns:
            # Verifica se a coluna está no índice (comum após agregações)
            if self.raw_df.index.name == 'TOTAL_FLIGHTS':
                self.raw_df = self.raw_df.reset_index()
            else:
                raise ValueError(f"❌ ERRO CRÍTICO: A coluna 'TOTAL_FLIGHTS' é obrigatória e não foi encontrada. Colunas presentes: {self.raw_df.columns.tolist()}")

        # Seleciona apenas colunas numéricas e preenche nulos
        self.X = self.raw_df.select_dtypes(include=[np.number]).fillna(0)
        
        # Log das features usadas para clareza
        self.features_used = self.X.columns.tolist()
        print(f"🔍 Features selecionadas para clusterização: {self.features_used}")

        self.scaler = StandardScaler()
        # Verifica se há dados suficientes
        if self.X.shape[0] > 0:
            self.X_scaled = self.scaler.fit_transform(self.X)
        else:
            self.X_scaled = None
        
        self.ks = []
        self.inertias = []
        self.silhouettes = []
        self.model = None
        self.score = None

    def find_optimal_k(self, max_k: int = 10, plot_path: str = None):
        """
        Avalia K de 2 até max_k calculando inércia e silhouette.
        Se plot_path for fornecido, salva o gráfico do cotovelo.
        """
        if self.X_scaled is None:
            print("Dados insuficientes para encontrar K ótimo.")
            return

        n_samples = self.X_scaled.shape[0]
        # Garante que não tentamos mais clusters do que amostras (evita erro em testes pequenos)
        limit_k = min(max_k, n_samples - 1)
        
        if limit_k < 2:
            print(f"Dados insuficientes para clusterização (n_samples={n_samples}).")
            return

        self.ks = list(range(2, limit_k + 1))
        self.inertias = []
        self.silhouettes = []

        print(f"🔎 Buscando K ótimo entre 2 e {limit_k}...")
        for k in self.ks:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            self.inertias.append(kmeans.inertia_)
            try:
                score = silhouette_score(self.X_scaled, labels)
                self.silhouettes.append(score)
            except Exception:
                self.silhouettes.append(float('nan'))
            
        if plot_path:
            plt.figure(figsize=(8, 4))
            plt.plot(self.ks, self.inertias, marker='o')
            plt.title('Método do Cotovelo (Elbow Method)')
            plt.xlabel('Número de Clusters (K)')
            plt.ylabel('Inércia')
            plt.grid(True)
            plt.savefig(plot_path)
            plt.close()

    def plot_silhouette(self, plot_path: str = None):
        """Plota ou salva gráfico de Silhouette Scores."""
        if not self.silhouettes:
            return

        plt.figure(figsize=(8, 4))
        plt.plot(self.ks, self.silhouettes, marker='o', color='orange')
        plt.title('Silhouette Scores (Quanto maior, melhor)')
        plt.xlabel('k')
        plt.ylabel('Silhouette')
        plt.grid(True)
        if plot_path:
            plt.savefig(plot_path)
            plt.close()
        else:
            plt.show()

    def get_best_k(self) -> int:
        """Retorna o K com melhor Silhouette Score."""
        if not self.silhouettes:
            return 2
        
        # Ignora NaNs
        valid_scores = [s if not np.isnan(s) else -1 for s in self.silhouettes]
        best_idx = np.argmax(valid_scores)
        best_k = self.ks[best_idx]
        print(f"🏆 Melhor K encontrado: {best_k} (Silhouette: {valid_scores[best_idx]:.3f})")
        return best_k

    def train_clustering(self, k: int) -> pd.DataFrame:
        """Treina o K-Means com o K escolhido e retorna DataFrame com clusters."""
        if self.X_scaled is None:
            return self.raw_df

        print(f"🚀 Treinando K-Means final com K={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.X_scaled)
        self.model = kmeans
        
        # Adiciona o cluster ao dataframe original para interpretação
        df_out = self.raw_df.copy()
        df_out['CLUSTER'] = clusters

        # Safety Check: Se houver um índice nomeado (ex: AIRPORT ou TOTAL_FLIGHTS), reseta para virar coluna
        if df_out.index.name is not None:
            print(f"⚠️ AVISO: Índice '{df_out.index.name}' detectado. Resetando índice para transformá-lo em coluna persistente.")
            df_out = df_out.reset_index()

        # Padronização: Renomeia ORIGIN_AIRPORT para AIRPORT se necessário para compatibilidade com consumidor
        if 'ORIGIN_AIRPORT' in df_out.columns and 'AIRPORT' not in df_out.columns:
            print("🔄 Renomeando 'ORIGIN_AIRPORT' para 'AIRPORT' para compatibilidade.")
            df_out.rename(columns={'ORIGIN_AIRPORT': 'AIRPORT'}, inplace=True)
        
        # Padroniza nomes das colunas para maiúsculas e garante tipos numéricos
        df_out.columns = df_out.columns.str.upper().str.strip()
        if 'TOTAL_FLIGHTS' in df_out.columns:
            df_out['TOTAL_FLIGHTS'] = pd.to_numeric(df_out['TOTAL_FLIGHTS'], errors='coerce').fillna(0)

        print(f"💾 Retornando DataFrame com colunas: {df_out.columns.tolist()}")
        print(f"📊 Tipos de dados das colunas:\n{df_out.dtypes}")
        
        try:
            score = silhouette_score(self.X_scaled, clusters)
            self.score = score
            print(f"✅ Clusterização concluída. Silhouette Score Final: {score:.3f}")
        except Exception:
            self.score = None
            pass
            
        return df_out
