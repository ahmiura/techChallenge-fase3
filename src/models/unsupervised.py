import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class UnsupervisedModeler:
    """
    Gerencia a clusterização (K-Means).
    """
    def __init__(self, df_aggregated: pd.DataFrame):
        self.raw_df = df_aggregated
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.raw_df)
        
    def find_optimal_k(self, max_k: int = 10):
        """Método do Cotovelo (Elbow Method)."""
        inertias = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
            
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, max_k + 1), inertias, marker='o')
        plt.title('Método do Cotovelo (Elbow Method)')
        plt.xlabel('Número de Clusters (K)')
        plt.ylabel('Inércia')
        plt.show()

    def train_clustering(self, k: int):
        """Treina o K-Means com o K escolhido."""
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.X_scaled)
        
        # Adiciona o cluster ao dataframe original para interpretação
        self.raw_df['CLUSTER'] = clusters
        score = silhouette_score(self.X_scaled, clusters)
        
        print(f"✅ Clusterização concluída com K={k}. Silhouette Score: {score:.3f}")
        return self.raw_df
