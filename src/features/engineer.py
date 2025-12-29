import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from src.utils.mlflow_client import active_run, log_param


class FeatureEngineer:
    """
    Responsável por transformar dados brutos em features para ML.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.label_encoders = {}

    def create_target_classification(self, threshold: int = 15) -> pd.DataFrame:
        """Cria a variável alvo binária: 1 se atrasou > threshold, 0 caso contrário."""
        # Garante que ARRIVAL_DELAY seja numérico antes da comparação
        self.df['ARRIVAL_DELAY'] = pd.to_numeric(self.df['ARRIVAL_DELAY'], errors='coerce').fillna(0)
        self.df['IS_DELAYED'] = (self.df['ARRIVAL_DELAY'] > threshold).astype(int)
        print(f"🎯 Target criado: 'IS_DELAYED' (> {threshold} min).")
        return self.df

    def prepare_features_supervised(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara X e y para modelos supervisionados."""
        # Seleção de Features
        features = ['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'DISTANCE', 'SCHEDULED_DEPARTURE']
        
        # Encoding de Categóricas (Label Encoding para simplicidade de Tree-models)
        # Em produção, OneHotEncoding seria melhor para Regressão Logística
        if 'AIRLINE' in features:
            le = LabelEncoder()
            self.df['AIRLINE'] = le.fit_transform(self.df['AIRLINE'].astype(str))
            self.label_encoders['AIRLINE'] = le
            
        # Registrar as features selecionadas no MLflow (só se houver run ativo)
        print("📝 Registrando features no MLflow...")
        try:
            if active_run() is not None:
                log_param("selected_features", features)
        except Exception:
            # Em ambientes sem MLflow configurado, seguir sem erro
            pass
        
        X = self.df[features].fillna(0)
        y = self.df['IS_DELAYED']
        
        return X, y

    def prepare_data_unsupervised(self) -> pd.DataFrame:
        """
        Agrega dados por Aeroporto de Origem para clusterização.
        Não faz sentido clusterizar voos individuais para 'perfilar aeroportos'.
        """
        print("🔄 Agregando dados por Aeroporto...")
        agg_dict = {}
        if 'ARRIVAL_DELAY' in self.df.columns:
            agg_dict['ARRIVAL_DELAY'] = 'mean'
        if 'DEPARTURE_DELAY' in self.df.columns:
            agg_dict['DEPARTURE_DELAY'] = 'mean'
        if 'DISTANCE' in self.df.columns:
            agg_dict['DISTANCE'] = 'mean'
        if 'AIRLINE' in self.df.columns:
            agg_dict['AIRLINE'] = 'nunique'
        
        # Para contar o volume de voos, preferir uma coluna sempre presente
        count_col = 'FLIGHT_NUMBER' if 'FLIGHT_NUMBER' in self.df.columns else (self.df.columns[0] if len(self.df.columns) > 0 else None)
        if count_col:
            agg_dict[count_col] = 'count'
        
        airport_profile = self.df.groupby('ORIGIN_AIRPORT').agg(agg_dict)
        
        # Normaliza nome da coluna de total de voos
        if count_col and count_col != 'FLIGHT_NUMBER':
            airport_profile = airport_profile.rename(columns={count_col: 'TOTAL_FLIGHTS'})
        elif 'FLIGHT_NUMBER' in airport_profile.columns:
            airport_profile = airport_profile.rename(columns={'FLIGHT_NUMBER': 'TOTAL_FLIGHTS'})
        
        # Filtra aeroportos muito pequenos para evitar ruído
        if 'TOTAL_FLIGHTS' in airport_profile.columns:
            airport_profile = airport_profile[airport_profile['TOTAL_FLIGHTS'] > 50]
        
        return airport_profile
