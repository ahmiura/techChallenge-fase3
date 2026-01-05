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
        # Cria features de hora a partir das colunas de tempo (HHMM)
        if 'SCHEDULED_DEPARTURE' in self.df.columns:
            self.df['DEPARTURE_HOUR'] = self.df['SCHEDULED_DEPARTURE'].astype(str).str.zfill(4).str[:2]
            self.df['DEPARTURE_HOUR'] = pd.to_numeric(self.df['DEPARTURE_HOUR'], errors='coerce').fillna(0).astype(int)
            print("✨ Feature 'DEPARTURE_HOUR' criada.")

        if 'SCHEDULED_ARRIVAL' in self.df.columns:
            self.df['ARRIVAL_HOUR'] = self.df['SCHEDULED_ARRIVAL'].astype(str).str.zfill(4).str[:2]
            self.df['ARRIVAL_HOUR'] = pd.to_numeric(self.df['ARRIVAL_HOUR'], errors='coerce').fillna(0).astype(int)
            print("✨ Feature 'ARRIVAL_HOUR' criada.")

        # Cria feature de Rota (Origem-Destino)
        if 'ORIGIN_AIRPORT' in self.df.columns and 'DESTINATION_AIRPORT' in self.df.columns:
            self.df['ROUTE'] = self.df['ORIGIN_AIRPORT'].astype(str) + '-' + self.df['DESTINATION_AIRPORT'].astype(str)
            print("✨ Feature 'ROUTE' criada.")

        # Cria feature de Fim de Semana (Sábado=6, Domingo=7)
        if 'DAY_OF_WEEK' in self.df.columns:
            self.df['IS_WEEKEND'] = self.df['DAY_OF_WEEK'].isin([6, 7]).astype(int)
            print("✨ Feature 'IS_WEEKEND' criada.")

        # Cria feature de Estação do Ano (Hemisfério Norte)
        if 'MONTH' in self.df.columns:
            # 1=Inverno, 2=Primavera, 3=Verão, 4=Outono
            self.df['SEASON'] = self.df['MONTH'].apply(lambda x: 1 if x in [12, 1, 2] else (2 if x in [3, 4, 5] else (3 if x in [6, 7, 8] else 4)))
            print("✨ Feature 'SEASON' criada.")

        # Seleção de Features
        features = [
            'MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'DISTANCE', 
            'DEPARTURE_HOUR', 'ARRIVAL_HOUR', 'SCHEDULED_TIME',
            'ROUTE', 'IS_WEEKEND', 'SEASON'
        ]
        
        # Encoding de Categóricas (Label Encoding para simplicidade de Tree-models)
        # Em produção, OneHotEncoding seria melhor para Regressão Logística
        for col in ['AIRLINE', 'ROUTE']:
            if col not in features or col not in self.df.columns:
                continue
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        
        # Garante que todas as features selecionadas existam no DataFrame
        existing_features = [f for f in features if f in self.df.columns]
        if len(existing_features) != len(features):
            print(f"⚠️ Aviso: Algumas features não foram encontradas. Usando: {existing_features}")

        # Registrar as features selecionadas no MLflow (só se houver run ativo)
        print("📝 Registrando features no MLflow...")
        try:
            if active_run() is not None:
                log_param("selected_features", existing_features)
        except Exception:
            # Em ambientes sem MLflow configurado, seguir sem erro
            pass

        X = self.df[existing_features].fillna(0)
        y = self.df['IS_DELAYED']
        
        return X, y

    def prepare_data_unsupervised(self, min_flights: int = 50) -> pd.DataFrame:
        """
        Agrega dados por Aeroporto de Origem para clusterização.
        min_flights: filtra aeroportos com número mínimo de voos (útil para reduzir ruído).
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
            try:
                airport_profile = airport_profile[airport_profile['TOTAL_FLIGHTS'] > int(min_flights)]
            except Exception:
                airport_profile = airport_profile[airport_profile['TOTAL_FLIGHTS'] > 50]

        return airport_profile
