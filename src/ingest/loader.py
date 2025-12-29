import pandas as pd
from sqlalchemy import create_engine

class FlightDataLoader:
    """
    Responsável por carregar e limpar os dados brutos.
    Princípio: Single Responsibility (S do SOLID).
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load_data(self, sample_size: int = None) -> pd.DataFrame:
        """Carrega o CSV. Opcionalmente faz amostragem para performance."""
        print(f"🔄 Carregando dados de {self.file_path}...")
        # Lendo colunas essenciais para economizar memória
        cols = [
            'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 
            'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DISTANCE', 
            'SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY', 
            'CANCELLED', 'DIVERTED', 'FLIGHT_NUMBER'
        ]
        
        try:
            self.df = pd.read_csv(self.file_path, usecols=cols)
            print(f"📊 Quantidade original de linhas no arquivo: {self.df.shape[0]}")
            
            # Filtros iniciais de consistência
            self.df = self.df[self.df['CANCELLED'] == 0] # Remove cancelados
            self.df = self.df[self.df['DIVERTED'] == 0]  # Remove desviados
            
            # Preenche nulos de atraso com 0 (voos pontuais/adiantados às vezes vêm NaN)
            self.df['ARRIVAL_DELAY'] = self.df['ARRIVAL_DELAY'].fillna(0)
            # Preenche nulos em partida também se existir
            if 'DEPARTURE_DELAY' in self.df.columns:
                self.df['DEPARTURE_DELAY'] = self.df['DEPARTURE_DELAY'].fillna(0)

            if sample_size:
                self.df = self.df.sample(n=sample_size, random_state=42)
                print(f"⚠️ Amostragem aplicada: {sample_size} linhas.")
                
            print(f"✅ Dados carregados. Shape: {self.df.shape}")
            return self.df
            
        except FileNotFoundError:
            print("❌ Arquivo não encontrado.")
            return pd.DataFrame()

    def save_to_db(self, table_name: str, db_url: str):
        """Salva o dataframe atual no banco de dados."""
        if self.df is not None and not self.df.empty:
            print(f"💾 Salvando dados na tabela '{table_name}'...")
            engine = create_engine(db_url)
            # Pandas aceita um DBAPI connection (com .cursor) ou um SQLAlchemy
            # connectable. Dependendo da versão, Connection/Engine podem não
            # expor .cursor(), então tentamos raw_connection() (DBAPI) primeiro
            # e, em fallback, usamos o engine diretamente.
            # Implementação robusta para Postgres: criar tabela simples e usar COPY
            # via DBAPI para carregar os dados em blocos (mais estável que passar
            # objetos Connection/Engine inconsistentes para pandas em alguns ambientes).
            try:
                raw_conn = engine.raw_connection()
                cur = raw_conn.cursor()
                try:
                    # Drop + create table como TEXT para garantir escrita simples
                    cur.execute(f"DROP TABLE IF EXISTS {table_name};")
                    cols_ddl = ', '.join([f'"{c}" TEXT' for c in self.df.columns])
                    cur.execute(f"CREATE TABLE {table_name} ({cols_ddl});")

                    # Usar COPY para inserção rápida
                    from io import StringIO
                    csv_buffer = StringIO()
                    self.df.to_csv(csv_buffer, index=False, header=False)
                    csv_buffer.seek(0)
                    cur.copy_expert(f"COPY {table_name} ({', '.join([f'\"{c}\"' for c in self.df.columns])}) FROM STDIN WITH CSV", csv_buffer)
                    raw_conn.commit()
                finally:
                    try:
                        cur.close()
                    except Exception:
                        pass
                    try:
                        raw_conn.close()
                    except Exception:
                        pass
            except Exception:
                # Fallback: usar pandas.to_sql com engine (pior caso)
                self.df.to_sql(table_name, con=engine, if_exists='replace', index=False, method='multi')
            print("✅ Dados salvos com sucesso no banco.")
