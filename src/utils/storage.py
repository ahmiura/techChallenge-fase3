import pandas as pd
from sqlalchemy import create_engine, text
from typing import Optional


def get_engine(db_url: str):
    """Retorna um SQLAlchemy engine para a URL informada."""
    return create_engine(db_url)


def read_table(engine, table_name: str) -> pd.DataFrame:
    """Lê toda a tabela para um DataFrame usando SQLAlchemy Connection.

    Evita passar diretamente objetos que não exponham .cursor() para pandas
    em alguns ambientes.
    """
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT * FROM {table_name}"))
        rows = result.fetchall()
        cols = result.keys()
    return pd.DataFrame(rows, columns=cols)


def save_df(engine, df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
    """Salva um DataFrame no banco. Para Postgres usa COPY via raw_connection
    para maior compatibilidade e performance; fallback para pandas.to_sql.
    """
    try:
        raw_conn = engine.raw_connection()
        cur = raw_conn.cursor()
        try:
            cur.execute(f"DROP TABLE IF EXISTS {table_name};")
            cols_ddl = ', '.join([f'"{c}" TEXT' for c in df.columns])
            cur.execute(f"CREATE TABLE {table_name} ({cols_ddl});")

            from io import StringIO
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False, header=False)
            csv_buffer.seek(0)
            cur.copy_expert(f"COPY {table_name} ({', '.join([f'\"{c}\"' for c in df.columns])}) FROM STDIN WITH CSV", csv_buffer)
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
        df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
