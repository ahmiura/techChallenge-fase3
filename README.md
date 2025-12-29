## Flight Delay MLOps Pipeline

Este repositório contém um pipeline simples de ML para análise de atrasos de voos usando:

- Airflow (DAG em `dags/flight_pipeline.py`)
- Postgres (persistência: `raw_flights`, `silver_flights`, `gold_features`)
- MLflow (registro de features e modelos)
- Docker Compose para orquestrar serviços

Pré-requisitos

- Docker e Docker Compose instalados
- Copiar/editar o arquivo `.env` com variáveis (já existe um `.env` de exemplo)

Como subir o stack

```bash
# Na raiz do projeto
docker-compose up --build
```

Acessos

- Airflow Web UI: http://localhost:8080
- MLflow UI: http://localhost:5000
- Postgres: localhost:5432 (usuário/senha/DB conforme `.env`)

Configuração recomendada

- No Airflow UI, em Admin -> Variables, defina `TARGET_DATABASE_CONN` e `MLFLOW_TRACKING_URI` caso queira sobrescrever valores do `.env`.
- O DAG `flight_mlops_pipeline` está agendado para rodar diariamente por padrão.

Notas

- `requirements.txt` já inclui os pacotes necessários; o Dockerfile da imagem do Airflow deve instalar estas dependências.
- O DAG tenta registrar amostras e importâncias de feature no MLflow para rastreabilidade.

Próximos passos sugeridos

- Adicionar testes unitários para `src/data_loader.py` e `src/features.py`.
- Criar um script de ingestão incremental (CDC) se os dados crescerem.
- Adicionar monitoramento/alertas no Airflow para falhas de tasks.
