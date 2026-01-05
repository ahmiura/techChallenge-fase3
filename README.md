## Tech Challenge Fase 3 - Flight Delay MLOps Pipeline

Este repositório contém a solução para a Fase 3 do Tech Challenge, implementando um pipeline de MLOps para análise e predição de atrasos de voos.

A arquitetura utiliza:
- **Apache Airflow**: Orquestração de tarefas (DAGs).
- **MLflow**: Rastreamento de experimentos, métricas e registro de modelos.
- **PostgreSQL**: Armazenamento de dados (Raw, Silver, Gold) e metadados.
- **Jupyter Notebook**: Ambiente de desenvolvimento e exploração.
- **Docker Compose**: Orquestração dos serviços.

### Estrutura do Projeto

- `dags/`: Definições dos DAGs do Airflow.
- `src/`: Código fonte auxiliar (data loaders, feature engineering).
- `data/`: Diretório para dados locais (montado nos contêineres).
- `notebooks/`: Notebooks Jupyter.
- `mlruns/`: Armazenamento de artefatos do MLflow.

### Pipeline de Dados (Airflow DAG)

O pipeline `flight_mlops_pipeline` executa as seguintes etapas:

1.  **Ingestão**: Carrega dados brutos do CSV para a tabela `raw_flights`.
2.  **Pré-processamento**: Limpa e move dados para `silver_flights`.
3.  **EDA Automatizada**: Gera estatísticas e gráficos (distribuição, correlação) e registra no MLflow.
4.  **Engenharia de Features**:
    -   Criação do target `IS_DELAYED` (> 15 min).
    -   Novas features: `DEPARTURE_HOUR`, `ARRIVAL_HOUR`, `ROUTE`, `IS_WEEKEND`, `SEASON`.
    -   Salva dados prontos em `gold_features`.
5.  **Modelagem**:
    -   **Supervisionada**: Treina Random Forest (Simples e Tunado) e XGBoost (Tunado).
    -   **Não-Supervisionada**: Agrupa aeroportos usando K-Means (perfil de atraso e volume).
6.  **Avaliação e Registro**: Compara modelos baseados no F1-Score e registra o melhor no MLflow Model Registry.

### Variáveis de Ambiente

Além das credenciais do banco, as seguintes variáveis controlam o pipeline (definidas no `docker-compose.yml` ou `.env`):

-   `SAMPLE_SIZE`: Quantidade de linhas para carregar do CSV (ex: `50000`). Se vazio, carrega tudo.
-   `MAX_TRAIN_ROWS`: Limite de linhas para treinamento para evitar OOM (ex: `200000`).
-   `TARGET_DATABASE_CONN`: String de conexão SQLAlchemy para o banco de dados de destino.
-   `MLFLOW_TRACKING_URI`: URL do servidor MLflow.

### Pré-requisitos

- Docker e Docker Compose instalados
- Arquivo `.env` configurado na raiz do projeto com as variáveis de ambiente necessárias.

### Como subir o stack

```bash
# Na raiz do projeto
docker compose up --build
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
