FROM apache/airflow:2.9.3

# Copia o requirements e instala apenas dependências extras.
# Evita reinstalar 'apache-airflow' dentro da imagem base.
COPY requirements.txt /requirements.txt

# Filtra linhas com 'apache-airflow' (caso exista) e instala o restante
RUN grep -vE "^\s*apache-airflow" /requirements.txt > /tmp/requirements-extras.txt \
	&& pip install --no-cache-dir -r /tmp/requirements-extras.txt \
	&& rm -f /tmp/requirements-extras.txt