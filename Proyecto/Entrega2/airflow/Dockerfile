# Dockerfile para Airflow - SodAI Drinks
FROM apache/airflow:2.8.0-python3.10

# Cambiar a usuario root temporalmente para instalaciones
USER root

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Volver a usuario airflow
USER airflow

# Copiar requirements ML
COPY config/requirements-ml.txt /tmp/requirements-ml.txt

# Instalar dependencias de Python adicionales
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements-ml.txt

# Copiar DAGs y scripts
COPY dags/ /opt/airflow/dags/

# Crear directorios necesarios
RUN mkdir -p /opt/airflow/data /opt/airflow/models /opt/airflow/logs

# Variables de entorno
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=False
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True