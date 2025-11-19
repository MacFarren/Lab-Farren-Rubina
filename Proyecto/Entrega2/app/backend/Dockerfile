FROM python:3.10-slim

WORKDIR /app

# Dependencias del sistema para healthchecks y librerías nativas
RUN apt-get update \
	&& apt-get install -y --no-install-recommends curl libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY main.py .

# Crear directorios necesarios
RUN mkdir -p /app/models /app/data

# Exponer puerto
EXPOSE 8000

# Variables de entorno por defecto
ENV MODEL_PATH=/app/models/recommendation_model.pkl
ENV DATA_PATH=/app/data
ENV PYTHONPATH=/app

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]