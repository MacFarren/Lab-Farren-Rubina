FROM python:3.10-slim

WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación
COPY app.py .

# Exponer puerto
EXPOSE 7860

# Variables de entorno
ENV API_BASE_URL=http://backend:8000
ENV HOST=0.0.0.0
ENV PORT=7860

# Comando para ejecutar
CMD ["python", "app.py"]