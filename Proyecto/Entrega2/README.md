# SodAI Drinks - Entrega Parcial 2 (MLOps)

Este repositorio contiene la entrega 2 del proyecto MDS7202. Incluye un pipeline productivo en Apache Airflow, una aplicación web (FastAPI + Gradio) y la orquestación dockerizada descrita en el enunciado `enunciado_entrega2 (2).ipynb`.

## 1. Estructura Entregada

```
entrega_2/
 airflow/                # DAG, scripts y configs del pipeline
 app/
    backend/            # API FastAPI dockerizada
    frontend/           # UI Gradio dockerizada
    docker-compose.yml  # Compose interno opcional
 data/                   # Datos crudos, limpios, features y predicciones
 models/                 # Modelos entrenados, metadata y runs
 nginx/                  # Reverse proxy para /, /api y /airflow
 docker-compose.yml      # Stack completo (Airflow + app)
 docs/                   # Diagramas y capturas de apoyo
 conclusiones.md         # Reflexión solicitada en el enunciado
```

La carpeta `bonus/` no se entrega porque no se implementaron los extras.

## 2. Checklist del Enunciado

| Requisito | Implementación | Evidencia |
| --- | --- | --- |
| Pipeline completo (extracción  features  drift  retrain  evaluación  interpretabilidad  predicciones) | `airflow/dags/sodai_recommendation_pipeline.py` + scripts en `airflow/dags/scripts/` | Ver sección 4 y Airflow UI (`http://localhost/airflow`) |
| Detección de drift y reentrenamiento condicional | `drift_detection.py` calcula PSI + KS y decide entre `retrain_model` y `load_existing_model` | JSON en `models/pipeline_runs/` registra la decisión |
| Optuna + SHAP | `model_training.py` usa Optuna-LightGBM; `interpretability.py` genera importancia SHAP | Artefactos en `models/feature_names.pkl` y `models/feature_importance.json` |
| Tracking organizado + export del modelo | Persistencia local en `models/model_metadata.json`, `models/pipeline_runs/*.json`, `models/recommendation_model.pkl` y gráficos en `models/interpretability/` | Detalle en sección 5 |
| Predicciones para la semana siguiente | `prediction_generator.py` escribe parquet en `data/predictions/` | Archivos `data/predictions/predictions_demo_*.parquet` |
| Documentación del pipeline | Este README + `airflow/README.md` | Carpeta raíz |
| App web dockerizada (FastAPI + Gradio) + reverse proxy | Código en `app/backend`, `app/frontend` y `nginx/nginx.conf`; orquestado en `docker-compose.yml` | Sección 6 |
| Conclusiones escritas | `conclusiones.md` | Carpeta raíz |

## 3. Ejecución

### 3.1 Prerrequisitos

- Docker Desktop (Compose v2) con 8 GB de RAM asignados.
- Puertos libres: 80, 443, 5432, 6379, 7860, 8000, 8081.
- Archivos `clientes.parquet`, `productos.parquet` y `transacciones.parquet` ubicados en `data/`.

### 3.2 Pasos

```powershell
cd C:\Users\Seba\Desktop\Proyecto\Entrega2
Get-ChildItem data
Get-ChildItem models

docker compose up --build

docker compose ps
```

URLs una vez que todos los servicios estén `healthy`:

| Servicio | URL | Notas |
| --- | --- | --- |
| Gradio frontend | http://localhost | Formularios de predicción y recomendaciones |
| FastAPI backend | http://localhost/api/docs | Documentación interactiva |
| Airflow Webserver | http://localhost/airflow (admin/admin) | Monitoreo y ejecución del DAG |

## 4. Pipeline de Airflow

### 4.1 Tareas del DAG `sodai_recommendation_pipeline`

1. `extract_and_validate_data`: valida esquemas, controla duplicados y deja parquet limpios en `data/clean/`.
2. `create_features`: genera más de 70 features (cliente, producto, interacción y temporales) y escribe `training_dataset.parquet` + `preprocessing_pipeline.pkl` en `data/features/`.
3. `detect_data_drift`: compara `reference_features.parquet` vs `current_features.parquet` usando PSI y KS; devuelve la bandera `should_retrain`.
4. Rama condicional:
   - `retrain_model`: usa Optuna (LightGBM + 40 trials) y guarda `recommendation_model.pkl`, `feature_names.pkl` y `model_metadata.json`.
   - `load_existing_model`: garantiza que exista un modelo listo y comparte su ruta.
5. `evaluate_model`: calcula AUC, precision, recall, F1 y métricas top-K del dataset actual.
6. `analyze_interpretability`: ejecuta SHAP y guarda importancia global en `models/feature_importance.json`.
7. `generate_predictions`: crea predicciones semanales y las guarda en `data/predictions/`.
8. `log_pipeline_metrics`: consolida métricas, hiperparámetros y rutas de artefactos en `models/pipeline_runs/pipeline_run_<timestamp>.json`.

### 4.2 Configuración relevante

- Schedule semanal (`@weekly`) y `catchup=False`.
- Variables manejadas en código: `SODAI_DATA_PATH`, `SODAI_MODELS_PATH`, `DRIFT_THRESHOLD`, `OPTIMIZATION_TRIALS`.
- Airflow se ejecuta con Postgres (metadatos) y Redis (cola Celery) definidos en `docker-compose.yml`.
- `airflow/README.md` y `docs/` contienen diagrama mermaid y capturas de la UI.

## 5. Tracking sin MLflow

| Información | Ruta | Formato |
| --- | --- | --- |
| Métricas e hiperparámetros por ejecución | `models/pipeline_runs/pipeline_run_<timestamp>.json` | JSON con secciones `data_stats`, `drift`, `training`, `evaluation`, `artifacts` |
| Metadata del modelo vigente | `models/model_metadata.json` | JSON (versión, hiperparámetros óptimos, fecha) |
| Modelo exportado | `models/recommendation_model.pkl` + `models/feature_names.pkl` | Joblib/PKL |
| Interpretabilidad | `models/feature_importance.json` (y PNG futuros en `models/interpretability/`) | JSON + imágenes |
| Predicciones | `data/predictions/predictions_demo_<timestamp>.parquet` | Parquet |

Estos artefactos se comparten entre Airflow y el backend mediante bind mounts, asegurando que la API siempre use el modelo más reciente.

## 6. Aplicación Web

### Backend (FastAPI)

- Archivo principal: `app/backend/main.py`.
- Endpoints clave: `/health`, `/predict`, `/predict/batch`, `/recommendations`, `/customers/{id}/info`, `/products/{id}/info`.
- Carga `models/recommendation_model.pkl`, `models/feature_names.pkl` y los parquet limpios en `data/clean/`.
- Dockerfile expone el servicio en el puerto 8000 y define un healthcheck.

### Frontend (Gradio)

- Archivo principal: `app/frontend/app.py`.
- Pestañas: predicción individual, recomendaciones por cliente y sección de ayuda.
- Usa `API_BASE_URL=http://backend:8000` para comunicarse con FastAPI.
- Dockerfile expone el puerto 7860 y declara healthcheck.

### Nginx

- `nginx/nginx.conf` mapea `/` al frontend, `/api/` al backend y `/airflow/` al Airflow Webserver.

## 7. Datos y Artefactos

- `data/clean/`: parquet validados de clientes, productos y transacciones.
- `data/features/`: datasets de entrenamiento, referencia y current features.
- `data/predictions/`: salidas del pipeline para la semana futura.
- `models/`: modelo entrenado, codificadores, importancia de features y runs.

## 8. Validación y Troubleshooting

```powershell
# Levantar stack y ver logs en foreground
cd C:\Users\Seba\Desktop\Proyecto\Entrega2
docker compose up --build

# Trigger manual del DAG (desde el webserver)
docker compose exec airflow-webserver airflow dags trigger sodai_recommendation_pipeline

docker compose exec airflow-webserver airflow dags list-runs -d sodai_recommendation_pipeline

docker compose exec airflow-webserver airflow tasks log sodai_recommendation_pipeline generate_predictions <run_id>

# Probar API
Invoke-RestMethod -Method POST -Uri "http://localhost/api/predict" -Body '{"customer_id":256017,"product_id":34092}' -ContentType "application/json"
```

Si algún contenedor falla, inspeccionar con `docker compose logs <servicio>` y confirmar que `data/` contenga los tres parquet originales.

## 9. Próximos Pasos

1. Reincorporar MLflow para obtener el bonus opcional y centralizar el tracking.
2. Automatizar la generación del video solicitado en el enunciado y agregar el enlace en este README.
3. Completar la carpeta `bonus/` con recsys o chatbot conversacional para el puntaje adicional.
4. Agregar pruebas automatizadas (`pytest`) para la API y los scripts críticos del pipeline.

---
Autores: Maximiliano Farren y Sebastián Rubina.
Curso: MDS7202 - Laboratorio de Programación Científica para Ciencia de Datos.
Fecha de actualización: 19/11/2025.
