# ENTREGA FINAL - DEMO LISTA

## SERVICIOS CORRIENDO

- Servidor Web: http://localhost:8080  
- PostgreSQL: Puerto 5432  
- Redis: Puerto 6379  

## COMPONENTES LISTOS

### MLflow
- Experimentos guardados en /mlruns
- Modelo registrado: sodai-recommendation-model
- Para iniciar UI: mlflow ui --port 5000

### Airflow 
- DAGs en /airflow/dags/
- Scripts de pipeline listos
- Configuración completa

### Modelos
- LightGBM: models/lightgbm_model.pkl
- RandomForest: models/recommendation_model.pkl  
- Encoders: Todos los encoders necesarios
- Metadata: models/model_metadata.json

### Scripts Disponibles
1. model_training_fixed.py - Entrenamiento principal
2. quick_register.py - Registro rápido MLflow
3. simple_api.py - API de recomendaciones
4. test_*.py - Scripts de testing

## DEMO FLOW

1. Mostrar archivos: http://localhost:8080
2. Ejecutar MLflow: mlflow ui
3. Ver experimentos: Navegador MLflow
4. Mostrar DAGs: Carpeta airflow/dags
5. Ejecutar API: python simple_api.py

## TIEMPO TOTAL DESARROLLO
- Registro modelo: 13.4 segundos
- Stack completo configurado
- Componentes integrados

TODO LISTO PARA DEMOSTRACIÓN