# Archivo: dag_dynamic.py
# DAG dinámico para la Sección 2 del Laboratorio 9 - Airflow
# Este DAG implementa branching, triggers y entrenamiento de múltiples modelos en paralelo

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Importar las funciones del archivo hiring_dynamic_functions
from hiring_dynamic_functions import (
    create_folders, 
    load_and_merge, 
    split_data, 
    train_model, 
    evaluate_models
)

# Configuración por defecto del DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 1),  # 1 de octubre de 2024
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# 1. Definir el DAG con programación mensual y backfill habilitado
dag = DAG(
    'hiring_dynamic_monthly',  # DAG ID interpretable
    default_args=default_args,
    description='Pipeline dinámico para predicción de contrataciones con múltiples modelos',
    schedule_interval='0 15 5 * *',  # Día 5 de cada mes a las 15:00 UTC (cron: min hour day month dow)
    catchup=True,  # Habilitar backfill para ejecutar desde fechas pasadas
    max_active_runs=1,  # Evitar ejecuciones concurrentes
    tags=['hiring', 'ml', 'dynamic', 'monthly'],
)

# 2. Marcador de inicio del pipeline
start_pipeline = EmptyOperator(
    task_id='start_pipeline',
    dag=dag,
)

# 3. Crear carpetas para la ejecución (raw, preprocessed, splits, models)
create_folders_task = PythonOperator(
    task_id='create_folders',
    python_callable=create_folders,
    provide_context=True,
    dag=dag,
)

# 4. Función para el branching basado en la fecha
def decide_download_strategy(**kwargs):
    """
    Decide qué archivos descargar basándose en la fecha de ejecución.
    - Antes del 1 nov 2024: solo data_1.csv
    - Desde el 1 nov 2024: data_1.csv y data_2.csv
    """
    execution_date = kwargs['ds']  # Formato YYYY-MM-DD
    execution_datetime = datetime.strptime(execution_date, '%Y-%m-%d')
    cutoff_date = datetime(2024, 11, 1)
    
    print(f"Fecha de ejecución: {execution_date}")
    print(f"Fecha de corte: {cutoff_date.strftime('%Y-%m-%d')}")
    
    if execution_datetime < cutoff_date:
        print("Ejecutando antes del 1 de noviembre - descargando solo data_1.csv")
        return 'download_data1_only'
    else:
        print("Ejecutando desde el 1 de noviembre - descargando ambos datasets")
        return 'download_both_datasets'

# Branching operator
branching_download = BranchPythonOperator(
    task_id='decide_download_strategy',
    python_callable=decide_download_strategy,
    provide_context=True,
    dag=dag,
)

# Tareas de descarga - Opción 1: Solo data_1.csv (antes de noviembre)
download_data1_only = BashOperator(
    task_id='download_data1_only',
    bash_command='''
    echo "Descargando solo data_1.csv (ejecución antes del 1 nov 2024)"
    curl -o /opt/airflow/{{ ds }}/raw/data_1.csv https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv
    echo "data_1.csv descargado exitosamente"
    ls -la /opt/airflow/{{ ds }}/raw/
    ''',
    dag=dag,
)

# Tareas de descarga - Opción 2: Ambos datasets (desde noviembre)
download_both_datasets = BashOperator(
    task_id='download_both_datasets',
    bash_command='''
    echo "Descargando ambos datasets (ejecución desde el 1 nov 2024)"
    
    echo "Descargando data_1.csv..."
    curl -o /opt/airflow/{{ ds }}/raw/data_1.csv https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv
    
    echo "Descargando data_2.csv..."
    curl -o /opt/airflow/{{ ds }}/raw/data_2.csv https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv
    
    echo "Ambos datasets descargados exitosamente"
    ls -la /opt/airflow/{{ ds }}/raw/
    ''',
    dag=dag,
)

# 5. Concatenar datasets disponibles con trigger especial
load_and_merge_task = PythonOperator(
    task_id='load_and_merge',
    python_callable=load_and_merge,
    provide_context=True,
    trigger_rule=TriggerRule.ONE_SUCCESS,  # Se ejecuta si al menos una tarea anterior es exitosa
    dag=dag,
)

# 6. Aplicar hold out split
split_data_task = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    provide_context=True,
    dag=dag,
)

# 7. Entrenamiento de 3 modelos en paralelo

# Función wrapper para entrenar RandomForest
def train_random_forest(**kwargs):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    return train_model(model, model_name='RandomForest', **kwargs)

# Función wrapper para entrenar SVM
def train_svm(**kwargs):
    model = SVC(
        kernel='rbf',
        C=1.0,
        random_state=42,
        class_weight='balanced',
        probability=True  # Para obtener probabilidades
    )
    return train_model(model, model_name='SVM', **kwargs)

# Función wrapper para entrenar Regresión Logística
def train_logistic_regression(**kwargs):
    model = LogisticRegression(
        C=1.0,
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    )
    return train_model(model, model_name='LogisticRegression', **kwargs)

# Tareas de entrenamiento en paralelo
train_rf_task = PythonOperator(
    task_id='train_random_forest',
    python_callable=train_random_forest,
    provide_context=True,
    dag=dag,
)

train_svm_task = PythonOperator(
    task_id='train_svm',
    python_callable=train_svm,
    provide_context=True,
    dag=dag,
)

train_lr_task = PythonOperator(
    task_id='train_logistic_regression',
    python_callable=train_logistic_regression,
    provide_context=True,
    dag=dag,
)

# 8. Evaluar todos los modelos y seleccionar el mejor
evaluate_models_task = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    provide_context=True,
    trigger_rule=TriggerRule.ALL_SUCCESS,  # Solo se ejecuta si los 3 entrenamientos son exitosos
    dag=dag,
)

# Marcador de finalización
end_pipeline = EmptyOperator(
    task_id='end_pipeline',
    dag=dag,
)

# Definir las dependencias del DAG

# Flujo principal
start_pipeline >> create_folders_task >> branching_download

# Branching para descarga
branching_download >> [download_data1_only, download_both_datasets]

# Ambas opciones de descarga confluyen en load_and_merge
[download_data1_only, download_both_datasets] >> load_and_merge_task

# Flujo secuencial hasta split
load_and_merge_task >> split_data_task

# Entrenamiento en paralelo
split_data_task >> [train_rf_task, train_svm_task, train_lr_task]

# Evaluación final requiere que todos los entrenamientos terminen
[train_rf_task, train_svm_task, train_lr_task] >> evaluate_models_task

# Finalización
evaluate_models_task >> end_pipeline


# Configuraciones adicionales para el DAG

# Documentación del DAG
dag.doc_md = """
# DAG Dinámico de Predicción de Contrataciones

Este DAG implementa un pipeline más sofisticado con las siguientes características:

## Funcionalidades Principales:

1. **Programación Mensual**: Se ejecuta el día 5 de cada mes a las 15:00 UTC
2. **Backfill Habilitado**: Puede ejecutar tareas desde fechas pasadas
3. **Branching Condicional**: Descarga diferentes datasets según la fecha
4. **Entrenamiento Paralelo**: Tres modelos se entrenan simultáneamente
5. **Evaluación Automática**: Selecciona automáticamente el mejor modelo

## Lógica de Branching:

- **Antes del 1 Nov 2024**: Solo descarga `data_1.csv`
- **Desde el 1 Nov 2024**: Descarga `data_1.csv` y `data_2.csv`

## Modelos Entrenados:

1. **Random Forest**: Modelo ensemble robusto
2. **SVM**: Support Vector Machine con kernel RBF  
3. **Logistic Regression**: Modelo lineal baseline

## Triggers Especiales:

- `load_and_merge`: ONE_SUCCESS (mínimo un archivo disponible)
- `evaluate_models`: ALL_SUCCESS (los 3 modelos deben completarse)

## Estructura de Carpetas:

```
{fecha}/
├── raw/          # Datos originales descargados
├── preprocessed/ # Datos concatenados y preparados  
├── splits/       # Conjuntos de entrenamiento/prueba
└── models/       # Modelos entrenados y mejor modelo
```
"""
