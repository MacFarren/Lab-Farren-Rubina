from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Importar las funciones personalizadas
from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface

# Configuración por defecto del DAG
default_args = {
    'owner': 'hiring_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 1),  # 1 de octubre de 2024
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definir el DAG
dag = DAG(
    dag_id='hiring_lineal',  # ID fácil de reconocer
    default_args=default_args,
    description='Pipeline ML para predicción de contratación - Flujo lineal',
    schedule_interval=None,  # Ejecución manual únicamente
    catchup=False,  # Sin backfill - no ejecuta fechas pasadas
    max_active_runs=1,  # Solo una ejecución a la vez
    tags=['ml', 'hiring', 'prediction', 'linear'],
)

# Tarea 0: Inicio del pipeline
start_pipeline = EmptyOperator(
    task_id='start_pipeline',
    dag=dag,
)

# Tarea 1: Crear estructura de carpetas
create_folders_task = PythonOperator(
    task_id='create_folders',
    python_callable=create_folders,
    op_kwargs={
        'ds': "{{ ds }}",
        'ds_nodash': "{{ ds_nodash }}",
        'dag_id': "{{ dag.dag_id }}",
        'task_id': "{{ task.task_id }}",
    },
    dag=dag,
)

# Tarea 2: Descargar/preparar datos (simulado con bash)
download_data = BashOperator(
    task_id='download_data',
        bash_command='''
        echo "Descargando data_1.csv para la ejecución {{ ds }}..."
        mkdir -p "{{ ds }}/raw"
        curl -fsSL -o "{{ ds }}/raw/data_1.csv" "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        if [ $? -ne 0 ]; then
            echo "Error descargando data_1.csv"
            exit 1
        fi
        echo "Archivo descargado:" && ls -l "{{ ds }}/raw/data_1.csv"
        ''',
    dag=dag,
)

# Tarea 3: Dividir datos en train/test
split_data_task = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    op_kwargs={
        'ds': "{{ ds }}",
        'ds_nodash': "{{ ds_nodash }}",
        'dag_id': "{{ dag.dag_id }}",
        'task_id': "{{ task.task_id }}",
    },
    dag=dag,
)

# Tarea 4: Preprocesar y entrenar modelo
preprocess_and_train_task = PythonOperator(
    task_id='preprocess_and_train',
    python_callable=preprocess_and_train,
    op_kwargs={
        'ds': "{{ ds }}",
        'ds_nodash': "{{ ds_nodash }}",
        'dag_id': "{{ dag.dag_id }}",
        'task_id': "{{ task.task_id }}",
    },
    dag=dag,
)

# Tarea 5: Lanzar interfaz Gradio
gradio_interface_task = PythonOperator(
    task_id='gradio_interface',
    python_callable=gradio_interface,
    op_kwargs={
        'ds': "{{ ds }}",
        'ds_nodash': "{{ ds_nodash }}",
        'dag_id': "{{ dag.dag_id }}",
        'task_id': "{{ task.task_id }}",
    },
    dag=dag,
)

# Definir dependencias del flujo lineal
start_pipeline >> create_folders_task >> download_data >> split_data_task >> preprocess_and_train_task >> gradio_interface_task
