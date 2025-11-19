import sys
sys.path.append('/opt/airflow/dags/scripts')
from prediction_generator import PredictionGenerator

print(' Prueba con MLflow URI correcta...')

try:
    generator = PredictionGenerator(
        data_path='/opt/airflow/data',
        model_uri='models:/sodai-recommendation-model/1',
        output_path='/opt/airflow/data/predictions'
    )
    
    print(' Ejecutando generate_weekly_predictions...')
    result = generator.generate_weekly_predictions()
    print(' ÉXITO!')
    print(f' Resultados: {result}')
    
except Exception as e:
    print(f' Error: {e}')
    import traceback
    traceback.print_exc()
