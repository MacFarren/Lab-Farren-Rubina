#!/usr/bin/env python3
"""
Test directo de la función de predicción
"""
import os
import sys
import json
import tempfile

sys.path.append('/opt/airflow/OneDrive/Documentos/LabMDS/Lab-Farren-Rubina/Lab9/dags')
os.chdir('/opt/airflow/OneDrive/Documentos/LabMDS/Lab-Farren-Rubina/Lab9')

# Importar la función de predicción
from gradio_robust import predict_hiring

def test_prediction():
    """Test de la función de predicción con vale_data.json"""
    
    # Datos de prueba (vale_data.json)
    test_data = [{
        "Age": 25,
        "Gender": 1,
        "EducationLevel": 3,
        "ExperienceYears": 2,
        "PreviousCompanies": 2,
        "DistanceFromCompany": 40,
        "InterviewScore": 98.0,
        "SkillScore": 97.0,
        "PersonalityScore": 99.0,
        "RecruitmentStrategy": 2
    }]
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f, indent=2)
        temp_file_path = f.name
    
    print(f"Archivo temporal creado: {temp_file_path}")
    
    # Simular objeto file de Gradio
    class MockFile:
        def __init__(self, path):
            self.name = path
    
    mock_file = MockFile(temp_file_path)
    
    # Ejecutar predicción
    print("=== EJECUTANDO PREDICCIÓN ===")
    result = predict_hiring(mock_file)
    
    # Mostrar resultado
    print(result)
    
    # Limpiar archivo temporal
    os.unlink(temp_file_path)
    
    return result

if __name__ == "__main__":
    try:
        result = test_prediction()
        print("\n=== TEST COMPLETADO EXITOSAMENTE ===")
        if "CONTRATAR" in result and "75" in result:
            print("✅ Predicción correcta detectada")
        else:
            print("❌ Predicción inesperada")
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()