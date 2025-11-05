#!/usr/bin/env python3
import sys
import os
sys.path.append('/opt/airflow/OneDrive/Documentos/LabMDS/Lab-Farren-Rubina/Lab9/dags')

print("=== TEST DE SISTEMA ===")

try:
    import gradio
    print("✓ Gradio importado correctamente")
except ImportError as e:
    print(f"✗ Error importando Gradio: {e}")

try:
    import joblib
    print("✓ Joblib importado correctamente")
except ImportError as e:
    print(f"✗ Error importando Joblib: {e}")

try:
    import pandas
    print("✓ Pandas importado correctamente")
except ImportError as e:
    print(f"✗ Error importando Pandas: {e}")

# Cambiar al directorio correcto
os.chdir('/opt/airflow/OneDrive/Documentos/LabMDS/Lab-Farren-Rubina/Lab9')

# Verificar modelos
model_path = '2025-11-04/models'
if os.path.exists(model_path):
    models = os.listdir(model_path)
    print(f"✓ Directorio de modelos encontrado: {len(models)} archivos")
    for model in models:
        print(f"  - {model}")
else:
    print("✗ Directorio de modelos no encontrado")

# Verificar DAGs
dag_path = '/opt/airflow/dags'
if os.path.exists(dag_path):
    dags = [f for f in os.listdir(dag_path) if f.endswith('.py')]
    print(f"✓ Directorio DAGs encontrado: {len(dags)} archivos")
    for dag in dags:
        print(f"  - {dag}")
else:
    print("✗ Directorio DAGs no encontrado")

print("\n=== ESTADO DEL SISTEMA ===")
print("Airflow UI: http://localhost:8080 (admin/admin)")
print("Gradio UI: http://localhost:7864")
print("Sistema listo para uso!")