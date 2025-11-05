#!/usr/bin/env python3
# Ejecutar entrenamiento completo de modelos
import os
import sys
sys.path.append('/opt/airflow/OneDrive/Documentos/LabMDS/Lab-Farren-Rubina/Lab9/dags')

from hiring_dynamic_functions import *

def main():
    print("=== Iniciando entrenamiento de modelos ===")
    
    # Importar los modelos
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    
    # Simular contexto de kwargs para las funciones
    kwargs = {'ds': '2025-11-04'}
    
    # 1. Crear carpetas
    print("1. Creando estructura de carpetas...")
    create_folders(**kwargs)
    
    # 2. Cargar y combinar datos
    print("2. Cargando y combinando datos...")
    load_and_merge(**kwargs)
    
    # 3. Split de datos
    print("3. Dividiendo datos...")
    split_data(**kwargs)
    
    # 4. Entrenar modelos
    print("4. Entrenando RandomForest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_model(rf_model, **kwargs)
    
    print("5. Entrenando SVM...")
    svm_model = SVC(kernel='rbf', random_state=42)
    train_model(svm_model, **kwargs)
    
    print("6. Entrenando LogisticRegression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    train_model(lr_model, **kwargs)
    
    # 5. Evaluar modelos
    print("7. Evaluando modelos...")
    evaluate_models(**kwargs)
    
    print("=== ¡Entrenamiento completado! ===")
    print("Modelos guardados en: /opt/airflow/2025-11-04/models/")

if __name__ == "__main__":
    main()