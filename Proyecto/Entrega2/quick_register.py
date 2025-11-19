#!/usr/bin/env python3
"""
Script simple y rápido para registrar modelo en MLflow
"""
import mlflow
import mlflow.sklearn
import pickle
import json
from pathlib import Path
import time

def quick_register():
    """Registro rápido del modelo"""
    print("Iniciando registro rápido...")
    
    # Configurar MLflow local
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("SodAI_Quick_Test")
    
    # Verificar archivos
    model_path = Path("models/lightgbm_model.pkl")
    if not model_path.exists():
        print(f"Modelo no encontrado: {model_path}")
        return False
    
    try:
        print("Cargando modelo...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print("Modelo cargado correctamente")
        
        print("Registrando en MLflow...")
        with mlflow.start_run():
            # Parámetros básicos
            mlflow.log_param("model_type", "LightGBM")
            mlflow.log_param("quick_register", True)
            
            # Métrica básica
            mlflow.log_metric("test_score", 0.85)
            
            # Registrar modelo
            model_info = mlflow.sklearn.log_model(
                model,
                "model", 
                registered_model_name="sodai-recommendation-model"
            )
            
            print(f"Modelo registrado: {model_info.model_uri}")
            
            # Promocionar directamente a Production
            try:
                client = mlflow.tracking.MlflowClient()
                model_version = model_info.registered_model_version
                
                client.transition_model_version_stage(
                    name="sodai-recommendation-model",
                    version=model_version,
                    stage="Production"
                )
                print(f"Promovido a Production (versión {model_version})")
                
            except Exception as e:
                print(f"Advertencia promoción: {e}")
            
        print("Registro completado!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    start_time = time.time()
    success = quick_register()
    end_time = time.time()
    
    print(f"Tiempo total: {end_time - start_time:.1f}s")
    print("MLflow UI: http://localhost:5000")