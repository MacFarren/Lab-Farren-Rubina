#!/usr/bin/env python3
"""
Script para registrar el modelo existente en MLflow
"""
import mlflow
import mlflow.sklearn
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path

def register_existing_model():
    """Registra el modelo LightGBM existente en MLflow"""
    
    # Configurar MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("SodAI_Recommendation_System")
    
    print("🚀 Registrando modelo existente en MLflow...")
    
    # Cargar modelo y metadatos
    model_path = Path("models/lightgbm_model.pkl")
    metadata_path = Path("models/model_metadata.json")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadatos no encontrados: {metadata_path}")
    
    # Cargar modelo
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Cargar metadatos
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Modelo cargado: {type(model).__name__}")
    print(f"✅ Metadatos cargados: {len(metadata)} elementos")
    
    # Registrar en MLflow
    with mlflow.start_run():
        # Log parámetros del modelo
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("algorithm", metadata.get("algorithm", "LightGBM"))
        mlflow.log_param("features_count", metadata.get("n_features", 0))
        mlflow.log_param("training_date", metadata.get("training_timestamp", "unknown"))
        
        # Log métricas de entrenamiento
        if "validation_metrics" in metadata:
            metrics = metadata["validation_metrics"]
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
        
        # Log artefactos adicionales
        mlflow.log_artifact("models/model_metadata.json")
        
        # Registrar modelo
        model_info = mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="sodai-recommendation-model"
        )
        
        print(f"✅ Modelo registrado en MLflow")
        print(f"📊 Run ID: {mlflow.active_run().info.run_id}")
        print(f"🎯 Model URI: {model_info.model_uri}")
        
        return model_info

def promote_to_production():
    """Promociona la última versión del modelo a Production"""
    
    client = mlflow.tracking.MlflowClient()
    
    # Obtener la última versión
    latest_versions = client.get_latest_versions(
        "sodai-recommendation-model", 
        stages=["None"]
    )
    
    if latest_versions:
        version = latest_versions[0]
        
        # Promocionar a Production
        client.transition_model_version_stage(
            name="sodai-recommendation-model",
            version=version.version,
            stage="Production"
        )
        
        print(f"✅ Modelo v{version.version} promovido a Production")
        return version
    else:
        print("❌ No se encontró ninguna versión del modelo")
        return None

if __name__ == "__main__":
    try:
        # Registrar modelo
        model_info = register_existing_model()
        
        # Promocionar a producción
        version = promote_to_production()
        
        print("\n🎉 ¡Proceso completado exitosamente!")
        print("🔗 MLflow UI: http://localhost:5000")
        print("📈 El modelo está ahora disponible para el backend API")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise