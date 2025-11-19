"""
Interpretabilidad - SodAI Drinks
===============================

Análisis de interpretabilidad con SHAP sin depender de MLflow.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, List
import logging
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterpretabilityAnalyzer:
    def __init__(self, data_path: str, model_uri: str, models_path: str):
        self.data_path = Path(data_path)
        self.model_uri = model_uri
        self.models_path = Path(models_path)

    def _resolve_model_path(self) -> Path:
        model_path = self.model_uri
        if isinstance(model_path, str) and model_path.startswith('file://'):
            model_path = model_path.replace('file://', '')
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
        return model_path

    def _load_model(self):
        model_path = self._resolve_model_path()
        return joblib.load(model_path)

    def _load_training_feature_names(self) -> List[str]:
        """Carga los nombres de features exactos del entrenamiento."""
        models_path = Path("/opt/airflow/models")
        feature_names_path = models_path / "feature_names.pkl"
        
        if feature_names_path.exists():
            try:
                with open(feature_names_path, 'rb') as f:
                    feature_names = pickle.load(f)
                logger.info(f"Cargadas {len(feature_names)} feature names del entrenamiento")
                return feature_names
            except Exception as e:
                logger.warning(f"Error cargando feature_names.pkl: {e}")
        
        logger.warning("No se encontró feature_names.pkl, usando detección automática")
        return None

    def generate_shap_analysis(self) -> Dict[str, Any]:
        """Genera análisis SHAP del modelo."""
        logger.info("Generando análisis de interpretabilidad...")

        # Cargar datos
        features_path = self.data_path / "features" / "training_dataset.parquet"
        df = pd.read_parquet(features_path)

        # Cargar feature_names exactas del entrenamiento
        feature_names = self._load_training_feature_names()
        
        if feature_names:
            # Usar exactamente las mismas features que durante el entrenamiento
            feature_columns = []
            for feature in feature_names:
                if feature in df.columns:
                    feature_columns.append(feature)
                else:
                    logger.warning(f"Feature '{feature}' del entrenamiento no encontrada")
            
            logger.info(f"Usando {len(feature_columns)} features exactas del entrenamiento")
        else:
            # Fallback: usar todas menos las de control
            feature_columns = [col for col in df.columns if col not in ['customer_id', 'product_id', 'target']]
            logger.info(f"Usando {len(feature_columns)} features detectadas automáticamente")

        # Muestra para SHAP (reducida para eficiencia)
        X_sample = df[feature_columns].head(100).copy()
        
        # Aplicar el mismo preprocesamiento que en entrenamiento
        # Manejar valores faltantes
        X_sample = X_sample.fillna(0)
        
        # Encoding de variables categóricas (igual que en entrenamiento)
        categorical_columns = X_sample.select_dtypes(include=['object', 'category']).columns
        logger.info(f"Codificando {len(categorical_columns)} columnas categóricas para SHAP")
        
        for col in categorical_columns:
            le = LabelEncoder()
            X_sample[col] = le.fit_transform(X_sample[col].fillna('unknown'))

        logger.info(f"Datos preparados para SHAP: {X_sample.shape}")

        # Cargar modelo
        model = self._load_model()

        # Crear explainer SHAP
        try:
            explainer = shap.TreeExplainer(model.named_steps['classifier'])
            X_sample_scaled = model.named_steps['scaler'].transform(X_sample)
            shap_values = explainer.shap_values(X_sample_scaled)

            # Si es clasificación binaria, tomar valores para clase positiva
            if len(shap_values) == 2:
                shap_values = shap_values[1]

            # Feature importance global
            feature_importance = np.abs(shap_values).mean(0)
            feature_importance_dict = dict(zip(feature_columns, feature_importance))
            top_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

            # Generar gráficos (opcional, puede fallar en entorno sin display)
            plots_generated = []
            try:
                # Crear directorio si no existe
                self.models_path.mkdir(parents=True, exist_ok=True)
                
                # Summary plot básico (solo datos, sin visualización)
                logger.info("Generando análisis de importancia de features...")
                
                # Guardar importancia de features como archivo
                importance_data = {
                    'feature_names': feature_columns,
                    'importance_values': feature_importance.tolist()
                }
                
                import json
                importance_path = self.models_path / "feature_importance.json"
                with open(importance_path, 'w') as f:
                    json.dump(importance_data, f, indent=2)
                plots_generated.append(str(importance_path))
                
            except Exception as e:
                logger.warning(f"Error generando gráficos SHAP: {e}")

            results = {
                'top_features': [feat[0] for feat in top_features[:10]],
                'feature_importance_scores': {feat[0]: float(feat[1]) for feat in top_features[:10]},
                'plots_generated': plots_generated,
                'total_features_analyzed': len(feature_columns),
                'shap_analysis_completed': True
            }

            logger.info(f"✅ Análisis SHAP completado. Top features: {results['top_features'][:5]}")
            return results

        except Exception as e:
            logger.error(f"Error en análisis SHAP: {e}")
            # Retornar resultado básico en caso de error
            return {
                'top_features': [],
                'feature_importance_scores': {},
                'plots_generated': [],
                'total_features_analyzed': len(feature_columns),
                'shap_analysis_completed': False,
                'error': str(e)
            }
