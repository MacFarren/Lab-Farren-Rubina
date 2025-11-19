"""
Módulo de Evaluación de Modelos - SodAI Drinks Recommendation System
==================================================================

Evaluador de modelos de recomendación con métricas específicas del dominio.
Carga modelos desde disco, sin dependencias de MLflow.

Autor: SodAI Drinks MLOps Team
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, List
import logging
from pathlib import Path
import joblib
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, data_path: str, model_uri: str, feature_names: list = None):
        self.data_path = Path(data_path)
        self.model_uri = model_uri
        self.feature_names = feature_names

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

    def evaluate_recommendation_model(self) -> Dict[str, Any]:
        """Evalúa modelo de recomendación con métricas específicas."""
        logger.info("Evaluando modelo de recomendación...")

        # Cargar datos de test
        features_path = self.data_path / "features" / "training_dataset.parquet"
        df = pd.read_parquet(features_path)

        # División simple para test (últimos 20%)
        test_size = int(len(df) * 0.2)
        test_df = df.tail(test_size)

        # Cargar feature_names del entrenamiento si no se proporcionaron
        if self.feature_names is None:
            self.feature_names = self._load_training_feature_names()

        # Si tenemos la lista de features del entrenamiento, usarla
        if self.feature_names:
            # Usar exactamente las mismas features que durante el entrenamiento
            feature_columns = []
            for feature in self.feature_names:
                if feature in df.columns:
                    feature_columns.append(feature)
                else:
                    logger.warning(f"Feature '{feature}' del entrenamiento no encontrada en datos de evaluación")
            
            logger.info(f"Usando {len(feature_columns)} features exactas del entrenamiento")
        else:
            # Fallback: usar solo columnas numéricas
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_columns if col not in ['customer_id', 'product_id', 'target']]
            logger.info(f"Usando {len(feature_columns)} features numéricas detectadas automáticamente")

        # Seleccionar features y aplicar el mismo preprocesamiento que en entrenamiento
        X_test = test_df[feature_columns].copy()
        
        # Manejar valores faltantes
        X_test = X_test.fillna(0)
        
        # Encoding de variables categóricas (igual que en entrenamiento)
        categorical_columns = X_test.select_dtypes(include=['object', 'category']).columns
        logger.info(f"Codificando {len(categorical_columns)} columnas categóricas: {list(categorical_columns)}")
        
        for col in categorical_columns:
            le = LabelEncoder()
            X_test[col] = le.fit_transform(X_test[col].fillna('unknown'))
            logger.info(f"Columna '{col}' codificada con {len(le.classes_)} categorías")
        
        y_test = test_df['target']

        logger.info(f"Evaluando con {len(feature_columns)} features en {len(X_test)} muestras")
        logger.info(f"Tipos de datos finales en X_test: {X_test.dtypes.value_counts()}")
        logger.info(f"Shape final de X_test: {X_test.shape}")

        # Cargar modelo
        model = self._load_model()

        # Predicciones
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Métricas básicas
        auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        # Métricas de recomendación (Precision@K, Recall@K)
        precision_at_k, recall_at_k = self._calculate_ranking_metrics(y_test, y_pred_proba)

        # Coverage (diversidad de recomendaciones)
        coverage = self._calculate_coverage(test_df, y_pred_proba)

        results = {
            'auc': auc,
            'average_precision': avg_precision,
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k,
            'coverage': coverage
        }

        logger.info(f"Evaluación completada: AUC={auc:.4f}, Coverage={coverage:.4f}")
        return results

    def _calculate_ranking_metrics(self, y_true, y_scores):
        """Calcula Precision@K y Recall@K."""
        k_values = [5, 10, 20, 50]
        precision_at_k = {}
        recall_at_k = {}

        # Ordenar por score descendente
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_true = y_true.iloc[sorted_indices]

        for k in k_values:
            if k <= len(sorted_true):
                top_k_true = sorted_true.iloc[:k]
                precision_at_k[k] = top_k_true.mean()
                recall_at_k[k] = top_k_true.sum() / y_true.sum() if y_true.sum() > 0 else 0
            else:
                precision_at_k[k] = 0
                recall_at_k[k] = 0

        return precision_at_k, recall_at_k

    def _calculate_coverage(self, test_df, y_scores):
        """Calcula coverage (diversidad de productos recomendados)."""
        # Simular recomendaciones top-N por cliente
        test_df_copy = test_df.copy()
        test_df_copy['score'] = y_scores

        # Top 10 por cliente
        top_recommendations = test_df_copy.groupby('customer_id').apply(
            lambda x: x.nlargest(10, 'score')['product_id'].tolist()
        )

        # Productos únicos recomendados
        all_recommended = set()
        for recs in top_recommendations:
            all_recommended.update(recs)

        # Coverage = productos recomendados / productos totales
        total_products = test_df['product_id'].nunique()
        coverage = len(all_recommended) / total_products

        return coverage
