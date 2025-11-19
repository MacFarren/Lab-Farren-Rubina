"""
Módulo de Entrenamiento de Modelos - SodAI Drinks Recommendation System
====================================================================

Este módulo maneja el entrenamiento y optimización de hiperparámetros
del modelo de recomendación usando Optuna, sin dependencias de MLflow.

Funcionalidades:
- Entrenamiento de modelos de clasificación para recomendación
- Optimización de hiperparámetros con Optuna
- Validación cruzada estratificada
- Persistencia local de modelos y metadatos

Autor: SodAI Drinks MLOps Team
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
import logging
from pathlib import Path
import pickle
import json

# ML imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# LightGBM es el único modelo permitido según el enunciado
import lightgbm as lgb
import joblib

# Optuna para optimización
import optuna
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Entrenador de modelos para el sistema de recomendación.
    """
    
    def __init__(self, data_path: str, models_path: str, 
                 experiment_name: str = "sodai-recommendation-system",
                 optimization_trials: int = 10, cv_folds: int = 3):  # Valores más rápidos por defecto
        """
        Inicializa el entrenador de modelos.
        
        Args:
            data_path: Ruta al directorio de datos
            models_path: Ruta para guardar modelos
            experiment_name: Nombre lógico para el experimento (solo logging)
            optimization_trials: Número de trials para optimización (reducido para velocidad)
            cv_folds: Número de folds para validación cruzada (reducido para velocidad)
        """
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.experiment_name = experiment_name
        self.optimization_trials = optimization_trials
        self.cv_folds = cv_folds
        
        # Crear directorios si no existen
        self.models_path.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"ModelTrainer inicializado: {experiment_name}")
    
    def train_recommendation_model(self) -> Dict[str, Any]:
        """
        Entrena el modelo de recomendación con optimización completa.
        
        Returns:
            Diccionario con resultados del entrenamiento
        """
        logger.info("Iniciando entrenamiento del modelo de recomendación...")

        # 1. Cargar y preparar datos
        X_train, X_test, y_train, y_test, feature_names = self._prepare_training_data()

        logger.info(
            "Datos preparados para entrenamiento: train=%s, test=%s, features=%s",
            len(X_train),
            len(X_test),
            len(feature_names)
        )

        # 2. Optimización de hiperparámetros
        best_params, best_score = self._optimize_hyperparameters(X_train, y_train)

        logger.info("Mejores hiperparámetros encontrados: %s (AUC CV: %.4f)", best_params, best_score)

        # 3. Entrenar modelo final
        final_model = self._train_final_model(X_train, y_train, best_params)

        # 4. Evaluar modelo
        evaluation_results = self._evaluate_model(final_model, X_test, y_test)

        logger.info("Evaluación en test: %s", evaluation_results)

        # 5. Guardar modelo y artifacts en disco
        model_uri = self._register_model(final_model, feature_names)
        self._save_training_artifacts(final_model, feature_names, best_params, evaluation_results)

        results = {
            'model_uri': model_uri,
            'best_params': best_params,
            'best_auc': best_score,
            'test_auc': evaluation_results['auc'],
            'feature_names': feature_names,
            'training_timestamp': datetime.now().isoformat(),
            'model_type': 'lightgbm'
        }

        logger.info("✅ Entrenamiento completado. AUC: %.4f", evaluation_results['auc'])

        return results
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepara los datos para entrenamiento."""
        logger.info("Preparando datos de entrenamiento...")
        
        # Cargar dataset de entrenamiento
        features_path = self.data_path / "features" / "training_dataset.parquet"
        
        if not features_path.exists():
            raise FileNotFoundError(f"Dataset de entrenamiento no encontrado: {features_path}")
        
        df = pd.read_parquet(features_path)
        logger.info(f"Dataset cargado: {df.shape}")
        
        # Separar features y target
        feature_columns = [col for col in df.columns if col not in ['customer_id', 'product_id', 'target']]
        X = df[feature_columns]
        y = df['target']
        
        # Manejar valores faltantes
        X = X.fillna(0)
        
        # Encoding de variables categóricas si las hay
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna('unknown'))
        
        # División train/test estratificada
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"División completada: Train {X_train.shape}, Test {X_test.shape}")
        
        return X_train.values, X_test.values, y_train.values, y_test.values, feature_columns
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Dict, float]:
        """Optimiza hiperparámetros usando Optuna."""
        logger.info(f"Iniciando optimización de hiperparámetros ({self.optimization_trials} trials)...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 16, 128, step=8),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 80, step=5),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
                'random_state': 42,
                'n_jobs': -1,
            }

            model = lgb.LGBMClassifier(**params)

            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)

            # Guardar información para inspección
            trial.set_user_attr('lightgbm_params', params)

            return scores.mean()
        
        # Crear estudio de optimización
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Ejecutar optimización con timeout de 10 minutos máximo
        study.optimize(objective, n_trials=self.optimization_trials, timeout=600)
        
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"✅ Optimización completada. Mejor AUC: {best_score:.4f}")
        logger.info(f"Mejores parámetros (LightGBM): {best_params}")
        
        return best_params, best_score
    
    def _train_final_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                          best_params: Dict) -> Pipeline:
        """Entrena el modelo final con los mejores parámetros."""
        logger.info("Entrenando modelo final...")

        model = lgb.LGBMClassifier(**best_params)

        pipeline = Pipeline([
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train)

        logger.info("Modelo final entrenado: lightgbm")

        return pipeline
    
    def _evaluate_model(self, model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evalúa el modelo en datos de test."""
        logger.info("Evaluando modelo...")
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas
        metrics = {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        logger.info(f"Métricas de evaluación: {metrics}")
        
        return metrics
    
    def _register_model(self, model: Pipeline, feature_names: List[str]) -> str:
        """Guarda el modelo entrenado en disco y devuelve su ruta."""
        logger.info("Guardando modelo entrenado en disco...")

        model_path = self.models_path / "recommendation_model.pkl"
        joblib.dump(model, model_path)

        uri = model_path.as_posix()
        logger.info("Modelo guardado en %s", uri)

        # Persistir feature names para procesos aguas abajo
        feature_path = self.models_path / "feature_names.pkl"
        with open(feature_path, 'wb') as f:
            pickle.dump(feature_names, f)
        logger.info("Feature names guardadas en %s", feature_path)

        return uri
    
    def _save_training_artifacts(self, model: Pipeline, feature_names: List[str], 
                               best_params: Dict, evaluation_results: Dict) -> None:
        """Guarda artifacts adicionales del entrenamiento."""
        logger.info("Guardando artifacts de entrenamiento...")
        
        # Metadatos del modelo
        model_metadata = {
            'training_timestamp': datetime.now().isoformat(),
            'feature_names': feature_names,
            'best_params': best_params,
            'evaluation_results': evaluation_results,
            'model_type': 'lightgbm',
            'num_features': len(feature_names)
        }
        
        # Guardar metadatos
        metadata_path = self.models_path / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        logger.info(f"Artifacts guardados en {self.models_path}")

# Función de utilidad para uso independiente
def train_sodai_model(data_path: str, models_path: str, 
                     experiment_name: str = "sodai-recommendation",
                     trials: int = 50) -> Dict[str, Any]:
    """
    Función de conveniencia para entrenar modelo de SodAI.
    
    Args:
        data_path: Ruta al directorio de datos
        models_path: Ruta para guardar modelos
        experiment_name: Nombre del experimento
        trials: Número de trials de optimización
        
    Returns:
        Diccionario con resultados del entrenamiento
    """
    trainer = ModelTrainer(data_path, models_path, experiment_name, trials)
    return trainer.train_recommendation_model()

if __name__ == "__main__":
    # Ejemplo de uso
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenador de modelos SodAI Drinks')
    parser.add_argument('--data-path', required=True, help='Ruta al directorio de datos')
    parser.add_argument('--models-path', required=True, help='Ruta para guardar modelos')
    parser.add_argument('--experiment-name', default='sodai-recommendation', help='Nombre del experimento')
    parser.add_argument('--trials', type=int, default=50, help='Número de trials de optimización')
    args = parser.parse_args()
    
    result = train_sodai_model(args.data_path, args.models_path, args.experiment_name, args.trials)
    print("Entrenamiento completado:")
    print(f"- AUC en test: {result['test_auc']:.4f}")
    print("- Modelo entrenado: lightgbm")
    print(f"- URI del modelo: {result['model_uri']}")
