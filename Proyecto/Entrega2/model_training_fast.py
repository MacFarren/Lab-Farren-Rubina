"""
Módulo de Entrenamiento de Modelos - SodAI Drinks Recommendation System
====================================================================

Este módulo maneja el entrenamiento y optimización de hiperparámetros
del modelo de recomendación usando MLflow para tracking y Optuna para
optimización automática.

Funcionalidades:
- Entrenamiento de modelos de clasificación para recomendación
- Optimización de hiperparámetros con Optuna
- Tracking de experimentos con MLflow
- Validación cruzada estratificada
- Registro automático de modelos

Autor: SodAI Drinks MLOps Team
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import pickle
import json

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import joblib

# MLflow para tracking
import mlflow
import mlflow.sklearn
import mlflow.tracking

# Optuna para optimización
import optuna
# Removemos MLflowCallback para evitar conflictos con runs activos

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
                 optimization_trials: int = 5, cv_folds: int = 2):  # Valores más rápidos por defecto
        """
        Inicializa el entrenador de modelos.
        
        Args:
            data_path: Ruta al directorio de datos
            models_path: Ruta para guardar modelos
            experiment_name: Nombre del experimento en MLflow
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
        
        # Configurar MLflow
        self._setup_mlflow()
        
        logger.info(f"ModelTrainer inicializado: {experiment_name}")
    
    def _setup_mlflow(self):
        """Configura MLflow para tracking."""
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"Experimento MLflow configurado: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"Error configurando MLflow: {e}")
    
    def train_recommendation_model(self) -> Dict[str, Any]:
        """
        Entrena el modelo de recomendación con optimización completa.
        
        Returns:
            Diccionario con resultados del entrenamiento
        """
        logger.info("Iniciando entrenamiento del modelo de recomendación...")
        
        with mlflow.start_run(run_name=f"recommendation_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # 1. Cargar y preparar datos
            X_train, X_test, y_train, y_test, feature_names = self._prepare_training_data()
            
            mlflow.log_params({
                'train_samples': len(X_train),
                'test_samples': len(X_test), 
                'num_features': len(feature_names),
                'positive_ratio_train': y_train.mean(),
                'positive_ratio_test': y_test.mean()
            })
            
            # 2. Optimización de hiperparámetros
            best_params, best_score = self._optimize_hyperparameters(X_train, y_train)
            
            mlflow.log_params(best_params)
            mlflow.log_metric('best_cv_auc', best_score)
            
            # 3. Entrenar modelo final
            final_model = self._train_final_model(X_train, y_train, best_params)
            
            # 4. Evaluar modelo
            evaluation_results = self._evaluate_model(final_model, X_test, y_test)
            
            mlflow.log_metrics(evaluation_results)
            
            # 5. Registrar modelo en MLflow
            model_uri = self._register_model(final_model, feature_names)
            
            # 6. Guardar artifacts adicionales
            self._save_training_artifacts(final_model, feature_names, best_params, evaluation_results)
            
            results = {
                'model_uri': model_uri,
                'best_params': best_params,
                'best_auc': best_score,
                'test_auc': evaluation_results['auc'],
                'feature_names': feature_names,
                'training_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Entrenamiento completado. AUC: {evaluation_results['auc']:.4f}")
            
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
        
        # No usar MLflow callback para evitar conflictos con el run activo
        
        def objective(trial):
            # Espacio de búsqueda simplificado para mayor velocidad
            model_type = trial.suggest_categorical('model_type', ['lightgbm'])  # Removimos gradient boosting
            
            if model_type == 'lightgbm':
                params = {
                    'model_type': 'lightgbm',
                    'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),  # Valores fijos
                    'max_depth': trial.suggest_categorical('max_depth', [5, 10, 15]),  # Valores fijos
                    'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10]),  # Valores fijos
                    'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 4]),  # Valores fijos
                    'random_state': 42,
                      'verbosity': -1
                }
                model = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k != 'model_type'})
                
            else:  # logistic regression
                params = {
                    'model_type': 'logistic',
                    'C': trial.suggest_categorical('C', [0.1, 1.0, 10.0]),  # Valores fijos
                    'penalty': trial.suggest_categorical('penalty', ['l2']),  # Solo L2 para simplicidad
                    'solver': 'lbfgs',  # Más rápido que liblinear para L2
                    'random_state': 42,
                      'verbosity': -1,
                    'max_iter': 1000  # Límite de iteraciones para velocidad
                }
                model = LogisticRegression(**{k: v for k, v in params.items() if k != 'model_type'})
            
            # Validación cruzada estratificada
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=2)  # Limitamos jobs para evitar sobrecarga
            
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
        logger.info(f"Mejores parámetros: {best_params}")
        
        return best_params, best_score
    
    def _train_final_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                          best_params: Dict) -> Pipeline:
        """Entrena el modelo final con los mejores parámetros."""
        logger.info("Entrenando modelo final...")
        
        # Crear modelo según tipo seleccionado
        model_type = best_params['model_type']
        model_params = {k: v for k, v in best_params.items() if k != 'model_type'}
        
        if model_type == 'lightgbm':
            model = lgb.LGBMClassifier(**model_params)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(**model_params)
        else:  # logistic
            model = LogisticRegression(**model_params)
        
        # Crear pipeline con escalado
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        # Entrenar
        pipeline.fit(X_train, y_train)
        
        logger.info(f"Modelo final entrenado: {model_type}")
        
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
        """Registra el modelo en MLflow."""
        logger.info("Registrando modelo en MLflow...")
        
        # Guardar modelo
        model_name = "sodai-recommendation-model"
        
        # Registrar en MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=np.zeros((1, len(feature_names)))
        )
        
        # Crear URI del modelo
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        
        logger.info(f"Modelo registrado: {model_uri}")
        
        return model_uri
    
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
            'model_type': best_params.get('model_type', 'unknown'),
            'num_features': len(feature_names)
        }
        
        # Guardar metadatos
        metadata_path = self.models_path / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Guardar modelo localmente también
        model_path = self.models_path / "recommendation_model.pkl"
        joblib.dump(model, model_path)
        
        # Guardar feature names
        feature_path = self.models_path / "feature_names.pkl"
        with open(feature_path, 'wb') as f:
            pickle.dump(feature_names, f)
        
        # Log artifacts en MLflow
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(feature_path))
        
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
    print(f"- Mejor modelo: {result['best_params']['model_type']}")
    print(f"- URI del modelo: {result['model_uri']}")





