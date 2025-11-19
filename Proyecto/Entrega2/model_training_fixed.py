"""
SodAI Drinks - Entrenamiento de Modelo de Recomendación
Sistema MLOps para recomendación de bebidas para SodAI
Versión con LightGBM optimizada y preprocesamiento corregido
"""

import logging
import json
import pickle
import joblib
from typing import Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import optuna
import mlflow
import mlflow.sklearn
import lightgbm as lgb

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suprimir warnings de Optuna para logs más limpios
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    """
    Entrenador de modelos para el sistema de recomendación de SodAI.
    
    Características:
    - Optimización automática de hiperparámetros con Optuna
    - Integración con MLflow para tracking
    - Soporte para múltiples algoritmos de ML
    - Validación cruzada estratificada
    - Pipeline completo de preprocessing
    """

    def __init__(self, data_path: str, models_path: str, 
                 experiment_name: str = "sodai-recommendation",
                 optimization_trials: int = 5, cv_folds: int = 2):  # Valores más rápidos por defecto
        """
        Inicializa el entrenador de modelos.
        
        Args:
            data_path: Ruta al directorio de datos
            models_path: Ruta al directorio donde guardar modelos
            experiment_name: Nombre del experimento MLflow
            optimization_trials: Número de trials para optimización
            cv_folds: Número de folds para validación cruzada
        """
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.experiment_name = experiment_name
        self.optimization_trials = optimization_trials
        self.cv_folds = cv_folds
        
        # Crear directorios si no existen
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar MLflow
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Configura MLflow para tracking."""
        logger.info(f"Configurando MLflow - Experimento: {self.experiment_name}")
        mlflow.set_experiment(self.experiment_name)

    def train_recommendation_model(self) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de entrenamiento.
        
        Returns:
            Dict con resultados del entrenamiento
        """
        logger.info("🚀 Iniciando entrenamiento del modelo de recomendación...")

        with mlflow.start_run():
            # 1. Preparar datos
            X_train, X_test, y_train, y_test, feature_names = self._prepare_training_data()

            # 2. Optimizar hiperparámetros
            logger.info("🔍 Optimizando hiperparámetros...")
            best_params, best_score = self._optimize_hyperparameters(X_train, y_train)

            # Log de parámetros y métricas en MLflow
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
        """Prepara los datos para entrenamiento con preprocessing mejorado."""
        logger.info("Preparando datos de entrenamiento...")

        # Cargar dataset de entrenamiento
        features_path = self.data_path / "features" / "training_dataset.parquet"

        if not features_path.exists():
            raise FileNotFoundError(f"Dataset de entrenamiento no encontrado: {features_path}")

        df = pd.read_parquet(features_path)
        logger.info(f"Dataset cargado: {df.shape}")

        # Separar features y target
        feature_columns = [col for col in df.columns if col not in ['customer_id', 'product_id', 'target']]
        X = df[feature_columns].copy()
        y = df['target']

        # PREPROCESSING MEJORADO PARA LIGHTGBM
        logger.info("Aplicando preprocessing para LightGBM...")
        
        # 1. Identificar tipos de columnas
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = [col for col in X.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        logger.info(f"Columnas categóricas detectadas: {categorical_columns}")
        logger.info(f"Columnas de fecha detectadas: {datetime_columns}")
        
        # 2. Manejar valores faltantes primero
        X = X.fillna(-1)
        
        # 3. Convertir columnas de fecha a timestamp numérico
        for col in datetime_columns:
            if col in X.columns:
                try:
                    X[col] = pd.to_datetime(X[col], errors='coerce').astype('int64') / 10**9
                    X[col] = X[col].fillna(-1)
                except Exception as e:
                    logger.warning(f"No se pudo convertir columna de fecha {col}: {e}")
                    X[col] = X[col].astype('float64')
        
        # 4. Encoding de variables categóricas
        encoders = {}
        for col in categorical_columns:
            if col in X.columns:
                logger.info(f"Codificando columna categórica: {col}")
                le = LabelEncoder()
                try:
                    X[col] = le.fit_transform(X[col].astype(str).fillna('unknown'))
                    encoders[col] = le
                except Exception as e:
                    logger.warning(f"Error codificando {col}: {e}")
                    X[col] = 0
        
        # 5. Asegurar que todas las columnas sean numéricas
        for col in X.columns:
            if X[col].dtype == 'object':
                logger.warning(f"Columna {col} sigue siendo object, convirtiendo a numérico")
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1)
        
        # 6. Guardar encoders para uso posterior
        encoders_path = self.models_path / "label_encoders.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)
        
        logger.info(f"Tipos de datos después del preprocessing:")
        logger.info(f"{X.dtypes.value_counts()}")
        
        # División train/test estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"División completada - Train: {X_train.shape}, Test: {X_test.shape}")

        return X_train.values, X_test.values, y_train.values, y_test.values, feature_columns

    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Dict, float]:
        """Optimiza hiperparámetros usando Optuna."""
        logger.info(f"Iniciando optimización con {self.optimization_trials} trials...")

        def objective(trial):
            # Seleccionar tipo de modelo
            model_type = trial.suggest_categorical('model_type', ['lightgbm', 'gradient_boosting', 'logistic'])

            if model_type == 'lightgbm':
                params = {
                    'model_type': model_type,
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k != 'model_type'})

            elif model_type == 'gradient_boosting':
                params = {
                    'model_type': model_type,
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': 42
                }
                model = GradientBoostingClassifier(**{k: v for k, v in params.items() if k != 'model_type'})

            else:  # logistic
                params = {
                    'model_type': model_type,
                    'C': trial.suggest_float('C', 0.001, 100.0, log=True),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
                    'max_iter': 1000,
                    'random_state': 42
                }
                model = LogisticRegression(**{k: v for k, v in params.items() if k != 'model_type'})

            # Crear pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])

            # Validación cruzada
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

            return scores.mean()

        # Ejecutar optimización
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.optimization_trials)

        best_params = study.best_params
        best_score = study.best_value

        logger.info(f"Mejores parámetros: {best_params}")
        logger.info(f"Mejor AUC: {best_score:.4f}")

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
        """Evalúa el modelo entrenado."""
        logger.info("Evaluando modelo...")

        # Predicciones
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Métricas
        auc = roc_auc_score(y_test, y_pred_proba)

        # Log de evaluación
        logger.info(f"AUC Score: {auc:.4f}")

        metrics = {
            'auc': auc,
        }

        logger.info(f"Evaluación completada. Métricas: {metrics}")

        return metrics

    def _register_model(self, model: Pipeline, feature_names: List[str]) -> str:
        """Registra modelo en MLflow."""
        logger.info("Registrando modelo en MLflow...")

        # Nombre del modelo
        model_name = "sodai-recommendation-model"

        # Registrar en MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=pd.DataFrame(np.zeros((1, len(feature_names))))
        )

        # Crear URI del modelo
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        logger.info(f"Modelo registrado. URI: {model_uri}")

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
        model_path = self.models_path / "lightgbm_model.pkl"
        joblib.dump(model, model_path)

        # Guardar feature names
        feature_path = self.models_path / "feature_names.pkl"
        with open(feature_path, 'wb') as f:
            pickle.dump(feature_names, f)

        # Log artifacts en MLflow
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(feature_path))

        logger.info(f"Artifacts guardados en: {self.models_path}")


# Función de utilidad para uso independiente
def train_sodai_model(data_path: str, models_path: str,
                     experiment_name: str = "sodai-recommendation",
                     trials: int = 5) -> Dict[str, Any]:
    """
    Función de utilidad para entrenar modelo de SodAI.
    
    Args:
        data_path: Directorio de datos
        models_path: Directorio para guardar modelos
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

    parser = argparse.ArgumentParser(description='Entrena el modelo de recomendación para SodAI Drinks')
    parser.add_argument('--data-path', default='/opt/airflow/data', help='Directorio de datos')
    parser.add_argument('--models-path', default='/opt/airflow/models', help='Directorio para guardar modelos')
    parser.add_argument('--experiment-name', default='sodai-recommendation', help='Nombre del experimento')
    parser.add_argument('--trials', type=int, default=50, help='Número de trials de optimización')
    args = parser.parse_args()

    result = train_sodai_model(args.data_path, args.models_path, args.experiment_name, args.trials)
    print("Entrenamiento completado:")
    print(f"- AUC Test: {result['test_auc']:.4f}")
    print(f"- Mejor modelo: {result['best_params']['model_type']}")
    print(f"- URI modelo: {result['model_uri']}")
