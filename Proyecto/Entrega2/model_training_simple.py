"""
SodAI Drinks - Entrenamiento con LightGBM
Versión simplificada para el DAG de Airflow
"""

import logging
import json
import pickle
import joblib
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import optuna
import lightgbm as lgb

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModelTrainer:
    def __init__(self, data_path: str, models_path: str, optimization_trials: int = 5, cv_folds: int = 2):
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.optimization_trials = optimization_trials
        self.cv_folds = cv_folds
        self.models_path.mkdir(parents=True, exist_ok=True)

    def train_recommendation_model(self):
        """Pipeline completo de entrenamiento"""
        logger.info("🚀 Iniciando entrenamiento LightGBM...")
        
        # 1. Preparar datos
        X_train, X_test, y_train, y_test, feature_names = self._prepare_training_data()
        
        # 2. Optimizar hiperparámetros
        best_params, best_score = self._optimize_hyperparameters(X_train, y_train)
        
        # 3. Entrenar modelo final
        final_model = self._train_final_model(X_train, y_train, best_params)
        
        # 4. Evaluar modelo
        evaluation_results = self._evaluate_model(final_model, X_test, y_test)
        
        # 5. Guardar artifacts
        self._save_training_artifacts(final_model, feature_names, best_params, evaluation_results)
        
        results = {
            'best_params': best_params,
            'best_auc': best_score,
            'test_auc': evaluation_results['auc'],
            'feature_names': feature_names,
            'training_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Entrenamiento completado. AUC: {evaluation_results['auc']:.4f}")
        return results

    def _prepare_training_data(self):
        """Prepara los datos con preprocessing para LightGBM"""
        logger.info("Preparando datos de entrenamiento...")
        
        # Cargar dataset
        features_path = self.data_path / "features" / "training_dataset.parquet"
        if not features_path.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {features_path}")
            
        df = pd.read_parquet(features_path)
        logger.info(f"Dataset cargado: {df.shape}")
        
        # Separar features y target
        feature_columns = [col for col in df.columns if col not in ['customer_id', 'product_id', 'target']]
        X = df[feature_columns].copy()
        y = df['target']
        
        # PREPROCESSING PARA LIGHTGBM
        logger.info("Aplicando preprocessing para LightGBM...")
        
        # Identificar tipos de columnas
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = [col for col in X.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        logger.info(f"Columnas categóricas: {categorical_columns}")
        logger.info(f"Columnas de fecha: {datetime_columns}")
        
        # Manejar valores faltantes
        X = X.fillna(-1)
        
        # Convertir fechas a timestamp numérico
        for col in datetime_columns:
            if col in X.columns:
                try:
                    X[col] = pd.to_datetime(X[col], errors='coerce').astype('int64') / 10**9
                    X[col] = X[col].fillna(-1)
                    logger.info(f"Convertida fecha {col}")
                except Exception as e:
                    logger.warning(f"Error convirtiendo fecha {col}: {e}")
                    X[col] = X[col].astype('float64')
        
        # Encoding de variables categóricas
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
        
        # Asegurar tipos numéricos
        for col in X.columns:
            if X[col].dtype == 'object':
                logger.warning(f"Convirtiendo {col} a numérico")
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1)
        
        logger.info(f"Tipos finales: {X.dtypes.value_counts()}")
        
        # Guardar encoders
        encoders_path = self.models_path / "label_encoders.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train.values, X_test.values, y_train.values, y_test.values, feature_columns

    def _optimize_hyperparameters(self, X_train, y_train):
        """Optimiza hiperparámetros con Optuna"""
        logger.info(f"Optimizando con {self.optimization_trials} trials...")
        
        def objective(trial):
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
            
            # Pipeline con scaling
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

    def _train_final_model(self, X_train, y_train, best_params):
        """Entrena el modelo final"""
        logger.info("Entrenando modelo final...")
        
        model_type = best_params['model_type']
        model_params = {k: v for k, v in best_params.items() if k != 'model_type'}
        
        if model_type == 'lightgbm':
            model = lgb.LGBMClassifier(**model_params)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(**model_params)
        else:  # logistic
            model = LogisticRegression(**model_params)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        logger.info(f"Modelo final entrenado: {model_type}")
        
        return pipeline

    def _evaluate_model(self, model, X_test, y_test):
        """Evalúa el modelo"""
        logger.info("Evaluando modelo...")
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"AUC Score: {auc:.4f}")
        
        return {'auc': auc}

    def _save_training_artifacts(self, model, feature_names, best_params, evaluation_results):
        """Guarda artifacts del entrenamiento"""
        logger.info("Guardando artifacts...")
        
        # Metadatos
        metadata = {
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
            json.dump(metadata, f, indent=2)
        
        # Guardar modelo
        model_path = self.models_path / "lightgbm_model.pkl"
        joblib.dump(model, model_path)
        
        # Guardar feature names
        feature_path = self.models_path / "feature_names.pkl"
        with open(feature_path, 'wb') as f:
            pickle.dump(feature_names, f)
        
        logger.info(f"Artifacts guardados en: {self.models_path}")

def train_sodai_model(data_path: str, models_path: str, experiment_name: str = "sodai-recommendation", trials: int = 5):
    """Función principal de entrenamiento"""
    trainer = ModelTrainer(data_path, models_path, trials)
    return trainer.train_recommendation_model()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Entrena el modelo LightGBM para SodAI')
    parser.add_argument('--data-path', default='/opt/airflow/data', help='Directorio de datos')
    parser.add_argument('--models-path', default='/opt/airflow/models', help='Directorio para modelos')
    parser.add_argument('--experiment-name', default='sodai-recommendation', help='Nombre del experimento')
    parser.add_argument('--trials', type=int, default=5, help='Número de trials de optimización')
    args = parser.parse_args()
    
    result = train_sodai_model(args.data_path, args.models_path, args.experiment_name, args.trials)
    print("Entrenamiento completado:")
    print(f"- AUC Test: {result['test_auc']:.4f}")
    print(f"- Mejor modelo: {result['best_params']['model_type']}")