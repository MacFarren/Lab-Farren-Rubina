""""""

SodAI Drinks - Entrenamiento de Modelo de Recomendación  Módulo de Entrenamiento de Modelos - SodAI Drinks Recommendation System

Sistema MLOps para recomendación de bebidas para SodAI====================================================================

Versión con LightGBM optimizada - MODO DEGRADADO sin MLflow

"""Este módulo maneja el entrenamiento y optimización de hiperparámetros

del modelo de recomendación usando MLflow para tracking y Optuna para

import loggingoptimización automática.

import json

import pickleFuncionalidades:

import joblib- Entrenamiento de modelos de clasificación para recomendación

from typing import Dict, Any, List, Tuple- Optimización de hiperparámetros con Optuna

from pathlib import Path- Tracking de experimentos con MLflow

from datetime import datetime- Validación cruzada estratificada

- Registro automático de modelos

import pandas as pd

import numpy as npAutor: SodAI Drinks MLOps Team

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score"""

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.pipeline import Pipelineimport pandas as pd

from sklearn.linear_model import LogisticRegressionimport numpy as np

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifierfrom datetime import datetime

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrixfrom typing import Dict, Any, List, Tuple, Optional

import optunaimport logging

import lightgbm as lgbfrom pathlib import Path

import pickle

# Configurar loggingimport json

logging.basicConfig(

    level=logging.INFO,# ML imports

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

)from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Suprimir warnings de Optuna para logs más limpiosfrom sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)from sklearn.pipeline import Pipeline

import lightgbm as lgb

# Variable para controlar MLflow (deshabilitado por problemas de conexión)import joblib

MLFLOW_ENABLED = False

# MLflow para tracking

#import mlflow (deshabilitado temporalmente)

class ModelTrainer:#import mlflow (deshabilitado temporalmente)

    """#import mlflow (deshabilitado temporalmente)

    Entrenador de modelos para el sistema de recomendación de SodAI.

    # Optuna para optimización

    Características:import optuna

    - Optimización automática de hiperparámetros con Optuna# Removemos MLflowCallback para evitar conflictos con runs activos

    - Integración con MLflow para tracking (cuando esté disponible)

    - Soporte para múltiples algoritmos de MLimport warnings

    - Validación cruzada estratificadawarnings.filterwarnings('ignore')

    - Pipeline completo de preprocessing

    """# Configurar logging

logging.basicConfig(level=logging.INFO)

    def __init__(self, data_path: str, models_path: str, logger = logging.getLogger(__name__)

                 experiment_name: str = "sodai-recommendation",

                 optimization_trials: int = 5, cv_folds: int = 2):class ModelTrainer:

        """    """

        Inicializa el entrenador de modelos.    Entrenador de modelos para el sistema de recomendación.

            """

        Args:    

            data_path: Ruta al directorio de datos    def __init__(self, data_path: str, models_path: str, 

            models_path: Ruta al directorio donde guardar modelos                 experiment_name: str = "sodai-recommendation-system",

            experiment_name: Nombre del experimento MLflow                 optimization_trials: int = 5, cv_folds: int = 2):  # Valores más rápidos por defecto

            optimization_trials: Número de trials para optimización        """

            cv_folds: Número de folds para validación cruzada        Inicializa el entrenador de modelos.

        """        

        self.data_path = Path(data_path)        Args:

        self.models_path = Path(models_path)            data_path: Ruta al directorio de datos

        self.experiment_name = experiment_name            models_path: Ruta para guardar modelos

        self.optimization_trials = optimization_trials            experiment_name: Nombre del experimento en MLflow

        self.cv_folds = cv_folds            optimization_trials: Número de trials para optimización (reducido para velocidad)

                    cv_folds: Número de folds para validación cruzada (reducido para velocidad)

        # Crear directorios si no existen        """

        self.models_path.mkdir(parents=True, exist_ok=True)        self.data_path = Path(data_path)

                self.models_path = Path(models_path)

        # Configurar MLflow solo si está habilitado        self.experiment_name = experiment_name

        if MLFLOW_ENABLED:        self.optimization_trials = optimization_trials

            self._setup_mlflow()        self.cv_folds = cv_folds

        

    def _setup_mlflow(self):        # Crear directorios si no existen

        """Configura MLflow para tracking."""        self.models_path.mkdir(exist_ok=True, parents=True)

        try:        

            import mlflow        # Configurar MLflow

            import mlflow.sklearn        self._setup_mlflow()

            logger.info(f"Configurando MLflow - Experimento: {self.experiment_name}")        

            mlflow.set_experiment(self.experiment_name)        logger.info(f"ModelTrainer inicializado: {experiment_name}")

        except Exception as e:    

            logger.warning(f"MLflow no disponible: {e}")    def _setup_mlflow(self):

            global MLFLOW_ENABLED        """Configura MLflow para tracking."""

            MLFLOW_ENABLED = False        try:

            #mlflow.set_experiment(self.experiment_name)

    def train_recommendation_model(self) -> Dict[str, Any]:            logger.info(f"Experimento MLflow configurado: {self.experiment_name}")

        """        except Exception as e:

        Ejecuta el pipeline completo de entrenamiento.            logger.warning(f"Error configurando MLflow: {e}")

            

        Returns:    def train_recommendation_model(self) -> Dict[str, Any]:

            Dict con resultados del entrenamiento        """

        """        Entrena el modelo de recomendación con optimización completa.

        logger.info("🚀 Iniciando entrenamiento del modelo de recomendación...")        

        Returns:

        if MLFLOW_ENABLED:            Diccionario con resultados del entrenamiento

            import mlflow        """

            with mlflow.start_run():        logger.info("Iniciando entrenamiento del modelo de recomendación...")

                return self._train_with_mlflow()        

        else:        with #mlflow.start_run(run_name=f"recommendation_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

            return self._train_without_mlflow()            # 1. Cargar y preparar datos

            X_train, X_test, y_train, y_test, feature_names = self._prepare_training_data()

    def _train_without_mlflow(self) -> Dict[str, Any]:            

        """Entrenamiento sin MLflow"""            #mlflow.log_params({

        logger.info("⚠️ Entrenamiento en modo degradado (sin MLflow)")                'train_samples': len(X_train),

                        'test_samples': len(X_test), 

        # 1. Preparar datos                'num_features': len(feature_names),

        X_train, X_test, y_train, y_test, feature_names = self._prepare_training_data()                'positive_ratio_train': y_train.mean(),

                'positive_ratio_test': y_test.mean()

        # 2. Optimizar hiperparámetros            })

        logger.info("🔍 Optimizando hiperparámetros...")            

        best_params, best_score = self._optimize_hyperparameters(X_train, y_train)            # 2. Optimización de hiperparámetros

            best_params, best_score = self._optimize_hyperparameters(X_train, y_train)

        # 3. Entrenar modelo final            

        final_model = self._train_final_model(X_train, y_train, best_params)            #mlflow.log_params(best_params)

            #mlflow.log_metric('best_cv_auc', best_score)

        # 4. Evaluar modelo            

        evaluation_results = self._evaluate_model(final_model, X_test, y_test)            # 3. Entrenar modelo final

            final_model = self._train_final_model(X_train, y_train, best_params)

        # 5. Guardar artifacts            

        self._save_training_artifacts(final_model, feature_names, best_params, evaluation_results)            # 4. Evaluar modelo

            evaluation_results = self._evaluate_model(final_model, X_test, y_test)

        results = {            

            'model_uri': 'local_file',  # Sin MLflow            #mlflow.log_metrics(evaluation_results)

            'best_params': best_params,            

            'best_auc': best_score,            # 5. Registrar modelo en MLflow

            'test_auc': evaluation_results['auc'],            model_uri = self._register_model(final_model, feature_names)

            'feature_names': feature_names,            

            'training_timestamp': datetime.now().isoformat()            # 6. Guardar artifacts adicionales

        }            self._save_training_artifacts(final_model, feature_names, best_params, evaluation_results)

            

        logger.info(f"✅ Entrenamiento completado. AUC: {evaluation_results['auc']:.4f}")            results = {

                'model_uri': model_uri,

        return results                'best_params': best_params,

                'best_auc': best_score,

    def _train_with_mlflow(self) -> Dict[str, Any]:                'test_auc': evaluation_results['auc'],

        """Entrenamiento con MLflow completo"""                'feature_names': feature_names,

        import mlflow                'training_timestamp': datetime.now().isoformat()

                    }

        # 1. Preparar datos            

        X_train, X_test, y_train, y_test, feature_names = self._prepare_training_data()            logger.info(f"✅ Entrenamiento completado. AUC: {evaluation_results['auc']:.4f}")

            

        # 2. Optimizar hiperparámetros            return results

        logger.info("🔍 Optimizando hiperparámetros...")    

        best_params, best_score = self._optimize_hyperparameters(X_train, y_train)    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:

        """Prepara los datos para entrenamiento."""

        # Log de parámetros y métricas en MLflow        logger.info("Preparando datos de entrenamiento...")

        mlflow.log_params(best_params)        

        mlflow.log_metric('best_cv_auc', best_score)        # Cargar dataset de entrenamiento

        features_path = self.data_path / "features" / "training_dataset.parquet"

        # 3. Entrenar modelo final        

        final_model = self._train_final_model(X_train, y_train, best_params)        if not features_path.exists():

            raise FileNotFoundError(f"Dataset de entrenamiento no encontrado: {features_path}")

        # 4. Evaluar modelo        

        evaluation_results = self._evaluate_model(final_model, X_test, y_test)        df = pd.read_parquet(features_path)

        logger.info(f"Dataset cargado: {df.shape}")

        mlflow.log_metrics(evaluation_results)        

        # Separar features y target

        # 5. Registrar modelo en MLflow        feature_columns = [col for col in df.columns if col not in ['customer_id', 'product_id', 'target']]

        model_uri = self._register_model(final_model, feature_names)        X = df[feature_columns]

        y = df['target']

        # 6. Guardar artifacts adicionales        

        self._save_training_artifacts(final_model, feature_names, best_params, evaluation_results)        # Manejar valores faltantes

        X = X.fillna(0)

        results = {        

            'model_uri': model_uri,        # Encoding de variables categóricas si las hay

            'best_params': best_params,        categorical_columns = X.select_dtypes(include=['object', 'category']).columns

            'best_auc': best_score,        for col in categorical_columns:

            'test_auc': evaluation_results['auc'],            le = LabelEncoder()

            'feature_names': feature_names,            X[col] = le.fit_transform(X[col].fillna('unknown'))

            'training_timestamp': datetime.now().isoformat()        

        }        # División train/test estratificada

        from sklearn.model_selection import train_test_split

        logger.info(f"✅ Entrenamiento completado. AUC: {evaluation_results['auc']:.4f}")        

        X_train, X_test, y_train, y_test = train_test_split(

        return results            X, y, test_size=0.2, random_state=42, stratify=y

        )

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:        

        """Prepara los datos para entrenamiento con preprocessing mejorado."""        logger.info(f"División completada: Train {X_train.shape}, Test {X_test.shape}")

        logger.info("Preparando datos de entrenamiento...")        

        return X_train.values, X_test.values, y_train.values, y_test.values, feature_columns

        # Cargar dataset de entrenamiento    

        features_path = self.data_path / "features" / "training_dataset.parquet"    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Dict, float]:

        """Optimiza hiperparámetros usando Optuna."""

        if not features_path.exists():        logger.info(f"Iniciando optimización de hiperparámetros ({self.optimization_trials} trials)...")

            raise FileNotFoundError(f"Dataset de entrenamiento no encontrado: {features_path}")        

        # No usar MLflow callback para evitar conflictos con el run activo

        df = pd.read_parquet(features_path)        

        logger.info(f"Dataset cargado: {df.shape}")        def objective(trial):

            # Espacio de búsqueda simplificado para mayor velocidad

        # Separar features y target            model_type = trial.suggest_categorical('model_type', ['lightgbm'])  # Removimos gradient boosting

        feature_columns = [col for col in df.columns if col not in ['customer_id', 'product_id', 'target']]            

        X = df[feature_columns].copy()            if model_type == 'lightgbm':

        y = df['target']                params = {

                    'model_type': 'lightgbm',

        # PREPROCESSING MEJORADO PARA LIGHTGBM                    'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),  # Valores fijos

        logger.info("Aplicando preprocessing para LightGBM...")                    'max_depth': trial.suggest_categorical('max_depth', [5, 10, 15]),  # Valores fijos

                            'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10]),  # Valores fijos

        # 1. Identificar tipos de columnas                    'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 4]),  # Valores fijos

        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()                    'random_state': 42,

        datetime_columns = [col for col in X.columns if 'date' in col.lower() or 'time' in col.lower()]                      'verbosity': -1

                        }

        logger.info(f"Columnas categóricas detectadas: {categorical_columns}")                model = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k != 'model_type'})

        logger.info(f"Columnas de fecha detectadas: {datetime_columns}")                

                    else:  # logistic regression

        # 2. Manejar valores faltantes primero                params = {

        X = X.fillna(-1)                    'model_type': 'logistic',

                            'C': trial.suggest_categorical('C', [0.1, 1.0, 10.0]),  # Valores fijos

        # 3. Convertir columnas de fecha a timestamp numérico                    'penalty': trial.suggest_categorical('penalty', ['l2']),  # Solo L2 para simplicidad

        for col in datetime_columns:                    'solver': 'lbfgs',  # Más rápido que liblinear para L2

            if col in X.columns:                    'random_state': 42,

                try:                      'verbosity': -1,

                    X[col] = pd.to_datetime(X[col], errors='coerce').astype('int64') / 10**9                    'max_iter': 1000  # Límite de iteraciones para velocidad

                    X[col] = X[col].fillna(-1)                }

                except Exception as e:                model = LogisticRegression(**{k: v for k, v in params.items() if k != 'model_type'})

                    logger.warning(f"No se pudo convertir columna de fecha {col}: {e}")            

                    X[col] = X[col].astype('float64')            # Validación cruzada estratificada

                    cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        # 4. Encoding de variables categóricas            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=2)  # Limitamos jobs para evitar sobrecarga

        encoders = {}            

        for col in categorical_columns:            return scores.mean()

            if col in X.columns:        

                logger.info(f"Codificando columna categórica: {col}")        # Crear estudio de optimización

                le = LabelEncoder()        study = optuna.create_study(

                try:            direction='maximize',

                    X[col] = le.fit_transform(X[col].astype(str).fillna('unknown'))            sampler=optuna.samplers.TPESampler(seed=42)

                    encoders[col] = le        )

                except Exception as e:        

                    logger.warning(f"Error codificando {col}: {e}")        # Ejecutar optimización con timeout de 10 minutos máximo

                    X[col] = 0        study.optimize(objective, n_trials=self.optimization_trials, timeout=600)

                

        # 5. Asegurar que todas las columnas sean numéricas        best_params = study.best_params

        for col in X.columns:        best_score = study.best_value

            if X[col].dtype == 'object':        

                logger.warning(f"Columna {col} sigue siendo object, convirtiendo a numérico")        logger.info(f"✅ Optimización completada. Mejor AUC: {best_score:.4f}")

                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1)        logger.info(f"Mejores parámetros: {best_params}")

                

        # 6. Guardar encoders para uso posterior        return best_params, best_score

        encoders_path = self.models_path / "label_encoders.pkl"    

        with open(encoders_path, 'wb') as f:    def _train_final_model(self, X_train: np.ndarray, y_train: np.ndarray, 

            pickle.dump(encoders, f)                          best_params: Dict) -> Pipeline:

                """Entrena el modelo final con los mejores parámetros."""

        logger.info(f"Tipos de datos después del preprocessing:")        logger.info("Entrenando modelo final...")

        logger.info(f"{X.dtypes.value_counts()}")        

                # Crear modelo según tipo seleccionado

        # División train/test estratificada        model_type = best_params['model_type']

        X_train, X_test, y_train, y_test = train_test_split(        model_params = {k: v for k, v in best_params.items() if k != 'model_type'}

            X, y, test_size=0.2, random_state=42, stratify=y        

        )        if model_type == 'lightgbm':

            model = lgb.LGBMClassifier(**model_params)

        logger.info(f"División completada - Train: {X_train.shape}, Test: {X_test.shape}")        elif model_type == 'gradient_boosting':

            model = GradientBoostingClassifier(**model_params)

        return X_train.values, X_test.values, y_train.values, y_test.values, feature_columns        else:  # logistic

            model = LogisticRegression(**model_params)

    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Dict, float]:        

        """Optimiza hiperparámetros usando Optuna."""        # Crear pipeline con escalado

        logger.info(f"Iniciando optimización con {self.optimization_trials} trials...")        pipeline = Pipeline([

            ('scaler', StandardScaler()),

        def objective(trial):            ('classifier', model)

            # Seleccionar tipo de modelo        ])

            model_type = trial.suggest_categorical('model_type', ['lightgbm', 'gradient_boosting', 'logistic'])        

        # Entrenar

            if model_type == 'lightgbm':        pipeline.fit(X_train, y_train)

                params = {        

                    'model_type': model_type,        logger.info(f"Modelo final entrenado: {model_type}")

                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),        

                    'max_depth': trial.suggest_int('max_depth', 3, 10),        return pipeline

                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),    

                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),    def _evaluate_model(self, model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:

                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),        """Evalúa el modelo en datos de test."""

                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),        logger.info("Evaluando modelo...")

                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),        

                    'random_state': 42,        # Predicciones

                    'verbose': -1        y_pred = model.predict(X_test)

                }        y_pred_proba = model.predict_proba(X_test)[:, 1]

                model = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k != 'model_type'})        

        # Métricas

            elif model_type == 'gradient_boosting':        metrics = {

                params = {            'auc': roc_auc_score(y_test, y_pred_proba),

                    'model_type': model_type,            'precision': precision_score(y_test, y_pred),

                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),            'recall': recall_score(y_test, y_pred),

                    'max_depth': trial.suggest_int('max_depth', 3, 8),            'f1_score': f1_score(y_test, y_pred)

                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),        }

                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),        

                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),        logger.info(f"Métricas de evaluación: {metrics}")

                    'random_state': 42        

                }        return metrics

                model = GradientBoostingClassifier(**{k: v for k, v in params.items() if k != 'model_type'})    

    def _register_model(self, model: Pipeline, feature_names: List[str]) -> str:

            else:  # logistic        """Registra el modelo en #mlflow."""

                params = {        logger.info("Registrando modelo en #mlflow...")

                    'model_type': model_type,        

                    'C': trial.suggest_float('C', 0.001, 100.0, log=True),        # Guardar modelo

                    'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),        model_name = "sodai-recommendation-model"

                    'max_iter': 1000,        

                    'random_state': 42        # Registrar en MLflow

                }        #mlflow.sklearn.log_model(

                model = LogisticRegression(**{k: v for k, v in params.items() if k != 'model_type'})            sk_model=model,

            artifact_path="model",

            # Crear pipeline            registered_model_name=model_name,

            pipeline = Pipeline([            input_example=np.zeros((1, len(feature_names)))

                ('scaler', StandardScaler()),        )

                ('classifier', model)        

            ])        # Crear URI del modelo

        run_id = #mlflow.active_run().info.run_id

            # Validación cruzada        model_uri = f"runs:/{run_id}/model"

            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)        

            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)        logger.info(f"Modelo registrado: {model_uri}")

        

            return scores.mean()        return model_uri

    

        # Ejecutar optimización    def _save_training_artifacts(self, model: Pipeline, feature_names: List[str], 

        study = optuna.create_study(direction='maximize')                               best_params: Dict, evaluation_results: Dict) -> None:

        study.optimize(objective, n_trials=self.optimization_trials)        """Guarda artifacts adicionales del entrenamiento."""

        logger.info("Guardando artifacts de entrenamiento...")

        best_params = study.best_params        

        best_score = study.best_value        # Metadatos del modelo

        model_metadata = {

        logger.info(f"Mejores parámetros: {best_params}")            'training_timestamp': datetime.now().isoformat(),

        logger.info(f"Mejor AUC: {best_score:.4f}")            'feature_names': feature_names,

            'best_params': best_params,

        return best_params, best_score            'evaluation_results': evaluation_results,

            'model_type': best_params.get('model_type', 'unknown'),

    def _train_final_model(self, X_train: np.ndarray, y_train: np.ndarray,             'num_features': len(feature_names)

                          best_params: Dict) -> Pipeline:        }

        """Entrena el modelo final con los mejores parámetros."""        

        logger.info("Entrenando modelo final...")        # Guardar metadatos

        metadata_path = self.models_path / "model_metadata.json"

        # Crear modelo según tipo seleccionado        with open(metadata_path, 'w') as f:

        model_type = best_params['model_type']            json.dump(model_metadata, f, indent=2)

        model_params = {k: v for k, v in best_params.items() if k != 'model_type'}        

        # Guardar modelo localmente también

        if model_type == 'lightgbm':        model_path = self.models_path / "recommendation_model.pkl"

            model = lgb.LGBMClassifier(**model_params)        joblib.dump(model, model_path)

        elif model_type == 'gradient_boosting':        

            model = GradientBoostingClassifier(**model_params)        # Guardar feature names

        else:  # logistic        feature_path = self.models_path / "feature_names.pkl"

            model = LogisticRegression(**model_params)        with open(feature_path, 'wb') as f:

            pickle.dump(feature_names, f)

        # Crear pipeline con escalado        

        pipeline = Pipeline([        # Log artifacts en MLflow

            ('scaler', StandardScaler()),        #mlflow.log_artifact(str(metadata_path))

            ('classifier', model)        #mlflow.log_artifact(str(model_path))

        ])        #mlflow.log_artifact(str(feature_path))

        

        # Entrenar        logger.info(f"Artifacts guardados en {self.models_path}")

        pipeline.fit(X_train, y_train)

# Función de utilidad para uso independiente

        logger.info(f"Modelo final entrenado: {model_type}")def train_sodai_model(data_path: str, models_path: str, 

                     experiment_name: str = "sodai-recommendation",

        return pipeline                     trials: int = 50) -> Dict[str, Any]:

    """

    def _evaluate_model(self, model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:    Función de conveniencia para entrenar modelo de SodAI.

        """Evalúa el modelo entrenado."""    

        logger.info("Evaluando modelo...")    Args:

        data_path: Ruta al directorio de datos

        # Predicciones        models_path: Ruta para guardar modelos

        y_pred_proba = model.predict_proba(X_test)[:, 1]        experiment_name: Nombre del experimento

        y_pred = model.predict(X_test)        trials: Número de trials de optimización

        

        # Métricas    Returns:

        auc = roc_auc_score(y_test, y_pred_proba)        Diccionario con resultados del entrenamiento

    """

        # Log de evaluación    trainer = ModelTrainer(data_path, models_path, experiment_name, trials)

        logger.info(f"AUC Score: {auc:.4f}")    return trainer.train_recommendation_model()



        metrics = {if __name__ == "__main__":

            'auc': auc,    # Ejemplo de uso

        }    import argparse

    

        logger.info(f"Evaluación completada. Métricas: {metrics}")    parser = argparse.ArgumentParser(description='Entrenador de modelos SodAI Drinks')

    parser.add_argument('--data-path', required=True, help='Ruta al directorio de datos')

        return metrics    parser.add_argument('--models-path', required=True, help='Ruta para guardar modelos')

    parser.add_argument('--experiment-name', default='sodai-recommendation', help='Nombre del experimento')

    def _register_model(self, model: Pipeline, feature_names: List[str]) -> str:    parser.add_argument('--trials', type=int, default=50, help='Número de trials de optimización')

        """Registra modelo en MLflow."""    args = parser.parse_args()

        if not MLFLOW_ENABLED:    

            return "mlflow_disabled"    result = train_sodai_model(args.data_path, args.models_path, args.experiment_name, args.trials)

                print("Entrenamiento completado:")

        import mlflow    print(f"- AUC en test: {result['test_auc']:.4f}")

        import mlflow.sklearn    print(f"- Mejor modelo: {result['best_params']['model_type']}")

            print(f"- URI del modelo: {result['model_uri']}")

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

        # Log artifacts en MLflow si está habilitado
        if MLFLOW_ENABLED:
            import mlflow
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