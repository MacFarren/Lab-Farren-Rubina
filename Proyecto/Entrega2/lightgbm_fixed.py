"""
Script de entrenamiento LightGBM con preprocesamiento completo
"""

import os
import sys
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightGBMTrainer:
    def __init__(self, data_path: str, models_path: str):
        self.data_path = data_path
        self.models_path = models_path
        
        # Crear directorios si no existen
        os.makedirs(self.models_path, exist_ok=True)
        
        logger.info("LightGBMTrainer inicializado exitosamente")
    
    def load_and_preprocess_data(self):
        """Cargar y preprocesar datos"""
        features_path = os.path.join(self.data_path, 'features/training_dataset.parquet')
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"No se encontraron features en {features_path}")
        
        logger.info(f"Cargando features desde {features_path}")
        df = pd.read_parquet(features_path)
        
        # Identificar columna target
        target_cols = ['target', 'compra_siguiente', 'y']
        target_col = None
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            raise ValueError(f"No se encontró columna target. Columnas disponibles: {list(df.columns)}")
        
        logger.info(f"Usando '{target_col}' como variable target")
        
        # Separar features y target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Preprocesar features
        X_processed = self.preprocess_features(X)
        
        logger.info(f"Datos procesados: {X_processed.shape[0]} filas, {X_processed.shape[1]} features")
        logger.info(f"Distribución target: {y.value_counts().to_dict()}")
        
        return X_processed, y
    
    def preprocess_features(self, X):
        """Preprocesar features para LightGBM"""
        X_processed = X.copy()
        
        # Manejar valores faltantes
        X_processed = X_processed.fillna(-1)
        
        # Convertir columnas object a numéricas
        object_cols = X_processed.select_dtypes(include=['object']).columns
        
        if len(object_cols) > 0:
            logger.info(f"Convirtiendo columnas categóricas: {list(object_cols)}")
            
            for col in object_cols:
                le = LabelEncoder()
                # Manejar valores faltantes en LabelEncoder
                mask = X_processed[col].notna()
                if mask.any():
                    X_processed.loc[mask, col] = le.fit_transform(X_processed.loc[mask, col].astype(str))
                    # Guardar encoder para futuro uso
                    joblib.dump(le, os.path.join(self.models_path, f'encoder_{col}.pkl'))
                
                # Convertir a numérico
                X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(-1)
        
        # Convertir columnas datetime si existen
        datetime_cols = X_processed.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            logger.info(f"Convirtiendo columnas datetime: {list(datetime_cols)}")
            for col in datetime_cols:
                X_processed[col] = X_processed[col].astype('int64') // 10**9  # convertir a timestamp
        
        # Asegurar que todas las columnas son numéricas
        for col in X_processed.columns:
            X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(-1)
        
        # Verificar tipos de datos finales
        non_numeric = X_processed.select_dtypes(exclude=['int', 'float', 'bool']).columns
        if len(non_numeric) > 0:
            logger.warning(f"Columnas aún no numéricas: {list(non_numeric)}")
            for col in non_numeric:
                X_processed[col] = X_processed[col].astype('float64')
        
        logger.info("Preprocesamiento completado exitosamente")
        return X_processed
    
    def train_model(self):
        """Entrenar modelo LightGBM"""
        try:
            # Cargar y preprocesar datos
            X, y = self.load_and_preprocess_data()
            
            # Split de datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info("Iniciando entrenamiento con LightGBM...")
            
            # Parámetros optimizados para velocidad
            model = lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                boosting_type='gbdt',
                num_leaves=15,  # Reducido para velocidad
                max_depth=4,    # Reducido para velocidad
                learning_rate=0.1,
                n_estimators=50,  # Muy reducido para testing
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                force_row_wise=True  # Para evitar warnings
            )
            
            # Entrenar
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='binary_logloss',
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Evaluar
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"🎯 AUC Score: {auc:.4f}")
            
            # Guardar modelo
            model_path = os.path.join(self.models_path, 'lightgbm_model.pkl')
            joblib.dump(model, model_path)
            logger.info(f"💾 Modelo guardado en: {model_path}")
            
            return {
                'model_path': model_path,
                'auc_score': auc,
                'status': 'success',
                'n_features': X.shape[1],
                'n_samples': X.shape[0]
            }
            
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'failed',
                'error': str(e)
            }

def run_training():
    """Función principal de entrenamiento"""
    logger.info("🚀 Iniciando entrenamiento LightGBM...")
    
    trainer = LightGBMTrainer('/opt/airflow/data', '/opt/airflow/models')
    result = trainer.train_model()
    
    if result['status'] == 'success':
        logger.info("✅ Entrenamiento completado exitosamente!")
        logger.info(f"📊 Resultados:")
        logger.info(f"   - AUC Score: {result['auc_score']:.4f}")
        logger.info(f"   - Features: {result['n_features']}")
        logger.info(f"   - Muestras: {result['n_samples']}")
        logger.info(f"   - Modelo: {result['model_path']}")
    else:
        logger.error("❌ Entrenamiento falló!")
        logger.error(f"Error: {result['error']}")
        raise Exception(f"Entrenamiento falló: {result['error']}")
    
    return result

if __name__ == "__main__":
    run_training()