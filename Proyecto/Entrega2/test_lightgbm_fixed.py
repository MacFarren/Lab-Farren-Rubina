"""
Test script para verificar que LightGBM funcione sin MLflow
"""

import logging
import pickle
import joblib
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_lightgbm_training():
    """Test rápido de entrenamiento con LightGBM"""
    logger.info("🚀 Iniciando test de entrenamiento LightGBM...")
    
    # Rutas
    data_path = Path("/opt/airflow/data")
    models_path = Path("/opt/airflow/models")
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Cargar dataset
    features_path = data_path / "features" / "training_dataset.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {features_path}")
        
    df = pd.read_parquet(features_path)
    logger.info(f"Dataset cargado: {df.shape}")
    
    # Separar features y target
    feature_columns = [col for col in df.columns if col not in ['customer_id', 'product_id', 'target']]
    X = df[feature_columns].copy()
    y = df['target']
    
    # PREPROCESSING MEJORADO
    logger.info("Aplicando preprocessing para LightGBM...")
    
    # 1. Identificar tipos de columnas
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_columns = [col for col in X.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    logger.info(f"Columnas categóricas: {categorical_columns}")
    logger.info(f"Columnas de fecha: {datetime_columns}")
    
    # 2. Manejar valores faltantes
    X = X.fillna(-1)
    
    # 3. Convertir fechas a timestamp numérico
    for col in datetime_columns:
        if col in X.columns:
            try:
                X[col] = pd.to_datetime(X[col], errors='coerce').astype('int64') / 10**9
                X[col] = X[col].fillna(-1)
                logger.info(f"Convertida fecha {col} a timestamp")
            except Exception as e:
                logger.warning(f"Error convirtiendo fecha {col}: {e}")
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
    
    # 5. Asegurar tipos numéricos
    for col in X.columns:
        if X[col].dtype == 'object':
            logger.warning(f"Convirtiendo {col} a numérico")
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1)
    
    logger.info(f"Tipos finales: {X.dtypes.value_counts()}")
    logger.info(f"Features shape: {X.shape}")
    
    # 6. División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 7. Entrenar LightGBM
    logger.info("Entrenando LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=50,
        random_state=42,
        verbose=-1
    )
    
    # Pipeline con scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    # Entrenar
    start_time = datetime.now()
    pipeline.fit(X_train, y_train)
    training_time = datetime.now() - start_time
    
    # Evaluar
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Guardar modelo
    model_path = models_path / "lightgbm_model.pkl"
    joblib.dump(pipeline, model_path)
    
    # Guardar encoders
    encoders_path = models_path / "label_encoders.pkl"
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    
    # Resultados
    logger.info("✅ ENTRENAMIENTO COMPLETADO!")
    logger.info(f"📊 AUC Score: {auc_score:.4f}")
    logger.info(f"⏱️ Tiempo de entrenamiento: {training_time}")
    logger.info(f"🔢 Features procesadas: {len(feature_columns)}")
    logger.info(f"📈 Samples de entrenamiento: {len(X_train)}")
    logger.info(f"💾 Modelo guardado en: {model_path}")
    
    return {
        'auc_score': auc_score,
        'training_time': str(training_time),
        'num_features': len(feature_columns),
        'num_samples': len(X_train),
        'model_path': str(model_path)
    }

if __name__ == "__main__":
    try:
        result = test_lightgbm_training()
        print("\n" + "="*50)
        print("RESULTADO DEL TEST:")
        print(f"AUC Score: {result['auc_score']:.4f}")
        print(f"Tiempo: {result['training_time']}")
        print(f"Features: {result['num_features']}")
        print(f"Samples: {result['num_samples']}")
        print("="*50)
    except Exception as e:
        logger.error(f"Error en el test: {e}")
        raise