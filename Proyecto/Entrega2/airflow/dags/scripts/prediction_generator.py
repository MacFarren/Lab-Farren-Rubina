"""
Generador de Predicciones OPTIMIZADO - SodAI Drinks
===================================================

Versi?n ultra-r?pida para demo/testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from pathlib import Path
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionGenerator:
    def __init__(self, data_path: str, model_uri: str, output_path: str):
        self.data_path = Path(data_path)
        self.model_uri = model_uri
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)

    def generate_weekly_predictions(self) -> Dict[str, Any]:
        """Genera predicciones semanales - VERSI?N ULTRA R?PIDA."""
        logger.info(" Generando predicciones OPTIMIZADAS (demo mode)...")
        
        start_time = datetime.now()
        
        try:
            # 1. Cargar modelo
            logger.info(" Cargando modelo...")
            model = joblib.load(self.model_uri.replace('file://', ''))
            
            # 2. Cargar datos base (muestra m?nima)
            features_path = self.data_path / "features" / "training_dataset.parquet"       
            df = pd.read_parquet(features_path)
            
            # 3. DEMO MODE: Solo 5 clientes x 5 productos = 25 predicciones
            logger.info(" DEMO MODE: Generando muestra ultra-peque?a...")
            unique_customers = df['customer_id'].unique()[:5]
            unique_products = df['product_id'].unique()[:5]
            
            logger.info(f"Scope: {len(unique_customers)} clientes x {len(unique_products)} productos = {len(unique_customers) * len(unique_products)} predicciones")
            
            # 4. Crear grid de predicci?n m?nima
            prediction_grid = []
            for customer in unique_customers:
                for product in unique_products:
                    prediction_grid.append({'customer_id': customer, 'product_id': product})
            
            prediction_df = pd.DataFrame(prediction_grid)
            
            # 5. Features simplificadas (valores promedio)
            feature_columns = [col for col in df.columns if col not in ['customer_id', 'product_id', 'target']]
            feature_means = df[feature_columns].mean()
            
            for col in feature_columns:
                prediction_df[col] = feature_means[col]
            
            # 6. Predecir
            logger.info(" Generando predicciones...")
            X_pred = prediction_df[feature_columns].fillna(0)
            predictions = model.predict_proba(X_pred)[:, 1]
            prediction_df['prediction_score'] = predictions
            
            # 7. Top recomendaciones (todas las que tenemos en demo)
            top_predictions = prediction_df.nlargest(25, 'prediction_score')
            
            # 8. Guardar resultados
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prediction_file = self.output_path / f"predictions_demo_{timestamp}.parquet"        
            top_predictions.to_parquet(prediction_file, index=False)
            
            # 9. Estad?sticas
            top_products = prediction_df.groupby('product_id')['prediction_score'].mean().nlargest(5)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            results = {
                'total_predictions': len(prediction_df),
                'total_recommendations': len(top_predictions),
                'unique_customers': len(unique_customers),
                'unique_products': len(unique_products),
                'target_week': (datetime.now() + timedelta(weeks=1)).strftime('%Y-%m-%d'), 
                'top_products': top_products.index.tolist(),
                'avg_prediction_score': float(predictions.mean()),
                'max_prediction_score': float(predictions.max()),
                'min_prediction_score': float(predictions.min()),
                'prediction_file': str(prediction_file),
                'execution_time_seconds': execution_time,
                'mode': 'DEMO_OPTIMIZED'
            }
            
            logger.info(f" Predicciones completadas en {execution_time:.2f} segundos!")
            logger.info(f" {results['total_predictions']} predicciones, score promedio: {results['avg_prediction_score']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f" Error en generaci?n de predicciones: {str(e)}")
            return {
                'total_predictions': 0,
                'error': str(e),
                'execution_time_seconds': (datetime.now() - start_time).total_seconds()
            }
