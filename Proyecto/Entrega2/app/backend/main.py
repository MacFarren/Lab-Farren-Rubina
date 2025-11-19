"""
Backend API - SodAI Drinks Recommendation System
===============================================

API FastAPI para servir el modelo de recomendación de productos.
Incluye endpoints para predicciones individuales y batch.

Autor: SodAI Drinks MLOps Team
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from pathlib import Path
import pickle
import mlflow.sklearn

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/recommendation_model.pkl')
DATA_PATH = os.getenv('DATA_PATH', '/app/data')
MLFLOW_MODEL_URI = os.getenv('MLFLOW_MODEL_URI', 'models:/sodai-recommendation-model/Production')

app = FastAPI(
    title="SodAI Drinks Recommendation API",
    description="API para el sistema de recomendación de productos de SodAI Drinks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Modelos de datos de entrada
class CustomerProductRequest(BaseModel):
    """Solicitud de predicción para un par cliente-producto."""
    customer_id: int = Field(..., description="ID del cliente")
    product_id: int = Field(..., description="ID del producto")

class BatchPredictionRequest(BaseModel):
    """Solicitud de predicción en lote."""
    customer_product_pairs: List[CustomerProductRequest] = Field(..., description="Lista de pares cliente-producto")

class CustomerRecommendationRequest(BaseModel):
    """Solicitud de recomendaciones para un cliente."""
    customer_id: int = Field(..., description="ID del cliente")
    top_k: int = Field(default=10, description="Número de recomendaciones")

# Modelos de respuesta
class PredictionResponse(BaseModel):
    """Respuesta de predicción."""
    customer_id: int
    product_id: int
    prediction_score: float
    recommendation_confidence: str
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Respuesta de predicción en lote."""
    predictions: List[PredictionResponse]
    total_predictions: int
    processing_time_ms: int

class RecommendationResponse(BaseModel):
    """Respuesta de recomendaciones."""
    customer_id: int
    recommendations: List[Dict[str, Any]]
    total_recommendations: int
    timestamp: str

class HealthResponse(BaseModel):
    """Respuesta de health check."""
    status: str
    model_loaded: bool
    data_available: bool
    timestamp: str

# Variables globales para el modelo
model = None
feature_names = []
customer_data = None
product_data = None
feature_defaults = {}

@app.on_event("startup")
async def load_model_and_data():
    """Carga el modelo y datos al iniciar la aplicación."""
    global model, feature_names, customer_data, product_data, feature_defaults
    
    logger.info("Inicializando SodAI Drinks Recommendation API...")
    
    try:
        # Cargar modelo desde MLflow (preferido) o local
        if MLFLOW_MODEL_URI and MLFLOW_MODEL_URI.startswith('models:/'):
            try:
                logger.info(f"Cargando modelo desde MLflow: {MLFLOW_MODEL_URI}")
                model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
            except Exception as e:
                logger.warning(f"Error cargando desde MLflow: {e}. Intentando carga local...")
                model = None
        
        if model is None:
            # Cargar modelo local
            if Path(MODEL_PATH).exists():
                logger.info(f"Cargando modelo local: {MODEL_PATH}")
                import joblib
                model = joblib.load(MODEL_PATH)
            else:
                logger.error("No se encontró modelo disponible")
                raise FileNotFoundError("Modelo no disponible")
        
        # Cargar feature names
        feature_names_path = Path(MODEL_PATH).parent / "feature_names.pkl"
        if feature_names_path.exists():
            with open(feature_names_path, 'rb') as f:
                feature_names = pickle.load(f)
        else:
            # Feature names por defecto basados en el feature engineering
            feature_names = [
                'total_orders', 'unique_products_bought', 'total_transactions',
                'avg_items_per_transaction', 'days_since_last_purchase',
                'unique_customers', 'total_items_sold', 'customer_penetration'
            ]
            logger.warning("Usando feature names por defecto")
        
        # Cargar datos de clientes y productos
        customers_path = Path(DATA_PATH) / "clean" / "clientes_clean.parquet"
        products_path = Path(DATA_PATH) / "clean" / "productos_clean.parquet"
        
        if customers_path.exists():
            customer_data = pd.read_parquet(customers_path)
            logger.info(f"Datos de clientes cargados: {len(customer_data)} clientes")
        
        if products_path.exists():
            product_data = pd.read_parquet(products_path)
            logger.info(f"Datos de productos cargados: {len(product_data)} productos")
        
        # Crear valores por defecto para features
        feature_defaults = {feature: 0.0 for feature in feature_names}
        
        logger.info("✅ Inicialización completada exitosamente")
        
    except Exception as e:
        logger.error(f"Error en inicialización: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de health check."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        data_available=customer_data is not None and product_data is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: CustomerProductRequest):
    """Predice la probabilidad de compra para un par cliente-producto."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    logger.info(f"Predicción individual: customer_id={request.customer_id}, product_id={request.product_id}")
    
    try:
        # Crear features para la predicción
        features = create_features_for_prediction(request.customer_id, request.product_id)
        
        # Realizar predicción
        prediction_score = model.predict_proba([features])[0][1]
        
        # Determinar nivel de confianza
        if prediction_score >= 0.7:
            confidence = "high"
        elif prediction_score >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            customer_id=request.customer_id,
            product_id=request.product_id,
            prediction_score=float(prediction_score),
            recommendation_confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Realiza predicciones en lote para múltiples pares cliente-producto."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    logger.info(f"Predicción en lote: {len(request.customer_product_pairs)} pares")
    
    start_time = datetime.now()
    
    try:
        predictions = []
        
        # Preparar features para todos los pares
        features_list = []
        for pair in request.customer_product_pairs:
            features = create_features_for_prediction(pair.customer_id, pair.product_id)
            features_list.append(features)
        
        # Realizar predicciones en lote
        if features_list:
            prediction_scores = model.predict_proba(features_list)[:, 1]
            
            for i, pair in enumerate(request.customer_product_pairs):
                score = float(prediction_scores[i])
                
                # Determinar confianza
                if score >= 0.7:
                    confidence = "high"
                elif score >= 0.4:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                predictions.append(PredictionResponse(
                    customer_id=pair.customer_id,
                    product_id=pair.product_id,
                    prediction_score=score,
                    recommendation_confidence=confidence,
                    timestamp=datetime.now().isoformat()
                ))
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            processing_time_ms=int(processing_time)
        )
        
    except Exception as e:
        logger.error(f"Error en predicción batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción batch: {str(e)}")

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: CustomerRecommendationRequest):
    """Obtiene las top-K recomendaciones para un cliente específico."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    if product_data is None:
        raise HTTPException(status_code=503, detail="Datos de productos no disponibles")
    
    logger.info(f"Generando {request.top_k} recomendaciones para cliente {request.customer_id}")
    
    try:
        # Obtener lista de productos disponibles
        available_products = product_data['product_id'].tolist()[:100]  # Limitamos para demo
        
        # Crear pares cliente-producto
        customer_product_pairs = [
            (request.customer_id, product_id) for product_id in available_products
        ]
        
        # Generar features y predicciones
        features_list = []
        for customer_id, product_id in customer_product_pairs:
            features = create_features_for_prediction(customer_id, product_id)
            features_list.append(features)
        
        if features_list:
            # Obtener predicciones
            prediction_scores = model.predict_proba(features_list)[:, 1]
            
            # Crear recomendaciones con información de productos
            recommendations_data = []
            for i, (customer_id, product_id) in enumerate(customer_product_pairs):
                product_info = product_data[product_data['product_id'] == product_id]
                
                if not product_info.empty:
                    product_row = product_info.iloc[0]
                    
                    recommendations_data.append({
                        'product_id': int(product_id),
                        'prediction_score': float(prediction_scores[i]),
                        'product_brand': product_row.get('brand', 'Unknown'),
                        'product_category': product_row.get('category', 'Unknown'),
                        'product_segment': product_row.get('segment', 'Unknown'),
                        'product_size': product_row.get('size', 'Unknown')
                    })
            
            # Ordenar por score descendente y tomar top-K
            recommendations_data.sort(key=lambda x: x['prediction_score'], reverse=True)
            top_recommendations = recommendations_data[:request.top_k]
            
        else:
            top_recommendations = []
        
        return RecommendationResponse(
            customer_id=request.customer_id,
            recommendations=top_recommendations,
            total_recommendations=len(top_recommendations),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generando recomendaciones: {e}")
        raise HTTPException(status_code=500, detail=f"Error generando recomendaciones: {str(e)}")

@app.get("/customers/{customer_id}/info")
async def get_customer_info(customer_id: int):
    """Obtiene información de un cliente específico."""
    if customer_data is None:
        raise HTTPException(status_code=503, detail="Datos de clientes no disponibles")
    
    customer_info = customer_data[customer_data['customer_id'] == customer_id]
    
    if customer_info.empty:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    
    customer_row = customer_info.iloc[0]
    
    return {
        'customer_id': int(customer_row['customer_id']),
        'region_id': int(customer_row.get('region_id', 0)),
        'zone_id': int(customer_row.get('zone_id', 0)),
        'customer_type': customer_row.get('customer_type', 'Unknown'),
        'location_x': float(customer_row.get('X', 0)),
        'location_y': float(customer_row.get('Y', 0))
    }

@app.get("/products/{product_id}/info")
async def get_product_info(product_id: int):
    """Obtiene información de un producto específico."""
    if product_data is None:
        raise HTTPException(status_code=503, detail="Datos de productos no disponibles")
    
    product_info = product_data[product_data['product_id'] == product_id]
    
    if product_info.empty:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    
    product_row = product_info.iloc[0]
    
    return {
        'product_id': int(product_row['product_id']),
        'brand': product_row.get('brand', 'Unknown'),
        'category': product_row.get('category', 'Unknown'),
        'sub_category': product_row.get('sub_category', 'Unknown'),
        'segment': product_row.get('segment', 'Unknown'),
        'package': product_row.get('package', 'Unknown'),
        'size': product_row.get('size', 'Unknown')
    }

def create_features_for_prediction(customer_id: int, product_id: int) -> List[float]:
    """
    Crea features para predicción. Simplificado para demo.
    En producción, esto debería usar los mismos features que el entrenamiento.
    """
    # Para simplificar, usar valores por defecto con alguna variación
    features = []
    
    for feature_name in feature_names:
        if feature_name in feature_defaults:
            # Simular features basadas en IDs (para demo)
            base_value = feature_defaults[feature_name]
            
            # Agregar variación basada en customer_id y product_id
            variation = (customer_id % 100) / 100.0 + (product_id % 50) / 50.0
            feature_value = base_value + variation
            
            features.append(feature_value)
        else:
            features.append(0.0)
    
    # Asegurar que tenemos el número correcto de features
    while len(features) < len(feature_names):
        features.append(0.0)
    
    return features[:len(feature_names)]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)