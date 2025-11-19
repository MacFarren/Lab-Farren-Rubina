from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
from typing import Dict, List

app = FastAPI(title="SodAI Drinks Recommendation API", version="1.0.0")

# Permitir CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def get_demo_page():
    """Página de demostración del sistema"""
    with open("/app/demo.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api")
def root():
    return {"message": "SodAI Drinks MLOps API está ejecutándose!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "recommendation-api"}

@app.get("/data/info")
def get_data_info():
    """Información sobre los datos cargados"""
    try:
        # Intentar cargar los archivos parquet
        transacciones = pd.read_parquet("/app/data/transacciones.parquet")
        clientes = pd.read_parquet("/app/data/clientes.parquet")
        productos = pd.read_parquet("/app/data/productos.parquet")
        
        return {
            "transacciones": {
                "rows": len(transacciones),
                "columns": list(transacciones.columns),
                "sample": transacciones.head(3).to_dict('records')
            },
            "clientes": {
                "rows": len(clientes),
                "columns": list(clientes.columns),
                "sample": clientes.head(3).to_dict('records')
            },
            "productos": {
                "rows": len(productos),
                "columns": list(productos.columns),
                "sample": productos.head(3).to_dict('records')
            }
        }
    except Exception as e:
        return {"error": f"Error al cargar datos: {str(e)}"}

@app.post("/recommend/{customer_id}")
def get_recommendations(customer_id: int, top_k: int = 5):
    """Obtener recomendaciones para un cliente específico"""
    try:
        # Simulamos recomendaciones básicas por ahora
        transacciones = pd.read_parquet("/app/data/transacciones.parquet")
        productos = pd.read_parquet("/app/data/productos.parquet")
        
        # Productos más populares como recomendación simple
        popular_products = (transacciones.groupby('product_id')
                          .size()
                          .sort_values(ascending=False)
                          .head(top_k)
                          .index.tolist())
        
        # Obtener información de productos
        recommended_products = productos[productos['product_id'].isin(popular_products)]
        
        return {
            "customer_id": customer_id,
            "recommendations": recommended_products.to_dict('records')[:top_k],
            "algorithm": "popularity_based",
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Error generando recomendaciones: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)