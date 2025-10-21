from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd, pickle, os
from optimize import optimize_model

app = FastAPI(title="Water Potability API")

class WaterSample(BaseModel):
    ph: Optional[float] = Field(None)
    Hardness: Optional[float] = None
    Solids: Optional[float] = None
    Chloramines: Optional[float] = None
    Sulfate: Optional[float] = None
    Conductivity: Optional[float] = None
    Organic_carbon: Optional[float] = None
    Trihalomethanes: Optional[float] = None
    Turbidity: Optional[float] = None

class PredictRequest(BaseModel):
    samples: List[WaterSample]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
def train(csv_path: str = "water_potability.csv", n_trials: int = 30,
          test_size: float = 0.2, val_size: float = 0.2, seed: int = 42):
    try:
        optimize_model(csv_path, n_trials, test_size, val_size, seed)
        return {"message": "Entrenamiento completado",
                "model_saved": os.path.exists("models/best_model.pkl")}
    except Exception as e:
        raise HTTPException(500, f"Error entrenando: {e}")

@app.post("/predict")
def predict(req: PredictRequest):
    path = "models/best_model.pkl"
    if not os.path.exists(path):
        raise HTTPException(400, "No existe models/best_model.pkl. Ejecuta /train primero.")
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        X = pd.DataFrame([s.model_dump() for s in req.samples])
        pred = model.predict(X)
        return {"predictions": [int(x) for x in pred]}
    except Exception as e:
        raise HTTPException(500, f"Error al predecir: {e}")
    







@app.get("/")
def home():
    return {
        "titulo": "API de Potabilidad de Agua",
        "modelo": "XGBoost + Optuna, pipeline con imputación",
        "metricas": ["valid_f1 en MLflow"],
        "endpoints": {
            "GET /": "Esta descripción",
            "GET /health": "Ping del servicio",
            "POST /train": "Entrena/optimiza y registra en MLflow",
            "POST /potabilidad": "Predice 0/1 para una muestra",
            "POST /predict": "Predice para batch de muestras (lista)",
        }
    }









@app.post("/potabilidad")
def potabilidad(sample: WaterSample):
    """
    Recibe una sola muestra y devuelve {"potabilidad": 0|1}
    """
    pkl_path = "models/best_model.pkl"
    if not os.path.exists(pkl_path):
        raise HTTPException(status_code=400, detail="No existe models/best_model.pkl. Ejecuta /train primero.")

    try:
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        X = pd.DataFrame([sample.model_dump()])
        pred = model.predict(X)
        return {"potabilidad": int(pred[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")
    






if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


