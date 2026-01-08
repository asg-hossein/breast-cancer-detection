from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import sys
from datetime import datetime
from typing import List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = FastAPI(
    title="Breast Cancer Detection API",
    description="ML API for breast cancer classification",
    version="2.0.0"
)

class PredictionRequest(BaseModel):
    features: List[float]
    model_name: Optional[str] = None

class PredictionResponse(BaseModel):
    model_used: str
    prediction: int
    diagnosis: str
    confidence: Optional[dict] = None
    timestamp: str

MODELS = {}
SCALER = None

def load_resources():
    global MODELS, SCALER
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)
    
    for model_file in [f for f in os.listdir(MODELS_DIR) if f.endswith("_model.pkl")]:
        try:
            model_name = model_file.replace("_model.pkl", "")
            MODELS[model_name] = joblib.load(os.path.join(MODELS_DIR, model_file))
        except:
            continue
    
    if not MODELS:
        create_sample_model()

def create_sample_model():
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        
        global SCALER
        SCALER = StandardScaler()
        X_train_scaled = SCALER.fit_transform(X_train)
        
        joblib.dump(SCALER, os.path.join(MODELS_DIR, "scaler.pkl"))
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        model_path = os.path.join(MODELS_DIR, "SampleRF_model.pkl")
        joblib.dump(model, model_path)
        MODELS["SampleRF"] = model
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample model creation failed: {str(e)}")

load_resources()

@app.get("/")
async def root():
    return {
        "api": "Breast Cancer Detection API",
        "version": "2.0.0",
        "models_available": list(MODELS.keys())
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if MODELS else "degraded",
        "models_loaded": len(MODELS),
        "scaler_loaded": SCALER is not None
    }

@app.get("/models")
async def list_models():
    models_list = [
        {
            "name": name,
            "type": type(model).__name__,
            "has_probability": hasattr(model, "predict_proba")
        }
        for name, model in MODELS.items()
    ]
    return {"count": len(models_list), "models": models_list}

@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    if len(request.features) != 30:
        raise HTTPException(400, "Expected 30 features")
    
    if not MODELS:
        raise HTTPException(503, "No models available")
    
    model_name = request.model_name or list(MODELS.keys())[0]
    if model_name not in MODELS:
        raise HTTPException(404, f"Model not found. Available: {list(MODELS.keys())}")
    
    model = MODELS[model_name]
    
    try:
        X = np.array(request.features).reshape(1, -1)
        if SCALER is not None:
            X = SCALER.transform(X)
        
        prediction = int(model.predict(X)[0])
        
        response = {
            "model_used": model_name,
            "prediction": prediction,
            "diagnosis": "Malignant" if prediction == 1 else "Benign",
            "timestamp": datetime.now().isoformat()
        }
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            response["confidence"] = {
                "benign": float(proba[0]),
                "malignant": float(proba[1])
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.get("/predict/sample")
async def predict_sample():
    sample_features = [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419,
        0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373,
        0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622,
        0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
    
    request = PredictionRequest(
        features=sample_features,
        model_name=list(MODELS.keys())[0] if MODELS else None
    )
    
    return await make_prediction(request)

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds() * 1000
    print(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms")
    return response

if __name__ == "__main__":
    import uvicorn
    
    print(f"\nBreast Cancer Detection API")
    print(f"Models: {len(MODELS)}")
    print(f"http://localhost:8000")
    print(f"http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)