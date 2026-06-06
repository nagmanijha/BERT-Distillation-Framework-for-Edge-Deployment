import time
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Distilled BERT Inference API")

class InferenceRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: InferenceRequest):
    """
    Simulates an ONNX Runtime inference pass for the quantized model.
    """
    start_time = time.time()
    
    # Simulate ONNX Runtime inference delay (approx 11ms for quantized 4-layer student)
    time.sleep(0.011) 
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Mock output
    return {
        "model": "tinybert-int8-onnx",
        "latency_ms": round(latency_ms, 2),
        "prediction": "positive",
        "confidence": 0.94
    }

if __name__ == "__main__":
    import uvicorn
    # To run: python -m src.deployment.api
    uvicorn.run(app, host="0.0.0.0", port=8000)
