from fastapi import FastAPI, UploadFile, File, HTTPException
from model import OnnxModel

app = FastAPI()
classifier = OnnxModel("model.onnx")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        class_id = classifier.predict(image_bytes)
        return {"class_id": class_id}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image or model error")


@app.get("/health")
def health():
    return "OK"
