import base64
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from inference import run_inference
from model import EfficientUNet

MODEL_PATH = os.environ.get("MODEL_PATH", "model/model_image_segmentation3.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="Flood Segmentation API",
    description="Deteksi dan segmentasi area banjir dari foto menggunakan EfficientUNet",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: EfficientUNet | None = None


def get_model() -> EfficientUNet:
    global _model
    if _model is None:
        model = EfficientUNet(pretrained=False)
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model


@app.on_event("startup")
async def startup_event():
    get_model()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="File kosong")

    model = get_model()

    try:
        result = run_inference(model, image_bytes, DEVICE)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inferensi gagal: {exc}") from exc

    overlay_b64 = base64.b64encode(result.pop("overlay_bytes")).decode()

    return JSONResponse({
        **result,
        "overlay_image": f"data:image/png;base64,{overlay_b64}",
    })
