from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.model import CLASS_NAMES, CNN, SIGNAL_LENGTH

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "cnn_sqrt.pt"

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the server starts, not on every request.
    model = CNN()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    model.eval()  # turns dropout off
    state["model"] = model
    yield
    state.clear()


app = FastAPI(title="ECG Heartbeat Classifier", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # The default 422 body echoes the rejected input, which crashes the response (a 500)
    # when the input contains NaN. Report only where the error is and why.
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


class BeatRequest(BaseModel):
    # Exactly 187 finite numbers (the MIT-BIH beats are scaled to 0-1).
    beat: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(
        min_length=SIGNAL_LENGTH, max_length=SIGNAL_LENGTH
    )


class PredictionResponse(BaseModel):
    label: str
    class_index: int
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: BeatRequest):
    x = torch.tensor([request.beat], dtype=torch.float32)  # (1, 187): a batch of one beat
    with torch.no_grad():
        logits = state["model"](x)                          # (1, 5)
        probs = torch.softmax(logits, dim=1)[0]             # (5,) probabilities summing to 1
    class_index = int(probs.argmax())
    return PredictionResponse(
        label=CLASS_NAMES[class_index],
        class_index=class_index,
        probabilities={name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    )
