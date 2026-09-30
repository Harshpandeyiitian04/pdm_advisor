"""FastAPI service. Run with `python -m uvicorn app.api:app --reload`."""
from pathlib import Path
from typing import Dict, List
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import data as D
from src.decision import band

ART = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"
app = FastAPI(title="Predictive Maintenance Advisor", version="1.0")
_bundle = None


def bundle():
    global _bundle
    if _bundle is None:
        if not ART.exists():
            raise HTTPException(503, "Model not trained yet. Run: python -m src.train")
        _bundle = joblib.load(ART)
    return _bundle


class History(BaseModel):
    """Chronological readings for one engine, including cycle and trained sensors."""
    cycles: List[Dict[str, float]]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": ART.exists()}


@app.post("/predict")
def predict(h: History):
    b = bundle()
    df = pd.DataFrame(h.cycles)
    missing = [c for c in ["cycle"] + b["sensors"] if c not in df.columns]
    if missing:
        raise HTTPException(422, f"Missing columns: {missing}")
    df["unit"] = 1
    feats = D.last_cycle(D.build_features(df, b["sensors"], b["window"]))
    rul = float(np.clip(b["model"].predict(feats[b["features"]])[0], 0, b["cap"]))
    return {"predicted_rul_cycles": round(rul, 1), "band": band(rul), "model": b["name"],
            "note": "RUL is capped at %d cycles; treat 125 as 'healthy, at least this much'." % b["cap"]}
