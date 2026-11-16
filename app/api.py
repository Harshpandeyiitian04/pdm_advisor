# This file is the web service. Other programs can ask it how much life one engine has left.
# Run it with: python -m uvicorn app.api:app --reload

# These lines bring in paths, lists, model loading, numbers, tables, the web framework, and the urgency label.
from pathlib import Path
from typing import Dict, List
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import data as D
from src.decision import band

# Where the trained model file is expected to be.
ART = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"
# Create the web app and a place to remember the model after the first load.
app = FastAPI(title="Predictive Maintenance Advisor", version="1.0")
_bundle = None


# This loads the saved model the first time someone asks for a prediction, then reuses it.
def bundle():
    global _bundle
    if _bundle is None:
        # Stop with a clear message if nobody has trained a model yet.
        if not ART.exists():
            raise HTTPException(503, "Model not trained yet. Run: python -m src.train")
        _bundle = joblib.load(ART)
    return _bundle


# This is the shape of one prediction request: a time-ordered list of readings for a single engine.
class History(BaseModel):
    """A time-ordered list of readings for one engine, including the cycle number and the sensors the model was trained on."""
    cycles: List[Dict[str, float]]


# This answers a simple "are you running" check, and says whether a trained model file is present.
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": ART.exists()}


# This takes one engine's history and returns a life-left guess plus an urgency label.
@app.post("/predict")
def predict(h: History):
    # Load the model and turn the incoming readings into a table.
    b = bundle()
    df = pd.DataFrame(h.cycles)
    # Stop if the cycle column or a needed sensor is missing.
    missing = [c for c in ["cycle"] + b["sensors"] if c not in df.columns]
    if missing:
        raise HTTPException(422, f"Missing columns: {missing}")
    # Treat the whole request as one engine.
    df["unit"] = 1
    # Build the clues and keep only the newest cycle.
    feats = D.last_cycle(D.build_features(df, b["sensors"], b["window"]))
    # Guess the life left and keep it inside the allowed range.
    rul = float(np.clip(b["model"].predict(feats[b["features"]])[0], 0, b["cap"]))
    # Send back the guess, how urgent it is, which model was used, and a note about the cap.
    return {"predicted_rul_cycles": round(rul, 1), "band": band(rul), "model": b["name"],
            "note": "RUL is capped at %d cycles; treat 125 as 'healthy, at least this much'." % b["cap"]}
