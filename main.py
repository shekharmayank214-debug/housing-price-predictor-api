from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

app = FastAPI(
    title="Housing Price Predictor API",
    description="Predict California housing prices using a Linear Regression model trained on the California Housing dataset.",
    version="1.0.0",
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    median_income: float = Field(..., description="Median income in block group (×$10,000)", examples=[3.5])
    house_age: float = Field(..., description="Median house age in block group", examples=[20.0])
    avg_rooms: float = Field(..., description="Average number of rooms per household", examples=[6.0])
    avg_bedrooms: float = Field(..., description="Average number of bedrooms per household", examples=[1.0])
    population: float = Field(..., description="Block group population", examples=[1200.0])
    avg_occupancy: float = Field(..., description="Average number of household members", examples=[3.0])
    latitude: float = Field(..., description="Block group latitude", examples=[37.5])
    longitude: float = Field(..., description="Block group longitude", examples=[-122.0])


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted median house value (in $100,000s)")
    currency_note: str = "Values are in units of $100,000 (e.g. 2.5 = $250,000)"


class HealthResponse(BaseModel):
    status: str
    model_trained: bool
    features: list[str]


# ── Model training at startup ─────────────────────────────────────────────────
scaler = StandardScaler()
model = LinearRegression()
FEATURE_NAMES: list[str] = []
model_ready = False


@app.on_event("startup")
def train_model():
    global model_ready, FEATURE_NAMES
    housing = fetch_california_housing()
    FEATURE_NAMES = list(housing.feature_names)

    X = scaler.fit_transform(housing.data)
    y = housing.target

    model.fit(X, y)
    model_ready = True


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Housing Price Predictor API",
        "docs": "/docs",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="healthy" if model_ready else "loading",
        model_trained=model_ready,
        features=FEATURE_NAMES,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: PredictionRequest):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is still training, try again shortly.")

    features = np.array(
        [[
            req.median_income,
            req.house_age,
            req.avg_rooms,
            req.avg_bedrooms,
            req.population,
            req.avg_occupancy,
            req.latitude,
            req.longitude,
        ]]
    )

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]

    return PredictionResponse(predicted_price=round(float(prediction), 4))
