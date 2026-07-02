# Housing Price Predictor API

A REST API that predicts California housing prices using Linear Regression.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check & model status |
| POST | `/predict` | Make a price prediction |

## Example Request

```json
POST /predict
{
  "median_income": 3.5,
  "house_age": 20.0,
  "avg_rooms": 6.0,
  "avg_bedrooms": 1.0,
  "population": 1200.0,
  "avg_occupancy": 3.0,
  "latitude": 37.5,
  "longitude": -122.0
}
```

## Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/shekharmayank214-debug/housing-price-predictor-api)
