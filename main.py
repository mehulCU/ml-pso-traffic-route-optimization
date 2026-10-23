from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from api.services.prediction_service import predict_next_probe_count
from api.services.routing_service import get_route_comparison
from api.services.pso_service import get_pso_route_comparison
from api.services.traffic_data_service import get_observation_count
from api.services.rag_service import answer_project_question

app = FastAPI(
    title="Dynamic Traffic Route Optimization API",
    description="Traffic forecasting and dynamic route optimization.",
    version="1.0.0",
)


class TrafficObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probe_count: float = Field(ge=0)
    hour_sin: float
    hour_cos: float
    day_of_week: int = Field(ge=0, le=6)
    speed_limit: float = Field(gt=0)
    frc: float
    distance: float = Field(gt=0)


class PredictionRequest(BaseModel):
    history: list[TrafficObservation] = Field(
        min_length=24,
        max_length=24,
        description="Exactly 24 hourly observations in chronological order.",
    )


class PredictionResponse(BaseModel):
    predicted_probe_count: float
    forecast_horizon: str
    observations_used: int

class AssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question about the traffic optimization project.",
    )


@app.get("/")
def home():
    return {
        "message": "Dynamic Traffic Route Optimization API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "traffic-optimization-api",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        history = [
            observation.model_dump()
            for observation in request.history
        ]
        result = predict_next_probe_count(history)

        return PredictionResponse(
            predicted_probe_count=round(result, 4),
            forecast_horizon="next hourly step",
            observations_used=len(history),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the API server logs.",
        ) from exc


class RouteRequest(BaseModel):
    source: int
    destination: int
    forecast_timestamp: str = Field(
        description="Forecast hour, e.g. 2024-08-26T10:00:00"
    )


@app.post("/route")
def calculate_route(request: RouteRequest):
    try:
        return get_route_comparison(
            source=request.source,
            destination=request.destination,
            forecast_timestamp=request.forecast_timestamp,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Route calculation failed. Check the API server logs.",
        )


@app.post("/route/pso")
def calculate_pso_route(request: RouteRequest):
    try:
        return get_pso_route_comparison(
            source=request.source,
            destination=request.destination,
            forecast_timestamp=request.forecast_timestamp,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="PSO route calculation failed. Check the API server logs.",
        )


@app.get("/database/health")
def database_health_check():
    try:
        count = get_observation_count()
        return {
            "status": "connected",
            "database": "traffic_optimization",
            "observations": count,
        }
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable. Check the database connection and server logs.",
        )


@app.get("/database/observations")
def database_observations(limit: int = 10):
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=422,
            detail="limit must be between 1 and 100",
        )

    try:
        from api.services.traffic_data_service import get_recent_observations

        observations = get_recent_observations(limit)
        return {
            "count": len(observations),
            "observations": [
                {
                    **row,
                    "timestamp": row["timestamp"].isoformat(),
                }
                for row in observations
            ],
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Unable to retrieve traffic observations from the database.",
        )


@app.post("/assistant/ask")
def ask_project_assistant(request: AssistantRequest):
    try:
        return answer_project_question(request.question)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Assistant failed. Check the API server logs.",
        )
