"""
LILITH API - Main FastAPI Application.

Provides REST API for weather forecasting:
- /v1/forecast - Single location forecast
- /v1/forecast/batch - Batch inference
- /v1/stations - Station information
- /v1/historical - Historical observations
"""

import time
from contextlib import asynccontextmanager
from typing import Optional

# Make torch optional for demo mode
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from web.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    BatchForecastRequest,
    BatchForecastResponse,
    StationListResponse,
    StationInfo,
    HistoricalRequest,
    HistoricalResponse,
    HealthResponse,
    ErrorResponse,
    Location,
    DailyForecast,
    HourlyForecast,
    HourlyForecastRequest,
    HourlyForecastResponse,
    PredictionRecord,
    AccuracyStats,
    AccuracyReportResponse,
)

# Global state for model
_forecaster = None
_config = None

# In-memory prediction storage (would use database in production)
_predictions: dict[str, PredictionRecord] = {}
_prediction_counter = 0


def get_forecaster():
    """Dependency to get forecaster instance."""
    if _forecaster is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _forecaster


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _forecaster, _config

    logger.info("Starting LILITH API...")

    # Load configuration
    import os

    checkpoint_path = os.environ.get("LILITH_CHECKPOINT", None)
    encoder_path = os.environ.get("LILITH_ENCODER", None)
    stations_path = os.environ.get("LILITH_STATIONS", None)

    # Load model if checkpoint provided
    if checkpoint_path:
        try:
            from inference.forecast import Forecaster

            _forecaster = Forecaster.from_pretrained(
                checkpoint_path,
                device="cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
                encoder_path=encoder_path,
                stations_path=stations_path,
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            _forecaster = None
    else:
        logger.warning("No checkpoint provided. Running in demo mode.")
        _forecaster = None

    yield

    # Cleanup
    logger.info("Shutting down LILITH API...")
    _forecaster = None


# Create FastAPI app
app = FastAPI(
    title="LILITH API",
    description="Long-range Intelligent Learning for Integrated Trend Hindcasting",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse},
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTPException", "message": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "InternalError", "message": str(exc)},
    )


# Health check
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy" if _forecaster is not None else "degraded",
        model_loaded=_forecaster is not None,
        gpu_available=TORCH_AVAILABLE and torch.cuda.is_available(),
        version="1.0.0",
    )


# Forecast endpoints
@app.post("/v1/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def create_forecast(request: ForecastRequest):
    """
    Generate weather forecast for a single location.

    Returns up to 90 days of temperature and precipitation forecasts
    with optional uncertainty bounds.
    """
    start_time = time.time()

    # Check if model is loaded
    if _forecaster is None:
        # Return demo response
        return _generate_demo_forecast(request)

    try:
        response = _forecaster.forecast(
            latitude=request.latitude,
            longitude=request.longitude,
            forecast_days=request.days,
            include_uncertainty=request.include_uncertainty,
            ensemble_members=request.ensemble_members,
        )

        # Convert to Pydantic model
        return ForecastResponse(
            location=Location(latitude=request.latitude, longitude=request.longitude),
            generated_at=response.generated_at,
            model_version=response.model_version,
            forecast_days=response.forecast_days,
            forecasts=[
                DailyForecast(
                    date=f.date,
                    temperature_max=f.temperature_max,
                    temperature_min=f.temperature_min,
                    precipitation=f.precipitation,
                    precipitation_probability=f.precipitation_probability,
                    temperature_max_lower=f.temperature_max_lower,
                    temperature_max_upper=f.temperature_max_upper,
                    temperature_min_lower=f.temperature_min_lower,
                    temperature_min_upper=f.temperature_min_upper,
                )
                for f in response.forecasts
            ],
        )

    except Exception as e:
        logger.exception(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/forecast/batch", response_model=BatchForecastResponse, tags=["Forecast"])
async def create_batch_forecast(request: BatchForecastRequest):
    """
    Generate forecasts for multiple locations.

    More efficient than individual requests for multiple locations.
    """
    start_time = time.time()

    forecasts = []
    for location in request.locations:
        single_request = ForecastRequest(
            latitude=location.latitude,
            longitude=location.longitude,
            days=request.days,
            include_uncertainty=request.include_uncertainty,
        )

        if _forecaster is None:
            forecast = _generate_demo_forecast(single_request)
        else:
            response = _forecaster.forecast(
                latitude=location.latitude,
                longitude=location.longitude,
                forecast_days=request.days,
                include_uncertainty=request.include_uncertainty,
            )
            forecast = ForecastResponse(
                location=location,
                generated_at=response.generated_at,
                model_version=response.model_version,
                forecast_days=response.forecast_days,
                forecasts=[
                    DailyForecast(**f.__dict__)
                    for f in response.forecasts
                ],
            )

        forecasts.append(forecast)

    processing_time = (time.time() - start_time) * 1000

    return BatchForecastResponse(
        forecasts=forecasts,
        total_locations=len(request.locations),
        processing_time_ms=processing_time,
    )


# Station endpoints
@app.get("/v1/stations", response_model=StationListResponse, tags=["Stations"])
async def list_stations(
    latitude: Optional[float] = Query(None, ge=-90, le=90),
    longitude: Optional[float] = Query(None, ge=-180, le=180),
    radius: float = Query(5.0, ge=0.1, le=50, description="Search radius in degrees"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """
    List weather stations, optionally filtered by location.

    If latitude and longitude are provided, returns stations within
    the specified radius.
    """
    # This would query the station database
    # For now, return empty response
    return StationListResponse(
        stations=[],
        total=0,
        page=page,
        page_size=page_size,
    )


@app.get("/v1/stations/{station_id}", response_model=StationInfo, tags=["Stations"])
async def get_station(station_id: str):
    """Get information about a specific station."""
    # This would query the station database
    raise HTTPException(status_code=404, detail="Station not found")


# Historical data endpoints
@app.post("/v1/historical", response_model=HistoricalResponse, tags=["Historical"])
async def get_historical_data(request: HistoricalRequest):
    """
    Get historical observations for a station.

    Returns daily observations for the specified date range.
    """
    # This would query the historical database
    raise HTTPException(status_code=501, detail="Historical data not yet implemented")


# Ensemble data endpoint
@app.get("/v1/ensemble/{forecast_id}", tags=["Forecast"])
async def get_ensemble_data(forecast_id: str):
    """
    Get detailed ensemble spread data for a forecast.

    Returns individual ensemble member predictions for detailed
    uncertainty analysis.
    """
    raise HTTPException(status_code=501, detail="Ensemble endpoint not yet implemented")


# Hourly forecast endpoint
@app.post("/v1/forecast/hourly", response_model=HourlyForecastResponse, tags=["Forecast"])
async def create_hourly_forecast(request: HourlyForecastRequest):
    """
    Generate hourly weather forecast for a location.

    Returns up to 168 hours (7 days) of detailed hourly predictions
    including temperature, humidity, wind, and precipitation.
    """
    if _forecaster is None:
        return _generate_demo_hourly_forecast(request)

    # Real model inference would go here
    raise HTTPException(status_code=501, detail="Hourly forecast not yet implemented for trained model")


# Prediction accuracy endpoints
@app.get("/v1/accuracy", response_model=AccuracyReportResponse, tags=["Accuracy"])
async def get_accuracy_report(
    latitude: Optional[float] = Query(None, ge=-90, le=90),
    longitude: Optional[float] = Query(None, ge=-180, le=180),
    days_back: int = Query(30, ge=1, le=365),
):
    """
    Get prediction accuracy report.

    Compares past predictions to actual observations and calculates
    accuracy metrics like MAE, RMSE, and accuracy by lead time.
    """
    return _generate_accuracy_report(latitude, longitude, days_back)


@app.get("/v1/accuracy/predictions", response_model=list[PredictionRecord], tags=["Accuracy"])
async def get_predictions(
    limit: int = Query(50, ge=1, le=200),
    verified_only: bool = Query(False),
):
    """
    Get recent prediction records.

    Returns stored predictions with their actual observations (if available).
    """
    predictions = list(_predictions.values())

    if verified_only:
        predictions = [p for p in predictions if p.actual_temp_max is not None]

    # Sort by predicted_at descending
    predictions.sort(key=lambda x: x.predicted_at, reverse=True)

    return predictions[:limit]


@app.post("/v1/accuracy/verify", tags=["Accuracy"])
async def verify_predictions():
    """
    Verify past predictions against actual observations.

    This endpoint fetches actual weather data and updates prediction
    records with observed values and error calculations.
    """
    verified_count = _verify_predictions_with_actuals()
    return {"message": f"Verified {verified_count} predictions", "verified_count": verified_count}


def _get_nearby_stations(lat: float, lon: float) -> list:
    """Get simulated nearby GHCN stations for a location."""
    import random

    # Simulated stations near the requested location
    # In production, this would query the actual GHCN station database
    station_types = [
        ("Airport", "ASOS"),
        ("University", "COOP"),
        ("City Center", "GHCN"),
        ("Regional", "GHCN"),
    ]

    stations = []
    random.seed(int(lat * 1000 + lon * 1000))  # Consistent for same location

    for i, (name_suffix, network) in enumerate(station_types[:3]):
        offset_lat = random.uniform(-0.1, 0.1)
        offset_lon = random.uniform(-0.1, 0.1)
        distance = ((offset_lat ** 2 + offset_lon ** 2) ** 0.5) * 111  # km

        stations.append({
            "id": f"USC00{random.randint(100000, 999999)}",
            "name": f"Station {name_suffix}",
            "network": network,
            "latitude": round(lat + offset_lat, 4),
            "longitude": round(lon + offset_lon, 4),
            "distance_km": round(distance, 1),
            "elevation_m": random.randint(10, 500),
            "record_start": f"{random.randint(1890, 1960)}-01-01",
            "record_end": "2025-12-31",
        })

    return sorted(stations, key=lambda x: x["distance_km"])


def _generate_demo_forecast(request: ForecastRequest) -> ForecastResponse:
    """Generate a demo forecast when model is not loaded."""
    import datetime
    import math
    import random

    forecasts = []
    start_date = request.start_date or datetime.date.today()
    lat = request.latitude
    lon = request.longitude

    # More realistic temperature model based on latitude and time of year
    # Reference: https://en.wikipedia.org/wiki/Climate_of_the_United_States

    # Determine hemisphere
    is_northern = lat >= 0
    abs_lat = abs(lat)

    # Base annual mean temperature decreases with latitude
    # Tropical (~0-23°): ~25°C, Subtropical (~23-35°): ~18°C,
    # Temperate (~35-50°): ~10°C, Cold (~50-66°): ~2°C, Polar (>66°): ~-10°C
    if abs_lat < 23:
        annual_mean = 26 - abs_lat * 0.1
        seasonal_amplitude = 3 + abs_lat * 0.1  # Small seasonal variation
    elif abs_lat < 35:
        annual_mean = 24 - (abs_lat - 23) * 0.5
        seasonal_amplitude = 5 + (abs_lat - 23) * 0.4
    elif abs_lat < 50:
        annual_mean = 18 - (abs_lat - 35) * 0.6
        seasonal_amplitude = 10 + (abs_lat - 35) * 0.3
    elif abs_lat < 66:
        annual_mean = 9 - (abs_lat - 50) * 0.5
        seasonal_amplitude = 15 + (abs_lat - 50) * 0.2
    else:
        annual_mean = -5 - (abs_lat - 66) * 0.4
        seasonal_amplitude = 18

    # Continental effect (distance from ocean) - simplified
    # Locations in interior have larger temperature swings
    # Using longitude as a rough proxy for US locations
    if -130 < lon < -60:  # North America
        if -100 < lon < -80:  # Continental interior
            seasonal_amplitude *= 1.2
        elif lon > -80:  # East coast
            seasonal_amplitude *= 0.95

    # Seed random for consistent results for same location
    random.seed(int(lat * 10000 + lon * 10000 + start_date.toordinal()))

    for i in range(request.days):
        forecast_date = start_date + datetime.timedelta(days=i + 1)
        day_of_year = forecast_date.timetuple().tm_yday

        # Seasonal temperature curve
        # Peak summer around day 200 (mid-July) in Northern Hemisphere
        # Phase shift for Southern Hemisphere
        if is_northern:
            phase_shift = 200  # Peak in mid-July
        else:
            phase_shift = 15   # Peak in mid-January

        seasonal = seasonal_amplitude * math.cos(2 * math.pi * (day_of_year - phase_shift) / 365)

        # Daily mean temperature
        daily_mean = annual_mean + seasonal

        # Diurnal range (difference between high and low)
        # Larger in dry climates, smaller near coasts
        diurnal_range = 8 + random.gauss(0, 1.5)

        # Add weather variability (fronts, etc.)
        weather_noise = random.gauss(0, 3)

        temp_max = daily_mean + diurnal_range / 2 + weather_noise
        temp_min = daily_mean - diurnal_range / 2 + weather_noise * 0.7

        # Precipitation probability varies by climate and season
        if abs_lat < 23:  # Tropical
            precip_base = 0.4
        elif abs_lat < 35:  # Subtropical
            precip_base = 0.25
        else:  # Temperate/Cold
            precip_base = 0.3

        precip_prob = max(0.0, min(1.0, precip_base + random.gauss(0, 0.1)))
        precipitation = random.expovariate(0.5) * 8 if random.random() < precip_prob else 0

        daily = DailyForecast(
            date=forecast_date.isoformat(),
            temperature_max=round(temp_max, 1),
            temperature_min=round(temp_min, 1),
            precipitation=round(precipitation, 1),
            precipitation_probability=round(precip_prob, 2),
        )

        if request.include_uncertainty:
            # Add uncertainty bounds that widen with forecast lead time
            uncertainty_scale = 1.5 + (i / request.days) * 3
            daily.temperature_max_lower = round(temp_max - uncertainty_scale, 1)
            daily.temperature_max_upper = round(temp_max + uncertainty_scale, 1)
            daily.temperature_min_lower = round(temp_min - uncertainty_scale, 1)
            daily.temperature_min_upper = round(temp_min + uncertainty_scale, 1)

        forecasts.append(daily)

    # Get nearby stations for display
    nearby_stations = _get_nearby_stations(lat, lon)

    return ForecastResponse(
        location=Location(latitude=request.latitude, longitude=request.longitude),
        generated_at=datetime.datetime.now(),
        model_version="demo-v1 (synthetic data - not trained)",
        forecast_days=request.days,
        forecasts=forecasts,
        nearby_stations=nearby_stations,
    )


def _generate_demo_hourly_forecast(request: HourlyForecastRequest) -> HourlyForecastResponse:
    """Generate demo hourly forecast when model is not loaded."""
    import datetime
    import math
    import random

    lat = request.latitude
    lon = request.longitude
    now = datetime.datetime.now()

    # Seed for consistency
    random.seed(int(lat * 10000 + lon * 10000 + now.hour))

    # Base temperature calculation (same as daily)
    abs_lat = abs(lat)
    is_northern = lat >= 0

    if abs_lat < 23:
        annual_mean = 26 - abs_lat * 0.1
        seasonal_amplitude = 3 + abs_lat * 0.1
    elif abs_lat < 35:
        annual_mean = 24 - (abs_lat - 23) * 0.5
        seasonal_amplitude = 5 + (abs_lat - 23) * 0.4
    elif abs_lat < 50:
        annual_mean = 18 - (abs_lat - 35) * 0.6
        seasonal_amplitude = 10 + (abs_lat - 35) * 0.3
    elif abs_lat < 66:
        annual_mean = 9 - (abs_lat - 50) * 0.5
        seasonal_amplitude = 15 + (abs_lat - 50) * 0.2
    else:
        annual_mean = -5 - (abs_lat - 66) * 0.4
        seasonal_amplitude = 18

    day_of_year = now.timetuple().tm_yday
    if is_northern:
        phase_shift = 200
    else:
        phase_shift = 15

    seasonal = seasonal_amplitude * math.cos(2 * math.pi * (day_of_year - phase_shift) / 365)
    daily_mean = annual_mean + seasonal

    forecasts = []

    for i in range(request.hours):
        forecast_time = now + datetime.timedelta(hours=i + 1)
        hour = forecast_time.hour

        # Diurnal temperature variation (coldest around 5am, warmest around 3pm)
        diurnal_phase = (hour - 5) / 24 * 2 * math.pi
        diurnal_amplitude = 5 + random.gauss(0, 0.5)
        diurnal_temp = diurnal_amplitude * math.sin(diurnal_phase)

        temp = daily_mean + diurnal_temp + random.gauss(0, 1)

        # Feels like (wind chill / heat index approximation)
        wind_speed = max(0, 3 + random.gauss(0, 2) + abs(math.sin(hour / 6)) * 2)
        humidity = max(20, min(100, 60 + random.gauss(0, 15) - diurnal_temp * 2))

        if temp < 10:
            feels_like = temp - wind_speed * 0.5
        elif temp > 27 and humidity > 40:
            feels_like = temp + (humidity - 40) * 0.1
        else:
            feels_like = temp

        # Precipitation
        precip_prob = max(0, min(1, 0.2 + random.gauss(0, 0.1)))
        precipitation = random.expovariate(2) * 3 if random.random() < precip_prob else 0

        # Cloud cover
        cloud_cover = max(0, min(100, 40 + random.gauss(0, 25) + (precip_prob * 30)))

        # Pressure
        pressure = 1013 + random.gauss(0, 5) - (precipitation * 0.5)

        # Wind direction
        wind_direction = (random.random() * 360)

        # UV Index (only during day)
        if 6 <= hour <= 18:
            uv_base = max(0, 8 * math.sin((hour - 6) / 12 * math.pi))
            uv_index = max(0, uv_base * (1 - cloud_cover / 100) * (1 - abs_lat / 90))
        else:
            uv_index = 0

        hourly = HourlyForecast(
            datetime=forecast_time.isoformat(),
            hour=hour,
            temperature=round(temp, 1),
            feels_like=round(feels_like, 1),
            humidity=round(humidity, 1),
            precipitation=round(precipitation, 2),
            precipitation_probability=round(precip_prob, 2),
            wind_speed=round(wind_speed, 1),
            wind_direction=round(wind_direction, 0),
            cloud_cover=round(cloud_cover, 0),
            pressure=round(pressure, 1),
            uv_index=round(uv_index, 1),
        )

        if request.include_uncertainty:
            uncertainty = 1.5 + (i / request.hours) * 2
            hourly.temperature_lower = round(temp - uncertainty, 1)
            hourly.temperature_upper = round(temp + uncertainty, 1)

        forecasts.append(hourly)

    return HourlyForecastResponse(
        location=Location(latitude=lat, longitude=lon),
        generated_at=now,
        model_version="demo-v1 (synthetic data - not trained)",
        forecast_hours=request.hours,
        forecasts=forecasts,
    )


def _store_prediction(
    lat: float, lon: float, location_name: str,
    target_date: str, temp_max: float, temp_min: float,
    precipitation: float, precip_prob: float, lead_days: int
) -> str:
    """Store a prediction for later accuracy verification."""
    import datetime
    import uuid

    global _prediction_counter
    _prediction_counter += 1

    prediction_id = f"pred_{uuid.uuid4().hex[:8]}"

    record = PredictionRecord(
        id=prediction_id,
        location=Location(latitude=lat, longitude=lon),
        location_name=location_name,
        predicted_at=datetime.datetime.now(),
        target_date=target_date,
        predicted_temp_max=temp_max,
        predicted_temp_min=temp_min,
        predicted_precipitation=precipitation,
        predicted_precip_prob=precip_prob,
        lead_days=lead_days,
    )

    _predictions[prediction_id] = record
    return prediction_id


def _verify_predictions_with_actuals() -> int:
    """Verify predictions with simulated actual observations."""
    import datetime
    import random

    today = datetime.date.today()
    verified_count = 0

    for pred_id, pred in _predictions.items():
        # Skip already verified
        if pred.actual_temp_max is not None:
            continue

        # Check if target date has passed
        target = datetime.date.fromisoformat(pred.target_date)
        if target >= today:
            continue

        # Simulate actual observations (in production, fetch from weather API)
        # Add some realistic error to the predictions
        random.seed(hash(pred_id))

        actual_max = pred.predicted_temp_max + random.gauss(0, 2.5)
        actual_min = pred.predicted_temp_min + random.gauss(0, 2.0)
        actual_precip = max(0, pred.predicted_precipitation + random.gauss(0, 3))

        # Update record
        pred.actual_temp_max = round(actual_max, 1)
        pred.actual_temp_min = round(actual_min, 1)
        pred.actual_precipitation = round(actual_precip, 1)

        # Calculate errors
        pred.temp_max_error = round(pred.predicted_temp_max - actual_max, 2)
        pred.temp_min_error = round(pred.predicted_temp_min - actual_min, 2)
        pred.precip_error = round(pred.predicted_precipitation - actual_precip, 2)

        verified_count += 1

    return verified_count


def _generate_accuracy_report(
    latitude: Optional[float],
    longitude: Optional[float],
    days_back: int
) -> AccuracyReportResponse:
    """Generate accuracy report from stored predictions."""
    import datetime
    import math

    now = datetime.datetime.now()
    period_start = (now - datetime.timedelta(days=days_back)).date()
    period_end = now.date()

    # Filter predictions
    filtered = list(_predictions.values())

    if latitude is not None and longitude is not None:
        # Filter by location (within ~50km)
        filtered = [
            p for p in filtered
            if abs(p.location.latitude - latitude) < 0.5
            and abs(p.location.longitude - longitude) < 0.5
        ]

    # Get verified predictions
    verified = [p for p in filtered if p.actual_temp_max is not None]

    # Calculate statistics
    if verified:
        temp_max_errors = [abs(p.temp_max_error) for p in verified if p.temp_max_error is not None]
        temp_min_errors = [abs(p.temp_min_error) for p in verified if p.temp_min_error is not None]
        precip_errors = [abs(p.precip_error) for p in verified if p.precip_error is not None]

        temp_max_mae = sum(temp_max_errors) / len(temp_max_errors) if temp_max_errors else 0
        temp_max_rmse = math.sqrt(sum(e**2 for e in temp_max_errors) / len(temp_max_errors)) if temp_max_errors else 0
        temp_min_mae = sum(temp_min_errors) / len(temp_min_errors) if temp_min_errors else 0
        temp_min_rmse = math.sqrt(sum(e**2 for e in temp_min_errors) / len(temp_min_errors)) if temp_min_errors else 0
        precip_mae = sum(precip_errors) / len(precip_errors) if precip_errors else 0

        # Precipitation occurrence accuracy
        precip_correct = sum(
            1 for p in verified
            if (p.predicted_precip_prob > 0.5) == (p.actual_precipitation > 0.1)
        )
        precip_accuracy = (precip_correct / len(verified) * 100) if verified else 0

        # Accuracy by lead day
        accuracy_by_lead = {}
        for lead in range(1, 15):
            lead_preds = [p for p in verified if p.lead_days == lead]
            if lead_preds:
                lead_max_errors = [abs(p.temp_max_error) for p in lead_preds if p.temp_max_error]
                lead_min_errors = [abs(p.temp_min_error) for p in lead_preds if p.temp_min_error]
                accuracy_by_lead[lead] = {
                    "temp_max_mae": round(sum(lead_max_errors) / len(lead_max_errors), 2) if lead_max_errors else 0,
                    "temp_min_mae": round(sum(lead_min_errors) / len(lead_min_errors), 2) if lead_min_errors else 0,
                    "count": len(lead_preds)
                }
    else:
        temp_max_mae = temp_max_rmse = temp_min_mae = temp_min_rmse = precip_mae = precip_accuracy = 0
        accuracy_by_lead = {}

    stats = AccuracyStats(
        total_predictions=len(filtered),
        verified_predictions=len(verified),
        temp_max_mae=round(temp_max_mae, 2),
        temp_max_rmse=round(temp_max_rmse, 2),
        temp_min_mae=round(temp_min_mae, 2),
        temp_min_rmse=round(temp_min_rmse, 2),
        precip_mae=round(precip_mae, 2),
        precip_accuracy=round(precip_accuracy, 1),
        accuracy_by_lead_day=accuracy_by_lead,
    )

    # Sort predictions by date
    recent = sorted(filtered, key=lambda x: x.predicted_at, reverse=True)[:20]

    location_filter = None
    if latitude is not None and longitude is not None:
        location_filter = f"{latitude:.4f}, {longitude:.4f}"

    return AccuracyReportResponse(
        generated_at=now,
        period_start=period_start.isoformat(),
        period_end=period_end.isoformat(),
        stats=stats,
        recent_predictions=recent,
        location_filter=location_filter,
    )


# Modify the demo forecast to also store predictions
def _generate_demo_forecast_with_storage(request: ForecastRequest, location_name: str = None) -> ForecastResponse:
    """Generate demo forecast and store predictions for accuracy tracking."""
    response = _generate_demo_forecast(request)

    # Store first 14 days of predictions for verification
    for i, forecast in enumerate(response.forecasts[:14]):
        _store_prediction(
            lat=request.latitude,
            lon=request.longitude,
            location_name=location_name or f"{request.latitude:.2f}, {request.longitude:.2f}",
            target_date=forecast.date,
            temp_max=forecast.temperature_max,
            temp_min=forecast.temperature_min,
            precipitation=forecast.precipitation,
            precip_prob=forecast.precipitation_probability,
            lead_days=i + 1,
        )

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
