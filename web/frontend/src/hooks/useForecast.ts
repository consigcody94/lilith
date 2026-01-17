"use client";

import { useQuery } from "@tanstack/react-query";
import axios from "axios";

interface Location {
  latitude: number;
  longitude: number;
  name?: string;
}

interface DailyForecast {
  date: string;
  temperature_max: number;
  temperature_min: number;
  precipitation: number;
  precipitation_probability: number;
  temperature_max_lower?: number;
  temperature_max_upper?: number;
  temperature_min_lower?: number;
  temperature_min_upper?: number;
}

interface ForecastResponse {
  location: {
    latitude: number;
    longitude: number;
  };
  generated_at: string;
  model_version: string;
  forecast_days: number;
  forecasts: DailyForecast[];
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchForecast(location: Location): Promise<ForecastResponse> {
  const response = await axios.post<ForecastResponse>(`${API_URL}/v1/forecast`, {
    latitude: location.latitude,
    longitude: location.longitude,
    days: 90,
    include_uncertainty: true,
  });

  return response.data;
}

export function useForecast(location: Location) {
  return useQuery({
    queryKey: ["forecast", location.latitude, location.longitude],
    queryFn: () => fetchForecast(location),
    staleTime: 30 * 60 * 1000, // 30 minutes
    gcTime: 60 * 60 * 1000, // 1 hour (formerly cacheTime)
    retry: 2,
    refetchOnWindowFocus: false,
  });
}

// Hook for batch forecasts
export function useBatchForecast(locations: Location[]) {
  return useQuery({
    queryKey: ["batch-forecast", locations.map((l) => `${l.latitude},${l.longitude}`).join("|")],
    queryFn: async () => {
      const response = await axios.post(`${API_URL}/v1/forecast/batch`, {
        locations: locations.map((l) => ({
          latitude: l.latitude,
          longitude: l.longitude,
        })),
        days: 90,
        include_uncertainty: true,
      });
      return response.data;
    },
    enabled: locations.length > 0,
    staleTime: 30 * 60 * 1000,
  });
}

// Hourly forecast types
interface HourlyForecast {
  datetime: string;
  hour: number;
  temperature: number;
  feels_like: number;
  humidity: number;
  precipitation: number;
  precipitation_probability: number;
  wind_speed: number;
  wind_direction: number;
  cloud_cover: number;
  pressure: number;
  uv_index: number | null;
  temperature_lower?: number;
  temperature_upper?: number;
}

interface HourlyForecastResponse {
  location: {
    latitude: number;
    longitude: number;
  };
  generated_at: string;
  model_version: string;
  forecast_hours: number;
  forecasts: HourlyForecast[];
}

// Hook for hourly forecast
export function useHourlyForecast(location: Location, hours: number = 48) {
  return useQuery({
    queryKey: ["hourly-forecast", location.latitude, location.longitude, hours],
    queryFn: async () => {
      const response = await axios.post<HourlyForecastResponse>(`${API_URL}/v1/forecast/hourly`, {
        latitude: location.latitude,
        longitude: location.longitude,
        hours,
        include_uncertainty: true,
      });
      return response.data;
    },
    staleTime: 15 * 60 * 1000, // 15 minutes
    gcTime: 30 * 60 * 1000,
    retry: 2,
    refetchOnWindowFocus: false,
  });
}

// Accuracy tracking types
interface PredictionRecord {
  id: string;
  location: { latitude: number; longitude: number };
  location_name: string | null;
  predicted_at: string;
  target_date: string;
  predicted_temp_max: number;
  predicted_temp_min: number;
  predicted_precipitation: number;
  predicted_precip_prob: number;
  actual_temp_max: number | null;
  actual_temp_min: number | null;
  actual_precipitation: number | null;
  temp_max_error: number | null;
  temp_min_error: number | null;
  precip_error: number | null;
  lead_days: number;
}

interface AccuracyStats {
  total_predictions: number;
  verified_predictions: number;
  temp_max_mae: number;
  temp_max_rmse: number;
  temp_min_mae: number;
  temp_min_rmse: number;
  precip_mae: number;
  precip_accuracy: number;
  accuracy_by_lead_day: Record<number, { temp_max_mae: number; temp_min_mae: number; count: number }>;
}

interface AccuracyReportResponse {
  generated_at: string;
  period_start: string;
  period_end: string;
  stats: AccuracyStats;
  recent_predictions: PredictionRecord[];
  location_filter: string | null;
}

// Hook for accuracy report
export function useAccuracyReport(location?: Location, daysBack: number = 30) {
  return useQuery({
    queryKey: ["accuracy-report", location?.latitude, location?.longitude, daysBack],
    queryFn: async () => {
      const params = new URLSearchParams({ days_back: daysBack.toString() });
      if (location) {
        params.append("latitude", location.latitude.toString());
        params.append("longitude", location.longitude.toString());
      }
      const response = await axios.get<AccuracyReportResponse>(`${API_URL}/v1/accuracy?${params}`);
      return response.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 30 * 60 * 1000,
    retry: 1,
    refetchOnWindowFocus: false,
  });
}

// Export types for components
export type { HourlyForecast, HourlyForecastResponse, PredictionRecord, AccuracyStats, AccuracyReportResponse };
