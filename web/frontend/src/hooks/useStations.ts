"use client";

import { useQuery } from "@tanstack/react-query";
import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Station {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  elevation: number;
  state?: string;
  country: string;
  record_start: string;
  record_end: string;
}

interface StationForecast {
  station: Station;
  current_temp?: number;
  forecast_high: number;
  forecast_low: number;
  precipitation_probability: number;
  accuracy_score?: number;
  temp_error_avg?: number;
  last_updated: string;
}

interface StationsResponse {
  stations: StationForecast[];
  total: number;
  page: number;
  page_size: number;
  generated_at: string;
  model_version: string;
}

interface StationAccuracy {
  station_id: string;
  predictions_count: number;
  verified_count: number;
  temp_max_mae: number;
  temp_min_mae: number;
  precip_accuracy: number;
  last_7_days_mae: number;
  trend: "improving" | "stable" | "declining";
}

interface GlobalAccuracyStats {
  total_stations: number;
  total_predictions: number;
  verified_predictions: number;
  avg_temp_max_mae: number;
  avg_temp_min_mae: number;
  avg_precip_accuracy: number;
  best_performing_stations: string[];
  generated_at: string;
}

// Hook to fetch all stations with forecasts
export function useStations(page: number = 1, pageSize: number = 50) {
  return useQuery({
    queryKey: ["stations", page, pageSize],
    queryFn: async () => {
      const response = await axios.get<StationsResponse>(
        `${API_URL}/v1/stations/forecasts?page=${page}&page_size=${pageSize}`
      );
      return response.data;
    },
    staleTime: 10 * 60 * 1000, // 10 minutes
    gcTime: 30 * 60 * 1000,
    retry: 2,
  });
}

// Hook to fetch global accuracy stats
export function useGlobalAccuracy() {
  return useQuery({
    queryKey: ["global-accuracy"],
    queryFn: async () => {
      const response = await axios.get<GlobalAccuracyStats>(
        `${API_URL}/v1/accuracy/global`
      );
      return response.data;
    },
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    retry: 1,
  });
}

// Hook to fetch accuracy for specific stations
export function useStationAccuracy(stationIds: string[]) {
  return useQuery({
    queryKey: ["station-accuracy", stationIds.join(",")],
    queryFn: async () => {
      const response = await axios.post<StationAccuracy[]>(
        `${API_URL}/v1/accuracy/stations`,
        { station_ids: stationIds }
      );
      return response.data;
    },
    enabled: stationIds.length > 0,
    staleTime: 5 * 60 * 1000,
  });
}

// Export types
export type {
  Station,
  StationForecast,
  StationsResponse,
  StationAccuracy,
  GlobalAccuracyStats,
};
