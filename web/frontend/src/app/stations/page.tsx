"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import { GlassCard } from "@/components/ui/GlassCard";
import { WeatherBackground } from "@/components/ui/WeatherBackground";
import { useWeatherStore } from "@/stores/weatherStore";

// Mock data for 300 US stations - in production this comes from API
const generateMockStations = () => {
  const states = [
    { code: "NY", name: "New York", lat: 40.7, lon: -74.0 },
    { code: "CA", name: "California", lat: 36.7, lon: -119.4 },
    { code: "TX", name: "Texas", lat: 31.0, lon: -100.0 },
    { code: "FL", name: "Florida", lat: 27.6, lon: -81.5 },
    { code: "IL", name: "Illinois", lat: 40.6, lon: -89.3 },
    { code: "PA", name: "Pennsylvania", lat: 41.2, lon: -77.2 },
    { code: "OH", name: "Ohio", lat: 40.4, lon: -82.9 },
    { code: "GA", name: "Georgia", lat: 32.1, lon: -82.9 },
    { code: "NC", name: "North Carolina", lat: 35.6, lon: -79.8 },
    { code: "MI", name: "Michigan", lat: 44.3, lon: -85.6 },
    { code: "WA", name: "Washington", lat: 47.4, lon: -120.5 },
    { code: "AZ", name: "Arizona", lat: 34.0, lon: -111.1 },
    { code: "MA", name: "Massachusetts", lat: 42.4, lon: -71.3 },
    { code: "TN", name: "Tennessee", lat: 35.5, lon: -86.6 },
    { code: "CO", name: "Colorado", lat: 39.0, lon: -105.5 },
  ];

  const stations = [];
  let id = 0;

  for (const state of states) {
    const stationCount = Math.floor(Math.random() * 15) + 10;
    for (let i = 0; i < stationCount && id < 300; i++) {
      const latOffset = (Math.random() - 0.5) * 4;
      const lonOffset = (Math.random() - 0.5) * 6;

      // Generate seasonal base temperature
      const now = new Date();
      const dayOfYear = Math.floor((now.getTime() - new Date(now.getFullYear(), 0, 0).getTime()) / 86400000);
      const seasonalFactor = Math.cos(((dayOfYear - 200) / 365) * 2 * Math.PI);
      const baseTemp = 15 + (40 - Math.abs(state.lat)) * 0.4 + seasonalFactor * 15;

      const high = baseTemp + Math.random() * 8 - 2;
      const low = high - 8 - Math.random() * 6;

      // Generate accuracy metrics
      const tempError = 1.5 + Math.random() * 2;
      const precipAccuracy = 70 + Math.random() * 25;

      // Simulate actual observed temps (with realistic error from predictions)
      const actualError = (Math.random() - 0.5) * 6; // -3 to +3 degree error
      const actualHigh = high + actualError + (Math.random() - 0.5) * 2;
      const actualLow = low + actualError + (Math.random() - 0.5) * 2;

      // Current temperature (between low and high based on time of day)
      const hour = new Date().getHours();
      const dayProgress = Math.sin(((hour - 6) / 12) * Math.PI); // Peak at 2pm
      const currentTemp = actualLow + (actualHigh - actualLow) * Math.max(0, dayProgress);

      stations.push({
        id: `USC00${(300000 + id).toString().padStart(6, "0")}`,
        name: `${state.name} Station ${i + 1}`,
        state: state.code,
        latitude: state.lat + latOffset,
        longitude: state.lon + lonOffset,
        elevation: Math.floor(Math.random() * 2000) + 50,
        // Predicted values (from LILITH model)
        forecast_high: Math.round(high * 10) / 10,
        forecast_low: Math.round(low * 10) / 10,
        // Actual observed values (from station sensors)
        actual_high: Math.round(actualHigh * 10) / 10,
        actual_low: Math.round(actualLow * 10) / 10,
        current_temp: Math.round(currentTemp * 10) / 10,
        // Errors
        high_error: Math.round((high - actualHigh) * 10) / 10,
        low_error: Math.round((low - actualLow) * 10) / 10,
        precipitation_probability: Math.round(Math.random() * 100) / 100,
        temp_error_avg: Math.round(tempError * 100) / 100,
        precip_accuracy: Math.round(precipAccuracy * 10) / 10,
        predictions_count: Math.floor(Math.random() * 500) + 100,
        verified_count: Math.floor(Math.random() * 200) + 50,
        trend: ["improving", "stable", "declining"][Math.floor(Math.random() * 3)] as "improving" | "stable" | "declining",
        last_updated: new Date().toISOString(),
        last_observation: new Date(Date.now() - Math.random() * 300000).toISOString(), // Within last 5 min
      });

      id++;
    }
  }

  return stations;
};

function convertTemp(celsius: number, unit: "C" | "F"): number {
  if (unit === "F") {
    return Math.round((celsius * 9 / 5 + 32) * 10) / 10;
  }
  return Math.round(celsius * 10) / 10;
}

function getTrendIcon(trend: string) {
  switch (trend) {
    case "improving":
      return <span className="text-green-400">↑</span>;
    case "declining":
      return <span className="text-red-400">↓</span>;
    default:
      return <span className="text-yellow-400">→</span>;
  }
}

function getAccuracyColor(mae: number): string {
  if (mae < 2) return "text-green-400";
  if (mae < 3) return "text-yellow-400";
  if (mae < 4) return "text-orange-400";
  return "text-red-400";
}

export default function StationsPage() {
  const { temperatureUnit, setTemperatureUnit } = useWeatherStore();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedState, setSelectedState] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<"name" | "accuracy" | "temp">("name");
  const [viewMode, setViewMode] = useState<"grid" | "table">("grid");

  // Generate mock stations (would come from API in production)
  const allStations = useMemo(() => generateMockStations(), []);

  // Filter and sort stations
  const filteredStations = useMemo(() => {
    let filtered = allStations;

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (s) =>
          s.name.toLowerCase().includes(query) ||
          s.id.toLowerCase().includes(query) ||
          s.state.toLowerCase().includes(query)
      );
    }

    if (selectedState) {
      filtered = filtered.filter((s) => s.state === selectedState);
    }

    // Sort
    switch (sortBy) {
      case "accuracy":
        filtered = [...filtered].sort((a, b) => a.temp_error_avg - b.temp_error_avg);
        break;
      case "temp":
        filtered = [...filtered].sort((a, b) => b.forecast_high - a.forecast_high);
        break;
      default:
        filtered = [...filtered].sort((a, b) => a.name.localeCompare(b.name));
    }

    return filtered;
  }, [allStations, searchQuery, selectedState, sortBy]);

  // Get unique states for filter
  const states = useMemo(() => {
    const uniqueStates = [...new Set(allStations.map((s) => s.state))];
    return uniqueStates.sort();
  }, [allStations]);

  // Calculate global stats
  const globalStats = useMemo(() => {
    const totalPredictions = allStations.reduce((sum, s) => sum + s.predictions_count, 0);
    const totalVerified = allStations.reduce((sum, s) => sum + s.verified_count, 0);
    const avgMAE = allStations.reduce((sum, s) => sum + s.temp_error_avg, 0) / allStations.length;
    const avgPrecipAcc = allStations.reduce((sum, s) => sum + s.precip_accuracy, 0) / allStations.length;

    return {
      totalStations: allStations.length,
      totalPredictions,
      totalVerified,
      avgMAE: Math.round(avgMAE * 100) / 100,
      avgPrecipAcc: Math.round(avgPrecipAcc * 10) / 10,
    };
  }, [allStations]);

  return (
    <main className="relative min-h-screen overflow-hidden">
      <WeatherBackground condition="clear" />

      <div className="relative z-10 min-h-screen">
        {/* Header */}
        <header className="w-full py-4 px-4 border-b border-white/10">
          <div className="max-w-[1800px] mx-auto flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/" className="text-white/60 hover:text-white transition-colors">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
              </Link>
              <div>
                <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                  <svg className="w-8 h-8 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  Station Command Center
                </h1>
                <p className="text-white/50 text-sm">Real-time predictions & accuracy tracking for all stations</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {/* Temperature Toggle */}
              <div className="flex bg-white/10 backdrop-blur-sm rounded-lg p-1">
                <button
                  onClick={() => setTemperatureUnit("C")}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                    temperatureUnit === "C"
                      ? "bg-purple-500 text-white shadow-lg"
                      : "text-white/70 hover:text-white"
                  }`}
                >
                  °C
                </button>
                <button
                  onClick={() => setTemperatureUnit("F")}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                    temperatureUnit === "F"
                      ? "bg-purple-500 text-white shadow-lg"
                      : "text-white/70 hover:text-white"
                  }`}
                >
                  °F
                </button>
              </div>

              {/* View Mode Toggle */}
              <div className="flex bg-white/10 backdrop-blur-sm rounded-lg p-1">
                <button
                  onClick={() => setViewMode("grid")}
                  className={`px-3 py-1.5 rounded-md text-sm transition-all ${
                    viewMode === "grid"
                      ? "bg-white/20 text-white"
                      : "text-white/70 hover:text-white"
                  }`}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                  </svg>
                </button>
                <button
                  onClick={() => setViewMode("table")}
                  className={`px-3 py-1.5 rounded-md text-sm transition-all ${
                    viewMode === "table"
                      ? "bg-white/20 text-white"
                      : "text-white/70 hover:text-white"
                  }`}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="max-w-[1800px] mx-auto px-4 py-6">
          {/* Global Stats Dashboard */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
            <GlassCard className="text-center" glow>
              <p className="text-white/50 text-xs uppercase tracking-wider mb-1">Total Stations</p>
              <p className="text-3xl font-bold text-white">{globalStats.totalStations}</p>
            </GlassCard>
            <GlassCard className="text-center" glow>
              <p className="text-white/50 text-xs uppercase tracking-wider mb-1">Predictions Made</p>
              <p className="text-3xl font-bold text-purple-400">{globalStats.totalPredictions.toLocaleString()}</p>
            </GlassCard>
            <GlassCard className="text-center" glow>
              <p className="text-white/50 text-xs uppercase tracking-wider mb-1">Verified</p>
              <p className="text-3xl font-bold text-cyan-400">{globalStats.totalVerified.toLocaleString()}</p>
            </GlassCard>
            <GlassCard className="text-center" glow>
              <p className="text-white/50 text-xs uppercase tracking-wider mb-1">Avg Temp MAE</p>
              <p className={`text-3xl font-bold ${getAccuracyColor(globalStats.avgMAE)}`}>
                {globalStats.avgMAE}°{temperatureUnit}
              </p>
            </GlassCard>
            <GlassCard className="text-center" glow>
              <p className="text-white/50 text-xs uppercase tracking-wider mb-1">Precip Accuracy</p>
              <p className="text-3xl font-bold text-green-400">{globalStats.avgPrecipAcc}%</p>
            </GlassCard>
          </div>

          {/* Filters */}
          <GlassCard className="mb-6">
            <div className="flex flex-wrap items-center gap-4">
              {/* Search */}
              <div className="flex-1 min-w-[200px]">
                <div className="relative">
                  <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  <input
                    type="text"
                    placeholder="Search stations..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                  />
                </div>
              </div>

              {/* State Filter */}
              <select
                value={selectedState || ""}
                onChange={(e) => setSelectedState(e.target.value || null)}
                className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500/50"
              >
                <option value="" className="bg-slate-900">All States</option>
                {states.map((state) => (
                  <option key={state} value={state} className="bg-slate-900">
                    {state}
                  </option>
                ))}
              </select>

              {/* Sort */}
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as "name" | "accuracy" | "temp")}
                className="px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500/50"
              >
                <option value="name" className="bg-slate-900">Sort by Name</option>
                <option value="accuracy" className="bg-slate-900">Sort by Accuracy</option>
                <option value="temp" className="bg-slate-900">Sort by Temperature</option>
              </select>

              <p className="text-white/50 text-sm">
                Showing {filteredStations.length} of {allStations.length} stations
              </p>
            </div>
          </GlassCard>

          {/* Stations Display */}
          {viewMode === "grid" ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
              {filteredStations.map((station) => (
                <GlassCard
                  key={station.id}
                  className="cursor-pointer hover:scale-[1.02] transition-transform"
                  hover
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="font-semibold text-white text-sm line-clamp-1">{station.name}</h3>
                      <p className="text-white/40 text-xs">{station.id}</p>
                    </div>
                    <span className="px-2 py-0.5 bg-white/10 rounded text-xs text-white/60">
                      {station.state}
                    </span>
                  </div>

                  {/* Current Temperature (Live) */}
                  <div className="text-center mb-3">
                    <div className="flex items-center justify-center gap-2">
                      <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                      <span className="text-white/50 text-xs uppercase tracking-wider">Current</span>
                    </div>
                    <p className="text-3xl font-bold text-white mt-1">
                      {convertTemp(station.current_temp, temperatureUnit)}°
                    </p>
                  </div>

                  {/* Predicted vs Actual */}
                  <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                    {/* High Temps */}
                    <div className="bg-white/5 rounded-lg p-2">
                      <p className="text-white/40 mb-1">High</p>
                      <div className="flex justify-between items-center">
                        <div>
                          <p className="text-white/50">Pred:</p>
                          <p className="text-red-400 font-semibold">{convertTemp(station.forecast_high, temperatureUnit)}°</p>
                        </div>
                        <div>
                          <p className="text-white/50">Actual:</p>
                          <p className="text-orange-400 font-semibold">{convertTemp(station.actual_high, temperatureUnit)}°</p>
                        </div>
                      </div>
                      <div className="mt-1 flex items-center justify-center">
                        <span className={`text-xs font-medium ${Math.abs(station.high_error) < 2 ? "text-green-400" : Math.abs(station.high_error) < 4 ? "text-yellow-400" : "text-red-400"}`}>
                          {station.high_error > 0 ? "+" : ""}{station.high_error}° error
                        </span>
                      </div>
                    </div>
                    {/* Low Temps */}
                    <div className="bg-white/5 rounded-lg p-2">
                      <p className="text-white/40 mb-1">Low</p>
                      <div className="flex justify-between items-center">
                        <div>
                          <p className="text-white/50">Pred:</p>
                          <p className="text-blue-400 font-semibold">{convertTemp(station.forecast_low, temperatureUnit)}°</p>
                        </div>
                        <div>
                          <p className="text-white/50">Actual:</p>
                          <p className="text-cyan-400 font-semibold">{convertTemp(station.actual_low, temperatureUnit)}°</p>
                        </div>
                      </div>
                      <div className="mt-1 flex items-center justify-center">
                        <span className={`text-xs font-medium ${Math.abs(station.low_error) < 2 ? "text-green-400" : Math.abs(station.low_error) < 4 ? "text-yellow-400" : "text-red-400"}`}>
                          {station.low_error > 0 ? "+" : ""}{station.low_error}° error
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Station Stats */}
                  <div className="pt-2 border-t border-white/10">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-1">
                        <span className="text-white/50">Avg Error:</span>
                        <span className={getAccuracyColor(station.temp_error_avg)}>
                          ±{station.temp_error_avg}°
                        </span>
                        {getTrendIcon(station.trend)}
                      </div>
                      <span className="text-white/40">
                        {new Date(station.last_observation).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                      </span>
                    </div>
                  </div>
                </GlassCard>
              ))}
            </div>
          ) : (
            /* Table View */
            <GlassCard>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-white/50 text-sm border-b border-white/10">
                      <th className="pb-3 px-2">Station</th>
                      <th className="pb-3 px-2">State</th>
                      <th className="pb-3 px-2 text-center">Current</th>
                      <th className="pb-3 px-2 text-center">Pred High</th>
                      <th className="pb-3 px-2 text-center">Actual High</th>
                      <th className="pb-3 px-2 text-center">High Err</th>
                      <th className="pb-3 px-2 text-center">Pred Low</th>
                      <th className="pb-3 px-2 text-center">Actual Low</th>
                      <th className="pb-3 px-2 text-center">Low Err</th>
                      <th className="pb-3 px-2 text-center">Trend</th>
                      <th className="pb-3 px-2 text-right">Updated</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredStations.map((station) => (
                      <tr
                        key={station.id}
                        className="border-b border-white/5 hover:bg-white/5 cursor-pointer transition-colors"
                      >
                        <td className="py-3 px-2">
                          <p className="text-white text-sm font-medium">{station.name}</p>
                          <p className="text-white/40 text-xs">{station.id}</p>
                        </td>
                        <td className="py-3 px-2">
                          <span className="px-2 py-0.5 bg-white/10 rounded text-xs text-white/60">
                            {station.state}
                          </span>
                        </td>
                        <td className="py-3 px-2 text-center">
                          <div className="flex items-center justify-center gap-1">
                            <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
                            <span className="text-white font-bold">{convertTemp(station.current_temp, temperatureUnit)}°</span>
                          </div>
                        </td>
                        <td className="py-3 px-2 text-center text-red-400 font-medium">
                          {convertTemp(station.forecast_high, temperatureUnit)}°
                        </td>
                        <td className="py-3 px-2 text-center text-orange-400 font-medium">
                          {convertTemp(station.actual_high, temperatureUnit)}°
                        </td>
                        <td className={`py-3 px-2 text-center font-medium ${Math.abs(station.high_error) < 2 ? "text-green-400" : Math.abs(station.high_error) < 4 ? "text-yellow-400" : "text-red-400"}`}>
                          {station.high_error > 0 ? "+" : ""}{station.high_error}°
                        </td>
                        <td className="py-3 px-2 text-center text-blue-400 font-medium">
                          {convertTemp(station.forecast_low, temperatureUnit)}°
                        </td>
                        <td className="py-3 px-2 text-center text-cyan-400 font-medium">
                          {convertTemp(station.actual_low, temperatureUnit)}°
                        </td>
                        <td className={`py-3 px-2 text-center font-medium ${Math.abs(station.low_error) < 2 ? "text-green-400" : Math.abs(station.low_error) < 4 ? "text-yellow-400" : "text-red-400"}`}>
                          {station.low_error > 0 ? "+" : ""}{station.low_error}°
                        </td>
                        <td className="py-3 px-2 text-center text-lg">
                          {getTrendIcon(station.trend)}
                        </td>
                        <td className="py-3 px-2 text-right text-white/40 text-xs">
                          {new Date(station.last_observation).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </GlassCard>
          )}
        </div>

        {/* Footer */}
        <footer className="border-t border-white/10 py-6 px-4 mt-8">
          <div className="max-w-[1800px] mx-auto flex items-center justify-between text-sm text-white/40">
            <div className="flex items-center gap-3">
              <img src="/images/logo.png" alt="L.I.L.I.T.H." className="h-8 w-auto opacity-50" />
              <span>Station Command Center</span>
            </div>
            <div className="flex items-center gap-4">
              <span>Last updated: {new Date().toLocaleString()}</span>
              <Link href="/" className="text-purple-400 hover:text-purple-300 transition-colors">
                ← Back to Forecast
              </Link>
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
}
