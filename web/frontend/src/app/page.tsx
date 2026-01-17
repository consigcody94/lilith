"use client";

import { useState } from "react";
import Link from "next/link";
import { GlassCard } from "@/components/ui/GlassCard";
import { TemperatureDisplay } from "@/components/forecast/TemperatureDisplay";
import { ForecastChart } from "@/components/charts/ForecastChart";
import { LocationSearch } from "@/components/LocationSearch";
import { DailyCards } from "@/components/forecast/DailyCards";
import { HourlyScroll } from "@/components/forecast/HourlyScroll";
import { WeatherBackground } from "@/components/ui/WeatherBackground";
import { Settings } from "@/components/Settings";
import { AccuracyTracker } from "@/components/accuracy/AccuracyTracker";
import { useForecast, useHourlyForecast, useAccuracyReport } from "@/hooks/useForecast";
import { useWeatherStore } from "@/stores/weatherStore";

export default function Home() {
  const [location, setLocation] = useState({
    latitude: 40.7128,
    longitude: -74.006,
    name: "New York, NY",
  });

  const [settingsOpen, setSettingsOpen] = useState(false);

  const { temperatureUnit, setTemperatureUnit } = useWeatherStore();
  const { data: forecast, isLoading, error } = useForecast(location);
  const { data: hourlyForecast, isLoading: hourlyLoading } = useHourlyForecast(location, 48);
  const { data: accuracyReport, isLoading: accuracyLoading } = useAccuracyReport(location, 30);

  return (
    <main className="relative min-h-screen overflow-hidden">
      {/* Dynamic weather background */}
      <WeatherBackground condition={forecast?.forecasts[0] ? "clear" : "clear"} />

      {/* Settings Panel */}
      <Settings isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />

      {/* Content */}
      <div className="relative z-10 min-h-screen flex flex-col">
        {/* Header */}
        <header className="w-full py-4 px-4">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            {/* Logo and Title - Centered */}
            <div className="flex-1" />
            <div className="flex flex-col items-center">
              <img
                src="/images/logo.png"
                alt="LILITH"
                className="h-28 w-auto object-contain drop-shadow-[0_0_20px_rgba(139,92,246,0.5)]"
              />
              <p className="text-white/60 text-sm tracking-wider -mt-1">
                90-Day Weather Forecasting
              </p>
            </div>

            {/* Controls - Right side */}
            <div className="flex-1 flex justify-end items-center gap-3">
              {/* Command Center Link */}
              <Link
                href="/stations"
                className="flex items-center gap-2 px-3 py-1.5 bg-purple-500/20 hover:bg-purple-500/30 border border-purple-500/30 rounded-lg text-purple-300 hover:text-purple-200 transition-colors text-sm font-medium"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span className="hidden sm:inline">Command Center</span>
              </Link>

              {/* Temperature Unit Toggle */}
              <div className="flex bg-white/10 backdrop-blur-sm rounded-lg p-1">
                <button
                  onClick={() => setTemperatureUnit("C")}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                    temperatureUnit === "C"
                      ? "bg-sky-500 text-white shadow-lg"
                      : "text-white/70 hover:text-white"
                  }`}
                >
                  °C
                </button>
                <button
                  onClick={() => setTemperatureUnit("F")}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                    temperatureUnit === "F"
                      ? "bg-sky-500 text-white shadow-lg"
                      : "text-white/70 hover:text-white"
                  }`}
                >
                  °F
                </button>
              </div>

              {/* Settings Button */}
              <button
                onClick={() => setSettingsOpen(true)}
                className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                title="Settings"
              >
                <svg
                  className="w-6 h-6 text-white/70 hover:text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                  />
                </svg>
              </button>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-8">
          {/* Location Search - Centered */}
          <div className="max-w-lg mx-auto mb-6">
            <LocationSearch onLocationSelect={setLocation} />
          </div>

          {/* Current Location Display */}
          <div className="text-center mb-6">
            <h2 className="text-2xl font-semibold text-white">{location.name}</h2>
            <p className="text-white/50 text-sm">
              {location.latitude.toFixed(4)}°N, {location.longitude.toFixed(4)}°W
            </p>
          </div>

          {/* Error Display */}
          {error && (
            <div className="max-w-lg mx-auto mb-6 p-4 bg-red-500/20 border border-red-500/50 rounded-xl text-red-200 text-center">
              Failed to load forecast. Please try again.
            </div>
          )}

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            {/* Current Conditions */}
            <GlassCard className="lg:col-span-1" glow hover>
              <h3 className="text-lg font-semibold text-white/90 mb-4 flex items-center gap-2">
                <span className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></span>
                Tomorrow
              </h3>
              {isLoading ? (
                <div className="animate-pulse">
                  <div className="h-32 bg-white/10 rounded-lg"></div>
                </div>
              ) : forecast?.forecasts[0] ? (
                <TemperatureDisplay
                  high={forecast.forecasts[0].temperature_max}
                  low={forecast.forecasts[0].temperature_min}
                  precipitation={forecast.forecasts[0].precipitation}
                  precipitationProbability={forecast.forecasts[0].precipitation_probability}
                  unit={temperatureUnit}
                />
              ) : (
                <p className="text-white/50">No forecast available</p>
              )}
            </GlassCard>

            {/* 90-Day Chart */}
            <GlassCard className="lg:col-span-2" glow>
              <h3 className="text-lg font-semibold text-white/90 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                </svg>
                90-Day Temperature Forecast
              </h3>
              {isLoading ? (
                <div className="animate-pulse h-64 bg-white/10 rounded-lg"></div>
              ) : forecast ? (
                <ForecastChart data={forecast.forecasts} unit={temperatureUnit} />
              ) : (
                <p className="text-white/50">No forecast available</p>
              )}
            </GlassCard>
          </div>

          {/* Hourly Forecast */}
          <GlassCard className="mb-6" hover>
            <h3 className="text-lg font-semibold text-white/90 mb-4 flex items-center gap-2">
              <svg className="w-5 h-5 text-sky-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              48-Hour Forecast
            </h3>
            {hourlyLoading ? (
              <div className="flex gap-3 overflow-hidden">
                {[...Array(12)].map((_, i) => (
                  <div key={i} className="flex-shrink-0 w-24 h-40 animate-pulse bg-white/10 rounded-xl" />
                ))}
              </div>
            ) : hourlyForecast ? (
              <HourlyScroll forecasts={hourlyForecast.forecasts} unit={temperatureUnit} />
            ) : (
              <p className="text-white/50">No hourly forecast available</p>
            )}
          </GlassCard>

          {/* Daily Forecast Cards */}
          <GlassCard className="mb-6" hover>
            <h3 className="text-lg font-semibold text-white/90 mb-4 flex items-center gap-2">
              <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              14-Day Extended Forecast
            </h3>
            {isLoading ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
                {[...Array(7)].map((_, i) => (
                  <div key={i} className="animate-pulse h-32 bg-white/10 rounded-lg"></div>
                ))}
              </div>
            ) : forecast ? (
              <DailyCards forecasts={forecast.forecasts.slice(0, 14)} unit={temperatureUnit} />
            ) : (
              <p className="text-white/50">No forecast available</p>
            )}
          </GlassCard>

          {/* Prediction Accuracy Tracker */}
          <GlassCard className="mb-6" variant="accent" glow hover>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white/90 flex items-center gap-2">
                <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Prediction Accuracy
              </h3>
              <span className="text-xs text-white/60 bg-white/10 px-3 py-1.5 rounded-full border border-white/10">
                Predictions vs Actual Weather
              </span>
            </div>
            <AccuracyTracker
              report={accuracyReport}
              isLoading={accuracyLoading}
              unit={temperatureUnit}
            />
          </GlassCard>

          {/* Footer */}
          <footer className="mt-12 pt-8 border-t border-white/10">
            <div className="max-w-4xl mx-auto">
              <div className="flex flex-col md:flex-row items-center justify-between gap-4 mb-6">
                <div className="flex items-center gap-3">
                  <img
                    src="/images/logo.png"
                    alt="L.I.L.I.T.H."
                    className="h-10 w-auto opacity-60"
                  />
                  <div className="text-left">
                    <p className="text-white/60 text-sm font-medium">L.I.L.I.T.H.</p>
                    <p className="text-white/40 text-xs">Open Source Weather AI</p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <Link
                    href="/stations"
                    className="flex items-center gap-2 text-purple-400/70 hover:text-purple-300 transition-colors"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    Command Center
                  </Link>
                  <Link
                    href="/historical"
                    className="flex items-center gap-2 text-white/50 hover:text-white transition-colors"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Historical
                  </Link>
                  <a
                    href="https://github.com/consigcody94/lilith"
                    className="flex items-center gap-2 text-white/50 hover:text-white transition-colors"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                      <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                    </svg>
                    GitHub
                  </a>
                </div>
              </div>
              <div className="flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-white/30">
                <p>
                  Model: <span className="text-purple-400/70">{forecast?.model_version || "demo-v1"}</span> |
                  Generated: <span className="text-sky-400/70">{forecast?.generated_at ? new Date(forecast.generated_at).toLocaleString() : "N/A"}</span>
                </p>
                <p className="text-white/20">
                  Built with GHCN public weather data
                </p>
              </div>
            </div>
          </footer>
        </div>
      </div>
    </main>
  );
}
