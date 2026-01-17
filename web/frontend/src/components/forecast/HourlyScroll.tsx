"use client";

import { useRef } from "react";
import { motion } from "framer-motion";
import { format, parseISO } from "date-fns";
import type { HourlyForecast } from "@/hooks/useForecast";

interface HourlyScrollProps {
  forecasts: HourlyForecast[];
  unit?: "C" | "F";
}

// Convert Celsius to Fahrenheit
function toFahrenheit(celsius: number): number {
  return celsius * 9 / 5 + 32;
}

// Convert temperature based on unit
function convertTemp(celsius: number, unit: "C" | "F"): number {
  return unit === "F" ? toFahrenheit(celsius) : celsius;
}

// Get weather icon based on conditions
function getWeatherIcon(hour: number, cloudCover: number, precipProb: number, precip: number): string {
  const isNight = hour < 6 || hour >= 20;

  if (precipProb > 0.6 && precip > 0.5) {
    return isNight ? "🌧️" : "🌧️";
  }
  if (precipProb > 0.4) {
    return isNight ? "🌧️" : "🌦️";
  }
  if (cloudCover > 70) {
    return isNight ? "☁️" : "☁️";
  }
  if (cloudCover > 40) {
    return isNight ? "🌙" : "⛅";
  }
  return isNight ? "🌙" : "☀️";
}

// Get wind direction arrow
function getWindArrow(degrees: number): string {
  const arrows = ["↓", "↙", "←", "↖", "↑", "↗", "→", "↘"];
  const index = Math.round(degrees / 45) % 8;
  return arrows[index];
}

// Get temperature color
function getTempColor(tempC: number): string {
  if (tempC < 0) return "text-blue-400";
  if (tempC < 10) return "text-cyan-400";
  if (tempC < 15) return "text-teal-400";
  if (tempC < 20) return "text-green-400";
  if (tempC < 25) return "text-yellow-400";
  if (tempC < 30) return "text-orange-400";
  return "text-red-400";
}

export function HourlyScroll({ forecasts, unit = "C" }: HourlyScrollProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  const scroll = (direction: "left" | "right") => {
    if (scrollRef.current) {
      const scrollAmount = 300;
      scrollRef.current.scrollBy({
        left: direction === "left" ? -scrollAmount : scrollAmount,
        behavior: "smooth",
      });
    }
  };

  return (
    <div className="relative">
      {/* Scroll buttons */}
      <button
        onClick={() => scroll("left")}
        className="absolute left-0 top-1/2 -translate-y-1/2 z-10 bg-slate-800/80 hover:bg-slate-700 p-2 rounded-full shadow-lg backdrop-blur-sm"
      >
        <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
      </button>
      <button
        onClick={() => scroll("right")}
        className="absolute right-0 top-1/2 -translate-y-1/2 z-10 bg-slate-800/80 hover:bg-slate-700 p-2 rounded-full shadow-lg backdrop-blur-sm"
      >
        <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </button>

      {/* Scrollable container */}
      <div
        ref={scrollRef}
        className="flex gap-3 overflow-x-auto scrollbar-hide px-8 py-2"
        style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
      >
        {forecasts.map((hour, index) => {
          const time = parseISO(hour.datetime);
          const icon = getWeatherIcon(hour.hour, hour.cloud_cover, hour.precipitation_probability, hour.precipitation);
          const displayTemp = convertTemp(hour.temperature, unit);
          const displayFeelsLike = convertTemp(hour.feels_like, unit);
          const tempColor = getTempColor(hour.temperature);
          const isNewDay = index === 0 || hour.hour === 0;

          return (
            <motion.div
              key={hour.datetime}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.02 }}
              className={`flex-shrink-0 w-24 bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-3 text-center hover:bg-white/10 transition-colors ${
                isNewDay ? "border-l-2 border-l-sky-500" : ""
              }`}
            >
              {/* Time */}
              <p className="text-xs text-white/50 mb-1">
                {isNewDay && format(time, "MMM d")}
              </p>
              <p className="text-sm font-medium text-white/80">
                {format(time, "h a")}
              </p>

              {/* Weather icon */}
              <div className="text-2xl my-2">{icon}</div>

              {/* Temperature */}
              <p className={`text-lg font-bold ${tempColor}`}>
                {Math.round(displayTemp)}°
              </p>
              <p className="text-xs text-white/50">
                Feels {Math.round(displayFeelsLike)}°
              </p>

              {/* Precipitation */}
              {hour.precipitation_probability > 0.1 && (
                <div className="mt-2 text-xs text-blue-400">
                  💧 {Math.round(hour.precipitation_probability * 100)}%
                </div>
              )}

              {/* Wind */}
              <div className="mt-1 text-xs text-white/50 flex items-center justify-center gap-1">
                <span>{getWindArrow(hour.wind_direction)}</span>
                <span>{Math.round(hour.wind_speed)} m/s</span>
              </div>

              {/* Humidity */}
              <div className="mt-1 text-xs text-white/40">
                {Math.round(hour.humidity)}% RH
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Gradient overlays */}
      <div className="absolute left-8 top-0 bottom-0 w-8 bg-gradient-to-r from-slate-900/50 to-transparent pointer-events-none" />
      <div className="absolute right-8 top-0 bottom-0 w-8 bg-gradient-to-l from-slate-900/50 to-transparent pointer-events-none" />
    </div>
  );
}
