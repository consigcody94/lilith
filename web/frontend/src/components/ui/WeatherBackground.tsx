"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";

interface WeatherBackgroundProps {
  condition: "clear" | "cloudy" | "rain" | "snow" | "storm";
}

export function WeatherBackground({ condition }: WeatherBackgroundProps) {
  const [particles, setParticles] = useState<{ id: number; x: number; delay: number }[]>([]);

  useEffect(() => {
    if (condition === "rain" || condition === "snow") {
      const newParticles = Array.from({ length: 50 }, (_, i) => ({
        id: i,
        x: Math.random() * 100,
        delay: Math.random() * 2,
      }));
      setParticles(newParticles);
    } else {
      setParticles([]);
    }
  }, [condition]);

  return (
    <div className="fixed inset-0 overflow-hidden -z-10">
      {/* Neural network background image */}
      <div
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{
          backgroundImage: `url("/images/background.png")`,
        }}
      />

      {/* Dark overlay for better text readability - enhanced gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900/80 via-purple-900/40 to-slate-900/90" />

      {/* Mesh gradient overlay for depth */}
      <div
        className="absolute inset-0 opacity-60"
        style={{
          background: `
            radial-gradient(ellipse 80% 50% at 20% 40%, rgba(120, 0, 255, 0.15), transparent),
            radial-gradient(ellipse 60% 40% at 80% 60%, rgba(0, 100, 255, 0.12), transparent),
            radial-gradient(ellipse 50% 30% at 50% 20%, rgba(180, 0, 200, 0.1), transparent)
          `
        }}
      />

      {/* Animated glow effect that moves across the background */}
      <motion.div
        className="absolute inset-0 opacity-40"
        animate={{
          background: [
            "radial-gradient(circle at 20% 30%, rgba(59, 130, 246, 0.3) 0%, transparent 40%)",
            "radial-gradient(circle at 80% 70%, rgba(59, 130, 246, 0.3) 0%, transparent 40%)",
            "radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.3) 0%, transparent 40%)",
            "radial-gradient(circle at 20% 30%, rgba(59, 130, 246, 0.3) 0%, transparent 40%)",
          ],
        }}
        transition={{
          duration: 15,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Secondary pulsing glow */}
      <motion.div
        className="absolute inset-0"
        animate={{
          opacity: [0.1, 0.2, 0.1],
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        style={{
          background: "radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.2) 0%, transparent 60%)",
        }}
      />

      {/* Rain particles */}
      {condition === "rain" &&
        particles.map((particle) => (
          <motion.div
            key={particle.id}
            className="absolute w-0.5 h-6 bg-gradient-to-b from-transparent to-blue-400/60"
            style={{ left: `${particle.x}%` }}
            initial={{ y: "-10%" }}
            animate={{ y: "110vh" }}
            transition={{
              duration: 1,
              repeat: Infinity,
              delay: particle.delay,
              ease: "linear",
            }}
          />
        ))}

      {/* Snow particles */}
      {condition === "snow" &&
        particles.map((particle) => (
          <motion.div
            key={particle.id}
            className="absolute w-2 h-2 rounded-full bg-white/80"
            style={{ left: `${particle.x}%` }}
            initial={{ y: "-10%", rotate: 0 }}
            animate={{ y: "110vh", rotate: 360 }}
            transition={{
              duration: 5 + Math.random() * 3,
              repeat: Infinity,
              delay: particle.delay,
              ease: "linear",
            }}
          />
        ))}

      {/* Noise texture overlay for subtle grain effect */}
      <div
        className="absolute inset-0 opacity-[0.02] mix-blend-overlay"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
        }}
      />

      {/* Vignette effect for depth */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: "radial-gradient(ellipse at center, transparent 0%, rgba(0,0,0,0.4) 100%)",
        }}
      />
    </div>
  );
}
