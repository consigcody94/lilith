"use client";

import { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  variant?: "default" | "dark" | "light" | "accent";
  hover?: boolean;
  glow?: boolean;
}

export function GlassCard({
  children,
  className,
  variant = "default",
  hover = false,
  glow = false,
}: GlassCardProps) {
  const variants = {
    default: "bg-white/[0.08] border-white/[0.15] shadow-[0_8px_32px_rgba(0,0,0,0.3)]",
    dark: "bg-black/30 border-white/10 shadow-[0_8px_32px_rgba(0,0,0,0.4)]",
    light: "bg-white/20 border-white/30 shadow-[0_8px_32px_rgba(255,255,255,0.1)]",
    accent: "bg-purple-900/20 border-purple-500/30 shadow-[0_8px_32px_rgba(139,92,246,0.15)]",
  };

  return (
    <div
      className={cn(
        "relative backdrop-blur-2xl border rounded-2xl p-6 transition-all duration-300",
        variants[variant],
        hover && "hover:bg-white/[0.12] hover:shadow-[0_12px_40px_rgba(0,0,0,0.4)] hover:scale-[1.01] hover:border-white/25",
        glow && "before:absolute before:inset-0 before:rounded-2xl before:bg-gradient-to-r before:from-purple-500/10 before:to-blue-500/10 before:blur-xl before:-z-10",
        className
      )}
    >
      {/* Inner glow effect */}
      <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-white/[0.08] via-transparent to-transparent pointer-events-none" />
      <div className="relative z-10">{children}</div>
    </div>
  );
}
