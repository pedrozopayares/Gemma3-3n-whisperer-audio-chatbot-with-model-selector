import React from "react";
import { IconSun, IconMoon } from "./Icons";

/**
 * Animated theme toggle button.
 *
 * Props:
 * - theme: "dark" | "light"
 * - onToggle(): switch theme
 */
export default function ThemeToggle({ theme, onToggle }) {
  const isDark = theme === "dark";

  return (
    <button
      onClick={onToggle}
      className="p-2 rounded-lg hover:bg-hover text-secondary hover:text-primary transition-colors"
      title={isDark ? "Cambiar a modo claro" : "Cambiar a modo oscuro"}
      aria-label={isDark ? "Cambiar a modo claro" : "Cambiar a modo oscuro"}
    >
      {isDark ? (
        <IconSun className="w-5 h-5" />
      ) : (
        <IconMoon className="w-5 h-5" />
      )}
    </button>
  );
}
