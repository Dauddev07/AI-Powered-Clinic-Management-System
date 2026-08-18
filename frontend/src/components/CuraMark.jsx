// Cura's own visual identity — a speech bubble with three dots, optionally paired
// with the small sparkle accent. Already hand-drawn independently in three places
// (FloatingChatButton, PrimaryNavMobile's chat tab, Landing's "Meet Cura" mock) —
// shared here so ChatPage (where Cura actually talks) can finally use the same mark
// instead of its own unrelated stethoscope glyph, and so any future spot never
// drifts from it either.
export function CuraBubbleIcon({ size = 24, className }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M4 12.5C4 7.81 8.03 4 13 4s9 3.81 9 8.5-4.03 8.5-9 8.5c-1.09 0-2.13-.19-3.1-.53L4 21l1.2-4.02A8.16 8.16 0 0 1 4 12.5Z" />
      <circle cx="9.5" cy="12.5" r="1" fill="currentColor" stroke="none" />
      <circle cx="13" cy="12.5" r="1" fill="currentColor" stroke="none" />
      <circle cx="16.5" cy="12.5" r="1" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function CuraSparkle({ size = 12, className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" width={size} height={size} fill="currentColor" aria-hidden="true">
      <path d="M12 2 13.8 8.9 21 12l-7.2 3.1L12 22l-1.8-6.9L3 12l7.2-3.1L12 2Z" />
    </svg>
  );
}
