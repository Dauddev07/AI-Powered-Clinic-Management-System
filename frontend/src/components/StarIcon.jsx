// The single star glyph shared by the chat feedback prompt's interactive StarRating
// (ChatPage.jsx) and the admin feedback table's read-only Stars (admin/Feedback.jsx) —
// those two components differ in behavior (interactive input vs. static display) enough
// that unifying them isn't a clean fit, but they were both hand-drawing this exact same
// SVG path independently, which is worth sharing.
export default function StarIcon({ filled, size = 24, className, strokeWidth = 1.6 }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
    >
      <path d="M12 2.5l2.9 6.06 6.6.86-4.85 4.63 1.24 6.6L12 17.6l-5.89 3.05 1.24-6.6-4.85-4.63 6.6-.86L12 2.5Z" />
    </svg>
  );
}
