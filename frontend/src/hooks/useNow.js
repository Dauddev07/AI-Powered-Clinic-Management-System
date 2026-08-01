import { useEffect, useState } from "react";

// A ticking clock for components that derive display state (e.g. "in
// progress" vs "upcoming") from comparing timestamps against the current
// time — without this, that derived state would only update on the next
// unrelated re-render or full page reload, not when a slot's start time
// actually arrives while the page sits open.
export function useNow(intervalMs = 30000) {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);

  return now;
}
