import { useEffect, useRef, useState } from "react";

const EASE = (t) => 1 - Math.pow(1 - t, 3);

// Animates a stat number counting up from its previous value to `target`
// whenever `target` changes (e.g. once dashboard data finishes loading, or a
// live count changes on the next poll) — turns a plain number swap into a
// small sign of a live, active dashboard instead of numbers just appearing.
// Returns `target` itself (no animation) while it's null/undefined, and
// immediately/synchronously while prefers-reduced-motion is set.
export function useCountUp(target, { duration = 800 } = {}) {
  const [value, setValue] = useState(target ?? 0);
  const fromRef = useRef(0);
  const rafRef = useRef(null);

  useEffect(() => {
    if (target === null || target === undefined) return;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduceMotion) {
      setValue(target);
      return;
    }

    const from = fromRef.current;
    if (from === target) {
      setValue(target);
      return;
    }
    const start = performance.now();

    function step(now) {
      const elapsed = now - start;
      const progress = Math.min(1, elapsed / duration);
      const eased = EASE(progress);
      setValue(Math.round(from + (target - from) * eased));
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(step);
      } else {
        fromRef.current = target;
      }
    }

    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, duration]);

  return target === null || target === undefined ? target : value;
}
