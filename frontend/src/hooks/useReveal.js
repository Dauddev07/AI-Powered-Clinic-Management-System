import { useCallback, useEffect, useRef } from "react";

// Shared scroll-reveal: one IntersectionObserver per calling component,
// toggling the global `.revealed` class (see index.css) on whatever DOM node
// is registered via the returned ref callback — added as soon as any part of
// it is visible, removed once it scrolls back out, so the fade/slide replays
// every time an element re-enters the viewport in either scroll direction.
// prefers-reduced-motion is handled entirely in CSS (the `.reveal` base style
// is a no-op under that media query), so this hook doesn't need to branch on it.
//
// threshold is 0, not a percentage of the target's own area — a percentage
// threshold requires that fraction of the ELEMENT's own height to be inside
// the viewport, which silently never fires for a target taller than the
// viewport itself (a real case here: some content lists render far taller
// than any screen). rootMargin's -15% still makes the reveal start a little
// before the element's edge actually reaches the viewport edge.
export function useReveal() {
  const observerRef = useRef(null);
  if (observerRef.current === null) {
    observerRef.current = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          entry.target.classList.toggle("revealed", entry.isIntersecting);
        }
      },
      { threshold: 0, rootMargin: "0px 0px -15% 0px" },
    );
  }

  useEffect(() => () => observerRef.current?.disconnect(), []);

  return useCallback((el) => {
    if (el) observerRef.current.observe(el);
  }, []);
}

// Small per-item stagger for grouped reveals (cards, rows) — capped at 5 so
// a long list doesn't end up with an oddly long tail delay.
export function revealDelayClass(i) {
  if (i <= 0) return "";
  return `reveal-delay-${Math.min(i, 5)}`;
}
