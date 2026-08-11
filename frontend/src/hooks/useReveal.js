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
    if (!el) return;
    // Reported live: a card sitting above the fold (e.g. the patient
    // dashboard's first two cards, visible on load with no scroll needed)
    // could get stuck permanently invisible on some loads. Root cause: the
    // IntersectionObserver's own initial report (fired right after
    // observe()) can be taken before layout has fully settled — a
    // font-display:swap webfont swapping in, or a skeleton→content swap,
    // shifting element position between the ref callback firing and the
    // observer's first measurement — and for an above-the-fold element
    // there's no later scroll event to ever re-trigger a correction.
    // Checking synchronously here, independent of the observer's own
    // timing, covers exactly that case: already-visible now means revealed
    // now, no race to lose.
    const rect = el.getBoundingClientRect();
    const alreadyVisible = rect.top < window.innerHeight && rect.bottom > 0;
    if (alreadyVisible) el.classList.add("revealed");
    observerRef.current.observe(el);
  }, []);
}

// Small per-item stagger for grouped reveals (cards, rows) — capped at 5 so
// a long list doesn't end up with an oddly long tail delay.
export function revealDelayClass(i) {
  if (i <= 0) return "";
  return `reveal-delay-${Math.min(i, 5)}`;
}
