import { useEffect, useState } from "react";

// Tracks whether a CSS media query currently matches — for the rare case a
// component needs to change actual JS behavior/content (not just styling) at a
// breakpoint, e.g. swapping in a shorter placeholder string that CSS alone can't
// truncate cleanly. Prefer a plain CSS media query for anything stylable; reach
// for this only when the difference is in the JS-rendered content itself.
export function useMediaQuery(query) {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);

  useEffect(() => {
    const mql = window.matchMedia(query);
    const onChange = () => setMatches(mql.matches);
    onChange();
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}
