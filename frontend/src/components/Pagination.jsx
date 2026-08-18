import styles from "./Pagination.module.css";

// Windowed page numbers with "…" gaps, e.g. [1, "…", 4, 5, 6, "…", 12] — keeps
// the control usable at any total page count instead of rendering one button
// per page.
function buildPageItems(page, totalPages) {
  const items = [];
  const add = (v) => items.push(v);
  const window = 1;

  add(1);
  if (page - window > 2) add("ellipsis-start");

  for (let p = Math.max(2, page - window); p <= Math.min(totalPages - 1, page + window); p++) {
    add(p);
  }

  if (page + window < totalPages - 1) add("ellipsis-end");
  if (totalPages > 1) add(totalPages);

  return items;
}

export default function Pagination({ page, pageSize, total, onPageChange }) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  if (totalPages <= 1) return null;

  const pageItems = buildPageItems(page, totalPages);
  const rangeStart = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const rangeEnd = Math.min(total, page * pageSize);

  // Reported live: clicking a page number (any of the 4 screens this component
  // is used on — BookAppointment, admin DoctorList/IngestionLogScreen/Feedback)
  // swapped the list content but left scroll position untouched, so a click
  // made near the bottom of a long list left the newly-loaded page's own top
  // (and this pagination control itself) off-screen above the viewport,
  // reading as broken/unresponsive. Centralized here (not in each of the 4
  // callers) since this is the one shared control they all funnel through —
  // fixing it here fixes every one of them at once, and any future page that
  // adopts Pagination gets this for free too.
  //
  // Reported live (2nd report, admin DoctorList only): landing on page 1 via
  // the "‹" Previous button specifically never scrolled, while the same
  // button's "›" Next sibling and the numbered page buttons worked fine.
  // Cause: only Previous disables itself (page<=1) as a *direct result* of
  // the very click that fires this handler — and Chromium silently drops a
  // `window.scrollTo(..., {behavior:"smooth"})` requested from a control that
  // becomes disabled/loses focus within the same tick, before the animation
  // ever starts. Next and the number buttons never disable themselves on
  // click, so they never hit this. Deferring to the next animation frame lets
  // that disabled-state re-render (and the resulting blur) settle first, so
  // the scroll request is no longer racing it.
  const goToPage = (newPage) => {
    onPageChange(newPage);
    requestAnimationFrame(() => {
      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
    });
  };

  return (
    <nav className={styles.pagination} aria-label="Pagination">
      <span className={styles.rangeText}>
        {rangeStart}–{rangeEnd} of {total}
      </span>

      <div className={styles.controls}>
        <button
          type="button"
          className={styles.navBtn}
          onClick={() => goToPage(page - 1)}
          disabled={page <= 1}
          aria-label="Previous page"
        >
          ‹
        </button>

        <div className={styles.pageList}>
          {pageItems.map((item, i) =>
            typeof item === "number" ? (
              <button
                type="button"
                key={item}
                className={item === page ? styles.pageBtnActive : styles.pageBtn}
                onClick={() => goToPage(item)}
                aria-current={item === page ? "page" : undefined}
                aria-label={`Page ${item}`}
              >
                {item}
              </button>
            ) : (
              <span key={`${item}-${i}`} className={styles.ellipsis}>
                …
              </span>
            ),
          )}
        </div>

        <button
          type="button"
          className={styles.navBtn}
          onClick={() => goToPage(page + 1)}
          disabled={page >= totalPages}
          aria-label="Next page"
        >
          ›
        </button>
      </div>
    </nav>
  );
}
