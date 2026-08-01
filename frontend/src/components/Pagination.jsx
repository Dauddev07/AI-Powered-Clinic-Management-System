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

  return (
    <nav className={styles.pagination} aria-label="Pagination">
      <span className={styles.rangeText}>
        {rangeStart}–{rangeEnd} of {total}
      </span>

      <div className={styles.controls}>
        <button
          type="button"
          className={styles.navBtn}
          onClick={() => onPageChange(page - 1)}
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
                onClick={() => onPageChange(item)}
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
          onClick={() => onPageChange(page + 1)}
          disabled={page >= totalPages}
          aria-label="Next page"
        >
          ›
        </button>
      </div>
    </nav>
  );
}
