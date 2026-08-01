import { Fragment, useEffect, useState } from "react";
import { fetchIngestionLogs } from "../../api/adminDoctors";
import { ApiError } from "../../api/client";
import StatusBadge from "../../components/StatusBadge";
import EmptyState from "../../components/EmptyState";
import Pagination from "../../components/Pagination";
import { useReveal } from "../../hooks/useReveal";
import { formatDateTimeLocal as formatDate } from "../../utils/formatDateTime";
import styles from "./AdminScreens.module.css";

const PAGE_SIZE = 10;

// preview_failed (abandoned at preview, nothing was ever confirmed) is styled as a
// neutral/muted badge, distinct from failed (a confirm attempt that was rejected) which
// stays red — so the two read as clearly different severities, not near-identical labels.
const STATUS_META = {
  success: { label: "Success", tone: "success" },
  failed: { label: "Failed", tone: "error" },
  preview_failed: { label: "Preview failed", tone: "neutral" },
};

export default function IngestionLogScreen() {
  const revealRef = useReveal();
  const [logs, setLogs] = useState(null);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);

  useEffect(() => {
    fetchIngestionLogs({ limit: PAGE_SIZE, offset: (page - 1) * PAGE_SIZE })
      .then((data) => {
        setLogs(data.items);
        setTotal(data.total);
      })
      .catch((err) => setError(err instanceof ApiError ? err.detail || err.message : "Failed to load ingestion log."));
  }, [page]);

  return (
    <div>
      <h1 className={styles.title}>Doctor CSV ingestion log</h1>
      <p className={styles.subtitle}>Past doctor roster imports for this clinic, newest first.</p>

      <div className={`${styles.card} reveal`} ref={revealRef}>
        {error && <p className={styles.errorText}>{error}</p>}

        {logs && total === 0 && (
          <EmptyState
            icon="inbox"
            message="No ingestions yet."
            actionLabel="Import a doctor CSV"
            actionTo="/admin/doctors/import"
          />
        )}

        {logs && total > 0 && (
          <>
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Filename</th>
                    <th>Status</th>
                    <th>Accepted</th>
                    <th>Rejected</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {logs.map((log) => {
                    const meta = STATUS_META[log.status] ?? { label: log.status, tone: "neutral" };
                    return (
                    <Fragment key={log.id}>
                      <tr>
                        <td data-label="Timestamp">{formatDate(log.created_at)}</td>
                        <td data-label="Filename">{log.filename ?? "—"}</td>
                        <td data-label="Status">
                          <StatusBadge tone={meta.tone} label={meta.label} />
                        </td>
                        <td data-label="Accepted">{log.rows_accepted ?? "—"}</td>
                        <td data-label="Rejected">{log.rows_rejected ?? "—"}</td>
                        <td data-label="">
                          {log.detail && (
                            <button
                              type="button"
                              className={styles.linkBtn}
                              onClick={() => setExpandedId(expandedId === log.id ? null : log.id)}
                            >
                              {expandedId === log.id ? "Hide detail" : "View detail"}
                            </button>
                          )}
                        </td>
                      </tr>
                      {expandedId === log.id && log.detail && (
                        <tr>
                          <td colSpan={6}>
                            <pre className={styles.detailBlock}>{log.detail}</pre>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <Pagination page={page} pageSize={PAGE_SIZE} total={total} onPageChange={setPage} />
          </>
        )}
      </div>
    </div>
  );
}
