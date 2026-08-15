import { Fragment, useEffect, useState } from "react";
import { fetchDoctors, updateDoctorStatus } from "../../api/adminDoctors";
import { ApiError } from "../../api/client";
import StatusBadge from "../../components/StatusBadge";
import EmptyState from "../../components/EmptyState";
import Pagination from "../../components/Pagination";
import Skeleton from "../../components/Skeleton";
import { useReveal } from "../../hooks/useReveal";
import { useMediaQuery } from "../../hooks/useMediaQuery";
import styles from "./AdminScreens.module.css";

const PAGE_SIZE = 10;
// Waits for the admin to pause typing before hitting the server — a search-as-you-type
// field firing one request per keystroke would otherwise spam /admin/doctors for every
// letter of, say, "cardiology".
const SEARCH_DEBOUNCE_MS = 300;

export default function DoctorList() {
  const revealRef = useReveal();
  // The full placeholder ("Search by doctor, department, or specialization…") gets
  // hard-clipped by the input's own width on narrow screens — a browser placeholder
  // has no text-wrap/ellipsis control, so the fix is a genuinely shorter string
  // below this breakpoint rather than trying to shrink font-size/padding further.
  const isNarrow = useMediaQuery("(max-width: 480px)");
  const [doctors, setDoctors] = useState(null);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [error, setError] = useState(null);
  const [busyId, setBusyId] = useState(null);
  const [rowErrors, setRowErrors] = useState({});

  const load = async (pageToLoad, q) => {
    try {
      const data = await fetchDoctors({ limit: PAGE_SIZE, offset: (pageToLoad - 1) * PAGE_SIZE, q });
      setDoctors(data.items);
      setTotal(data.total);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Could not load doctors.");
    }
  };

  // Debounce: only commit `searchInput` to `search` (which actually triggers the
  // fetch below) once the admin has stopped typing for a moment.
  useEffect(() => {
    const id = setTimeout(() => setSearch(searchInput.trim()), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(id);
  }, [searchInput]);

  // A new search term always jumps back to page 1 — staying on, say, page 3 of an
  // old unfiltered list would silently show "no results" against the new search's
  // (usually much shorter) result set instead of its actual first page.
  useEffect(() => {
    setPage(1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search]);

  useEffect(() => {
    load(page, search);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, search]);

  const handleToggle = async (doctor) => {
    setBusyId(doctor.id);
    setRowErrors((prev) => ({ ...prev, [doctor.id]: null }));
    try {
      const updated = await updateDoctorStatus(doctor.id, !doctor.is_active);
      setDoctors((prev) => prev.map((d) => (d.id === updated.id ? updated : d)));
    } catch (err) {
      // Surfaced verbatim — this is the same conflict reason the CSV import path
      // shows when a doctor's schedule change would orphan a confirmed appointment.
      setRowErrors((prev) => ({
        ...prev,
        [doctor.id]: err instanceof ApiError ? err.detail || err.message : "Could not update status.",
      }));
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div>
      <h1 className={styles.title}>Doctors</h1>
      <p className={styles.subtitle}>
        Manage each doctor's active status directly — no CSV re-upload needed for a doctor leaving or returning.
      </p>

      <div className={`${styles.card} reveal`} ref={revealRef}>
        <div className={styles.searchRow}>
          <svg className={styles.searchIcon} viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <circle cx="11" cy="11" r="7" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            type="text"
            className={styles.searchInput}
            placeholder={isNarrow ? "Search doctors…" : "Search by doctor, department, or specialization…"}
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            aria-label="Search doctors"
          />
          {searchInput && (
            <button
              type="button"
              className={styles.searchClearBtn}
              onClick={() => setSearchInput("")}
              aria-label="Clear search"
            >
              ✕
            </button>
          )}
        </div>

        {error && <p className={styles.errorText}>{error}</p>}

        {doctors === null && !error && <Skeleton rows={4} />}

        {doctors && total === 0 && search && (
          <EmptyState icon="search" message={`No doctors match "${search}".`} />
        )}

        {doctors && total === 0 && !search && (
          <EmptyState
            icon="search"
            message="No doctors yet. Import a CSV to add some."
            actionLabel="Import doctor CSV"
            actionTo="/admin/doctors/import"
          />
        )}

        {doctors && total > 0 && (
          <>
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Department</th>
                    <th>Doctor</th>
                    <th>Specialization</th>
                    <th>Status</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {doctors.map((d) => (
                    <Fragment key={d.id}>
                      <tr>
                        <td data-label="Department">{d.department_name}</td>
                        <td data-label="Doctor">{d.full_name}</td>
                        <td data-label="Specialization">{d.specialization || "—"}</td>
                        <td data-label="Status">
                          <StatusBadge tone={d.is_active ? "success" : "neutral"} label={d.is_active ? "Active" : "Inactive"} />
                        </td>
                        <td data-label="">
                          <button
                            type="button"
                            className={d.is_active ? styles.dangerBtn : styles.primaryBtn}
                            onClick={() => handleToggle(d)}
                            disabled={busyId === d.id}
                          >
                            {busyId === d.id ? "Working…" : d.is_active ? "Deactivate" : "Activate"}
                          </button>
                        </td>
                      </tr>
                      {rowErrors[d.id] && (
                        <tr>
                          <td colSpan={5}>
                            <p className={styles.errorText}>{rowErrors[d.id]}</p>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  ))}
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
