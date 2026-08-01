import { Fragment, useEffect, useState } from "react";
import { fetchDoctors, updateDoctorStatus } from "../../api/adminDoctors";
import { ApiError } from "../../api/client";
import StatusBadge from "../../components/StatusBadge";
import EmptyState from "../../components/EmptyState";
import Pagination from "../../components/Pagination";
import Skeleton from "../../components/Skeleton";
import { useReveal } from "../../hooks/useReveal";
import styles from "./AdminScreens.module.css";

const PAGE_SIZE = 10;

export default function DoctorList() {
  const revealRef = useReveal();
  const [doctors, setDoctors] = useState(null);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [error, setError] = useState(null);
  const [busyId, setBusyId] = useState(null);
  const [rowErrors, setRowErrors] = useState({});

  const load = async (pageToLoad) => {
    try {
      const data = await fetchDoctors({ limit: PAGE_SIZE, offset: (pageToLoad - 1) * PAGE_SIZE });
      setDoctors(data.items);
      setTotal(data.total);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Could not load doctors.");
    }
  };

  useEffect(() => {
    load(page);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page]);

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
        {error && <p className={styles.errorText}>{error}</p>}

        {doctors === null && !error && <Skeleton rows={4} />}

        {doctors && total === 0 && (
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
