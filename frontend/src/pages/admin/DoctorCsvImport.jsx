import { useRef, useState } from "react";
import { confirmDoctorCsv, previewDoctorCsv } from "../../api/adminDoctors";
import { ApiError } from "../../api/client";
import Modal from "../../components/Modal";
import SuccessCheck from "../../components/SuccessCheck";
import { useReveal } from "../../hooks/useReveal";
import styles from "./AdminScreens.module.css";

const WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

// Mirrors backend/app/services/doctor_csv.py's REQUIRED_COLUMNS + OPTIONAL_COLUMNS,
// in the same order — this is a read-only reference for admins, never itself used
// for validation, so it must be kept in sync with that module by hand if it changes.
const CSV_COLUMNS = [
  { name: "external_doctor_id", required: true, description: "Unique ID for this doctor in your own system. Must be unique within the file." },
  { name: "name", required: true, description: "Doctor's full name." },
  { name: "department", required: true, description: "Department name. Created automatically if it doesn't already exist." },
  { name: "specialty", required: false, description: "e.g. \"Cardiologist\". Leave blank if not applicable." },
  { name: "shift_days", required: true, description: "Comma-separated weekdays the shift repeats on, e.g. \"Mon,Wed,Fri\" (full names like \"Monday\" also accepted)." },
  { name: "shift_start", required: true, description: "Clinic-local start time, 24-hour HH:MM, e.g. \"09:00\"." },
  { name: "shift_end", required: true, description: "Clinic-local end time, 24-hour HH:MM. Must be after shift_start." },
  { name: "active", required: true, description: "Whether the doctor is active/bookable. Accepts true/false, yes/no, 1/0." },
  { name: "slot_length_minutes", required: false, description: "Length of each bookable slot, in minutes. Leave blank to use the clinic's default." },
  { name: "leave_dates", required: false, description: "Comma-separated dates (YYYY-MM-DD) the doctor is unavailable." },
];

// Matches public/sample-doctor-import.csv row-for-row — keep both in sync by hand.
const EXAMPLE_ROWS = [
  { external_doctor_id: "DOC-1001", name: "Dr. Jane Example", department: "Cardiology", specialty: "Cardiologist", shift_days: "Mon,Wed,Fri", shift_start: "09:00", shift_end: "17:00", active: "true", slot_length_minutes: "30", leave_dates: "2026-08-15" },
  { external_doctor_id: "DOC-1002", name: "Dr. John Sample", department: "General Medicine", specialty: "", shift_days: "Tue,Thu", shift_start: "08:30", shift_end: "14:00", active: "true", slot_length_minutes: "", leave_dates: "" },
  { external_doctor_id: "DOC-1003", name: "Dr. Alex Placeholder", department: "Pediatrics", specialty: "Pediatrician", shift_days: "Mon,Tue,Wed,Thu,Fri", shift_start: "10:00", shift_end: "18:00", active: "false", slot_length_minutes: "20", leave_dates: "2026-08-01,2026-08-02" },
];

export default function DoctorCsvImport() {
  const revealRef = useReveal();
  const fileInputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [confirmResult, setConfirmResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [templateOpen, setTemplateOpen] = useState(false);
  // Header-mapping suggestions awaiting explicit admin decisions, from the most
  // recent preview call: { suggestions: [{original_header, suggested_field, confidence}], unrecognizedHeaders: [] }
  const [headerMapping, setHeaderMapping] = useState(null);
  // Per-suggestion explicit decision: { [original_header]: true (accepted) | false (rejected) }
  const [mappingDecisions, setMappingDecisions] = useState({});
  // The confirmed mapping that actually produced the current `preview` — resent as-is to Confirm,
  // since confirm never trusts a prior preview and re-validates from scratch.
  const [appliedMapping, setAppliedMapping] = useState(null);

  const resetForNewFile = () => {
    setPreview(null);
    setConfirmResult(null);
    setError(null);
    setHeaderMapping(null);
    setMappingDecisions({});
    setAppliedMapping(null);
  };

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0] ?? null;
    setFile(selected);
    resetForNewFile();
  };

  const handleClearFile = () => {
    setFile(null);
    resetForNewFile();
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handlePreview = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setConfirmResult(null);
    setHeaderMapping(null);
    setMappingDecisions({});
    setAppliedMapping(null);
    try {
      const data = await previewDoctorCsv(file);
      if (data.needs_header_mapping) {
        setPreview(null);
        setHeaderMapping({ suggestions: data.header_suggestions, unrecognizedHeaders: data.unrecognized_headers });
      } else {
        setPreview(data);
      }
    } catch (err) {
      setPreview(null);
      setError(err instanceof ApiError ? err.detail || err.message : "Preview failed.");
    } finally {
      setBusy(false);
    }
  };

  const setMappingDecision = (originalHeader, accepted) => {
    setMappingDecisions((prev) => ({ ...prev, [originalHeader]: accepted }));
  };

  const allSuggestionsDecided =
    headerMapping != null &&
    headerMapping.suggestions.every((s) => mappingDecisions[s.original_header] !== undefined);

  const handleApplyMapping = async () => {
    if (!file || !headerMapping) return;
    const confirmedMapping = {};
    headerMapping.suggestions.forEach((s) => {
      if (mappingDecisions[s.original_header]) confirmedMapping[s.original_header] = s.suggested_field;
    });
    setBusy(true);
    setError(null);
    try {
      const data = await previewDoctorCsv(file, confirmedMapping);
      if (data.needs_header_mapping) {
        // Still unresolved (rejected suggestions stay suggested, or a header remains unrecognized).
        setHeaderMapping({ suggestions: data.header_suggestions, unrecognizedHeaders: data.unrecognized_headers });
        setAppliedMapping(null);
      } else {
        setHeaderMapping(null);
        setAppliedMapping(confirmedMapping);
        setPreview(data);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Preview failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleConfirm = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const data = await confirmDoctorCsv(file, appliedMapping);
      setConfirmResult(data);
      setPreview(null);
      setFile(null);
      setHeaderMapping(null);
      setAppliedMapping(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      // Keep the existing preview on screen — the admin shouldn't lose their place.
      setError(err instanceof ApiError ? err.detail || err.message : "Confirm failed.");
    } finally {
      setBusy(false);
    }
  };

  const canConfirm = preview && preview.rejected_count === 0 && preview.accepted_count > 0 && !busy;

  return (
    <div>
      <h1 className={styles.title}>Import doctors from CSV</h1>
      <p className={styles.subtitle}>
        Upload the clinic's doctor roster. Nothing is written until you review the preview and click Confirm.
      </p>

      <div className={`${styles.card} reveal`} ref={revealRef}>
        <div className={styles.uploadRow}>
          <label className={styles.fileChooseBtn}>
            Choose file
            <input
              ref={fileInputRef}
              type="file"
              // Reported live: the OS file picker opened but selecting a CSV never
              // registered (no filename appeared) — many mobile file-browsing apps
              // filter `accept` by MIME type rather than extension, and a CSV's
              // actual MIME type varies a lot by app/OS (text/csv,
              // application/vnd.ms-excel, text/comma-separated-values, or none at
              // all) instead of reliably being "text/csv". Listing the extension
              // AND every MIME type it commonly gets tagged with covers pickers
              // that filter either way.
              accept=".csv,text/csv,application/vnd.ms-excel,text/comma-separated-values,text/plain"
              onChange={handleFileChange}
              disabled={busy}
              className={styles.hiddenFileInput}
            />
          </label>

          <button
            type="button"
            className={styles.infoBtn}
            onClick={() => setTemplateOpen(true)}
            aria-haspopup="dialog"
          >
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <circle cx="12" cy="12" r="9" />
              <path d="M12 11v5.5M12 8v.01" />
            </svg>
            CSV format
          </button>

          {file && (
            <div className={styles.fileChip}>
              <span className={styles.fileChipName} title={file.name}>
                {file.name}
              </span>
              <button
                type="button"
                className={styles.fileChipClear}
                onClick={handleClearFile}
                disabled={busy}
                aria-label={`Clear selected file ${file.name}`}
                title="Clear selected file"
              >
                ✕
              </button>
            </div>
          )}

          <button type="button" className={styles.primaryBtn} onClick={handlePreview} disabled={!file || busy}>
            {busy && !preview ? "Uploading…" : "Preview"}
          </button>
        </div>

        {error && <p className={styles.errorText}>{error}</p>}

        {confirmResult && (
          <div className={styles.successBox}>
            <strong>
              <SuccessCheck />
              Import complete for {confirmResult.filename}
            </strong>
            <p>
              {confirmResult.inserted} inserted, {confirmResult.updated} updated,{" "}
              {confirmResult.marked_inactive} marked inactive, {confirmResult.rejected_count} rejected.
            </p>
            <p>
              {confirmResult.slots_inserted} new slot(s) generated
              {confirmResult.slots_flagged_for_review > 0
                ? `, ${confirmResult.slots_flagged_for_review} booked slot(s) flagged for admin review (shift no longer covers an existing appointment — nothing was cancelled).`
                : "."}
            </p>
          </div>
        )}

        {confirmResult && confirmResult.deactivation_skipped.length > 0 && (
          <div className={styles.noticeBox}>
            <p>
              <strong>
                {confirmResult.deactivation_skipped.length} doctor(s) omitted from the file were left active
                instead of being auto-deactivated:
              </strong>
            </p>
            {confirmResult.deactivation_skipped.map((d) => (
              <p key={d.external_doctor_id}>
                {d.full_name} ({d.external_doctor_id}): {d.reason}
              </p>
            ))}
          </div>
        )}
      </div>

      {headerMapping && (
        <div className={`${styles.card} reveal`} ref={revealRef}>
          <h2 className={styles.sectionTitle}>Column mapping needed</h2>
          <p className={styles.subtitle}>
            Some column headers in this file don't exactly match the expected names. Nothing is applied until you
            confirm each suggestion below.
          </p>

          {headerMapping.suggestions.length > 0 && (
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Column in your file</th>
                    <th>We think this means</th>
                    <th>Confidence</th>
                    <th>Your decision</th>
                  </tr>
                </thead>
                <tbody>
                  {headerMapping.suggestions.map((s) => {
                    const decision = mappingDecisions[s.original_header];
                    return (
                      <tr key={s.original_header}>
                        <td data-label="Column in your file">"{s.original_header}"</td>
                        <td data-label="We think this means">
                          <code>{s.suggested_field}</code>
                        </td>
                        <td data-label="Confidence">{Math.round(s.confidence * 100)}%</td>
                        <td data-label="Your decision">
                          <div className={styles.uploadRow}>
                            <button
                              type="button"
                              className={styles.primaryBtn}
                              aria-pressed={decision === true}
                              disabled={decision === true}
                              onClick={() => setMappingDecision(s.original_header, true)}
                            >
                              {decision === true ? "Confirmed" : "Confirm"}
                            </button>
                            <button
                              type="button"
                              className={styles.infoBtn}
                              aria-pressed={decision === false}
                              disabled={decision === false}
                              onClick={() => setMappingDecision(s.original_header, false)}
                            >
                              {decision === false ? "Rejected" : "Reject"}
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {headerMapping.unrecognizedHeaders.length > 0 && (
            <div className={styles.noticeBox}>
              <p>
                <strong>Unrecognized column(s) — no confident match found:</strong>{" "}
                {headerMapping.unrecognizedHeaders.join(", ")}
              </p>
              <p>Rename these columns in your CSV to a recognized name (see "CSV format" above) and re-upload.</p>
            </div>
          )}

          <button
            type="button"
            className={styles.primaryBtn}
            onClick={handleApplyMapping}
            disabled={!allSuggestionsDecided || headerMapping.unrecognizedHeaders.length > 0 || busy}
          >
            {busy ? "Applying…" : "Apply confirmed mappings"}
          </button>
        </div>
      )}

      {preview && (
        <div className={`${styles.card} reveal`} ref={revealRef}>
          <div className={styles.summaryRow}>
            <span>
              <strong>{preview.total_rows}</strong> rows
            </span>
            <span className={styles.successText}>
              <strong>{preview.accepted_count}</strong> accepted
            </span>
            <span className={styles.errorTextInline}>
              <strong>{preview.rejected_count}</strong> rejected
            </span>
          </div>

          {preview.warnings.length > 0 && (
            <div className={styles.noticeBox}>
              {preview.warnings.map((w, i) => (
                <p key={i}>{w}</p>
              ))}
            </div>
          )}

          {preview.rejected_count > 0 && (
            <p className={styles.warningText}>
              Fix the rejected rows below and re-upload before confirming. Nothing will be written while any row is
              rejected.
            </p>
          )}

          {preview.rejected_rows.length > 0 && (
            <>
              <h2 className={styles.sectionTitle}>Rejected rows</h2>
              <div className={styles.tableWrap}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Row</th>
                      <th>External ID</th>
                      <th>Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rejected_rows.map((r) => (
                      <tr key={r.row_number} className={styles.rejectedRow}>
                        <td data-label="Row">{r.row_number}</td>
                        <td data-label="External ID">{r.external_doctor_id ?? "—"}</td>
                        <td data-label="Reason">{r.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {preview.accepted_rows.length > 0 && (
            <>
              <h2 className={styles.sectionTitle}>Accepted rows</h2>
              <div className={styles.tableWrap}>
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>Row</th>
                      <th>External ID</th>
                      <th>Name</th>
                      <th>Department</th>
                      <th>Specialty</th>
                      <th>Shift days</th>
                      <th>Leaves</th>
                      <th>Active</th>
                    </tr>
                  </thead>
                  <tbody>
                    {preview.accepted_rows.map((r) => (
                      <tr key={r.row_number}>
                        <td data-label="Row">{r.row_number}</td>
                        <td data-label="External ID">{r.external_doctor_id}</td>
                        <td data-label="Name">{r.full_name}</td>
                        <td data-label="Department">
                          {r.department_name}
                          {r.department_will_be_created && <span className={styles.badge}>new</span>}
                        </td>
                        <td data-label="Specialty">{r.specialization ?? "—"}</td>
                        <td data-label="Shift days">{r.shift_days.map((d) => WEEKDAY_LABELS[d]).join(", ")}</td>
                        <td data-label="Leaves">{r.leave_dates_count}</td>
                        <td data-label="Active">{r.is_active ? "Yes" : "No"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          <button type="button" className={styles.primaryBtn} onClick={handleConfirm} disabled={!canConfirm}>
            {busy ? "Confirming…" : "Confirm import"}
          </button>
        </div>
      )}

      <Modal open={templateOpen} onClose={() => setTemplateOpen(false)} title="Expected CSV format">
        <p className={styles.modalIntro}>
          The header row must include these exact column names (case-sensitive). Column order in the file doesn't
          matter.
        </p>

        <div className={styles.tableWrap}>
          <table className={`${styles.table} ${styles.columnsTable}`}>
            <thead>
              <tr>
                <th>Column</th>
                <th>Required</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {CSV_COLUMNS.map((c) => (
                <tr key={c.name}>
                  <td data-label="Column">
                    <code>{c.name}</code>
                  </td>
                  <td data-label="Required">{c.required ? "Yes" : "No"}</td>
                  <td data-label="Description" className={styles.wrapCell}>{c.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h3 className={styles.sectionTitle}>Example rows</h3>
        <p className={styles.modalSubIntro}>Placeholder data — for format reference only.</p>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                {CSV_COLUMNS.map((c) => (
                  <th key={c.name}>{c.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {EXAMPLE_ROWS.map((row) => (
                <tr key={row.external_doctor_id}>
                  {CSV_COLUMNS.map((c) => (
                    <td key={c.name} data-label={c.name}>{row[c.name] || "—"}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <a href="/sample-doctor-import.csv" download="sample-doctor-import.csv" className={styles.sampleDownloadBtn}>
          Download sample CSV
        </a>
      </Modal>
    </div>
  );
}
