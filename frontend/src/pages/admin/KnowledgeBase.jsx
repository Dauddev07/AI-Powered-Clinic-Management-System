import { useEffect, useRef, useState } from "react";
import { deleteKbDocument, fetchKbDocuments, uploadKbDocument } from "../../api/adminKb";
import { ApiError } from "../../api/client";
import EmptyState from "../../components/EmptyState";
import Modal from "../../components/Modal";
import Skeleton from "../../components/Skeleton";
import SuccessCheck from "../../components/SuccessCheck";
import { useReveal } from "../../hooks/useReveal";
import { formatDateTimeLocal as formatDate } from "../../utils/formatDateTime";
import styles from "./AdminScreens.module.css";

export default function KnowledgeBase() {
  const revealRef = useReveal();
  const fileInputRef = useRef(null);
  const [documents, setDocuments] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [pendingDelete, setPendingDelete] = useState(null);
  const [deleteBusy, setDeleteBusy] = useState(false);

  const load = async () => {
    try {
      const data = await fetchKbDocuments();
      setDocuments(data.items);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Could not load knowledge base documents.");
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError(null);
    setUploadResult(null);
    try {
      const data = await uploadKbDocument(file);
      setUploadResult(data);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Upload failed.");
    } finally {
      setBusy(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDelete = async () => {
    if (!pendingDelete) return;
    setDeleteBusy(true);
    try {
      await deleteKbDocument(pendingDelete.id);
      setPendingDelete(null);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Delete failed.");
      setPendingDelete(null);
    } finally {
      setDeleteBusy(false);
    }
  };

  return (
    <div>
      <h1 className={styles.title}>Knowledge base</h1>
      <p className={styles.subtitle}>
        Upload PDF or DOCX documents (clinic timings, location, fees, policies, and other general clinic
        information) for the AI chatbot to draw on when answering logistics questions. Uploading a file with the
        same name as an existing document replaces it.
      </p>

      <div className={`${styles.card} reveal`} ref={revealRef}>
        <div className={styles.uploadRow}>
          <label className={styles.fileChooseBtn}>
            {busy ? "Uploading…" : "Upload document"}
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx"
              onChange={handleFileChange}
              disabled={busy}
              className={styles.hiddenFileInput}
            />
          </label>
        </div>

        {error && <p className={styles.errorText}>{error}</p>}

        {uploadResult && (
          <div className={styles.successBox}>
            <strong>
              <SuccessCheck />
              {uploadResult.replaced
                ? `Replaced existing document "${uploadResult.document.filename}"`
                : `Uploaded "${uploadResult.document.filename}"`}
            </strong>
            <p>{uploadResult.document.chunk_count} chunk(s) embedded and stored.</p>
          </div>
        )}
      </div>

      {documents === null && !error && (
        <div className={`${styles.card} reveal`} ref={revealRef}>
          <Skeleton rows={3} />
        </div>
      )}

      {documents && (
        <div className={`${styles.card} reveal`} ref={revealRef}>
          <h2 className={styles.sectionHeading}>Knowledge base documents</h2>

          {documents.length === 0 && <EmptyState icon="inbox" message="No documents yet." />}

          {documents.length > 0 && (
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Filename</th>
                    <th>Type</th>
                    <th>Uploaded</th>
                    <th>Chunks</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map((d) => (
                    <tr key={d.id}>
                      <td data-label="Filename">{d.filename}</td>
                      <td data-label="Type">{d.source_type.toUpperCase()}</td>
                      <td data-label="Uploaded">{formatDate(d.created_at)}</td>
                      <td data-label="Chunks">{d.chunk_count}</td>
                      <td data-label="">
                        <button type="button" className={styles.dangerBtn} onClick={() => setPendingDelete(d)}>
                          Remove
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      <Modal open={!!pendingDelete} onClose={() => setPendingDelete(null)} title="Remove document">
        <p className={styles.modalIntro}>
          Remove <strong>{pendingDelete?.filename}</strong>? This deletes it and all {pendingDelete?.chunk_count}{" "}
          of its chunks from the knowledge base. This cannot be undone.
        </p>
        <div className={styles.uploadRow}>
          <button type="button" className={styles.secondaryBtn} onClick={() => setPendingDelete(null)} disabled={deleteBusy}>
            Cancel
          </button>
          <button type="button" className={styles.dangerBtn} onClick={handleDelete} disabled={deleteBusy}>
            {deleteBusy ? "Removing…" : "Remove"}
          </button>
        </div>
      </Modal>
    </div>
  );
}
