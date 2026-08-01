import { apiFetch } from "./client";

export function fetchKbDocuments() {
  return apiFetch("/admin/kb/documents", { auth: true });
}

export function uploadKbDocument(file) {
  const formData = new FormData();
  formData.append("file", file);
  return apiFetch("/admin/kb/documents", { method: "POST", auth: true, formData });
}

export function deleteKbDocument(documentId) {
  return apiFetch(`/admin/kb/documents/${documentId}`, { method: "DELETE", auth: true });
}
