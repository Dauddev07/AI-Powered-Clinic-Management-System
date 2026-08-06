import { apiFetch } from "./client";

export function fetchPublicTopRatedDoctors() {
  return apiFetch("/clinics/top-rated-doctors");
}
