import { apiFetch } from "./client";

export function login(email, password) {
  return apiFetch("/auth/login", { method: "POST", body: { email, password } });
}

export function register(payload) {
  return apiFetch("/auth/register", { method: "POST", body: payload });
}

export function changePassword(oldPassword, newPassword) {
  return apiFetch("/auth/change-password", {
    method: "POST",
    auth: true,
    body: { old_password: oldPassword, new_password: newPassword },
  });
}

export function forgotPassword(email) {
  return apiFetch("/auth/forgot-password", { method: "POST", body: { email } });
}

export function verifyResetOtp(email, otp) {
  return apiFetch("/auth/verify-reset-otp", { method: "POST", body: { email, otp } });
}

export function resetPassword(email, otp, newPassword) {
  return apiFetch("/auth/reset-password", {
    method: "POST",
    body: { email, otp, new_password: newPassword },
  });
}

export function fetchPublicClinics() {
  return apiFetch("/clinics/public-list");
}

export function fetchMyAccount() {
  return apiFetch("/auth/me", { auth: true });
}

export function fetchMyProfile() {
  return apiFetch("/patients/me", { auth: true });
}

export function updateMyProfile(payload) {
  return apiFetch("/patients/me", { method: "PATCH", auth: true, body: payload });
}
