import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { fetchPublicClinics, register } from "../api/auth";
import { ApiError } from "../api/client";
import DateInput from "../components/DateInput";
import PasswordInput from "../components/PasswordInput";
import PhoneInput, { isPhoneDigitsComplete, toE164 } from "../components/PhoneInput";
import SuccessCheck from "../components/SuccessCheck";
import Logo from "../components/Logo";
import { useReveal } from "../hooks/useReveal";
import styles from "./Register.module.css";

const initialForm = {
  clinicId: "",
  fullName: "",
  email: "",
  phoneDigits: "",
  password: "",
  dob: "",
  gender: "",
};

const FULL_NAME_RE = /^[A-Za-z' -]+$/;
const MAX_DOB_YEARS_AGO = 120;

function toDateInputValue(date) {
  return date.toISOString().slice(0, 10);
}

function todayStr() {
  return toDateInputValue(new Date());
}

function minDobStr() {
  const d = new Date();
  d.setFullYear(d.getFullYear() - MAX_DOB_YEARS_AGO);
  return toDateInputValue(d);
}

export default function Register() {
  const navigate = useNavigate();
  const revealRef = useReveal();
  const [clinics, setClinics] = useState([]);
  const [clinicsError, setClinicsError] = useState(null);
  const [form, setForm] = useState(initialForm);
  const [fieldErrors, setFieldErrors] = useState({});
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    fetchPublicClinics()
      .then(setClinics)
      .catch(() => setClinicsError("Could not load the list of clinics. Please try again shortly."));
  }, []);

  const setField = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const validate = () => {
    const errs = {};
    if (!form.clinicId) errs.clinicId = "Please select a clinic.";

    if (!form.fullName.trim()) {
      errs.fullName = "Full name is required.";
    } else if (!FULL_NAME_RE.test(form.fullName)) {
      errs.fullName = "Full name can only contain letters, spaces, hyphens, and apostrophes.";
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) errs.email = "Enter a valid email.";

    if (!form.phoneDigits) {
      errs.phone = "Phone number is required.";
    } else if (!isPhoneDigitsComplete(form.phoneDigits)) {
      errs.phone = "Enter 10 digits, not starting with 0 (e.g. 3001234567).";
    }

    if (form.password.length < 8) {
      errs.password = "Password must be at least 8 characters.";
    } else if (!/[A-Za-z]/.test(form.password)) {
      errs.password = "Add at least one letter.";
    } else if (!/[0-9]/.test(form.password)) {
      errs.password = "Add at least one number.";
    }

    if (!form.dob) {
      errs.dob = "Date of birth is required.";
    } else if (form.dob >= todayStr()) {
      errs.dob = "Date of birth must be in the past.";
    } else if (form.dob < minDobStr()) {
      errs.dob = `Date of birth must be within the last ${MAX_DOB_YEARS_AGO} years.`;
    }

    return errs;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSuccess(false);

    const errs = validate();
    setFieldErrors(errs);
    if (Object.keys(errs).length > 0) return;

    setSubmitting(true);
    try {
      await register({
        clinic_id: form.clinicId,
        full_name: form.fullName,
        email: form.email,
        phone: toE164(form.phoneDigits),
        password: form.password,
        dob: form.dob,
        gender: form.gender || null,
      });
      setSuccess(true);
      setForm(initialForm);
      setTimeout(() => navigate("/login"), 1200);
    } catch (err) {
      // Backend is authoritative: surface its message verbatim, including 409 duplicates.
      setError(err instanceof ApiError ? err.detail || err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className={styles.page}>
      <div className={`${styles.card} reveal`} ref={revealRef}>
        <div className={styles.intro}>
          <Logo wrap />
          <h2 className={styles.introTitle}>Join Quick Check Clinic</h2>
          <p className={styles.introText}>
            Create a patient account to browse departments, book appointments, and manage your visits online.
          </p>
        </div>

        <div className={styles.formSide}>
          <h1 className={styles.title}>Register</h1>
          <p className={styles.subtitle}>Create your patient account</p>

          {clinicsError && <div className={styles.error}>{clinicsError}</div>}
          {error && (
            <div className={styles.error} role="alert">
              {error}
            </div>
          )}
          {success && (
            <div className={styles.success}>
              <SuccessCheck />
              Account created! Redirecting to log in…
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate>
            <div className={styles.formGrid}>
              <div className={`${styles.field} ${styles.fieldFull}`}>
                <label htmlFor="clinic">
                  Clinic / branch<span className={styles.requiredMark}>*</span>
                </label>
                <select id="clinic" required value={form.clinicId} onChange={setField("clinicId")}>
                  <option value="">Select your clinic…</option>
                  {clinics.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
                {fieldErrors.clinicId && <span className={styles.error}>{fieldErrors.clinicId}</span>}
              </div>

              <div className={styles.field}>
                <label htmlFor="fullName">
                  Full name<span className={styles.requiredMark}>*</span>
                </label>
                <input id="fullName" required value={form.fullName} onChange={setField("fullName")} />
                {fieldErrors.fullName && <span className={styles.error}>{fieldErrors.fullName}</span>}
              </div>

              <div className={styles.field}>
                <label htmlFor="email">
                  Email<span className={styles.requiredMark}>*</span>
                </label>
                <input
                  id="email"
                  type="email"
                  required
                  value={form.email}
                  onChange={setField("email")}
                  autoComplete="username"
                />
                {fieldErrors.email && <span className={styles.error}>{fieldErrors.email}</span>}
              </div>

              <div className={styles.field}>
                <label htmlFor="phone">
                  Phone<span className={styles.requiredMark}>*</span>
                </label>
                <PhoneInput
                  id="phone"
                  digits={form.phoneDigits}
                  onDigitsChange={(digits) => setForm((f) => ({ ...f, phoneDigits: digits }))}
                  required
                />
                {fieldErrors.phone && <span className={styles.error}>{fieldErrors.phone}</span>}
              </div>

              <div className={styles.field}>
                <label htmlFor="password">
                  Password<span className={styles.requiredMark}>*</span>
                </label>
                <PasswordInput
                  id="password"
                  required
                  value={form.password}
                  onChange={setField("password")}
                  autoComplete="new-password"
                />
                <span className={styles.fieldHint}>At least 8 characters, with a letter and a number.</span>
                {fieldErrors.password && <span className={styles.error}>{fieldErrors.password}</span>}
              </div>

              <div className={styles.field}>
                <label htmlFor="dob">
                  Date of birth<span className={styles.requiredMark}>*</span>
                </label>
                <DateInput
                  id="dob"
                  required
                  min={minDobStr()}
                  max={todayStr()}
                  value={form.dob}
                  onChange={setField("dob")}
                />
                <span className={styles.fieldHint}>Format: YYYY-MM-DD.</span>
                {fieldErrors.dob && <span className={styles.error}>{fieldErrors.dob}</span>}
              </div>

              <div className={styles.field}>
                <label htmlFor="gender">Gender</label>
                <select id="gender" value={form.gender} onChange={setField("gender")}>
                  <option value="">Prefer not to say</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="other">Other</option>
                </select>
              </div>
            </div>

            <button className={styles.submit} type="submit" disabled={submitting}>
              {submitting ? "Creating account…" : "Register"}
            </button>
          </form>

          <div className={styles.footer}>
            Already have an account? <Link to="/login">Log in</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
