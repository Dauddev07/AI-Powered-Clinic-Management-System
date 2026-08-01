import { useEffect, useState } from "react";
import { fetchMyProfile, updateMyProfile } from "../../api/auth";
import { ApiError } from "../../api/client";
import SuccessCheck from "../../components/SuccessCheck";
import Skeleton from "../../components/Skeleton";
import { useReveal } from "../../hooks/useReveal";
import dashStyles from "../Dashboard.module.css";
import styles from "./PatientScreens.module.css";

// Moved out of PatientDashboard (now an action hub) — reached from the "View
// Profile" item in the fixed account menu instead. Logic is unchanged from
// before: same fetchMyProfile/updateMyProfile calls, same validation rules.
export default function ViewProfile() {
  const revealRef = useReveal();
  const [profile, setProfile] = useState(null);
  const [form, setForm] = useState(null);
  const [fieldErrors, setFieldErrors] = useState({});
  const [status, setStatus] = useState(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchMyProfile()
      .then((data) => {
        setProfile(data);
        setForm({
          full_name: data.full_name,
          phone: data.phone || "",
          dob: data.dob || "",
          gender: data.gender || "",
        });
      })
      .catch((err) => setStatus({ ok: false, msg: err instanceof ApiError ? err.detail || err.message : "Could not load profile." }));
  }, []);

  const setField = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  // A phone is only required on this form once the patient already has one on file —
  // matches the backend rule that a legacy null phone stays optional until first set,
  // but can never be cleared again after that.
  const phoneRequired = Boolean(profile?.phone);

  const validate = () => {
    const errs = {};
    if (!form.full_name.trim()) {
      errs.full_name = "Full name is required.";
    }
    if (!form.dob) {
      errs.dob = "Date of birth is required.";
    }
    if (phoneRequired && !form.phone.trim()) {
      errs.phone = "Phone number is required.";
    }
    return errs;
  };

  const handleSave = async (e) => {
    e.preventDefault();
    setStatus(null);

    const errs = validate();
    setFieldErrors(errs);
    if (Object.keys(errs).length > 0) return;

    setSaving(true);
    try {
      const updated = await updateMyProfile({
        full_name: form.full_name,
        phone: form.phone || null,
        dob: form.dob || null,
        gender: form.gender || null,
      });
      setProfile(updated);
      setStatus({ ok: true, msg: "Profile updated." });
    } catch (err) {
      setStatus({ ok: false, msg: err instanceof ApiError ? err.detail || err.message : "Update failed." });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div>
      <h1 className={styles.title}>My profile</h1>
      <p className={styles.subtitle}>Your profile details.</p>

      <div className={`${styles.card} reveal`} ref={revealRef}>
        {!profile && !status && <Skeleton rows={2} />}
        {profile && (
          <>
            <p className={dashStyles.muted} style={{ marginTop: 0 }}>
              Signed in as {profile.email}
            </p>
            <form onSubmit={handleSave} noValidate>
              <div className={dashStyles.grid}>
                <div className={dashStyles.field}>
                  <label htmlFor="full_name">
                    Full name<span className={dashStyles.requiredMark}>*</span>
                  </label>
                  <input id="full_name" required value={form.full_name} onChange={setField("full_name")} />
                  {fieldErrors.full_name && <span className={dashStyles.fieldError}>{fieldErrors.full_name}</span>}
                </div>
                <div className={dashStyles.field}>
                  <label htmlFor="phone">
                    Phone{phoneRequired && <span className={dashStyles.requiredMark}>*</span>}
                  </label>
                  <input id="phone" required={phoneRequired} value={form.phone} onChange={setField("phone")} />
                  <span className={dashStyles.fieldHint}>10 digits, not starting with 0 (e.g. 3001234567).</span>
                  {fieldErrors.phone && <span className={dashStyles.fieldError}>{fieldErrors.phone}</span>}
                </div>
                <div className={dashStyles.field}>
                  <label htmlFor="dob">
                    Date of birth<span className={dashStyles.requiredMark}>*</span>
                  </label>
                  <input id="dob" type="date" required value={form.dob} onChange={setField("dob")} />
                  <span className={dashStyles.fieldHint}>Format: YYYY-MM-DD.</span>
                  {fieldErrors.dob && <span className={dashStyles.fieldError}>{fieldErrors.dob}</span>}
                </div>
                <div className={dashStyles.field}>
                  <label htmlFor="gender">Gender</label>
                  <select id="gender" value={form.gender} onChange={setField("gender")}>
                    <option value="">Prefer not to say</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                  </select>
                </div>
              </div>
              <button className={dashStyles.saveBtn} type="submit" disabled={saving}>
                {saving ? "Saving…" : "Save changes"}
              </button>
            </form>
          </>
        )}
        {status && (
          <p className={`${dashStyles.status} ${status.ok ? dashStyles.ok : dashStyles.err}`}>
            {status.ok && <SuccessCheck />}
            {status.msg}
          </p>
        )}
      </div>
    </div>
  );
}
