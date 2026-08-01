import { useEffect, useState } from "react";
import { fetchMyAccount } from "../../api/auth";
import { ApiError } from "../../api/client";
import { useReveal } from "../../hooks/useReveal";
import Skeleton from "../../components/Skeleton";
import { formatDateOnly as formatDate } from "../../utils/formatDateTime";
import styles from "./AdminScreens.module.css";

// Read-only — there's no admin-facing profile-edit endpoint (admin accounts are
// managed via the superadmin CLI), so this only ever displays whatever /auth/me
// already returns, unlike the patient's editable View Profile page.
export default function ViewProfile() {
  const revealRef = useReveal();
  const [account, setAccount] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchMyAccount()
      .then(setAccount)
      .catch((err) => setError(err instanceof ApiError ? err.detail || err.message : "Could not load profile."));
  }, []);

  return (
    <div>
      <h1 className={styles.title}>My profile</h1>
      <p className={styles.subtitle}>Your admin account details for this clinic.</p>

      <div className={`${styles.card} reveal`} ref={revealRef}>
        {error && <p className={styles.errorText}>{error}</p>}
        {!account && !error && <Skeleton rows={2} />}

        {account && (
          <div className={styles.profileGrid}>
            <div className={styles.profileField}>
              <span className={styles.profileLabel}>Full name</span>
              <span className={styles.profileValue}>{account.full_name}</span>
            </div>
            <div className={styles.profileField}>
              <span className={styles.profileLabel}>Email</span>
              <span className={styles.profileValue}>{account.email}</span>
            </div>
            <div className={styles.profileField}>
              <span className={styles.profileLabel}>Role</span>
              <span className={styles.profileValue}>Admin</span>
            </div>
            <div className={styles.profileField}>
              <span className={styles.profileLabel}>Phone</span>
              <span className={styles.profileValue}>{account.phone || "—"}</span>
            </div>
            <div className={styles.profileField}>
              <span className={styles.profileLabel}>Account created</span>
              <span className={styles.profileValue}>{formatDate(account.created_at)}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
