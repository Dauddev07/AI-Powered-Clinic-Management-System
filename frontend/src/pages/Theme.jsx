import { useReveal } from "../hooks/useReveal";
import { useTheme } from "../theme/ThemeContext";
import styles from "./Login.module.css";
import themeStyles from "./Theme.module.css";

// Reuses Login.module.css's .page/.card/.title/.subtitle/.submit — same
// pattern as ChangePassword.jsx — so this reads as one more account-settings
// screen in the same visual language, not a one-off. Reached from Settings ->
// Appearance (see SettingsMenu.jsx's SUBVIEWS.settings.items), same click-a-
// row-to-see-the-real-screen behavior as View profile / Change password,
// instead of the compact inline toggle this used to be.
export default function Theme() {
  const { isDark, toggleTheme } = useTheme();
  const revealRef = useReveal();

  return (
    <div className={styles.page}>
      <div className={`${styles.card} reveal`} ref={revealRef}>
        <h1 className={styles.title}>Theme</h1>
        <p className={styles.subtitle}>Choose how QuickCheck Clinic looks for you.</p>

        <div className={themeStyles.statusCard}>
          <span className={themeStyles.statusIcon} aria-hidden="true">
            {isDark ? "🌙" : "☀️"}
          </span>
          <div className={themeStyles.statusText}>
            <span className={themeStyles.statusLabel}>{isDark ? "Dark" : "Light"}</span>
            <span className={themeStyles.statusCaption}>
              {isDark
                ? "This is the default theme — every account starts on Dark."
                : "You've switched away from the default (Dark)."}
            </span>
          </div>
        </div>

        <button type="button" className={styles.submit} onClick={toggleTheme}>
          {isDark ? "Switch to Light" : "Switch to Dark"}
        </button>
      </div>
    </div>
  );
}
