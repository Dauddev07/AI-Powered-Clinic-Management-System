import { useTheme } from "../theme/ThemeContext";
import ThemeToggle from "../components/ThemeToggle";
import styles from "./Theme.module.css";

// Reached from Settings -> Appearance (see SettingsMenu.jsx's
// SUBVIEWS.settings.items), same click-a-row-to-see-the-real-screen behavior
// as View profile / Change password. Deliberately NOT a centered form like
// ChangePassword — there's only one real control here, so it sits as a
// single compact row near the top of the page instead of a big card
// vertically centered in the middle of the screen.
export default function Theme() {
  const { isDark, toggleTheme } = useTheme();

  return (
    <div className={styles.page}>
      <h1 className={styles.title}>Theme</h1>
      <div className={styles.row}>
        <span className={styles.icon} aria-hidden="true">
          {isDark ? "🌙" : "☀️"}
        </span>
        <span className={styles.label}>
          <span className={styles.labelName}>{isDark ? "Dark" : "Light"}</span>
          <span className={styles.caption}>
            {isDark ? "Default — tap to switch to Light" : "Tap to switch back to Dark, the default"}
          </span>
        </span>
        <ThemeToggle />
      </div>
    </div>
  );
}
