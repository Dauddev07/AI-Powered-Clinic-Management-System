import { useTheme } from "../theme/ThemeContext";
import styles from "./Theme.module.css";

// Reached from Settings -> Appearance (see SettingsMenu.jsx's
// SUBVIEWS.settings.items), same click-a-row-to-see-the-real-screen behavior
// as View profile / Change password. The whole row is the control — a real
// <button> wrapping icon/label/switch, so a tap anywhere on it (not just the
// small switch) toggles the theme, same "big hit target" pattern as
// SettingsMenu's own .menuItem rows. The switch on the right is purely a
// visual indicator (aria-hidden), not its own separately-focusable control —
// nesting an interactive element inside another <button> isn't valid HTML,
// and the whole row already carries the click/keyboard/focus behavior via
// native button semantics plus aria-pressed for the toggled state.
export default function Theme() {
  const { isDark, toggleTheme } = useTheme();

  return (
    <div className={styles.page}>
      <h1 className={styles.title}>Theme</h1>
      <button type="button" className={styles.card} onClick={toggleTheme} aria-pressed={isDark}>
        <span className={styles.icon} aria-hidden="true">
          {isDark ? "🌙" : "☀️"}
        </span>
        <span className={styles.label}>
          <span className={styles.labelName}>{isDark ? "Dark" : "Light"}</span>
          <span className={styles.caption}>
            {isDark ? "Default — tap to switch to Light" : "Tap to switch back to Dark, the default"}
          </span>
        </span>
        <span className={styles.switchTrack} data-on={isDark} aria-hidden="true">
          <span className={styles.switchKnob} />
        </span>
      </button>
    </div>
  );
}
