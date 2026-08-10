import StarIcon from "./StarIcon";
import styles from "./DoctorRatingBadge.module.css";

// A compact "★ 4.8 (32)" badge for contexts too small for the full 5-star
// row + stat card treatment used on the landing page / AdminHome's "top
// rated doctors" sections (see their own local DoctorStars) — the patient
// slot table and the chatbot's doctor-option cards just need a quick,
// glanceable signal next to a doctor's name. Renders nothing for a doctor
// with no ratings yet (averageRating is null), rather than a "0.0" or an
// empty star row that would read as "rated zero".
export default function DoctorRatingBadge({ averageRating, ratingCount, className }) {
  if (averageRating == null) return null;
  return (
    <span
      className={`${styles.badge} ${className || ""}`}
      aria-label={`Rated ${averageRating.toFixed(1)} out of 5${ratingCount ? ` from ${ratingCount} review${ratingCount === 1 ? "" : "s"}` : ""}`}
    >
      <StarIcon filled size={12} className={styles.icon} />
      <span className={styles.value}>{averageRating.toFixed(1)}</span>
      {ratingCount > 0 && <span className={styles.count}>({ratingCount})</span>}
    </span>
  );
}
