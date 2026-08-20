import { useEffect, useState } from "react";
import { fetchFeedback, fetchFeedbackInsights } from "../../api/adminFeedback";
import { ApiError } from "../../api/client";
import EmptyState from "../../components/EmptyState";
import Pagination from "../../components/Pagination";
import Skeleton from "../../components/Skeleton";
import StarIcon from "../../components/StarIcon";
import { useReveal } from "../../hooks/useReveal";
import { formatDateOnly, formatDateTimeLocal as formatDate } from "../../utils/formatDateTime";
import styles from "./AdminScreens.module.css";
import feedbackStyles from "./Feedback.module.css";

const PAGE_SIZE = 10;

const TONE_TABS = [
  { key: null, label: "All" },
  { key: "good", label: "Good" },
  { key: "neutral", label: "Neutral" },
  { key: "bad", label: "Needs attention" },
];

function toneOfRating(rating) {
  if (rating >= 4) return "good";
  if (rating === 3) return "neutral";
  return "bad";
}

function Stars({ rating, size = 15 }) {
  return (
    <span className={feedbackStyles.stars} aria-label={`${rating} out of 5 stars`}>
      {[1, 2, 3, 4, 5].map((n) => (
        <StarIcon
          key={n}
          filled={rating >= n}
          size={size}
          className={rating >= n ? feedbackStyles.starFilled : feedbackStyles.starEmpty}
        />
      ))}
    </span>
  );
}

function StatTile({ label, value, tone }) {
  return (
    <div className={`${feedbackStyles.statTile} ${tone ? feedbackStyles[`statTile_${tone}`] : ""}`}>
      <span className={feedbackStyles.statValue}>{value}</span>
      <span className={feedbackStyles.statLabel}>{label}</span>
    </div>
  );
}

export default function Feedback() {
  const revealRef = useReveal();
  const [items, setItems] = useState(null);
  const [total, setTotal] = useState(0);
  const [toneCounts, setToneCounts] = useState({ good: 0, neutral: 0, bad: 0 });
  const [averageRating, setAverageRating] = useState(null);
  const [page, setPage] = useState(1);
  const [tone, setTone] = useState(null);
  const [error, setError] = useState(null);
  const [insight, setInsight] = useState(null);
  const [insightError, setInsightError] = useState(null);

  useEffect(() => {
    fetchFeedback({ limit: PAGE_SIZE, offset: (page - 1) * PAGE_SIZE, tone })
      .then((data) => {
        setItems(data.items);
        setTotal(data.total);
        setToneCounts(data.tone_counts);
        setAverageRating(data.average_rating);
      })
      .catch((err) => setError(err instanceof ApiError ? err.detail || err.message : "Failed to load feedback."));
  }, [page, tone]);

  // digest is null both when there's no low-rating feedback with a reason to
  // synthesize yet and when nothing has ever been generated (no LLM key, or every
  // attempt failed) — neither is an error, so the card just doesn't render.
  useEffect(() => {
    fetchFeedbackInsights()
      .then(setInsight)
      .catch((err) =>
        setInsightError(err instanceof ApiError ? err.detail || err.message : "Could not load feedback insights."),
      );
  }, []);

  const totalFeedback = toneCounts.good + toneCounts.neutral + toneCounts.bad;
  const badPct = totalFeedback > 0 ? Math.round((toneCounts.bad / totalFeedback) * 100) : 0;

  return (
    <div>
      <h1 className={styles.title}>Patient feedback</h1>
      <p className={styles.subtitle}>Post-appointment ratings patients have submitted via Cura, newest first.</p>

      {insightError && <p className={styles.errorText}>{insightError}</p>}
      {insight?.digest && (
        <div className={feedbackStyles.digestCard}>
          <div className={feedbackStyles.digestTitleRow}>
            <span className={feedbackStyles.digestTitleIcon} aria-hidden="true">
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
                <circle cx="12" cy="12" r="5" />
              </svg>
            </span>
            <h2 className={feedbackStyles.digestTitle}>Recurring themes</h2>
          </div>
          <p className={feedbackStyles.digestText}>{insight.digest}</p>
          {insight.generated_at && (
            <div className={feedbackStyles.digestUpdated}>Updated {formatDateOnly(insight.generated_at)}</div>
          )}
        </div>
      )}

      {totalFeedback > 0 && (
        <div className={feedbackStyles.statGrid}>
          <StatTile label="Total feedback" value={totalFeedback} />
          <StatTile
            label="Average rating"
            value={averageRating !== null ? averageRating.toFixed(1) : "—"}
          />
          <StatTile label="Good (4–5★)" value={toneCounts.good} tone="good" />
          <StatTile label="Neutral (3★)" value={toneCounts.neutral} tone="neutral" />
          <StatTile label="Needs attention" value={`${toneCounts.bad} (${badPct}%)`} tone="bad" />
        </div>
      )}

      <div className={`${styles.card} reveal`} ref={revealRef}>
        {error && <p className={styles.errorText}>{error}</p>}

        {items === null && !error && <Skeleton rows={4} />}

        {totalFeedback > 0 && (
          <div className={feedbackStyles.toneTabs} role="tablist" aria-label="Filter by rating">
            {TONE_TABS.map((tab) => (
              <button
                key={tab.label}
                type="button"
                role="tab"
                aria-selected={tone === tab.key}
                className={`${feedbackStyles.toneTab} ${tone === tab.key ? feedbackStyles.toneTabActive : ""}`}
                onClick={() => {
                  setTone(tab.key);
                  setPage(1);
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>
        )}

        {items && total === 0 && (
          <EmptyState
            icon="inbox"
            message={tone ? `No ${tone === "bad" ? "low-rated" : tone} feedback yet.` : "No feedback submitted yet."}
          />
        )}

        {items && total > 0 && (
          <>
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Patient</th>
                    <th>Doctor</th>
                    <th>Rating</th>
                    <th>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item) => (
                    <tr
                      key={item.id}
                      className={toneOfRating(item.rating) === "bad" ? feedbackStyles.rowBad : ""}
                    >
                      <td data-label="Date">{formatDate(item.created_at)}</td>
                      <td data-label="Patient">{item.patient_name}</td>
                      <td data-label="Doctor">{item.doctor_name}</td>
                      <td data-label="Rating">
                        <Stars rating={item.rating} />
                      </td>
                      <td data-label="Reason">{item.reason ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <Pagination page={page} pageSize={PAGE_SIZE} total={total} onPageChange={setPage} />
          </>
        )}
      </div>
    </div>
  );
}
