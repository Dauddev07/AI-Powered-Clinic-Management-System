import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fetchMyAccount } from "../api/auth";
import { fetchAdminDashboardStats, fetchAppointmentsTrend, fetchTopRatedDoctors } from "../api/adminDashboard";
import { ApiError } from "../api/client";
import EmptyState from "../components/EmptyState";
import Skeleton from "../components/Skeleton";
import StarIcon from "../components/StarIcon";
import WelcomeBanner from "../components/WelcomeBanner";
import { useCountUp } from "../hooks/useCountUp";
import { useReveal, revealDelayClass } from "../hooks/useReveal";
import { useTheme } from "../theme/ThemeContext";
import styles from "./AdminHome.module.css";

// Rank-badge tone per position — gold/silver/bronze is the universal "top 3"
// convention, so it reads at a glance without needing the "#1/#2/#3" label to
// do all the work on its own.
const RANK_TONES = ["gold", "silver", "bronze"];

function getInitials(fullName) {
  const parts = fullName.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  const first = parts[0][0] || "";
  const last = parts.length > 1 ? parts[parts.length - 1][0] || "" : "";
  return (first + last).toUpperCase();
}

function DoctorStars({ rating, size = 15 }) {
  const rounded = Math.round(rating);
  return (
    <span className={styles.doctorStars} aria-label={`${rating} out of 5 stars`}>
      {[1, 2, 3, 4, 5].map((n) => (
        <StarIcon
          key={n}
          filled={rounded >= n}
          size={size}
          className={rounded >= n ? styles.starFilled : styles.starEmpty}
        />
      ))}
    </span>
  );
}

function TopRatedDoctorCard({ doctor, rank }) {
  const tone = RANK_TONES[rank] || "bronze";
  return (
    <div className={`${styles.doctorCard} ${styles[`doctorCard_${tone}`]}`}>
      {/* Clips only the corner glow (see .doctorCardGlow) to the card's own
          rounded corners — kept as a separate layer rather than overflow:
          hidden on .doctorCard itself, which used to clip the top of
          .rankBadge below (it deliberately sits half outside the card's top
          edge, see .rankBadge's own top: -12px). */}
      <span className={styles.doctorCardGlow} aria-hidden="true" />
      <span className={`${styles.rankBadge} ${styles[`rankBadge_${tone}`]}`}>#{rank + 1}</span>
      <div className={styles.doctorCardHeader}>
        <span className={styles.doctorAvatar} aria-hidden="true">
          {getInitials(doctor.doctor_name)}
        </span>
        <div className={styles.doctorCardHeaderText}>
          <div className={styles.doctorName}>{doctor.doctor_name}</div>
          <div className={styles.doctorDepartment}>{doctor.department_name}</div>
        </div>
      </div>
      <div className={styles.doctorRatingRow}>
        <DoctorStars rating={doctor.average_rating} />
        <span className={styles.doctorRatingValue}>{doctor.average_rating.toFixed(1)}</span>
      </div>
      <div className={styles.doctorStatRow}>
        <div className={styles.doctorStat}>
          <span className={styles.doctorStatValue}>{doctor.rating_count}</span>
          <span className={styles.doctorStatLabel}>{doctor.rating_count === 1 ? "rating" : "ratings"}</span>
        </div>
        <div className={styles.doctorStatDivider} aria-hidden="true" />
        <div className={styles.doctorStat}>
          <span className={styles.doctorStatValue}>{doctor.visit_count}</span>
          <span className={styles.doctorStatLabel}>{doctor.visit_count === 1 ? "visit" : "visits"}</span>
        </div>
      </div>
    </div>
  );
}

const WEEKDAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
// One/two-letter weekday initials for the X-axis on narrow screens — "Th" (not
// bare "T") disambiguates Thursday from Tuesday, the standard narrow-weekday
// convention calendar UIs use.
const WEEKDAY_INITIALS = ["Su", "M", "T", "W", "Th", "F", "Sa"];

// The trend endpoint returns plain "YYYY-MM-DD" clinic-local calendar dates —
// parsed and formatted entirely via UTC getters (never `new Date(iso)` +
// local getters) so the label can't shift a day depending on the viewer's
// own browser timezone offset.
function getUtcWeekday(isoDate) {
  const [year, month, day] = isoDate.split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, day)).getUTCDay();
}

function formatDayLabel(isoDate) {
  const day = Number(isoDate.split("-")[2]);
  return `${WEEKDAY_LABELS[getUtcWeekday(isoDate)]} ${day}`;
}

function formatDayLabelNarrow(isoDate) {
  return WEEKDAY_INITIALS[getUtcWeekday(isoDate)];
}

// Switches the X-axis to weekday initials on narrow screens (see the ticket:
// "M, T, W, Th" instead of "Mon 21") — re-evaluated on resize/orientation
// change via matchMedia, not just read once on mount.
// Fixed categorical order (validated for CVD-safety, see the data-viz palette
// reference) — kept as one fixed palette across both light/dark themes rather
// than swapping per theme; each color is saturated enough to stay legible
// against both a light card and a dark one. Assigned by slot order, never
// cycled/regenerated per render.
const PIE_COLORS = [
  "#3987e5", // blue
  "#d95926", // orange
  "#199e70", // aqua
  "#c98500", // yellow
  "#d55181", // magenta
  "#008300", // green
  "#9085e9", // violet
  "#e66767", // red
];
// Instructed live: the card is titled "Busiest doctors today", but with every
// doctor who saw a patient today plotted it stopped reading as a top-N
// leaderboard — capped to the busiest 3 (the backend already returns them
// busiest-first) instead of folding the rest into an "Other" slice.
const BUSIEST_DOCTORS_LIMIT = 3;

function useIsNarrowScreen(query) {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);

  useEffect(() => {
    const mql = window.matchMedia(query);
    const handleChange = (e) => setMatches(e.matches);
    mql.addEventListener("change", handleChange);
    return () => mql.removeEventListener("change", handleChange);
  }, [query]);

  return matches;
}

// recharts renders its axes/bars as SVG presentation attributes, which is a
// less universally reliable place to drop a raw `var(--x)` string than an
// inline CSS style, so the actual resolved color values are read here instead
// of guessing. Re-reads whenever `theme` changes (see ThemeContext) — with the
// light/dark toggle, these values are no longer fixed for the page's lifetime,
// so a mount-only read would leave the chart showing the PREVIOUS theme's
// colors (e.g. dark-theme axis text) against the newly switched background.
function useChartColors() {
  const { theme } = useTheme();
  const [colors, setColors] = useState({ accent: "#87977f", border: "#2c2d30", subtitle: "#7c7e83" });

  useEffect(() => {
    const computed = getComputedStyle(document.documentElement);
    const read = (name, fallback) => computed.getPropertyValue(name).trim() || fallback;
    setColors({
      accent: read("--app-accent", "#87977f"),
      border: read("--app-border", "#2c2d30"),
      subtitle: read("--app-text-subtitle", "#7c7e83"),
    });
  }, [theme]);

  return colors;
}

// The doctor-management and knowledge-base action cards that used to sit
// below the welcome banner have moved into the fixed account menu (see
// SettingsMenu.jsx, "Manage Doctors" and "Upload Documents") — this page is
// the welcome landing spot, plus (added here) the first real content: active
// doctor count and today's slot utilization (GET /admin/dashboard/stats),
// plus a 7-day booking-volume chart with today's headline count (GET
// /admin/dashboard/appointments-trend).
export default function AdminHome() {
  const [fullName, setFullName] = useState(null);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [trend, setTrend] = useState(null);
  const [trendError, setTrendError] = useState(null);
  const [topRatedDoctors, setTopRatedDoctors] = useState(null);
  const [topRatedError, setTopRatedError] = useState(null);
  const revealRef = useReveal();
  const chartColors = useChartColors();
  const isNarrowScreen = useIsNarrowScreen("(max-width: 480px)");

  useEffect(() => {
    fetchMyAccount()
      .then((data) => setFullName(data.full_name))
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetchAdminDashboardStats()
      .then(setStats)
      .catch((err) =>
        setError(err instanceof ApiError ? err.detail || err.message : "Could not load dashboard stats."),
      );
  }, []);

  useEffect(() => {
    fetchAppointmentsTrend()
      .then(setTrend)
      .catch((err) =>
        setTrendError(err instanceof ApiError ? err.detail || err.message : "Could not load the appointments trend."),
      );
  }, []);

  useEffect(() => {
    fetchTopRatedDoctors()
      .then(setTopRatedDoctors)
      .catch((err) =>
        setTopRatedError(err instanceof ApiError ? err.detail || err.message : "Could not load top rated doctors."),
      );
  }, []);

  const utilization = stats?.slot_utilization_today;
  const trendData = trend
    ? trend.map((d) => ({
        label: formatDayLabel(d.date),
        narrowLabel: formatDayLabelNarrow(d.date),
        count: d.count,
      }))
    : [];
  // The trend list is oldest-to-newest ending today, so the last entry is
  // always today's booking-activity count — no separate query needed for it.
  const bookedToday = trend ? trend[trend.length - 1].count : null;

  // Counting up from 0 (rather than the number just appearing the instant
  // data loads) gives the numbers a small "live dashboard" feel — each hook
  // call is a no-op (returns its target immediately) until its own value
  // stops being null, and again for anyone with prefers-reduced-motion set.
  const activeDoctorsAnimated = useCountUp(stats?.active_doctors_count ?? null);
  const utilizationPercentAnimated = useCountUp(utilization?.percentage ?? null);
  const bookedTodayAnimated = useCountUp(bookedToday);

  // Busiest-first per the backend's own ordering — top 3 only, see
  // BUSIEST_DOCTORS_LIMIT above.
  const busiestDoctors = (stats?.busiest_doctors_today || []).slice(0, BUSIEST_DOCTORS_LIMIT);
  const pieData = busiestDoctors.map((d) => ({ name: d.doctor_name, value: d.count }));

  return (
    <>
      <WelcomeBanner
        name={fullName}
        tagline="Use the account menu to manage doctors or upload knowledge base documents."
      />

      {error && <p className={styles.errorText}>{error}</p>}

      <div className={styles.sectionLabel}>Overview</div>
      <div className={styles.statsGrid}>
        <div className={`${styles.card} reveal`} ref={revealRef}>
          <div className={styles.cardTitleRow}>
            <span className={`${styles.titleIcon} ${styles.titleIcon_info}`} aria-hidden="true">
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 3v5a3 3 0 0 0 6 0V3M8 11v1a5 5 0 0 0 10 0V9" />
                <circle cx="18" cy="7" r="2" />
              </svg>
            </span>
            <h2 className={styles.cardTitle}>Active doctors</h2>
          </div>
          {stats === null && !error && <Skeleton rows={1} />}
          {stats && (
            <>
              <div className={styles.statValue}>{activeDoctorsAnimated}</div>
              <div className={styles.statLabel}>of {stats.total_doctors_count} total doctors</div>
            </>
          )}
        </div>

        <div className={`${styles.card} reveal ${revealDelayClass(1)}`} ref={revealRef}>
          <div className={styles.cardTitleRow}>
            <span className={`${styles.titleIcon} ${styles.titleIcon_warning}`} aria-hidden="true">
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 8v4l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
              </svg>
            </span>
            <h2 className={styles.cardTitle}>Slot utilization today</h2>
          </div>
          {stats === null && !error && <Skeleton rows={1} />}
          {stats && (
            <>
              <div className={styles.utilizationSummary}>
                <span className={styles.utilizationFraction}>
                  {utilization.booked} of {utilization.total} booked
                </span>
                <span className={styles.utilizationPercent}>{utilizationPercentAnimated}%</span>
              </div>
              <div
                className={styles.barTrack}
                title={`${utilization.booked} of ${utilization.total} slots booked today (${utilization.percentage}%)`}
              >
                <div className={styles.barFill} style={{ width: `${Math.min(100, utilization.percentage)}%` }} />
              </div>
            </>
          )}
        </div>

        <div className={`${styles.card} reveal ${revealDelayClass(1)}`} ref={revealRef}>
          <div className={styles.cardTitleRow}>
            <span className={`${styles.titleIcon} ${styles.titleIcon_success}`} aria-hidden="true">
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
                <path d="M12 3v18M3 12h18" opacity="0.5" />
              </svg>
            </span>
            <h2 className={styles.cardTitle}>Busiest doctors today</h2>
          </div>
          {stats === null && !error && <Skeleton rows={3} />}
          {stats && pieData.length === 0 && (
            <EmptyState icon="calendar" message="No appointments today yet." />
          )}
          {stats && pieData.length > 0 && (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80} paddingAngle={2}>
                  {pieData.map((entry, index) => (
                    <Cell
                      key={entry.name}
                      fill={PIE_COLORS[index]}
                      stroke="var(--app-bg-card)"
                      strokeWidth={2}
                    />
                  ))}
                </Pie>
                <Legend wrapperStyle={{ fontSize: 12, color: chartColors.subtitle }} />
                <Tooltip
                  contentStyle={{
                    background: "var(--app-bg-page)",
                    border: "1px solid var(--app-border)",
                    borderRadius: 10,
                  }}
                  labelStyle={{ color: "var(--app-text-title)", fontWeight: 600 }}
                  itemStyle={{ color: "var(--app-text-body)" }}
                  formatter={(value, name) => [value, name]}
                />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className={styles.sectionLabel}>Insights</div>

      <div className={`${styles.card} reveal ${revealDelayClass(2)}`} ref={revealRef}>
        <div className={styles.cardTitleRow}>
          <span className={`${styles.titleIcon} ${styles.titleIcon_warning}`} aria-hidden="true">
            <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2.5l2.9 6.06 6.6.86-4.85 4.63 1.24 6.6L12 17.6l-5.89 3.05 1.24-6.6-4.85-4.63 6.6-.86L12 2.5Z" />
            </svg>
          </span>
          <h2 className={styles.cardTitle}>Top rated doctors</h2>
        </div>
        {topRatedDoctors === null && !topRatedError && <Skeleton rows={3} />}
        {topRatedError && <p className={styles.errorText}>{topRatedError}</p>}
        {topRatedDoctors && topRatedDoctors.length === 0 && (
          <EmptyState icon="star" message="No doctor ratings yet." />
        )}
        {topRatedDoctors && topRatedDoctors.length > 0 && (
          <div className={styles.doctorGrid}>
            {topRatedDoctors.map((doctor, index) => (
              <TopRatedDoctorCard key={doctor.doctor_id} doctor={doctor} rank={index} />
            ))}
          </div>
        )}
      </div>

      <div className={`${styles.card} reveal ${revealDelayClass(2)}`} ref={revealRef}>
        <div className={styles.cardTitleRow}>
          <span className={`${styles.titleIcon} ${styles.titleIcon_info}`} aria-hidden="true">
            <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M7 3v3M17 3v3M4 9h16M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1Z" />
            </svg>
          </span>
          <h2 className={styles.cardTitle}>Appointments booked — last 7 days</h2>
        </div>
        {trend === null && !trendError && <Skeleton rows={4} />}
        {trendError && <p className={styles.errorText}>{trendError}</p>}
        {trend && (
          <>
            <div className={styles.utilizationSummary}>
              <span className={styles.utilizationFraction}>{bookedTodayAnimated} booked today</span>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={trendData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid vertical={false} stroke={chartColors.border} />
                <XAxis
                  dataKey={isNarrowScreen ? "narrowLabel" : "label"}
                  tick={{ fill: chartColors.subtitle, fontSize: 12 }}
                  axisLine={{ stroke: chartColors.border }}
                  tickLine={false}
                  interval={0}
                />
                <YAxis
                  allowDecimals={false}
                  tick={{ fill: chartColors.subtitle, fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                  width={36}
                />
                <Tooltip
                  cursor={{ fill: chartColors.border, opacity: 0.4 }}
                  contentStyle={{
                    background: "var(--app-bg-page)",
                    border: "1px solid var(--app-border)",
                    borderRadius: 10,
                  }}
                  labelStyle={{ color: "var(--app-text-title)", fontWeight: 600, marginBottom: 4 }}
                  itemStyle={{ color: "var(--app-text-body)" }}
                  labelFormatter={(_, payload) => payload?.[0]?.payload?.label ?? ""}
                  formatter={(value) => [value, "Booked"]}
                />
                <Bar dataKey="count" fill={chartColors.accent} radius={[4, 4, 0, 0]} maxBarSize={36} />
              </BarChart>
            </ResponsiveContainer>
          </>
        )}
      </div>
    </>
  );
}
