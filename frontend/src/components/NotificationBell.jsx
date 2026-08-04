import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { fetchMyNotifications, markAllNotificationsRead, markNotificationRead } from "../api/notifications";
import styles from "./NotificationBell.module.css";

// Where clicking a notification takes the patient — booked/rescheduled point at
// the still-active upcoming list, cancelled/auto-completed point at history since
// that appointment is no longer "upcoming" by the time its notification exists.
// Unknown/future types simply don't navigate (mark-as-read still happens).
const TYPE_DESTINATION = {
  appointment_booked: "/patient/appointments",
  appointment_rescheduled: "/patient/appointments",
  appointment_cancelled: "/patient/appointments/history",
  appointment_auto_completed: "/patient/appointments/history",
};

// Per-type icon + color family — lets a patient tell what happened at a glance
// (booked vs. rescheduled vs. cancelled vs. auto-completed) without reading the
// full message first, the same "icon + accent tone" language used for status on
// the appointment history timeline.
const TYPE_META = {
  appointment_booked: {
    tone: "info",
    icon: (
      <path d="M7 3v3M17 3v3M4 9h16M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1ZM9 15l2 2 4-4" />
    ),
  },
  appointment_rescheduled: {
    tone: "warning",
    icon: <path d="M3 12a9 9 0 1 0 2.64-6.36M3 5v5h5" />,
  },
  appointment_cancelled: {
    tone: "error",
    icon: <path d="M7 3v3M17 3v3M4 9h16M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1ZM10 13l4 4M14 13l-4 4" />,
  },
  appointment_auto_completed: {
    tone: "success",
    icon: <path d="M7 3v3M17 3v3M4 9h16M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1ZM8 14l3 3 5-5" />,
  },
};

const DEFAULT_TYPE_META = { tone: "neutral", icon: <path d="M12 8v4l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" /> };

function TypeIcon({ type }) {
  const meta = TYPE_META[type] || DEFAULT_TYPE_META;
  return (
    <span className={`${styles.itemIcon} ${styles[meta.tone]}`} aria-hidden="true">
      <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        {meta.icon}
      </svg>
    </span>
  );
}

// Coarse, day-level relative label for recent items, falling back to an exact
// date once it's more than a week old — matches the appointment history page's
// own relative-time convention, so the two features read consistently.
function formatWhen(iso) {
  const date = new Date(iso);
  const now = new Date();
  const diffDays = Math.floor((now.setHours(0, 0, 0, 0) - new Date(date).setHours(0, 0, 0, 0)) / 86400000);
  const time = date.toLocaleString(undefined, { hour: "numeric", minute: "2-digit" });
  if (diffDays <= 0) return `Today · ${time}`;
  if (diffDays === 1) return `Yesterday · ${time}`;
  if (diffDays < 7) return `${diffDays} days ago · ${time}`;
  return date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function dayBucketFor(iso, now) {
  const diffDays = Math.floor((now.setHours(0, 0, 0, 0) - new Date(iso).setHours(0, 0, 0, 0)) / 86400000);
  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return "This week";
  return "Earlier";
}

// Groups the (already newest-first) list into Today/Yesterday/This week/Earlier
// buckets without re-sorting — a Map preserves first-insertion key order, which
// for an already-sorted list is exactly the bucket order above.
function groupByDay(notifications) {
  const groups = new Map();
  for (const n of notifications) {
    const key = dayBucketFor(n.created_at, new Date());
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(n);
  }
  return Array.from(groups.entries());
}

function BellSkeletonRow() {
  return (
    <div className={styles.skeletonRow} aria-hidden="true">
      <span className={styles.skeletonIcon} />
      <span className={styles.skeletonLines}>
        <span className={styles.skeletonLine} style={{ width: "88%" }} />
        <span className={styles.skeletonLine} style={{ width: "45%" }} />
      </span>
    </div>
  );
}

// Header bell for the patient-facing in-app notification system — booking/
// reschedule/cancel/auto-complete events written server-side (see
// app/services/notifications.py). No websocket/real-time infra exists in this
// stack, so freshness comes from refetching on every navigation instead.
export default function NotificationBell() {
  const location = useLocation();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [notifications, setNotifications] = useState(null);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showUnreadOnly, setShowUnreadOnly] = useState(false);
  const panelRef = useRef(null);
  const launcherRef = useRef(null);

  const load = () => {
    fetchMyNotifications()
      .then((data) => {
        setNotifications(data.notifications);
        setUnreadCount(data.unread_count);
      })
      .catch(() => {});
  };

  // Refetches on every navigation (no websocket/real-time infra to push
  // updates instead) so the badge count stays current as the patient moves
  // through the app, e.g. right after booking or cancelling an appointment.
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname]);

  useEffect(() => {
    if (!open) return undefined;
    const handlePointerDown = (e) => {
      if (
        panelRef.current &&
        !panelRef.current.contains(e.target) &&
        launcherRef.current &&
        !launcherRef.current.contains(e.target)
      ) {
        setOpen(false);
      }
    };
    const handleKeyDown = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  // Always reopen on "All" rather than wherever the filter was left last time.
  useEffect(() => {
    if (!open) setShowUnreadOnly(false);
  }, [open]);

  const handleItemClick = async (notification) => {
    if (!notification.read_at) {
      setNotifications((prev) =>
        prev.map((n) => (n.id === notification.id ? { ...n, read_at: new Date().toISOString() } : n)),
      );
      setUnreadCount((c) => Math.max(0, c - 1));
      try {
        await markNotificationRead(notification.id);
      } catch {
        load();
      }
    }

    const destination = TYPE_DESTINATION[notification.type];
    if (destination) {
      setOpen(false);
      navigate(destination);
    }
  };

  const handleMarkAllRead = async () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read_at: n.read_at || new Date().toISOString() })));
    setUnreadCount(0);
    try {
      await markAllNotificationsRead();
    } catch {
      load();
    }
  };

  return (
    <div className={styles.wrapper}>
      <button
        type="button"
        className={styles.bellBtn}
        ref={launcherRef}
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={unreadCount > 0 ? `Notifications, ${unreadCount} unread` : "Notifications"}
      >
        <svg
          className={unreadCount > 0 ? styles.bellIconRing : ""}
          viewBox="0 0 24 24"
          width="24"
          height="24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9M13.73 21a2 2 0 0 1-3.46 0" />
        </svg>
        {unreadCount > 0 && (
          <span className={styles.badge} aria-hidden="true">
            {unreadCount > 99 ? "99+" : unreadCount}
          </span>
        )}
      </button>

      {open && (
        <div className={styles.panel} role="menu" aria-label="Notifications" ref={panelRef}>
          <div className={styles.panelHeader}>
            <span className={styles.panelHeading}>
              Notifications
              {unreadCount > 0 && <span className={styles.panelHeadingCount}>{unreadCount} new</span>}
            </span>
            <span className={styles.headerActions}>
              {unreadCount > 0 && (
                <button type="button" className={styles.markAllBtn} onClick={handleMarkAllRead}>
                  Mark all read
                </button>
              )}
              <button
                type="button"
                className={styles.closeBtn}
                onClick={() => setOpen(false)}
                aria-label="Close notifications"
              >
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M6 6l12 12M18 6L6 18" />
                </svg>
              </button>
            </span>
          </div>

          {notifications && notifications.length > 0 && (
            <div className={styles.filterBar} role="tablist" aria-label="Filter notifications">
              <button
                type="button"
                role="tab"
                aria-selected={!showUnreadOnly}
                className={`${styles.filterChip} ${!showUnreadOnly ? styles.filterChipActive : ""}`}
                onClick={() => setShowUnreadOnly(false)}
              >
                All
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={showUnreadOnly}
                className={`${styles.filterChip} ${showUnreadOnly ? styles.filterChipActive : ""}`}
                onClick={() => setShowUnreadOnly(true)}
                disabled={unreadCount === 0}
              >
                Unread{unreadCount > 0 ? ` (${unreadCount})` : ""}
              </button>
            </div>
          )}

          <div className={styles.panelBody}>
            {notifications === null && (
              <>
                <BellSkeletonRow />
                <BellSkeletonRow />
                <BellSkeletonRow />
              </>
            )}
            {notifications && notifications.length === 0 && (
              <div className={styles.emptyState}>
                <svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9M13.73 21a2 2 0 0 1-3.46 0" />
                </svg>
                <p className={styles.emptyText}>No notifications yet.</p>
              </div>
            )}
            {(() => {
              if (!notifications || notifications.length === 0) return null;
              const visible = showUnreadOnly ? notifications.filter((n) => !n.read_at) : notifications;
              if (visible.length === 0) {
                return (
                  <div className={styles.emptyState}>
                    <p className={styles.emptyText}>You're all caught up.</p>
                  </div>
                );
              }
              return groupByDay(visible).map(([bucket, items]) => (
                <div key={bucket}>
                  <div className={styles.dayHeading}>{bucket}</div>
                  {items.map((n) => (
                    <button
                      type="button"
                      key={n.id}
                      className={`${styles.item} ${n.read_at ? styles.itemRead : styles.itemUnread}`}
                      onClick={() => handleItemClick(n)}
                    >
                      <TypeIcon type={n.type} />
                      <span className={styles.itemBody}>
                        <span className={styles.itemMessage}>{n.message}</span>
                        <span className={styles.itemWhen}>{formatWhen(n.created_at)}</span>
                      </span>
                      {!n.read_at && <span className={styles.unreadDot} aria-hidden="true" />}
                    </button>
                  ))}
                </div>
              ));
            })()}
          </div>
        </div>
      )}
    </div>
  );
}
