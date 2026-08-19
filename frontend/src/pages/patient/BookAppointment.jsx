import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  bookAppointment,
  fetchDepartmentsWithSlots,
  fetchDoctorsWithSlots,
  fetchSlots,
} from "../../api/patientBooking";
import { ApiError } from "../../api/client";
import DateInput from "../../components/DateInput";
import DoctorRatingBadge from "../../components/DoctorRatingBadge";
import StatusBadge from "../../components/StatusBadge";
import EmptyState from "../../components/EmptyState";
import Skeleton from "../../components/Skeleton";
import SuccessCheck from "../../components/SuccessCheck";
import Pagination from "../../components/Pagination";
import { useReveal, revealDelayClass } from "../../hooks/useReveal";
import styles from "./PatientScreens.module.css";

const PAGE_SIZE = 20;

function formatDateHeading(iso, timeZone) {
  return new Date(iso).toLocaleDateString(undefined, {
    weekday: "long",
    month: "short",
    day: "numeric",
    timeZone,
  });
}

function formatTime(iso, timeZone) {
  return new Date(iso).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
    timeZone,
  });
}

function localDateKey(iso, timeZone) {
  return new Date(iso).toLocaleDateString("en-CA", { timeZone });
}

function addDaysToDateKey(dateKey, days) {
  const d = new Date(`${dateKey}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

const DATE_FILTERS = [
  { value: "", label: "All upcoming" },
  { value: "today", label: "Today" },
  { value: "week", label: "This week" },
  { value: "next7", label: "Next 7 days" },
  { value: "custom", label: "Specific date…" },
];

const TIME_OF_DAY_FILTERS = [
  { value: "", label: "Any time of day" },
  { value: "morning", label: "Morning (before 12:00)" },
  { value: "afternoon", label: "Afternoon (12:00–17:00)" },
  { value: "evening", label: "Evening (after 17:00)" },
];

// Shared dropdown for every filter field on this page (Department, Doctor, Date,
// Time of day) — replaces native <select> elements entirely. Reported live: a
// native <select> with enough options (Department's 12+ entries) triggers the
// OS/browser's own full-screen picker on mobile (iOS Safari in particular), which
// centers on the currently-selected option and can render starting above the top
// of the visible viewport, behind the browser chrome — that rendering is done by
// the OS, not this page, so no amount of our own CSS can reposition or clamp it.
// A custom, in-DOM dropdown (portaled to <body>, positioned via measured
// coordinates) sidesteps the native picker entirely and CAN be kept on-screen: see
// openMenu below, which flips the panel above the trigger when there isn't enough
// room below, and clamps its horizontal position so it never runs off either edge.
function FilterDropdown({ id, label, value, options, onChange, searchable = false, searchPlaceholder }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [panelStyle, setPanelStyle] = useState(null);
  const triggerRef = useRef(null);
  const panelRef = useRef(null);
  const searchInputRef = useRef(null);

  const selectedLabel = options.find((o) => o.value === value)?.label ?? options[0]?.label ?? "";

  const filteredOptions = useMemo(() => {
    if (!searchable) return options;
    const q = query.trim().toLowerCase();
    return q ? options.filter((o) => o.label.toLowerCase().includes(q)) : options;
  }, [options, query, searchable]);

  const openMenu = () => {
    const rect = triggerRef.current?.getBoundingClientRect();
    if (rect) {
      // The panel's real height isn't known until it renders (it depends on how
      // many options/whether the search input is shown), so this estimates a
      // worst-case height from the same CSS constants the panel itself uses
      // (comboboxList's 200px max-height + the search input + padding/shadow
      // margin) purely to decide FLIP DIRECTION — the panel is portaled and
      // position: fixed either way, so an estimate that's a little off just
      // means the flip decision is made a little conservatively, never a broken
      // layout.
      const estimatedPanelHeight = (searchable ? 260 : 220);
      const spaceBelow = window.innerHeight - rect.bottom;
      const spaceAbove = rect.top;
      const openAbove = spaceBelow < estimatedPanelHeight && spaceAbove > spaceBelow;
      const left = Math.min(Math.max(8, rect.left), window.innerWidth - rect.width - 8);
      setPanelStyle({
        left,
        width: rect.width,
        ...(openAbove ? { bottom: window.innerHeight - rect.top + 4 } : { top: rect.bottom + 4 }),
      });
    }
    setOpen(true);
    if (searchable) setTimeout(() => searchInputRef.current?.focus(), 0);
  };

  const closeMenu = () => {
    setOpen(false);
    setQuery("");
  };

  useEffect(() => {
    if (!open) return;
    const handlePointerDown = (e) => {
      const inTrigger = triggerRef.current && triggerRef.current.contains(e.target);
      const inPanel = panelRef.current && panelRef.current.contains(e.target);
      if (!inTrigger && !inPanel) closeMenu();
    };
    const handleKeyDown = (e) => {
      if (e.key === "Escape") closeMenu();
    };
    // `scroll` doesn't bubble, but a capture-phase listener on window still sees it
    // for every scrollable descendant, including the option list itself (it has
    // its own overflow-y: auto) — so scrolling *inside* the list must be ignored
    // here, or scrolling to reach a lower option would immediately close the panel.
    //
    // Reported live (small screens, searchable dropdowns only — Doctor's own
    // filter): openMenu() auto-focuses the search input the instant the panel
    // opens, which raises the on-screen keyboard — and on mobile that keyboard
    // appearing fires both a window `resize` (the visual viewport shrinks) and
    // often a `scroll` (the browser nudges the page to keep the focused input in
    // view), each of which used to be treated as "the user left/resized the
    // page," closing the panel an instant after it opened, before a single key
    // could be typed. Both handlers below now skip closing while focus is still
    // inside the panel (i.e. the search input itself) — that's exactly the
    // keyboard-driven case; a genuine tap/scroll *outside* the panel still blurs
    // the input first, so this can't mask the real close-on-outside-interaction
    // behavior handlePointerDown already provides.
    const focusIsInsidePanel = () =>
      panelRef.current && document.activeElement && panelRef.current.contains(document.activeElement);
    const handleScroll = (e) => {
      if (panelRef.current && panelRef.current.contains(e.target)) return;
      if (focusIsInsidePanel()) return;
      closeMenu();
    };
    const handleResize = () => {
      if (focusIsInsidePanel()) return;
      closeMenu();
    };
    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    window.addEventListener("scroll", handleScroll, true);
    window.addEventListener("resize", handleResize);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("scroll", handleScroll, true);
      window.removeEventListener("resize", handleResize);
    };
  }, [open]);

  return (
    <div className={styles.filterField}>
      <label htmlFor={id}>{label}</label>
      <div className={styles.combobox}>
        <button
          type="button"
          id={id}
          ref={triggerRef}
          className={styles.comboboxTrigger}
          onClick={() => (open ? closeMenu() : openMenu())}
          aria-haspopup="listbox"
          aria-expanded={open}
        >
          <span className={styles.comboboxTriggerText}>{selectedLabel}</span>
          <span className={styles.comboboxChevron} aria-hidden="true">
            ▾
          </span>
        </button>

        {open &&
          panelStyle &&
          createPortal(
            <div ref={panelRef} className={styles.comboboxPanel} style={panelStyle}>
              {searchable && (
                <input
                  ref={searchInputRef}
                  type="text"
                  className={styles.comboboxSearchInput}
                  placeholder={searchPlaceholder}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  aria-label={searchPlaceholder}
                />
              )}
              <ul className={styles.comboboxList} role="listbox" aria-label={label}>
                {filteredOptions.map((o) => (
                  <li
                    key={o.value}
                    role="option"
                    aria-selected={value === o.value}
                    className={value === o.value ? styles.comboboxOptionSelected : styles.comboboxOption}
                    onClick={() => {
                      onChange(o.value);
                      closeMenu();
                    }}
                  >
                    {o.label}
                  </li>
                ))}
                {searchable && filteredOptions.length === 0 && (
                  <li className={styles.comboboxEmpty}>No matches for "{query}"</li>
                )}
              </ul>
            </div>,
            document.body,
          )}
      </div>
    </div>
  );
}

export default function BookAppointment() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const revealRef = useReveal();
  const [departments, setDepartments] = useState([]);
  const [departmentId, setDepartmentId] = useState("");
  const [doctorOptions, setDoctorOptions] = useState([]);
  const [slotData, setSlotData] = useState(null);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [bookingSlotId, setBookingSlotId] = useState(null);
  const [message, setMessage] = useState(null);

  const [doctorId, setDoctorId] = useState("");
  const [dateFilter, setDateFilter] = useState("");
  const [customDate, setCustomDate] = useState("");
  const [timeOfDay, setTimeOfDay] = useState("");

  // Set once the clinic's timezone is known from the first response, so later
  // fetches can translate the selected date filter into a clinic-local calendar
  // range without waiting on a state update / re-render round trip.
  const clinicTzRef = useRef(null);

  const computeDateRange = (dateFilterValue, customDateValue, tz) => {
    if (!tz) return { rangeStart: null, rangeEnd: null };
    const today = localDateKey(new Date().toISOString(), tz);
    if (dateFilterValue === "today") {
      return { rangeStart: today, rangeEnd: today };
    }
    if (dateFilterValue === "week") {
      const dow = new Date(`${today}T00:00:00Z`).getUTCDay(); // 0 = Sunday
      const daysUntilSunday = dow === 0 ? 0 : 7 - dow;
      return { rangeStart: today, rangeEnd: addDaysToDateKey(today, daysUntilSunday) };
    }
    if (dateFilterValue === "next7") {
      return { rangeStart: today, rangeEnd: addDaysToDateKey(today, 6) };
    }
    if (dateFilterValue === "custom" && customDateValue) {
      return { rangeStart: customDateValue, rangeEnd: customDateValue };
    }
    return { rangeStart: null, rangeEnd: null };
  };

  const loadSlots = async ({ departmentId: deptId, doctorId: docId, dateFilter: df, customDate: cd, timeOfDay: tod, page: pageToLoad }) => {
    setLoading(true);
    setError(null);
    try {
      // The clinic-local date range is resolved server-side too (see GET /slots),
      // combined with — never in place of — its own start_utc >= now guard, so a
      // "Today" fetch made late in the day only ever returns what's left of it.
      const { rangeStart, rangeEnd } = computeDateRange(df, cd, clinicTzRef.current);
      const data = await fetchSlots(deptId || undefined, {
        doctorId: docId || undefined,
        dateFrom: rangeStart || undefined,
        dateTo: rangeEnd || undefined,
        timeOfDay: tod || undefined,
        limit: PAGE_SIZE,
        offset: (pageToLoad - 1) * PAGE_SIZE,
      });
      clinicTzRef.current = data.clinic_timezone;
      setSlotData(data);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.detail || err.message
          : "Could not load slots.",
      );
    } finally {
      setLoading(false);
    }
  };

  const loadDoctorOptions = async (deptId, df, cd) => {
    try {
      const { rangeStart, rangeEnd } = computeDateRange(df, cd, clinicTzRef.current);
      const data = await fetchDoctorsWithSlots(deptId || undefined, {
        dateFrom: rangeStart || undefined,
        dateTo: rangeEnd || undefined,
      });
      setDoctorOptions(data);
    } catch {
      // Non-critical — the doctor filter just stays at whatever it last had.
    }
  };

  useEffect(() => {
    // Landing page's department cards deep-link here as ?department=<name> for a
    // signed-in patient — resolved against the real backend department list
    // (matched by name, case-insensitively) since the card only knows the
    // clinic's static marketing name, never a department_id. No param is by far
    // the common case (direct navigation to this page), so that path keeps firing
    // both fetches in parallel exactly as before; only a present param waits on
    // the department list first, since it's needed to resolve the id.
    const deptNameFromUrl = searchParams.get("department");
    const departmentsPromise = fetchDepartmentsWithSlots()
      .then((depts) => {
        setDepartments(depts);
        return depts;
      })
      .catch(() => []);

    if (deptNameFromUrl) {
      departmentsPromise.then((depts) => {
        const match = depts.find((d) => d.name.toLowerCase() === deptNameFromUrl.toLowerCase());
        const initialId = match ? match.id : "";
        setDepartmentId(initialId);
        loadSlots({ departmentId: initialId, doctorId: "", dateFilter: "", customDate: "", timeOfDay: "", page: 1 });
        loadDoctorOptions(initialId, "", "");
      });
    } else {
      loadSlots({ departmentId: "", doctorId: "", dateFilter: "", customDate: "", timeOfDay: "", page: 1 });
      loadDoctorOptions("", "", "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleDepartmentChange = (value) => {
    setDepartmentId(value);
    setDoctorId("");
    setPage(1);
    loadSlots({ departmentId: value, doctorId: "", dateFilter, customDate, timeOfDay, page: 1 });
    loadDoctorOptions(value, dateFilter, customDate);
  };

  const selectDoctor = (value) => {
    setDoctorId(value);
    setPage(1);
    loadSlots({ departmentId, doctorId: value, dateFilter, customDate, timeOfDay, page: 1 });
  };

  const handleDateFilterChange = (value) => {
    setDateFilter(value);
    setPage(1);
    loadSlots({ departmentId, doctorId, dateFilter: value, customDate, timeOfDay, page: 1 });
    loadDoctorOptions(departmentId, value, customDate);
  };

  const handleCustomDateChange = (e) => {
    const value = e.target.value;
    setCustomDate(value);
    setPage(1);
    loadSlots({ departmentId, doctorId, dateFilter, customDate: value, timeOfDay, page: 1 });
    loadDoctorOptions(departmentId, dateFilter, value);
  };

  const handleTimeOfDayChange = (value) => {
    setTimeOfDay(value);
    setPage(1);
    loadSlots({ departmentId, doctorId, dateFilter, customDate, timeOfDay: value, page: 1 });
  };

  const handlePageChange = (newPage) => {
    setPage(newPage);
    loadSlots({ departmentId, doctorId, dateFilter, customDate, timeOfDay, page: newPage });
  };

  const handleBook = async (slotId) => {
    setBookingSlotId(slotId);
    setMessage(null);
    setError(null);
    try {
      await bookAppointment(slotId);
      // Booking succeeded — hand off to Upcoming Appointments with the
      // confirmation message rather than showing it on this page, since the
      // patient's next useful action is there, not back on the slot list.
      navigate("/patient/appointments", {
        state: { bookingMessage: "Appointment booked successfully." },
      });
      return;
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        // Lost the race — refresh immediately so the patient sees fresh
        // alternatives rather than a dead error sitting over a stale list.
        setMessage({
          ok: false,
          text: `${err.detail || "That slot was just taken."}`,
        });
        await loadSlots({ departmentId, doctorId, dateFilter, customDate, timeOfDay, page });
      } else {
        setError(
          err instanceof ApiError
            ? err.detail || err.message
            : "Booking failed.",
        );
      }
      // Unsuccessful booking — the message/error banner lives at the top of
      // the page, above the slot table the patient may have scrolled past.
      window.scrollTo({ top: 0, behavior: "smooth" });
    } finally {
      setBookingSlotId(null);
    }
  };

  // Slots that haven't started yet as of this exact render — a defensive guard on
  // top of the server's own future-only filter, so a slot that was still upcoming
  // when fetched but has since ticked into the past (a long-open tab, no refetch)
  // is never shown as a clickable option, no matter what filter is selected.
  const notStartedSlots = useMemo(() => {
    if (!slotData) return [];
    const now = Date.now();
    return slotData.slots.filter((slot) => new Date(slot.start_utc).getTime() > now);
  }, [slotData]);

  // Reset doctor filter if the selected doctor no longer has any bookable slots
  // (e.g. after switching departments or a refresh took their last open slot).
  useEffect(() => {
    if (doctorId && !doctorOptions.some((d) => d.id === doctorId)) {
      setDoctorId("");
    }
  }, [doctorOptions, doctorId]);

  // FilterDropdown works in {value, label} pairs — these translate each filter's
  // own data shape into that once per render rather than inside the shared
  // component, which has no idea what a "department" or "doctor" is.
  const departmentDropdownOptions = useMemo(
    () => [{ value: "", label: "All departments" }, ...departments.map((d) => ({ value: d.id, label: d.name }))],
    [departments],
  );
  const doctorDropdownOptions = useMemo(
    () => [{ value: "", label: "All doctors" }, ...doctorOptions.map((d) => ({ value: d.id, label: d.name }))],
    [doctorOptions],
  );

  const groupedByDate = useMemo(() => {
    if (!slotData) return [];
    const tz = slotData.clinic_timezone;
    const groups = new Map();
    for (const slot of notStartedSlots) {
      const dateKey = localDateKey(slot.start_utc, tz);
      if (!groups.has(dateKey)) {
        groups.set(dateKey, {
          dateKey,
          heading: formatDateHeading(slot.start_utc, tz),
          slots: [],
        });
      }
      groups.get(dateKey).slots.push(slot);
    }
    return Array.from(groups.values()).sort((a, b) =>
      a.dateKey.localeCompare(b.dateKey),
    );
  }, [slotData, notStartedSlots]);

  const anyFilterActive = Boolean(
    departmentId || doctorId || dateFilter || timeOfDay,
  );

  return (
    <div>
      <h1 className={styles.title}>Book an appointment</h1>
      <p className={styles.subtitle}>
        Browse available slots below. Already-taken slots are shown too, so you
        can see the full shape of each doctor's day.
      </p>
      <p className={styles.policyNote}>
        Once booked, cancellations and reschedules aren't possible within 2 hours of the appointment time — for
        last-minute changes after that, please contact the clinic directly.
      </p>

      <div className={`${styles.card} reveal`} ref={revealRef}>
        <div className={styles.filterRow}>
          <FilterDropdown
            id="department"
            label="Department"
            value={departmentId}
            options={departmentDropdownOptions}
            onChange={handleDepartmentChange}
          />

          <FilterDropdown
            id="doctor"
            label="Doctor"
            value={doctorId}
            options={doctorDropdownOptions}
            onChange={selectDoctor}
            searchable
            searchPlaceholder="Search doctor by name…"
          />

          <div className={styles.filterField}>
            <FilterDropdown
              id="dateFilter"
              label="Date"
              value={dateFilter}
              options={DATE_FILTERS}
              onChange={handleDateFilterChange}
            />
            {dateFilter === "custom" && (
              <DateInput
                className={styles.select}
                value={customDate}
                onChange={handleCustomDateChange}
              />
            )}
          </div>

          <FilterDropdown
            id="timeOfDay"
            label="Time of day"
            value={timeOfDay}
            options={TIME_OF_DAY_FILTERS}
            onChange={handleTimeOfDayChange}
          />
        </div>

        {message && (
          <p className={message.ok ? styles.successBox : styles.noticeBox}>
            {message.ok && <SuccessCheck />}
            {message.text}
          </p>
        )}
        {error && <p className={styles.errorText}>{error}</p>}
      </div>

      {loading && (
        <div className={styles.card}>
          <Skeleton rows={4} />
        </div>
      )}

      {!loading && groupedByDate.length === 0 && (
        <div className={`${styles.card} reveal`} ref={revealRef}>
          <EmptyState
            icon="calendar"
            message={`No upcoming slots${anyFilterActive ? " match these filters" : ""} right now.`}
          />
        </div>
      )}

      {!loading &&
        groupedByDate.map((group, i) => (
          <div
            key={group.dateKey}
            className={`${styles.card} reveal ${revealDelayClass(i)}`}
            ref={revealRef}
          >
            <h2 className={styles.dateHeading}>{group.heading}</h2>
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Doctor</th>
                    <th>Department</th>
                    <th>Status</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {group.slots.map((slot) => (
                    <tr
                      key={slot.id}
                      className={`${slot.is_bookable ? "" : styles.takenRow} ${bookingSlotId === slot.id ? styles.bookingRow : ""}`}
                    >
                      <td data-label="Time">
                        {formatTime(slot.start_utc, slotData.clinic_timezone)}
                      </td>
                      <td data-label="Doctor">
                        <span className={styles.slotDoctorCell}>
                          <span className={styles.slotDoctorName}>{slot.doctor_name}</span>
                          <DoctorRatingBadge
                            averageRating={slot.average_rating}
                            ratingCount={slot.rating_count}
                          />
                        </span>
                      </td>
                      <td data-label="Department">{slot.department_name}</td>
                      <td data-label="Status">
                        <StatusBadge tone={slot.is_bookable ? "info" : "neutral"} label={slot.is_bookable ? "Open" : "Taken"} />
                      </td>
                      <td data-label="">
                        {slot.is_bookable ? (
                          <button
                            type="button"
                            className={styles.primaryBtn}
                            onClick={() => handleBook(slot.id)}
                            disabled={bookingSlotId === slot.id}
                          >
                            {bookingSlotId === slot.id ? "Booking…" : "Book"}
                          </button>
                        ) : null}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}

      {!loading && slotData && slotData.total > 0 && (
        <div className={`${styles.card} reveal`} ref={revealRef}>
          <Pagination page={page} pageSize={PAGE_SIZE} total={slotData.total} onPageChange={handlePageChange} />
        </div>
      )}
    </div>
  );
}
