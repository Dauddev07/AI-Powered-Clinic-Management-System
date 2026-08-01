import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useNavigate } from "react-router-dom";
import {
  bookAppointment,
  fetchDepartmentsWithSlots,
  fetchDoctorsWithSlots,
  fetchSlots,
} from "../../api/patientBooking";
import { ApiError } from "../../api/client";
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

export default function BookAppointment() {
  const navigate = useNavigate();
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
  const [doctorMenuOpen, setDoctorMenuOpen] = useState(false);
  const [doctorQuery, setDoctorQuery] = useState("");
  const [doctorPanelStyle, setDoctorPanelStyle] = useState(null);
  // The trigger button's own ref (for position measurement + outside-click detection)
  // and a separate ref for the portaled panel — the panel is rendered into
  // document.body (see below) rather than as a normal DOM child of the trigger, so a
  // later sibling card can never visually cover it regardless of that card's own
  // stacking context (e.g. the "reveal" fade-in animation on each date's card).
  const doctorTriggerRef = useRef(null);
  const doctorPanelRef = useRef(null);
  const doctorSearchInputRef = useRef(null);
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
    fetchDepartmentsWithSlots()
      .then(setDepartments)
      .catch(() => {});
    loadSlots({ departmentId: "", doctorId: "", dateFilter: "", customDate: "", timeOfDay: "", page: 1 });
    loadDoctorOptions("", "", "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleDepartmentChange = (e) => {
    const value = e.target.value;
    setDepartmentId(value);
    setDoctorId("");
    setPage(1);
    loadSlots({ departmentId: value, doctorId: "", dateFilter, customDate, timeOfDay, page: 1 });
    loadDoctorOptions(value, dateFilter, customDate);
  };

  const selectDoctor = (value) => {
    setDoctorId(value);
    setDoctorQuery("");
    setDoctorMenuOpen(false);
    setPage(1);
    loadSlots({ departmentId, doctorId: value, dateFilter, customDate, timeOfDay, page: 1 });
  };

  const openDoctorMenu = () => {
    const rect = doctorTriggerRef.current?.getBoundingClientRect();
    if (rect) {
      // Panel is portaled to <body> and positioned fixed at these viewport
      // coordinates, so it's never confined to (or hidden behind) this card's
      // own stacking context.
      setDoctorPanelStyle({ top: rect.bottom + 4, left: rect.left, width: rect.width });
    }
    setDoctorMenuOpen(true);
    // Deferred so it runs after the panel (and its input) has actually mounted.
    setTimeout(() => doctorSearchInputRef.current?.focus(), 0);
  };

  const closeDoctorMenu = () => {
    setDoctorMenuOpen(false);
    setDoctorQuery("");
  };

  const handleDateFilterChange = (e) => {
    const value = e.target.value;
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

  const handleTimeOfDayChange = (e) => {
    const value = e.target.value;
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

  // Closes the doctor dropdown on an outside click/tap, Escape, or a scroll/resize
  // (rather than re-tracking position through those), without changing the applied
  // filter — only committed by actually picking an option below. The panel is
  // portaled to <body>, so "outside" means outside both the trigger AND the panel.
  useEffect(() => {
    if (!doctorMenuOpen) return;
    const handlePointerDown = (e) => {
      const inTrigger = doctorTriggerRef.current && doctorTriggerRef.current.contains(e.target);
      const inPanel = doctorPanelRef.current && doctorPanelRef.current.contains(e.target);
      if (!inTrigger && !inPanel) closeDoctorMenu();
    };
    const handleKeyDown = (e) => {
      if (e.key === "Escape") closeDoctorMenu();
    };
    // `scroll` doesn't bubble, but a capture-phase listener on window still sees it
    // for every scrollable descendant, including the doctor list itself (it has its
    // own overflow-y: auto) — so scrolling *inside* the list must be ignored here,
    // or every scroll to reach a lower doctor would immediately close the panel.
    const handleScroll = (e) => {
      if (doctorPanelRef.current && doctorPanelRef.current.contains(e.target)) return;
      closeDoctorMenu();
    };
    const handleResize = () => closeDoctorMenu();
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
  }, [doctorMenuOpen]);

  // The search bar inside the open dropdown narrows this same doctor list by name as
  // the patient types — it never fetches or filters slots on its own; the actual slot
  // filter stays entirely on `doctorId`, set only when an option below is picked.
  const filteredDoctorOptions = useMemo(() => {
    const query = doctorQuery.trim().toLowerCase();
    if (!query) return doctorOptions;
    return doctorOptions.filter((d) => d.name.toLowerCase().includes(query));
  }, [doctorOptions, doctorQuery]);

  const selectedDoctorLabel = doctorId
    ? doctorOptions.find((d) => d.id === doctorId)?.name ?? "All doctors"
    : "All doctors";

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

      <div className={`${styles.card} reveal`} ref={revealRef}>
        <div className={styles.filterRow}>
          <div className={styles.filterField}>
            <label htmlFor="department">Department</label>
            <select
              id="department"
              className={styles.select}
              value={departmentId}
              onChange={handleDepartmentChange}
            >
              <option value="">All departments</option>
              {departments.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          </div>

          <div className={styles.filterField}>
            <label htmlFor="doctor">Doctor</label>
            <div className={styles.combobox}>
              <button
                type="button"
                id="doctor"
                ref={doctorTriggerRef}
                className={styles.comboboxTrigger}
                onClick={() => (doctorMenuOpen ? closeDoctorMenu() : openDoctorMenu())}
                aria-haspopup="listbox"
                aria-expanded={doctorMenuOpen}
              >
                <span className={styles.comboboxTriggerText}>{selectedDoctorLabel}</span>
                <span className={styles.comboboxChevron} aria-hidden="true">
                  ▾
                </span>
              </button>

              {doctorMenuOpen &&
                doctorPanelStyle &&
                createPortal(
                  <div
                    ref={doctorPanelRef}
                    className={styles.comboboxPanel}
                    style={{
                      top: doctorPanelStyle.top,
                      left: doctorPanelStyle.left,
                      width: doctorPanelStyle.width,
                    }}
                  >
                    <input
                      ref={doctorSearchInputRef}
                      type="text"
                      className={styles.comboboxSearchInput}
                      placeholder="Search doctor by name…"
                      value={doctorQuery}
                      onChange={(e) => setDoctorQuery(e.target.value)}
                      aria-label="Search doctors"
                    />
                    <ul className={styles.comboboxList} role="listbox" aria-label="Doctors">
                      <li
                        role="option"
                        aria-selected={doctorId === ""}
                        className={doctorId === "" ? styles.comboboxOptionSelected : styles.comboboxOption}
                        onClick={() => selectDoctor("")}
                      >
                        All doctors
                      </li>
                      {filteredDoctorOptions.map((d) => (
                        <li
                          key={d.id}
                          role="option"
                          aria-selected={doctorId === d.id}
                          className={doctorId === d.id ? styles.comboboxOptionSelected : styles.comboboxOption}
                          onClick={() => selectDoctor(d.id)}
                        >
                          {d.name}
                        </li>
                      ))}
                      {filteredDoctorOptions.length === 0 && (
                        <li className={styles.comboboxEmpty}>No doctors match "{doctorQuery}"</li>
                      )}
                    </ul>
                  </div>,
                  document.body,
                )}
            </div>
          </div>

          <div className={styles.filterField}>
            <label htmlFor="dateFilter">Date</label>
            <select
              id="dateFilter"
              className={styles.select}
              value={dateFilter}
              onChange={handleDateFilterChange}
            >
              {DATE_FILTERS.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </select>
            {dateFilter === "custom" && (
              <input
                type="date"
                className={styles.select}
                value={customDate}
                onChange={handleCustomDateChange}
              />
            )}
          </div>

          <div className={styles.filterField}>
            <label htmlFor="timeOfDay">Time of day</label>
            <select
              id="timeOfDay"
              className={styles.select}
              value={timeOfDay}
              onChange={handleTimeOfDayChange}
            >
              {TIME_OF_DAY_FILTERS.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </select>
          </div>
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
                      <td data-label="Doctor">{slot.doctor_name}</td>
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
