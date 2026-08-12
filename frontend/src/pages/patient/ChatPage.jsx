import { useEffect, useRef, useState } from "react";
import { fetchMyProfile } from "../../api/auth";
import {
  deleteChatSession,
  fetchChatHistory,
  fetchChatSessions,
  fetchPendingFeedback,
  sendChatMessage,
  submitFeedback,
} from "../../api/chat";
import { ApiError } from "../../api/client";
import DoctorRatingBadge from "../../components/DoctorRatingBadge";
import Modal from "../../components/Modal";
import StarIcon from "../../components/StarIcon";
import styles from "./ChatPage.module.css";

const SESSION_STORAGE_KEY = "chat_session_id";
const SESSION_TITLE_MAX_LENGTH = 60;

// The booking/reschedule tools (task 6.2.4) prefix a confirmed-booking reply with
// this sentinel followed by a JSON payload — see app/services/chat_tools.py.
const BOOKING_MARKER = "BOOKING_CONFIRMED::";

// Symptom triage's doctor-matching step and the get_department_availability tool
// (task 6.2.3/6.2.4) prefix their reply with this sentinel followed by a JSON
// payload of { note, department_name, doctors: [{ doctor_id, doctor_name,
// specialization, slots: [{ slot_id, when }] }] } — see app/services/chat_tools.py.
const DOCTOR_OPTIONS_MARKER = "DOCTOR_OPTIONS::";

// A genuine cross-department question ("list every doctor across every department")
// calls get_department_availability more than once in one turn — the backend
// combines every call's real result into this marker's payload in code
// ({ departments: [{ department_name, note, doctors: [...] }, ...], unavailable: [{
// department_name, message }, ...] }), never summarized by the LLM itself. `note`
// is the same one-sentence triage reasoning DOCTOR_OPTIONS_MARKER shows for a
// single department — present when that department was reached by symptom
// inference, omitted when the patient named it directly.
// `unavailable` covers a real department that WAS checked but has nobody free right
// now — previously silently dropped, so a reply could name a department (e.g. "you
// may see Dermatology") with no card and no explanation for it. See
// app.services.chat_tools.combine_department_availability_results.
const DEPARTMENT_LIST_MARKER = "DEPARTMENT_LIST::";

// appointment_agent's own disambiguation question — several real candidates match
// what the patient typed (a doctor-name match, or a vague reference to more than one
// of the patient's own appointments) — payload { kind, question, candidates: [{
// doctor_name, department_name }, ...] }. See app/services/chat_markers.py and
// app/services/orchestrator/agents/appointment_agent.py.
const DOCTOR_DISAMBIGUATION_MARKER = "DOCTOR_DISAMBIGUATION::";

const SUGGESTIONS = [
  "What are your clinic's opening hours?",
  "I have a headache and mild fever — what should I do?",
  "How do I book an appointment?",
];

// A slot-pick sends its raw text — including "(slot_id: <uuid>)" — as the actual
// message the backend/LLM sees and saves to history (see selectSlot below: the
// model needs the literal id to know exactly which slot was chosen). The friendly
// "Book with Dr. X at ..." wording only exists for the instant the optimistic bubble
// is pushed locally — once history reloads from the server, it renders straight from
// that raw saved text. Stripped here at display time only, so the UUID never shows
// up either way, without changing what's actually sent or stored.
const SLOT_ID_SUFFIX_RE = /\s*\(slot_id:\s*[0-9a-f-]{36}\)\.?\s*$/i;

function stripSlotId(content) {
  return content.replace(SLOT_ID_SUFFIX_RE, ".");
}

function parseBookingConfirmation(content) {
  if (!content.startsWith(BOOKING_MARKER)) return null;
  try {
    return JSON.parse(content.slice(BOOKING_MARKER.length));
  } catch {
    return null;
  }
}

function parseDoctorOptions(content) {
  if (!content.startsWith(DOCTOR_OPTIONS_MARKER)) return null;
  try {
    return JSON.parse(content.slice(DOCTOR_OPTIONS_MARKER.length));
  } catch {
    return null;
  }
}

function parseDepartmentList(content) {
  if (!content.startsWith(DEPARTMENT_LIST_MARKER)) return null;
  try {
    return JSON.parse(content.slice(DEPARTMENT_LIST_MARKER.length));
  } catch {
    return null;
  }
}

function parseDoctorDisambiguation(content) {
  if (!content.startsWith(DOCTOR_DISAMBIGUATION_MARKER)) return null;
  try {
    return JSON.parse(content.slice(DOCTOR_DISAMBIGUATION_MARKER.length));
  } catch {
    return null;
  }
}

// Mirrors the backend's own title derivation (first user message, same length cap) so
// a freshly-started thread's sidebar entry never has to wait on a round trip to read
// correctly — see app/services/chat.py::_session_title.
function titleFromMessage(text) {
  const trimmed = text.trim();
  if (trimmed.length <= SESSION_TITLE_MAX_LENGTH) return trimmed;
  return `${trimmed.slice(0, SESSION_TITLE_MAX_LENGTH - 1).trimEnd()}…`;
}

function formatSessionTime(iso) {
  const date = new Date(iso);
  const diffMin = Math.floor((Date.now() - date.getTime()) / 60000);
  if (diffMin < 1) return "Just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay === 1) return "Yesterday";
  if (diffDay < 7) return `${diffDay}d ago`;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

// Small clinic-appropriate avatar (a stethoscope glyph) that sits next to every
// assistant bubble so a reply reads as coming from "the clinic assistant," not as an
// unstyled floating block of text.
function AssistantAvatar() {
  return (
    <span className={styles.assistantAvatar} aria-hidden="true">
      <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M5 3v5a3 3 0 0 0 6 0V3" />
        <path d="M8 11v1a5 5 0 0 0 10 0V9" />
        <circle cx="18" cy="7" r="2" />
      </svg>
    </span>
  );
}

function TypingIndicator() {
  return (
    <div className={`${styles.bubbleRow} ${styles.assistantRow}`}>
      <AssistantAvatar />
      <div className={`${styles.bubble} ${styles.assistantBubble} ${styles.typingBubble}`} aria-label="Assistant is typing">
        <span className={styles.typingDot} />
        <span className={styles.typingDot} />
        <span className={styles.typingDot} />
      </div>
    </div>
  );
}

function BookingConfirmationCard({ booking }) {
  return (
    <div className={styles.bookingCard}>
      <div className={styles.bookingCardHeader}>
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M20 6 9 17l-5-5" />
        </svg>
        Appointment confirmed
      </div>
      {booking.doctor_name && <div className={styles.bookingCardRow}>{booking.doctor_name}</div>}
      {booking.department_name && <div className={styles.bookingCardRow}>{booking.department_name}</div>}
      {booking.when && <div className={styles.bookingCardRow}>{booking.when}</div>}
    </div>
  );
}

function DoctorOptionsCard({ options, onSelectSlot, disabled }) {
  return (
    <div className={styles.doctorOptionsCard}>
      {options.note && <div className={styles.doctorOptionsNote}>{options.note}</div>}
      {options.department_name && (
        <div className={styles.doctorOptionsHeader}>{options.department_name}</div>
      )}
      {options.doctors.map((doctor) => (
        <DoctorGroup key={doctor.doctor_id} doctor={doctor} onSelectSlot={onSelectSlot} disabled={disabled} />
      ))}
    </div>
  );
}

function DoctorGroup({ doctor, onSelectSlot, disabled }) {
  return (
    <div className={styles.doctorOptionGroup}>
      <div className={styles.doctorOptionNameRow}>
        <div className={styles.doctorOptionName}>
          {doctor.doctor_name}
          {doctor.specialization ? ` — ${doctor.specialization}` : ""}
        </div>
        <DoctorRatingBadge averageRating={doctor.average_rating} ratingCount={doctor.rating_count} />
      </div>
      <div className={styles.doctorOptionSlots}>
        {doctor.slots.map((slot) => (
          <button
            key={slot.slot_id}
            type="button"
            className={styles.slotOptionBtn}
            disabled={disabled}
            onClick={() => onSelectSlot(doctor.doctor_name, slot.when, slot.slot_id)}
          >
            {slot.when}
          </button>
        ))}
      </div>
    </div>
  );
}

function DepartmentListCard({ list, onSelectSlot, disabled }) {
  return (
    <div className={styles.departmentListCard}>
      {list.departments.map((department) => (
        <div key={department.department_name} className={styles.departmentListSection}>
          {department.note && <div className={styles.doctorOptionsNote}>{department.note}</div>}
          <div className={styles.doctorOptionsHeader}>{department.department_name}</div>
          {department.doctors.map((doctor) => (
            <DoctorGroup key={doctor.doctor_id} doctor={doctor} onSelectSlot={onSelectSlot} disabled={disabled} />
          ))}
        </div>
      ))}
      {(list.unavailable || []).map((entry) => (
        <div key={entry.department_name} className={styles.departmentListSection}>
          <div className={styles.doctorOptionsHeader}>{entry.department_name}</div>
          <div className={styles.doctorOptionsNote}>{entry.message}</div>
        </div>
      ))}
    </div>
  );
}

function DoctorDisambiguationCard({ disambiguation, onSelectCandidate, disabled }) {
  const candidates = disambiguation.candidates || [];
  // Two appointments with the SAME doctor render identical "Dr. X — Dept" buttons
  // otherwise — indistinguishable to tap, and the backend can only tell them apart
  // by date/time anyway (see _match_candidate) — so show the time instead here.
  const sameDoctor =
    disambiguation.kind === "appointment" &&
    candidates.length > 1 &&
    new Set(candidates.map((c) => c.doctor_name)).size === 1;
  return (
    <div className={styles.doctorOptionsCard}>
      {disambiguation.question && <div className={styles.doctorOptionsNote}>{disambiguation.question}</div>}
      <div className={styles.doctorOptionSlots}>
        {candidates.map((candidate) => (
          <button
            key={`${candidate.doctor_name}-${candidate.department_name}-${candidate.when || ""}`}
            type="button"
            className={styles.slotOptionBtn}
            disabled={disabled}
            onClick={() => onSelectCandidate(candidate.doctor_name, candidate.department_name, candidate.when)}
          >
            {sameDoctor
              ? candidate.when
              : `${candidate.doctor_name}${candidate.department_name ? ` — ${candidate.department_name}` : ""}`}
          </button>
        ))}
      </div>
    </div>
  );
}

const RATING_LABELS = { 1: "Poor", 2: "Not great", 3: "Okay", 4: "Good", 5: "Excellent" };

function StarRating({ rating, onRate, disabled }) {
  const [hovered, setHovered] = useState(0);
  const displayed = hovered || rating;

  return (
    <div className={styles.starBlock}>
      <div
        className={styles.starRow}
        role="radiogroup"
        aria-label="Rate your experience"
        onMouseLeave={() => setHovered(0)}
      >
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            type="button"
            className={styles.starBtn}
            aria-label={`${n} star${n > 1 ? "s" : ""}`}
            aria-pressed={rating >= n}
            disabled={disabled}
            onMouseEnter={() => setHovered(n)}
            onFocus={() => setHovered(n)}
            onClick={() => onRate(n)}
          >
            <StarIcon filled={displayed >= n} size={30} />
          </button>
        ))}
      </div>
      <span className={styles.starRatingLabel} aria-live="polite">
        {displayed ? RATING_LABELS[displayed] : "Tap a star to rate"}
      </span>
    </div>
  );
}

// Rendered as a synthetic first bubble (never part of the real conversation transcript
// / never saved to conversation memory) whenever the patient has completed
// appointment(s) they haven't rated yet — fetched via GET /chat/pending-feedback and
// shown before anything else happens on the chat screen. Entirely self-contained:
// rating >= 3 submits immediately with a thank-you/neutral acknowledgement, rating <= 2
// asks for a reason first, then submits and shows an acknowledgement that it was passed
// along to the clinic.
function FeedbackPromptCard({ prompt, appointmentIds }) {
  const [rating, setRating] = useState(0);
  const [stage, setStage] = useState("rating"); // rating -> reason -> done -> skipped
  const [reason, setReason] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [ackMessage, setAckMessage] = useState(null);
  const [error, setError] = useState(null);

  const submit = (finalRating, finalReason) => {
    setSubmitting(true);
    setError(null);
    submitFeedback(appointmentIds, finalRating, finalReason)
      .then((res) => {
        setAckMessage(res.message);
        setStage("done");
      })
      .catch((err) => {
        setError(err instanceof ApiError ? err.detail || err.message : "Could not submit feedback.");
      })
      .finally(() => setSubmitting(false));
  };

  const handleRate = (n) => {
    setRating(n);
    if (n <= 2) {
      setStage("reason");
      return;
    }
    submit(n, "");
  };

  const finished = stage === "done" || stage === "skipped";

  return (
    <div className={`${styles.bubbleRow} ${styles.assistantRow}`}>
      <AssistantAvatar />
      <div className={`${styles.feedbackCard} ${finished ? styles.feedbackCardDone : ""}`}>
        {stage === "rating" && (
          <div className={styles.feedbackRatingRow}>
            <div className={styles.feedbackTextCol}>
              <div className={styles.feedbackHeader}>
                <span className={styles.feedbackHeaderIcon} aria-hidden="true">
                  <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2.5l2.9 6.06 6.6.86-4.85 4.63 1.24 6.6L12 17.6l-5.89 3.05 1.24-6.6-4.85-4.63 6.6-.86L12 2.5Z" />
                  </svg>
                </span>
                <span className={styles.feedbackHeaderTitle}>Quick feedback</span>
              </div>
              <p className={styles.feedbackPrompt}>{prompt}</p>
              <button type="button" className={styles.feedbackSkipBtn} onClick={() => setStage("skipped")}>
                Ask me later
              </button>
            </div>
            <div className={styles.feedbackActionCol}>
              <StarRating rating={rating} onRate={handleRate} disabled={submitting} />
            </div>
          </div>
        )}

        {stage === "reason" && (
          <div className={styles.feedbackReasonBlock}>
            <div className={styles.feedbackHeader}>
              <span className={styles.feedbackHeaderIcon} aria-hidden="true">
                <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2.5l2.9 6.06 6.6.86-4.85 4.63 1.24 6.6L12 17.6l-5.89 3.05 1.24-6.6-4.85-4.63 6.6-.86L12 2.5Z" />
                </svg>
              </span>
              <span className={styles.feedbackHeaderTitle}>Quick feedback</span>
            </div>
            <p className={styles.feedbackPrompt}>{prompt}</p>
            <StarRating rating={rating} onRate={setRating} disabled={submitting} />
            <label className={styles.feedbackReasonLabel} htmlFor="feedback-reason">
              Sorry to hear that — what went wrong?
            </label>
            <textarea
              id="feedback-reason"
              className={styles.feedbackReasonInput}
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (!submitting) submit(rating, reason.trim());
                }
              }}
              placeholder="Tell us what happened… (Enter to submit)"
              rows={2}
              disabled={submitting}
            />
            <button
              type="button"
              className={styles.feedbackSubmitBtn}
              disabled={submitting}
              onClick={() => submit(rating, reason.trim())}
            >
              {submitting ? "Sending…" : "Submit"}
            </button>
          </div>
        )}

        {finished && (
          <div className={styles.feedbackAckRow}>
            <span className={styles.feedbackAckIcon} aria-hidden="true">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20 6 9 17l-5-5" />
              </svg>
            </span>
            <p className={styles.feedbackAck}>
              {stage === "done" ? ackMessage : "No problem — we'll ask again next time."}
            </p>
          </div>
        )}
        {error && <p className={styles.historyError}>{error}</p>}
      </div>
    </div>
  );
}

// Browser-local time-of-day — chat is about when a message was sent from the
// viewer's own perspective (like any messaging app), unlike appointment times
// elsewhere in the app which always render against the clinic's timezone instead.
function formatMessageTime(iso) {
  if (!iso) return null;
  return new Date(iso).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

function ChatMessage({ message, onSelectSlot, onSelectCandidate, disabled, grouped }) {
  const isUser = message.role === "user";
  const booking = !isUser ? parseBookingConfirmation(message.content) : null;
  const doctorOptions = !isUser && !booking ? parseDoctorOptions(message.content) : null;
  const departmentList = !isUser && !booking && !doctorOptions ? parseDepartmentList(message.content) : null;
  const doctorDisambiguation =
    !isUser && !booking && !doctorOptions && !departmentList ? parseDoctorDisambiguation(message.content) : null;
  const time = formatMessageTime(message.createdAt);

  if (message.redFlag) {
    return (
      <div className={`${styles.bubbleRow} ${styles.assistantRow}`}>
        {grouped ? <span className={styles.avatarSpacer} aria-hidden="true" /> : <AssistantAvatar />}
        <div className={styles.redFlagBubble} role="alert">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M12 9v4M12 17h.01M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" />
          </svg>
          <span className={styles.redFlagText}>{message.content}</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`${styles.bubbleRow} ${isUser ? styles.userRow : styles.assistantRow} ${grouped ? styles.groupedRow : ""}`}>
      {!isUser && (grouped ? <span className={styles.avatarSpacer} aria-hidden="true" /> : <AssistantAvatar />)}
      <div className={styles.bubbleCol}>
        <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.assistantBubble} ${message.error ? styles.errorBubble : ""}`}>
          {booking ? (
            <BookingConfirmationCard booking={booking} />
          ) : doctorOptions ? (
            <DoctorOptionsCard options={doctorOptions} onSelectSlot={onSelectSlot} disabled={disabled} />
          ) : departmentList ? (
            <DepartmentListCard list={departmentList} onSelectSlot={onSelectSlot} disabled={disabled} />
          ) : doctorDisambiguation ? (
            <DoctorDisambiguationCard
              disambiguation={doctorDisambiguation}
              onSelectCandidate={onSelectCandidate}
              disabled={disabled}
            />
          ) : isUser ? (
            stripSlotId(message.content)
          ) : (
            message.content
          )}
        </div>
        {time && <span className={styles.messageTime}>{time}</span>}
      </div>
    </div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [firstName, setFirstName] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [sending, setSending] = useState(false);
  // Which thread (by viewedThreadTokenRef value, see below) the in-flight send
  // belongs to — lets the typing indicator show only on the thread that's actually
  // waiting on a reply, not on whatever thread the patient has since switched to.
  const [pendingThreadToken, setPendingThreadToken] = useState(null);
  const [historyError, setHistoryError] = useState(null);
  const [pendingFeedback, setPendingFeedback] = useState(null);
  const [openMenuSessionId, setOpenMenuSessionId] = useState(null);
  const [pendingDelete, setPendingDelete] = useState(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  // Reported live: a slot/doctor card stays clickable in the transcript even
  // after it's already been acted on — tapping an earlier "Book with Dr. X"
  // slot again (or an earlier disambiguation candidate) re-sends that same
  // stale selection. Indices (into the current `messages` array) of cards
  // that have had a selection made from them — once a message's index is in
  // here, its card is permanently disabled for the rest of this session view,
  // not just while `sending` is true. Cleared everywhere `messages` itself is
  // replaced wholesale (initial load, switching sessions, New Chat) so a
  // stale index from a previous thread can never disable an unrelated card.
  const [usedCardIndices, setUsedCardIndices] = useState(() => new Set());
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const topRef = useRef(null);
  const initialScrollDoneRef = useRef(false);
  const textareaRef = useRef(null);
  // Speech-to-text dictation for the message box — browser-native (Web Speech API),
  // no backend involvement at all. Only Chrome/Edge/Safari ship a working
  // implementation as of this writing (Firefox doesn't), so the mic button is
  // simply never rendered when the constructor isn't present rather than shown
  // disabled with an explanation nobody asked for.
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef(null);
  const speechRecognitionSupported =
    typeof window !== "undefined" && !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  // Bumped every time the visibly-displayed thread changes (New Chat, switching
  // sessions in the sidebar) so an in-flight sendChatMessage from a thread the
  // patient has since navigated away from can't paint its reply into whatever
  // thread happens to be on screen when the response lands, or wipe it out via a
  // stale setMessages. The reply itself is never lost — the backend already
  // persisted it under its own session_id regardless — this only guards which
  // thread's message list the frontend updates.
  const viewedThreadTokenRef = useRef(0);

  // Closes an open per-session "…" menu on an outside click — each menu's trigger and
  // popover share a `data-session-menu` wrapper, so anything outside every such
  // wrapper counts as "outside" regardless of which session's menu is open.
  useEffect(() => {
    if (!openMenuSessionId) return undefined;
    const handlePointerDown = (e) => {
      if (!e.target.closest("[data-session-menu]")) setOpenMenuSessionId(null);
    };
    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, [openMenuSessionId]);

  // Same source as the patient dashboard's "Welcome, {name}" heading (see
  // PatientDashboard.jsx) — kept independent of the sessions/history load below so a
  // slow profile fetch never blocks the chat itself from becoming usable.
  // Defensive baseline for whatever scroll position the browser happened to land on
  // navigating in (e.g. a mobile browser restoring/adjusting scroll while
  // --app-header-height corrects itself right after mount) — reset the page's own
  // scroll immediately, before anything else below decides where the chat content
  // itself should land.
  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, []);

  useEffect(() => {
    fetchMyProfile()
      .then((data) => setFirstName((data.full_name || "").trim().split(/\s+/)[0] || null))
      .catch(() => {});
  }, []);

  useEffect(() => {
    const storedSessionId = localStorage.getItem(SESSION_STORAGE_KEY);
    Promise.all([fetchChatSessions(), fetchChatHistory(storedSessionId)])
      .then(([sessionList, history]) => {
        setSessions(sessionList);
        setActiveSessionId(history.session_id);
        if (history.session_id) localStorage.setItem(SESSION_STORAGE_KEY, history.session_id);
        setMessages(
          history.messages.map((m) => ({
            role: m.role,
            content: m.content,
            createdAt: m.created_at,
            redFlag: m.red_flag,
          })),
        );
        setUsedCardIndices(new Set());
      })
      .catch((err) => {
        setHistoryError(err instanceof ApiError ? err.detail || err.message : "Could not load chat history.");
      })
      .finally(() => setLoadingHistory(false));
  }, []);

  // Fetched once per mount (once per chat-screen visit, not per message) — "before
  // anything" a completed, unrated appointment gets asked about, shown as a synthetic
  // first bubble rather than injected into the real message history. A failed fetch
  // just means no prompt shows, never blocks the rest of the chat screen.
  useEffect(() => {
    fetchPendingFeedback()
      .then(setPendingFeedback)
      .catch(() => setPendingFeedback({ appointments: [], prompt: null }));
  }, []);

  // First settle after both history and the pending-feedback prompt have loaded:
  // if there's a feedback prompt to answer, land at the very TOP of the thread (it
  // renders above all history, so a normal end-of-thread landing would leave it
  // scrolled out of view above everything else) rather than the usual bottom.
  // Every scroll after that first one — a new message arriving, sending state
  // changing — goes back to the normal bottom-follow behavior.
  useEffect(() => {
    if (loadingHistory || pendingFeedback === null) return;
    if (!initialScrollDoneRef.current) {
      initialScrollDoneRef.current = true;
      if (pendingFeedback.appointments.length > 0) {
        // Setting scrollTop directly (rather than topRef.scrollIntoView) is what
        // actually lands reliably at the very top — scrollIntoView only guarantees
        // the target becomes visible, which a browser can satisfy by leaving
        // scrollTop wherever it already was if the target's already on-screen, and
        // it only affects the nearest scrollable ancestor, not the page itself if a
        // mobile browser's own chrome has scrolled the viewport. Two nested rAFs
        // wait for the feedback card's real height to be in the layout (it mounts
        // in the same tick as this effect, so scrolling on the same frame can race
        // ahead of it) before both targets are forced to 0 explicitly.
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            messagesContainerRef.current?.scrollTo({ top: 0, behavior: "auto" });
            window.scrollTo({ top: 0, behavior: "auto" });
          });
        });
        return;
      }
    }
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, sending, pendingFeedback, loadingHistory]);

  // The textarea is disabled (and so unfocusable) while history is still loading —
  // once it's ready, focus it automatically so a patient can start typing straight
  // away on first load too, without having to click into the box first.
  useEffect(() => {
    if (!loadingHistory) requestAnimationFrame(() => textareaRef.current?.focus());
  }, [loadingHistory]);

  // Re-focus once a send completes (reply arrived or it errored) so the patient can
  // keep typing straight away without tapping back into the box. This is a real
  // effect (not called inline from the sendChatMessage promise's .finally()) so it's
  // guaranteed to run only after React has actually committed `disabled` clearing on
  // the textarea — calling .focus() from the promise callback directly raced ahead of
  // that commit on some devices, silently no-opping on a still-disabled element and
  // leaving the patient to tap the box manually.
  useEffect(() => {
    if (!sending) requestAnimationFrame(() => textareaRef.current?.focus());
  }, [sending]);

  const resizeTextarea = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  };

  // Toggling the mic: starts dictation into the SAME input box the patient would
  // otherwise type into, so nothing downstream (submitMessage, the slot-id
  // stripping, etc.) needs to know text arrived via speech rather than typing.
  // interimResults is on so the box fills in live as the patient talks, rather
  // than staying blank until they stop — closer to how dictation feels on a phone
  // keyboard. Stops itself on a natural pause (continuous: false) so there's no
  // need to also detect silence manually.
  const toggleListening = () => {
    if (!speechRecognitionSupported || sending) return;

    if (listening) {
      recognitionRef.current?.stop();
      return;
    }

    const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognitionCtor();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = false;

    recognition.onresult = (event) => {
      let transcript = "";
      for (let i = 0; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      setInput(transcript);
      requestAnimationFrame(resizeTextarea);
    };
    recognition.onerror = () => {
      recognitionRef.current = null;
      setListening(false);
    };
    recognition.onend = () => {
      recognitionRef.current = null;
      setListening(false);
    };

    recognitionRef.current = recognition;
    setListening(true);
    recognition.start();
  };

  // Reported live: calling recognition.stop() alone wasn't enough — Chrome sends
  // the audio to its own server for the FINAL transcription, and that response can
  // arrive a moment AFTER stop() returns, so onresult still fired once more,
  // refilling the input right after it had just been cleared on send. Detaching
  // the handlers (not just stopping) means a late-arriving result has nothing left
  // to call into, regardless of how long the network round-trip takes.
  const silenceListening = () => {
    const recognition = recognitionRef.current;
    if (!recognition) return;
    recognition.onresult = null;
    recognition.onerror = null;
    recognition.onend = null;
    try {
      recognition.stop();
    } catch {
      // Already stopped/ended on its own — nothing left to do.
    }
    recognitionRef.current = null;
    setListening(false);
  };

  // Stops any in-progress dictation if the patient navigates away mid-recording
  // (e.g. switches sessions) — a recognition session left running with no visible
  // mic indicator would otherwise keep listening in the background.
  useEffect(() => {
    return () => silenceListening();
  }, []);

  const selectSession = (sessionId) => {
    setSidebarOpen(false);
    if (sessionId === activeSessionId) return;

    viewedThreadTokenRef.current += 1;
    setLoadingHistory(true);
    setHistoryError(null);
    fetchChatHistory(sessionId)
      .then((data) => {
        setActiveSessionId(data.session_id);
        if (data.session_id) localStorage.setItem(SESSION_STORAGE_KEY, data.session_id);
        setMessages(
          data.messages.map((m) => ({
            role: m.role,
            content: m.content,
            createdAt: m.created_at,
            redFlag: m.red_flag,
          })),
        );
        setUsedCardIndices(new Set());
      })
      .catch((err) => {
        setHistoryError(err instanceof ApiError ? err.detail || err.message : "Could not load that conversation.");
      })
      .finally(() => setLoadingHistory(false));
  };

  // Starts a fresh, empty thread in the sidebar. The assistant doesn't actually lose
  // any context by doing this — see app/services/chat.py's module docstring: chat
  // memory is loaded across every one of the patient's sessions, not just the active
  // one, so "New Chat" only resets what's shown, never what the assistant remembers.
  const startNewChat = () => {
    viewedThreadTokenRef.current += 1;
    setActiveSessionId(null);
    localStorage.removeItem(SESSION_STORAGE_KEY);
    setMessages([]);
    setUsedCardIndices(new Set());
    setHistoryError(null);
    setInput("");
    setSidebarOpen(false);
    requestAnimationFrame(() => textareaRef.current?.focus());
  };

  const confirmDeleteSession = () => {
    if (!pendingDelete) return;
    setDeleteBusy(true);
    deleteChatSession(pendingDelete.session_id)
      .then(() => {
        setSessions((prev) => prev.filter((s) => s.session_id !== pendingDelete.session_id));
        if (pendingDelete.session_id === activeSessionId) startNewChat();
        setPendingDelete(null);
      })
      .catch((err) => {
        setHistoryError(err instanceof ApiError ? err.detail || err.message : "Could not delete that conversation.");
        setPendingDelete(null);
      })
      .finally(() => setDeleteBusy(false));
  };

  // Shared by both the free-text input and the doctor/slot option cards: a card's
  // "pick this slot" button sends its own plain-text message (naming the slot_id)
  // through the exact same chat turn as anything the patient types, so the backend
  // agent has one single message channel to reason over — see
  // app/services/llm.py's tool-use rules on only calling book_appointment once a
  // slot_id has been clearly referenced in the conversation.
  const submitMessage = (text, displayText) => {
    if (!text || sending) return;

    const threadToken = viewedThreadTokenRef.current;
    setMessages((prev) => [...prev, { role: "user", content: displayText ?? text, createdAt: new Date().toISOString() }]);
    setSending(true);
    setPendingThreadToken(threadToken);

    sendChatMessage(text, activeSessionId)
      .then((res) => {
        // Only paint the reply (and switch the active session/localStorage to it)
        // into the currently-displayed thread if the patient hasn't since
        // navigated away from it (New Chat / picked a different session) — see
        // viewedThreadTokenRef's comment. The sidebar entry below still updates
        // either way, so the reply's own thread reflects it whenever it's opened.
        const stillViewingThisThread = threadToken === viewedThreadTokenRef.current;
        if (stillViewingThisThread) {
          localStorage.setItem(SESSION_STORAGE_KEY, res.session_id);
          setActiveSessionId(res.session_id);
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: res.reply, redFlag: res.red_flag, createdAt: new Date().toISOString() },
          ]);
        }

        setSessions((prev) => {
          const nowIso = new Date().toISOString();
          const idx = prev.findIndex((s) => s.session_id === res.session_id);
          if (idx === -1) {
            return [{ session_id: res.session_id, title: titleFromMessage(displayText ?? text), last_message_at: nowIso }, ...prev];
          }
          const updated = [...prev];
          const [existing] = updated.splice(idx, 1);
          return [{ ...existing, last_message_at: nowIso }, ...updated];
        });
      })
      .catch((err) => {
        if (threadToken !== viewedThreadTokenRef.current) return;
        const detail = err instanceof ApiError ? err.detail || err.message : null;
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: detail || "Something went wrong. Please try again.",
            error: true,
            createdAt: new Date().toISOString(),
          },
        ]);
      })
      .finally(() => {
        setSending(false);
        setPendingThreadToken((cur) => (cur === threadToken ? null : cur));
      });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || sending) return;
    // See silenceListening's own comment — a plain .stop() alone wasn't enough,
    // since a final transcription result can still arrive after it returns.
    silenceListening();
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    submitMessage(trimmed);
  };

  // A patient clicking a slot option card is an explicit selection, not free text —
  // the message still travels as plain text (see submitMessage's comment above), but
  // shows a friendlier line in the transcript than the raw slot_id the model needs.
  const selectSlot = (doctorName, when, slotId) => {
    submitMessage(
      `I'd like to book the appointment with ${doctorName} at ${when} (slot_id: ${slotId}).`,
      `Book with ${doctorName} at ${when}`
    );
  };

  // A patient tapping one of appointment_agent's disambiguation candidates sends the
  // doctor's exact full name back as the message text — nothing else — so
  // find_doctors_by_name's exact-match tier resolves it directly on the next turn
  // instead of re-triggering the same disambiguation. The friendlier wording only
  // shows in the transcript (displayText), same pattern as selectSlot above. When
  // two candidates share a doctor (same-doctor reschedule/cancel case), the name
  // alone can't tell them apart on the backend either — the appointment's own
  // "when" text is included too, since that's what _match_candidate falls back to.
  const selectCandidate = (doctorName, departmentName, when) => {
    const whenSuffix = when ? ` on ${when}` : "";
    submitMessage(
      `${doctorName}${whenSuffix}`,
      `I mean ${doctorName}${departmentName ? ` (${departmentName})` : ""}${whenSuffix}.`
    );
  };

  const handleInputChange = (e) => {
    setInput(e.target.value);
    resizeTextarea();
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const showWelcome = !loadingHistory && messages.length === 0;

  const applySuggestion = (text) => {
    setInput(text);
    const el = textareaRef.current;
    if (el) {
      el.focus();
      requestAnimationFrame(resizeTextarea);
    }
  };

  return (
    <div className={styles.page}>
      <aside className={`${styles.sidebar} ${sidebarOpen ? styles.sidebarOpen : ""}`} aria-label="Chat history">
        <div className={styles.sidebarHeader}>
          <button type="button" className={styles.newChatBtn} onClick={startNewChat}>
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M12 5v14M5 12h14" />
            </svg>
            New chat
          </button>
          <button
            type="button"
            className={styles.sidebarCloseBtn}
            onClick={() => setSidebarOpen(false)}
            aria-label="Close chat history"
          >
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M6 6l12 12M18 6 6 18" />
            </svg>
          </button>
        </div>

        <nav className={styles.sessionList} aria-label="Previous conversations">
          {sessions.length === 0 && !loadingHistory && <p className={styles.sessionListEmpty}>No conversations yet</p>}
          {sessions.map((s) => (
            <div key={s.session_id} className={styles.sessionItemWrapper} data-session-menu>
              <button
                type="button"
                className={`${styles.sessionItem} ${s.session_id === activeSessionId ? styles.sessionItemActive : ""}`}
                onClick={() => selectSession(s.session_id)}
              >
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className={styles.sessionItemIcon}>
                  <path d="M4 12.5C4 7.81 8.03 4 13 4s9 3.81 9 8.5-4.03 8.5-9 8.5c-1.09 0-2.13-.19-3.1-.53L4 21l1.2-4.02A8.16 8.16 0 0 1 4 12.5Z" />
                </svg>
                <span className={styles.sessionItemText}>
                  <span className={styles.sessionItemTitle}>{s.title}</span>
                  <span className={styles.sessionItemTime}>{formatSessionTime(s.last_message_at)}</span>
                </span>
              </button>

              <button
                type="button"
                className={styles.sessionMenuBtn}
                aria-label="Chat options"
                aria-haspopup="menu"
                aria-expanded={openMenuSessionId === s.session_id}
                onClick={(e) => {
                  e.stopPropagation();
                  setOpenMenuSessionId((cur) => (cur === s.session_id ? null : s.session_id));
                }}
              >
                <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true">
                  <circle cx="12" cy="5" r="1.6" />
                  <circle cx="12" cy="12" r="1.6" />
                  <circle cx="12" cy="19" r="1.6" />
                </svg>
              </button>

              {openMenuSessionId === s.session_id && (
                <div className={styles.sessionMenuPopover} role="menu">
                  <button
                    type="button"
                    className={styles.sessionMenuDelete}
                    role="menuitem"
                    onClick={(e) => {
                      e.stopPropagation();
                      setOpenMenuSessionId(null);
                      setPendingDelete(s);
                    }}
                  >
                    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="M4 7h16M9 7V4.5A1.5 1.5 0 0 1 10.5 3h3A1.5 1.5 0 0 1 15 4.5V7M18.5 7 17.7 19a2 2 0 0 1-2 1.9H8.3a2 2 0 0 1-2-1.9L5.5 7" />
                    </svg>
                    Delete chat
                  </button>
                </div>
              )}
            </div>
          ))}
        </nav>
      </aside>

      {sidebarOpen && (
        <button type="button" className={styles.sidebarBackdrop} onClick={() => setSidebarOpen(false)} aria-label="Close chat history" />
      )}

      <div className={styles.main}>
        <div className={styles.topBar}>
          <button type="button" className={styles.sidebarToggle} onClick={() => setSidebarOpen(true)} aria-label="Show chat history">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <rect x="3" y="4" width="18" height="16" rx="2.5" />
              <line x1="9" y1="4" x2="9" y2="20" />
            </svg>
          </button>
          <span className={styles.topBarTitle}>AI Assistant</span>
          <button type="button" className={styles.topBarNewChatBtn} onClick={startNewChat} aria-label="Start new chat">
            <svg viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M12 5v14M5 12h14" />
            </svg>
          </button>
        </div>

        {/* Permanent, always-visible medical disclaimer — never dismissible. Warning-
            yellow banner, brightness dialed down a notch (see the CSS) so it stays
            noticeable without glaring against the rest of the dark theme. */}
        <div className={styles.disclaimer} role="note">
          <div className={styles.disclaimerInner}>
            <span className={styles.disclaimerIcon} aria-hidden="true">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M12 3.5 22 20H2L12 3.5Z" />
                <line x1="12" y1="10" x2="12" y2="14.5" />
                <circle cx="12" cy="17.25" r="0.9" fill="currentColor" stroke="none" />
              </svg>
            </span>
            <span className={styles.disclaimerText}>
              <span className={styles.disclaimerLabel}>Medical disclaimer</span>
              this assistant provides general information only and is not a substitute for professional medical
              advice, diagnosis, or treatment. In an emergency, contact the clinic or emergency services directly.
            </span>
          </div>
        </div>

        <div className={styles.messages} ref={messagesContainerRef}>
          <div ref={topRef} />
          {pendingFeedback && pendingFeedback.appointments.length > 0 && (
            <FeedbackPromptCard
              prompt={pendingFeedback.prompt}
              appointmentIds={pendingFeedback.appointments.map((a) => a.appointment_id)}
            />
          )}

          {showWelcome && (
            <div className={styles.welcome}>
              <div className={styles.welcomeIconHalo} aria-hidden="true">
                <div className={styles.welcomeIcon}>
                  <svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 12.5C4 7.81 8.03 4 13 4s9 3.81 9 8.5-4.03 8.5-9 8.5c-1.09 0-2.13-.19-3.1-.53L4 21l1.2-4.02A8.16 8.16 0 0 1 4 12.5Z" />
                  </svg>
                  <svg className={styles.welcomeSparkle} viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden="true">
                    <path d="M12 2 13.8 8.9 21 12l-7.2 3.1L12 22l-1.8-6.9L3 12l7.2-3.1L12 2Z" />
                  </svg>
                </div>
              </div>
              <h1 className={styles.welcomeHeading}>
                Hi{firstName ? ` ${firstName}` : ""}, how can I help you today?
              </h1>
              <p className={styles.welcomeSubtitle}>
                Ask about symptoms, clinic hours, or anything else about your care — I'll answer straight from
                your clinic's own knowledge base.
              </p>
              <div className={styles.suggestions}>
                {SUGGESTIONS.map((text) => (
                  <button
                    key={text}
                    type="button"
                    className={styles.suggestionChip}
                    onClick={() => applySuggestion(text)}
                  >
                    {text}
                  </button>
                ))}
              </div>
            </div>
          )}

          {historyError && <p className={styles.historyError}>{historyError}</p>}

          {messages.map((message, i) => (
            <ChatMessage
              key={i}
              message={message}
              onSelectSlot={(doctorName, when, slotId) => {
                // Marks THIS card (by its message index) used before the real
                // selection even lands — see usedCardIndices' own comment on
                // why a card must never be selectable a second time.
                setUsedCardIndices((prev) => new Set(prev).add(i));
                selectSlot(doctorName, when, slotId);
              }}
              onSelectCandidate={(doctorName, departmentName, when) => {
                setUsedCardIndices((prev) => new Set(prev).add(i));
                selectCandidate(doctorName, departmentName, when);
              }}
              disabled={sending || usedCardIndices.has(i)}
              grouped={i > 0 && messages[i - 1].role === message.role && !messages[i - 1].redFlag && !message.redFlag}
            />
          ))}

          {sending && pendingThreadToken === viewedThreadTokenRef.current && <TypingIndicator />}

          <div ref={messagesEndRef} />
        </div>

        <form className={styles.inputBar} onSubmit={handleSubmit}>
          <div className={styles.inputBarInner}>
            <textarea
              ref={textareaRef}
              className={styles.input}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Type a message…"
              rows={1}
              disabled={sending}
              aria-label="Chat message"
            />
            {speechRecognitionSupported && (
              <button
                type="button"
                className={`${styles.micBtn} ${listening ? styles.micBtnActive : ""}`}
                onClick={toggleListening}
                disabled={sending}
                aria-label={listening ? "Stop voice input" : "Start voice input"}
                aria-pressed={listening}
              >
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <rect x="9" y="2" width="6" height="12" rx="3" />
                  <path d="M5 10a7 7 0 0 0 14 0M12 19v3" />
                </svg>
              </button>
            )}
            <button type="submit" className={styles.sendBtn} disabled={sending || !input.trim()} aria-label="Send message">
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M22 2 11 13M22 2 15 22l-4-9-9-4 20-7Z" />
              </svg>
            </button>
          </div>
          {/* Shown only while actively recording, not at rest — a privacy disclosure
              only matters at the moment it's actually relevant, and showing it
              permanently would just be noise on every other visit to this screen. */}
          {listening && (
            <p className={styles.micPrivacyNote}>
              Listening… voice is transcribed by your browser's speech service, not stored by this app.
            </p>
          )}
        </form>
      </div>

      <Modal open={!!pendingDelete} onClose={() => setPendingDelete(null)} title="Delete chat">
        <p className={styles.modalIntro}>
          Delete <strong>{pendingDelete?.title}</strong>? This removes the whole conversation. This cannot be undone.
        </p>
        <div className={styles.modalActions}>
          <button type="button" className={styles.modalCancelBtn} onClick={() => setPendingDelete(null)} disabled={deleteBusy}>
            Cancel
          </button>
          <button type="button" className={styles.modalDeleteBtn} onClick={confirmDeleteSession} disabled={deleteBusy}>
            {deleteBusy ? "Deleting…" : "Delete"}
          </button>
        </div>
      </Modal>
    </div>
  );
}
