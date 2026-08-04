"""Sentinel markers prefixed onto certain chat replies so the frontend can parse a
JSON payload out of an otherwise plain-text message, without a richer response
schema. Kept in their own tiny module (rather than living in app.services.chat_tools,
where they originated) so a lightweight, high-frequency caller — e.g.
app.services.message_classifier, invoked on every chat turn — can check for a marker
without pulling in chat_tools' much heavier LangChain/tool-building imports.
"""

BOOKING_MARKER = "BOOKING_CONFIRMED::"
DOCTOR_OPTIONS_MARKER = "DOCTOR_OPTIONS::"

# A genuine cross-department question calls get_department_availability more than
# once in the same turn — this marker's payload combines every call's real result
# ({"departments": [{department_name, doctors: [...]}, ...]}), assembled in code
# (app.services.chat_tools.combine_department_availability_results), never written
# by the LLM itself.
DEPARTMENT_LIST_MARKER = "DEPARTMENT_LIST::"

# Internal-only marker (never reaches the frontend) distinguishing "this department
# is real but has no doctor with a free upcoming slot" from "this department name
# doesn't exist" — both used to come back from get_department_availability as a
# plain string, indistinguishable to combine_department_availability_results, which
# treats any non-DOCTOR_OPTIONS_MARKER result as noise to skip. That silently
# dropped a real, found-but-empty department from a multi-department reply (e.g. a
# combined neck-pain + itchy-skin turn showed Orthopedics but never mentioned that
# Dermatology was checked and had nothing open) — reported live. See
# app.services.chat_tools._get_department_availability_impl and
# combine_department_availability_results.
NO_SLOTS_MARKER = "NO_SLOTS::"

# Emitted by appointment_agent (see app.services.orchestrator.agents.appointment_agent)
# when it needs to ask the patient to disambiguate between multiple real candidates —
# either several doctors matching a typed name, or several of the patient's own
# appointments matching a vague description ("cancel my appointment" with 2 upcoming).
# Payload: {"kind": "doctor_name" | "appointment", "question": str, "candidates": [...]}.
# Exists specifically so a reply to THIS question is routed back to appointment_agent
# deterministically (see app.services.orchestrator.router's rule 1), the same way
# DOCTOR_OPTIONS_MARKER already removes any ambiguity for a slot-picking reply — never
# inferred from wording alone.
DOCTOR_DISAMBIGUATION_MARKER = "DOCTOR_DISAMBIGUATION::"

# NOTE: as of this marker's introduction, the frontend does not yet have a renderer
# for DOCTOR_DISAMBIGUATION_MARKER the way it does for the three markers above — a
# reply using it will show as raw marker-prefixed text until frontend support is
# added. Flagged explicitly rather than silently shipped; see the orchestrator
# migration notes.
