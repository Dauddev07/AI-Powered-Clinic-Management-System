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
