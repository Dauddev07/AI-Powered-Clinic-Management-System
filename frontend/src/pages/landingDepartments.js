// Shared department data for the public-facing pages (landing page strip and
// the dedicated Departments page). Kept as one list so both pages stay in
// sync — this mirrors the clinic's seeded department list, but isn't a live
// backend call: the departments-with-slots endpoint requires an authenticated
// patient session (it's scoped per-clinic), so it can't back an unauthenticated
// marketing page. Icons are plain inline SVG path data (24x24, stroke=currentColor),
// no icon library dependency, since this is a small, fixed set of 12 names.
export const DEPARTMENTS = [
  {
    name: "Cardiology",
    icon: "M12 21s-7-4.35-9.5-8.8C.9 8.6 2.2 5 5.6 5c1.9 0 3.3 1 4 2.3.7-1.3 2.1-2.3 4-2.3 3.4 0 4.7 3.6 3.1 7.2C19 16.65 12 21 12 21Z",
    description: "Diagnosis and ongoing care for heart and blood vessel conditions, including routine heart health checkups.",
  },
  {
    name: "General Medicine",
    icon: "M12 3v7M8.5 6.5h7M6 21h12a2 2 0 0 0 2-2v-3.5a5 5 0 0 0-2-4L15 9h-6l-3 2.5a5 5 0 0 0-2 4V19a2 2 0 0 0 2 2Z",
    description: "Routine checkups and care for everyday illnesses and general health concerns.",
  },
  {
    name: "Pediatrics",
    icon: "M12 3a3 3 0 0 1 3 3c0 1-.5 1.8-1 2.3 2.7.7 4.5 3 4.5 6.2V17a4 4 0 0 1-4 4H9.5a4 4 0 0 1-4-4v-2.5c0-3.2 1.8-5.5 4.5-6.2-.5-.5-1-1.3-1-2.3a3 3 0 0 1 3-3Z",
    description: "Routine and specialist care for infants, children, and adolescents.",
  },
  {
    name: "Orthopedics",
    icon: "M6.5 6.5a2.5 2.5 0 1 1 3.6 3.6l4.8 4.8a2.5 2.5 0 1 1-3.6 3.6L6.5 13.7a2.5 2.5 0 0 1 0-3.5l0-3.6ZM16 4l4 4-2 2-4-4 2-2ZM4 16l4 4-2 2-4-4 2-2Z",
    description: "Care for bones, joints, and muscles, including injuries and mobility issues.",
  },
  {
    name: "Gynecology",
    icon: "M12 3a5 5 0 0 1 5 5c0 2.2-1.4 4-3.3 4.7L14 14h2v2h-2v3h-2v-3H10v-2h2l.3-1.3C10.4 12 9 10.2 9 8a5 5 0 0 1 3-4.6Z",
    description: "Women's reproductive health, routine checkups, and related care.",
  },
  {
    name: "ENT",
    icon: "M8 12a5 5 0 0 1 10 0c0 3-2 4-2 6a2 2 0 1 1-4 0M8 12c0-3.5 2-6 6-6M8 12v2a4 4 0 0 0 4 4",
    description: "Diagnosis and treatment of ear, nose, and throat conditions.",
  },
  {
    name: "Dermatology",
    icon: "M12 3s5 5.5 5 9.5a5 5 0 0 1-10 0C7 8.5 12 3 12 3Z",
    description: "Care for skin, hair, and nail conditions, from everyday rashes to longer-term concerns.",
  },
  {
    name: "Neurology",
    icon: "M9 4a3 3 0 0 1 3 1 3 3 0 0 1 5 2c1 .3 1.8 1.2 1.8 2.4 0 .8-.4 1.5-1 2 .6.5 1 1.2 1 2 0 1.4-1.1 2.5-2.4 2.6A3 3 0 0 1 13 19a3 3 0 0 1-3-2M9 4A3 3 0 0 0 6 6a2.4 2.4 0 0 0-1.8 2.4c0 .8.4 1.5 1 2-.6.5-1 1.2-1 2C4.2 13.8 5.3 14.9 6.6 15A3 3 0 0 0 10 17M9 4v13",
    description: "Diagnosis and care for conditions affecting the brain, spine, and nervous system.",
  },
  {
    name: "Psychiatry",
    icon: "M12 3a7 7 0 0 0-7 7c0 2.4 1.2 4 2.3 5.1.5.5.7 1 .7 1.6V18a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-1.3c0-.6.2-1.1.7-1.6C17.8 14 19 12.4 19 10a7 7 0 0 0-7-7ZM10 20.5h4",
    description: "Assessment and ongoing care for mental health and emotional well-being.",
  },
  {
    name: "Ophthalmology",
    icon: "M2 12s3.8-6 10-6 10 6 10 6-3.8 6-10 6-10-6-10-6Z M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z",
    description: "Eye examinations and treatment for vision and eye health concerns.",
  },
  {
    name: "Dentistry",
    icon: "M8 3c-2.2 0-3.5 1.8-3.5 4.3 0 3 1 6.2 1.8 9 .4 1.4.9 2.7 1.9 2.7 1.2 0 1.3-2.5 1.8-4.3.3-1 .6-1.9 2-1.9s1.7.9 2 1.9c.5 1.8.6 4.3 1.8 4.3 1 0 1.5-1.3 1.9-2.7.8-2.8 1.8-6 1.8-9C19.5 4.8 18.2 3 16 3c-1.3 0-2.2.6-4 .6S9.3 3 8 3Z",
    description: "General dental checkups, cleanings, and treatment for common tooth and gum issues.",
  },
  {
    name: "Pulmonology",
    icon: "M12 3v6M12 9c-2 0-3.5 1.6-3.5 4v3a3 3 0 0 1-2 2.8V21h3.2c1 0 1.8-.7 2-1.7L12 15l.3 4.3c.2 1 1 1.7 2 1.7H17.5v-2.2A3 3 0 0 1 15.5 16v-3C15.5 10.6 14 9 12 9Z",
    description: "Diagnosis and care for lung and respiratory conditions.",
  },
];
