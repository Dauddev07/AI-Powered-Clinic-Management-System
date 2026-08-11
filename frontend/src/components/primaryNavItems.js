// Shared by PrimaryNavDesktop and PrimaryNavMobile so the two surfaces (a top
// row on wide screens, a fixed bottom tab bar on narrow ones — see each
// component's own comment) can never drift out of sync on which destinations
// exist or what icon/label represents them. Icon paths are the same ones
// already drawn elsewhere in the app (SettingsMenu.jsx's ICONS, the header's
// own home icon, PatientDashboard's count icons) — deliberately reused rather
// than redrawn, so an icon means the same thing wherever it appears.
const ICON_PATHS = {
  home: "M4 11.5 12 4l8 7.5M6 10v9a1 1 0 0 0 1 1h3v-6h4v6h3a1 1 0 0 0 1-1v-9",
  calendar: "M7 3v3M17 3v3M4 9h16M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1Z",
  clock: "M12 8v4l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  chat: "M4 12.5C4 7.81 8.03 4 13 4s9 3.81 9 8.5-4.03 8.5-9 8.5c-1.09 0-2.13-.19-3.1-.53L4 21l1.2-4.02A8.16 8.16 0 0 1 4 12.5Z",
  doctors:
    "M8 21v-2a4 4 0 0 1 4-4h0a4 4 0 0 1 4 4v2M12 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6ZM3 21v-1a3 3 0 0 1 3-3M21 21v-1a3 3 0 0 0-3-3M5.5 10a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5ZM18.5 10a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z",
  star: "M12 2.5l2.9 6.06 6.6.86-4.85 4.63 1.24 6.6L12 17.6l-5.89 3.05 1.24-6.6-4.85-4.63 6.6-.86L12 2.5Z",
};

export function getIconPath(icon) {
  return ICON_PATHS[icon];
}

// "home"/"dashboard" is the one item whose target depends on role (see
// dashboardPath in AppHeader.jsx) — it also doubles as the old header home
// button's refresh-on-click behavior, which PrimaryNav*'s own Link now takes
// over (see each component's own comment on why a Link to the current path
// still navigates).
// shortLabel is what PrimaryNavMobile's bottom tab bar uses (four labels have
// to fit side by side under a phone-width tab) — the desktop row has room
// for the full label, so it uses `label` instead (see each component).
export function getNavItems(role, dashboardPath) {
  if (role === "admin") {
    return [
      { key: "home", to: dashboardPath, label: "Dashboard", shortLabel: "Dashboard", icon: "home" },
      { key: "doctors", to: "/admin/doctors", label: "Doctors", shortLabel: "Doctors", icon: "doctors" },
      { key: "feedback", to: "/admin/feedback", label: "Feedback", shortLabel: "Feedback", icon: "star" },
    ];
  }
  return [
    { key: "home", to: dashboardPath, label: "Home", shortLabel: "Home", icon: "home" },
    { key: "book", to: "/patient/book", label: "Book Appointment", shortLabel: "Book", icon: "calendar" },
    { key: "upcoming", to: "/patient/appointments", label: "Upcoming", shortLabel: "Upcoming", icon: "clock" },
    { key: "chat", to: "/patient/chat", label: "Chat", shortLabel: "Chat", icon: "chat" },
  ];
}
