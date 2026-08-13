import { useLayoutEffect } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { AuthProvider } from "./auth/AuthContext";
import RequireAuth from "./auth/RequireAuth";
import AppHeader from "./components/AppHeader";
import Footer from "./components/Footer";
import PrimaryNavMobile from "./components/PrimaryNavMobile";
import { ToastProvider } from "./components/ToastContext";
import Landing from "./pages/Landing";
import Login from "./pages/Login";
import ForgotPassword from "./pages/ForgotPassword";
import Register from "./pages/Register";
import ChangePassword from "./pages/ChangePassword";
import AdminHome from "./pages/AdminHome";
import AdminLayout from "./pages/admin/AdminLayout";
import AdminViewProfile from "./pages/admin/ViewProfile";
import DoctorList from "./pages/admin/DoctorList";
import DoctorCsvImport from "./pages/admin/DoctorCsvImport";
import IngestionLogScreen from "./pages/admin/IngestionLogScreen";
import KnowledgeBase from "./pages/admin/KnowledgeBase";
import Feedback from "./pages/admin/Feedback";
import PatientLayout from "./pages/patient/PatientLayout";
import PatientDashboard from "./pages/patient/PatientDashboard";
import PatientViewProfile from "./pages/patient/ViewProfile";
import BookAppointment from "./pages/patient/BookAppointment";
import UpcomingAppointments from "./pages/patient/UpcomingAppointments";
import AppointmentHistory from "./pages/patient/AppointmentHistory";
import ChatPage from "./pages/patient/ChatPage";

export default function App() {
  const location = useLocation();

  // Every navigation must land at the top of the new page, not wherever the
  // previous page happened to be scrolled to — React Router (unlike a classic
  // full-page load) never resets scroll position on its own. Skipped when the
  // URL carries a hash (e.g. a footer link to "/#pricing") so Landing's own
  // hash-scroll effect below can still land on that anchor instead of being
  // immediately overridden back to the top. useLayoutEffect (not useEffect) so
  // this runs before the browser paints the new page, avoiding a visible flash
  // of the old scroll position first.
  useLayoutEffect(() => {
    if (location.hash) return;
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  }, [location.key, location.hash]);

  return (
    <ToastProvider>
      <AuthProvider>
        <AppHeader />
      {/* Keyed on location.key (a fresh id react-router mints per navigation,
          even a Link/navigate() to the *same* path) rather than pathname —
          so it remounts and replays the fade-in below on every navigation,
          including the header's home button clicked while already on the
          dashboard, which is meant to act like a refresh. Header/footer sit
          outside it and never remount, so open menus, scroll listeners, etc.
          on them are unaffected by route changes. */}
      <div key={location.key} className="page-transition">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/forgot-password" element={<ForgotPassword />} />
          <Route path="/register" element={<Register />} />
          <Route
            path="/change-password"
            element={
              <RequireAuth>
                <ChangePassword />
              </RequireAuth>
            }
          />
          <Route
            path="/patient"
            element={
              <RequireAuth role="patient">
                <PatientLayout />
              </RequireAuth>
            }
          >
            <Route index element={<PatientDashboard />} />
            <Route path="profile" element={<PatientViewProfile />} />
            <Route path="book" element={<BookAppointment />} />
            <Route path="appointments" element={<UpcomingAppointments />} />
            <Route path="appointments/history" element={<AppointmentHistory />} />
            <Route path="chat" element={<ChatPage />} />
          </Route>
          <Route
            path="/admin"
            element={
              <RequireAuth role="admin">
                <AdminLayout />
              </RequireAuth>
            }
          >
            <Route index element={<AdminHome />} />
            <Route path="profile" element={<AdminViewProfile />} />
            <Route path="doctors" element={<DoctorList />} />
            <Route path="doctors/import" element={<DoctorCsvImport />} />
            <Route path="doctors/ingestion-log" element={<IngestionLogScreen />} />
            <Route path="knowledge-base" element={<KnowledgeBase />} />
            <Route path="feedback" element={<Feedback />} />
          </Route>
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </div>
      {location.pathname !== "/patient/chat" && <Footer />}
      <PrimaryNavMobile />
      </AuthProvider>
    </ToastProvider>
  );
}
