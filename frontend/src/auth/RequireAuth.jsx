import { Navigate } from "react-router-dom";
import { useAuth } from "./AuthContext";

export default function RequireAuth({ role, children }) {
  const { isAuthenticated, isBootstrapping, user } = useAuth();

  // A reload with an expired access token but a still-valid refresh token is
  // mid-silent-refresh at this point (see AuthContext's mount effect) — rendering
  // nothing briefly beats a flash-redirect to /login for a session that's about
  // to come back.
  if (isBootstrapping) {
    return null;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (role && user.role !== role) {
    return <Navigate to="/login" replace />;
  }

  return children;
}
