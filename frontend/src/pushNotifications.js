// Web Push enable/disable flow — see backend app/services/push_notifications.py's
// own docstring for the full design (which notification types get pushed, why
// push is only ever sent after a commit, etc.). This module only handles the
// BROWSER side: service worker registration, permission, and subscribing/
// unsubscribing via PushManager.
import { fetchPushPublicKey, subscribeToPush, unsubscribeFromPush } from "./api/notifications";

// Registered once, at app startup (see main.jsx) — a no-op in any browser
// without service worker support (Firefox for iOS, very old browsers), same
// "just don't render/offer the feature" pattern as the speech-to-text mic button.
export function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  navigator.serviceWorker.register("/sw.js").catch(() => {
    // Registration can fail (e.g. served over plain http:// in some dev setups,
    // which service workers require https/localhost for) — push simply won't be
    // offered in that case, same silent-degrade pattern as everywhere else.
  });
}

// iOS Safari only supports Web Push for a site that's been added to the Home
// Screen — navigator.standalone is Safari's own (non-standard) flag for exactly
// that state, true only when launched from a Home Screen icon.
export function isIOS() {
  return /iphone|ipad|ipod/i.test(navigator.userAgent);
}

export function isStandalone() {
  return window.navigator.standalone === true || window.matchMedia("(display-mode: standalone)").matches;
}

// True when notifications are structurally possible right now — false either
// because the browser has no Push API at all, or because it's iOS Safari and the
// app hasn't been added to the Home Screen yet (see the two functions above).
export function pushNotificationsAvailable() {
  const hasApi = "serviceWorker" in navigator && "PushManager" in window;
  if (!hasApi) return false;
  if (isIOS() && !isStandalone()) return false;
  return true;
}

function urlBase64ToUint8Array(base64) {
  const padding = "=".repeat((4 - (base64.length % 4)) % 4);
  const base64Safe = (base64 + padding).replace(/-/g, "+").replace(/_/g, "/");
  const rawData = window.atob(base64Safe);
  return Uint8Array.from([...rawData].map((char) => char.charCodeAt(0)));
}

// Returns "granted" | "denied" | "unsupported" | "unavailable-needs-home-screen".
export async function getPushStatus() {
  if (!("serviceWorker" in navigator) || !("PushManager" in window)) return "unsupported";
  if (isIOS() && !isStandalone()) return "unavailable-needs-home-screen";
  if (Notification.permission === "denied") return "denied";
  const registration = await navigator.serviceWorker.ready;
  const subscription = await registration.pushManager.getSubscription();
  return subscription ? "granted" : "not-subscribed";
}

// Requests permission (shows the native browser prompt on first call) and, if
// granted, subscribes and saves the subscription server-side. Throws only on a
// genuine unexpected failure — the caller is expected to catch and show an
// inline message, same pattern as the mic's MIC_ERROR_MESSAGES.
export async function enablePushNotifications() {
  const { public_key: publicKey } = await fetchPushPublicKey();
  if (!publicKey) {
    throw new Error("Notifications aren't configured for this clinic yet.");
  }

  const permission = await Notification.requestPermission();
  if (permission !== "granted") {
    throw new Error(
      permission === "denied"
        ? "Notification permission was blocked. Enable it in your browser's site settings to turn this on."
        : "Notification permission wasn't granted."
    );
  }

  const registration = await navigator.serviceWorker.ready;
  let subscription = await registration.pushManager.getSubscription();
  if (!subscription) {
    subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(publicKey),
    });
  }

  await subscribeToPush(subscription.toJSON());
  return subscription;
}

export async function disablePushNotifications() {
  if (!("serviceWorker" in navigator)) return;
  const registration = await navigator.serviceWorker.ready;
  const subscription = await registration.pushManager.getSubscription();
  if (!subscription) return;

  const endpoint = subscription.endpoint;
  await subscription.unsubscribe();
  await unsubscribeFromPush(endpoint);
}
