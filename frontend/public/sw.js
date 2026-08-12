// Minimal service worker — its only job is receiving a Web Push event while the
// app itself may be fully closed, and showing/handling the resulting OS
// notification. Deliberately has NO offline-caching/fetch-interception logic:
// this app doesn't need offline page support, and adding one would risk serving
// stale API responses for a system where booking/appointment data must always be
// live. Registered from src/pushNotifications.js, scope "/" (the whole origin),
// which is also what makes "Add to Home Screen" install the app as a PWA at all.

self.addEventListener("push", (event) => {
  if (!event.data) return;

  let payload;
  try {
    payload = event.data.json();
  } catch {
    payload = { title: "Quick Check Clinic", body: event.data.text() };
  }

  event.waitUntil(
    self.registration.showNotification(payload.title || "Quick Check Clinic", {
      body: payload.body || "",
      icon: "/icon-192.png",
      badge: "/icon-192.png",
      tag: payload.tag,
      data: { url: payload.url || "/patient/appointments" },
    })
  );
});

// Tapping the notification focuses an already-open tab on this origin if one
// exists (rather than opening a duplicate), or opens a new one otherwise.
self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const targetUrl = event.notification.data?.url || "/patient/appointments";

  event.waitUntil(
    self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((clients) => {
      for (const client of clients) {
        if (client.url.includes(self.location.origin) && "focus" in client) {
          client.navigate(targetUrl);
          return client.focus();
        }
      }
      return self.clients.openWindow(targetUrl);
    })
  );
});
