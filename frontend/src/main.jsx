import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.jsx'
import { ThemeProvider } from './theme/ThemeContext.jsx'
import { registerServiceWorker } from './pushNotifications.js'

// Registered unconditionally at startup, not gated behind the patient ever
// enabling notifications — a service worker has to already be active for
// PushManager.subscribe() to work at all when they later do, and registering it
// early is also what makes "Add to Home Screen" install a real PWA (see
// public/manifest.json). No-ops safely in unsupported browsers.
registerServiceWorker()

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ThemeProvider>
  </StrictMode>,
)
