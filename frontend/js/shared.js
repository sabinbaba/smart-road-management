/**
 * shared.js — Smart Traffic (FIXED)
 */

const API_BASE = "http://localhost:8000";

function getToken() { return localStorage.getItem("token"); }
function getUsername() { return localStorage.getItem("username") || "User"; }

function authHeaders() { 
  const token = getToken();
  return token ? { "Authorization": `Bearer ${token}` } : {};
}

function requireAuth() {
  if (!getToken()) {
    window.location.href = "login.html";
  }
}

function logout() {
  localStorage.clear(); // ✅ cleaner
  window.location.href = "login.html";
}

function isActive(page) {
  const path = window.location.pathname.split("/").pop() || "";
  return page === path.replace(".html", "") ? "active" : "";
}

// ✅ FIX: handle undefined/null safely
function getStatusColor(status = "") {
  if (status.includes("ACCIDENT")) return "var(--accent-red)";
  if (status.includes("CONGESTION")) return "var(--accent-orange)";
  return "var(--accent-green)";
}

function getStatusClass(status = "") {
  if (status.includes("ACCIDENT")) return "accident";
  if (status.includes("CONGESTION")) return "congestion";
  return "normal";
}

function updateClock() {
  const clock = document.getElementById("headerClock");
  if (clock) clock.textContent = new Date().toTimeString().slice(0, 5);
}

function toggleUserMenu() {
  const menu = document.getElementById("userDropdown");
  if (menu) menu.classList.toggle("open"); // ✅ avoid null error
}

function showProfile() {
  alert("Profile settings coming soon!");
}

// ── INIT PAGE ─────────────────────────────
function initPage(pageName, title, breadcrumb) {
  document.title = `Smart Traffic — ${title}`;
  
  const header = document.getElementById("header");

  if (header) {
    header.innerHTML = `
      <div class="header-left">
        <div class="header-page-title">${title}</div>
        <div class="header-breadcrumb">${breadcrumb}</div>
      </div>
      <div class="header-right">
        <div class="live-badge" id="liveBadge">
          <div class="live-dot"></div> READY
        </div>
        <div class="header-clock" id="headerClock">--:--</div>
        <div class="user-menu" onclick="toggleUserMenu()">
          <div class="user-avatar" id="userAvatar">U</div>
          <div class="user-name" id="userName">User</div>
          <div style="font-size:16px;">▾</div>
        </div>
        <div class="user-dropdown" id="userDropdown">
          <div class="dropdown-item" onclick="showProfile()">Profile</div>
          <div class="dropdown-divider"></div>
          <div class="dropdown-item" style="color:var(--accent-red)" onclick="logout()">Sign Out</div>
        </div>
      </div>
    `;
  }

  // ✅ FIX: ensure elements exist AFTER rendering header
  setTimeout(() => {
    const avatar = document.getElementById("userAvatar");
    const name = document.getElementById("userName");

    if (avatar) avatar.textContent = getUsername().charAt(0).toUpperCase();
    if (name) name.textContent = getUsername();
  }, 0);

  renderSidebar();

  const footer = document.getElementById("footer");
  if (footer) {
    footer.innerHTML = `
      <div>© 2024 Smart Traffic</div>
      <div class="footer-right">
        <div class="footer-status">
          <div class="live-dot"></div>
          <span id="footerDetections">DETECTIONS: 0</span>
        </div>
      </div>
    `;
  }

  updateClock();
  setInterval(updateClock, 1000);

  if (
    window.location.pathname.includes("login") ||
    window.location.pathname.includes("signup")
  ) return;

  requireAuth();
}

// ── SIDEBAR ─────────────────────────────
function renderSidebar() {
  const sidebar = document.getElementById("sidebar");
  if (!sidebar) return;

  sidebar.innerHTML = `
    <div class="sidebar-brand">
      <div class="brand-logo">
        <svg viewBox="0 0 24 24">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"
            stroke="currentColor" stroke-width="1.5" fill="none"/>
        </svg>
      </div>
      <div class="brand-text">
        <div class="brand-title">Smart Traffic</div>
        <div class="brand-sub">Detection System</div>
      </div>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-section-label">MAIN</div>
      <a href="index.html" class="nav-item ${isActive('index')}">🏠 Dashboard</a>
      <a href="live.html" class="nav-item ${isActive('live')}">📡 Live Monitor</a>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-section-label">DATA</div>
      <a href="incidents.html" class="nav-item ${isActive('incidents')}">📋 Incidents</a>
      <a href="analytics.html" class="nav-item ${isActive('analytics')}">📊 Analytics</a>
      <a href="report.html" class="nav-item ${isActive('report')}">📈 Reports</a>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-section-label">CONFIG</div>
      <a href="settings.html" class="nav-item ${isActive('settings')}">⚙️ Settings</a>
    </div>

    <div class="sidebar-footer">
      <div class="cam-status">
        <div class="cam-dot"></div>
        <div class="cam-label">CAM READY</div>
      </div>
      <div class="cam-url">192.168.8.122:81/stream</div>
    </div>
  `;
}

// ── API ─────────────────────────────
async function apiFetch(endpoint) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      headers: authHeaders()
    });
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    console.error("API error:", e); // ✅ debugging help
    return null;
  }
}

// ── STATUS POLL ─────────────────────
let statusInterval;

function startStatusPoll(callback = () => {}) { // ✅ FIX default callback
  if (statusInterval) clearInterval(statusInterval);

  statusInterval = setInterval(async () => {
    const data = await apiFetch("/api/status");

    if (data) callback(data);

    const badge = document.getElementById("liveBadge");
    if (badge && data) {
      const status = data.status || "NORMAL";
      badge.innerHTML = `
        <div class="live-dot"></div>
        ${status}
      `;
      badge.style.borderColor = getStatusColor(status);
      badge.style.color = getStatusColor(status);
    }
  }, 5000);

  apiFetch("/api/status").then(callback);
}

// ── INIT ─────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  const path = window.location.pathname.split("/").pop() || "";

  const pageConfig = {
    "index.html": ["index", "Dashboard", "// HOME / DASHBOARD"],
    "live.html": ["live", "Live Monitor", "// MONITOR / LIVE FEED"],
    "incidents.html": ["incidents", "Incidents", "// INCIDENTS / LOG"],
    "analytics.html": ["analytics", "Analytics", "// ANALYTICS / INSIGHTS"],
    "report.html": ["report", "Reports", "// REPORTS / EXPORT"],
    "settings.html": ["settings", "Settings", "// SETTINGS / CONFIG"]
  };

  if (pageConfig[path]) {
    const [pageName, title, breadcrumb] = pageConfig[path];
    initPage(pageName, title, breadcrumb);
    startStatusPoll(); // now safe
  }
});