/**
 * shared.js — Smart Traffic Detection System
 * v3.0 — SSE (Server-Sent Events) for real-time updates
 */

const API_BASE = "http://localhost:8000";

// ─────────────────────────────────────────────
//  AUTH
// ─────────────────────────────────────────────
function getToken()    { return localStorage.getItem("token"); }
function getUsername() { return localStorage.getItem("username") || "User"; }
function getRole()     { return localStorage.getItem("role") || "user"; }

function authHeaders() {
  const token = getToken();
  return token
    ? { "Content-Type": "application/json", "Authorization": `Bearer ${token}` }
    : { "Content-Type": "application/json" };
}

function requireAuth() {
  if (!getToken()) {
    var redirectCount = parseInt(sessionStorage.getItem("_authRedirects") || "0");
    if (redirectCount >= 2) {
      sessionStorage.removeItem("_authRedirects");
      return;
    }
    sessionStorage.setItem("_authRedirects", redirectCount + 1);
    window.location.href = "login.html";
    return;
  }
  sessionStorage.removeItem("_authRedirects");
}

function logout() {
  localStorage.clear();
  sessionStorage.clear();
  window.location.href = "login.html";
}

function isActive(page) {
  const current = window.location.pathname.split("/").pop()
    .replace(".html", "") || "index";
  return page === current ? "active" : "";
}

// ─────────────────────────────────────────────
//  API FETCH
// ─────────────────────────────────────────────
async function apiFetch(endpoint, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      headers: authHeaders(),
      ...options
    });
    if (res.status === 401) { logout(); return null; }
    if (!res.ok) return null;
    return await res.json();
  } catch (err) {
    console.error(`[API] ${endpoint}:`, err);
    return null;
  }
}

// ─────────────────────────────────────────────
//  STATUS HELPERS
// ─────────────────────────────────────────────
function normalizeStatus(status) {
  const s = (status || "NORMAL").toString().toUpperCase();
  if (s.includes("ACCIDENT"))   return "ACCIDENT";
  if (s.includes("CONGESTION")) return "CONGESTION";
  return "NORMAL";
}

function getStatusClass(status) {
  if (!status) return "normal";
  const s = status.toUpperCase();
  if (s.includes("ACCIDENT"))   return "accident";
  if (s.includes("CONGESTION")) return "congestion";
  return "normal";
}

function getStatusColor(status) {
  const c = getStatusClass(status);
  if (c === "accident")   return "var(--accent-red)";
  if (c === "congestion") return "var(--accent-orange)";
  return "var(--accent-green)";
}

function statusBadge(status) {
  return `<span class="status-badge ${getStatusClass(status)}">${status || "NORMAL"}</span>`;
}

// ─────────────────────────────────────────────
//  SSE — REAL-TIME EVENT SOURCE
// ─────────────────────────────────────────────
let _sse = null;
let _sseReconnectTimer = null;

function startStatusPoll(onUpdate) {
  startSSE(onUpdate);
}

function startSSE(onUpdate) {
  if (_sse) { _sse.close(); _sse = null; }
  if (_sseReconnectTimer) clearTimeout(_sseReconnectTimer);

  console.log("[SSE] Connecting...");
  _sse = new EventSource(`${API_BASE}/api/events`);

  _sse.onopen = () => {
    console.log("[SSE] ✅ Connected");
    updateSSEIndicator(true);
  };

  _sse.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);

      const badge = document.getElementById("liveBadge");
      if (badge) {
        const label = normalizeStatus(data.status);
        badge.innerHTML     = `<div class="live-dot"></div> ${label}`;
        badge.style.color       = getStatusColor(label);
        badge.style.borderColor = getStatusColor(label);
      }

      if (typeof onUpdate === "function") onUpdate(data);
    } catch (e) {
      console.error("[SSE] Parse error:", e);
    }
  };

  _sse.onerror = () => {
    console.warn("[SSE] Lost — reconnecting in 3s...");
    updateSSEIndicator(false);
    _sse.close();
    _sse = null;
    _sseReconnectTimer = setTimeout(() => startSSE(onUpdate), 3000);
  };
}

function updateSSEIndicator(connected) {
  document.querySelectorAll(".cam-dot").forEach(d => {
    d.style.background = connected ? "var(--accent-green)" : "var(--accent-red)";
    d.style.boxShadow  = connected ? "0 0 6px var(--accent-green)" : "0 0 6px var(--accent-red)";
  });
  const lbl = document.getElementById("sseOnlineLabel");
  if (lbl) {
    lbl.textContent  = connected ? "● ONLINE" : "● OFFLINE";
    lbl.style.color  = connected ? "var(--accent-green)" : "var(--accent-red)";
  }
}

// ─────────────────────────────────────────────
//  HEADER
// ─────────────────────────────────────────────
function injectHeader(pageTitle, breadcrumb) {
  const el = document.getElementById("header");
  if (!el) return;
  el.innerHTML = `
    <div class="header-left">
      <div class="header-page-title">${pageTitle}</div>
      <div class="header-breadcrumb">${breadcrumb}</div>
    </div>
    <div class="header-right">
      <div class="live-badge" id="liveBadge">
        <div class="live-dot"></div> CONNECTING...
      </div>
      <div class="header-clock" id="headerClock">--:--:--</div>
      <div class="user-menu" onclick="toggleUserMenu()">
        <div class="user-avatar" id="userAvatar">U</div>
        <span class="user-name" id="userName">...</span>
        <span style="color:var(--text-dim); font-size:11px;">▾</span>
      </div>
      <div class="user-dropdown" id="userDropdown">
        <div class="dropdown-item" onclick="window.location.href='settings.html'">⚙ Settings</div>
        <div class="dropdown-divider"></div>
        <div class="dropdown-item danger" onclick="logout()">⏻ Sign Out</div>
      </div>
    </div>
  `;

  const username = getUsername();
  document.getElementById("userAvatar").textContent = username.charAt(0).toUpperCase();
  document.getElementById("userName").textContent   = username;

  function tick() {
    const el = document.getElementById("headerClock");
    if (el) el.textContent = new Date().toTimeString().slice(0, 8);
  }
  tick();
  if (!window._clockStarted) {
    window._clockStarted = true;
    setInterval(tick, 1000);
  }
}

function toggleUserMenu() {
  const menu = document.getElementById("userDropdown");
  if (menu) menu.classList.toggle("open");
}

document.addEventListener("click", (e) => {
  const menu = document.getElementById("userDropdown");
  const btn  = document.querySelector(".user-menu");
  if (menu && btn && !btn.contains(e.target)) menu.classList.remove("open");
});

// ─────────────────────────────────────────────
//  SIDEBAR
// ─────────────────────────────────────────────
function injectSidebar(activePage) {
  const el = document.getElementById("sidebar");
  if (!el) return;

  const groups = [
    { label: "MAIN", items: [
      { id:"index",     href:"index.html",     icon:svgDashboard(), label:"Dashboard",    sub:"Overview & stats"  },
      { id:"live",      href:"live.html",       icon:svgLive(),      label:"Live Monitor", sub:"ESP32-CAM feed"    },
    ]},
    { label: "DATA", items: [
      { id:"incidents", href:"incidents.html",  icon:svgList(),      label:"Incidents",    sub:"Detection log"     },
      { id:"analytics", href:"analytics.html",  icon:svgChart(),     label:"Analytics",    sub:"Charts & trends"   },
      { id:"report",    href:"report.html",     icon:svgReport(),    label:"Reports",      sub:"Export & download" },
    ]},
    { label: "CONFIG", items: [
      { id:"settings",  href:"settings.html",   icon:svgSettings(),  label:"Settings",     sub:"Config & account"  },
    ]},
  ];

  const navHTML = groups.map(g => `
    <div style="padding:14px 14px 4px; font-family:'Share Tech Mono',monospace;
      font-size:9px; color:var(--text-dim); letter-spacing:3px;">// ${g.label}</div>
    ${g.items.map(item => {
      const active = activePage === item.id;
      return `<a href="${item.href}" class="nav-item ${active ? "active" : ""}">
        <span class="nav-icon-svg">${item.icon}</span>
        <span class="nav-text">
          <span class="nav-label">${item.label}</span>
          <span class="nav-sub">${item.sub}</span>
        </span>
        ${active ? `<span class="nav-active-bar"></span>` : ""}
      </a>`;
    }).join("")}
  `).join("");

  el.innerHTML = `
    <div class="sidebar-brand">
      <div class="brand-logo">
        <svg viewBox="0 0 24 24" stroke="currentColor" fill="none" stroke-width="2">
          <path d="M12 2L2 7l10 5 10-5-10-5z"/>
          <path d="M2 17l10 5 10-5"/>
          <path d="M2 12l10 5 10-5"/>
        </svg>
      </div>
      <div class="brand-text">
        <div class="brand-title">SmartTraffic</div>
        <div class="brand-sub">DETECTION v3.0</div>
      </div>
    </div>
    <nav style="padding:4px 10px; flex:1; overflow-y:auto;">${navHTML}</nav>
    <div class="sidebar-footer">
      <div style="font-family:'Share Tech Mono',monospace; font-size:9px;
        color:var(--text-dim); letter-spacing:2px; margin-bottom:8px;">// SYSTEM</div>
      <div class="cam-status">
        <div class="cam-dot"></div>
        <div style="flex:1; min-width:0;">
          <div class="cam-label">ESP32-CAM</div>
          <div class="cam-url">192.168.8.122/capture</div>
        </div>
      </div>
      <div style="margin-top:8px; padding:8px 10px; background:rgba(0,0,0,0.2);
        border:1px solid var(--border); border-radius:6px;
        display:flex; align-items:center; gap:8px;">
        <div style="flex:1;">
          <div style="font-family:'Share Tech Mono',monospace; font-size:9px;
            color:var(--text-dim); letter-spacing:1px; margin-bottom:2px;">LIVE STREAM</div>
          <div style="font-family:'Share Tech Mono',monospace; font-size:9px;
            color:var(--text-dim);">SSE · instant push</div>
        </div>
        <span id="sseOnlineLabel" style="font-family:'Share Tech Mono',monospace;
          font-size:9px; color:var(--accent-orange); letter-spacing:1px;">● CONNECTING</span>
      </div>
      <div style="margin-top:8px; font-family:'Share Tech Mono',monospace;
        font-size:9px; color:var(--text-dim);">YOLO11 v3.0 · SQLite</div>
    </div>
  `;
}

function svgDashboard() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/>
    <rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>
  </svg>`;
}
function svgLive() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <circle cx="12" cy="12" r="3"/>
    <path d="M6.3 6.3a8 8 0 0 0 0 11.4M17.7 17.7a8 8 0 0 0 0-11.4"/>
  </svg>`;
}
function svgList() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/>
    <line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/>
    <line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>
  </svg>`;
}
function svgChart() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>
    <line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/>
  </svg>`;
}
function svgReport() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14 2 14 8 20 8"/>
    <line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
  </svg>`;
}
function svgSettings() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <circle cx="12" cy="12" r="3"/>
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06
      a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09
      A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83
      l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09
      A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83
      l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09
      a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83
      l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09
      a1.65 1.65 0 0 0-1.51 1z"/>
  </svg>`;
}

// ─────────────────────────────────────────────
//  FOOTER
// ─────────────────────────────────────────────
function injectFooter() {
  const el = document.getElementById("footer");
  if (!el) return;
  el.innerHTML = `
    <span>SMART TRAFFIC DETECTION SYSTEM © 2026</span>
    <div class="footer-right">
      <span id="footerDetections">DETECTIONS: —</span>
      <div class="footer-status">
        <div class="live-dot" style="width:5px;height:5px;"></div>
        <span>SSE LIVE</span>
      </div>
    </div>
  `;
}

// ─────────────────────────────────────────────
//  TOAST
// ─────────────────────────────────────────────
function showToast(message, type = "info", duration = 3000) {
  let container = document.getElementById("toastContainer");
  if (!container) {
    container = document.createElement("div");
    container.id        = "toastContainer";
    container.className = "toast-container";
    document.body.appendChild(container);
  }
  const toast     = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity    = "0";
    toast.style.transition = "opacity 0.3s";
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ─────────────────────────────────────────────
//  DATE HELPERS
// ─────────────────────────────────────────────
function formatDateTime(ts) {
  if (!ts) return "—";
  return new Date(ts).toLocaleString("en-US", {
    month:"short", day:"numeric",
    hour:"2-digit", minute:"2-digit", second:"2-digit"
  });
}

function timeAgo(ts) {
  if (!ts) return "—";
  const diff = (Date.now() - new Date(ts)) / 1000;
  if (diff < 60)    return `${Math.floor(diff)}s ago`;
  if (diff < 3600)  return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

// ─────────────────────────────────────────────
//  PAGE INIT
// ─────────────────────────────────────────────
function initPage(pageId, pageTitle, breadcrumb) {
  requireAuth();
  injectHeader(pageTitle, breadcrumb);
  injectSidebar(pageId);
  injectFooter();
}