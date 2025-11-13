import { useNavigate } from "@solidjs/router";
import "./TitleScreen.css";

export default function TitleScreen() {
  const navigate = useNavigate();
  return (
    <div class="title-screen">
      {/* Corner decorations */}
      <div class="corner-decoration top-left"></div>
      <div class="corner-decoration top-right"></div>
      <div class="corner-decoration bottom-left"></div>
      <div class="corner-decoration bottom-right"></div>

      {/* Main content */}
      <div class="title-screen-content">
        <h1 class="main-title">NEXUS COMMAND</h1>
        <p class="subtitle">TACTICAL OPERATIONS SYSTEM</p>

        <div class="menu-container">
          <div class="menu-item">
            <div class="menu-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            </div>
            <div class="menu-text">
              <div class="menu-title">NEW GAME</div>
              <div class="menu-description">Begin a new mission</div>
            </div>
            <div class="menu-arrow">→</div>
          </div>

          <div class="menu-item">
            <div class="menu-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M3 7h18M3 12h18M3 17h18" />
                <rect x="5" y="4" width="14" height="16" rx="2" />
              </svg>
            </div>
            <div class="menu-text">
              <div class="menu-title">LOAD GAME</div>
              <div class="menu-description">Continue previous operation</div>
            </div>
            <div class="menu-arrow">→</div>
          </div>

          <div class="menu-item">
            <div class="menu-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 1v6m0 6v6M23 12h-6m-6 0H5" />
              </svg>
            </div>
            <div class="menu-text">
              <div class="menu-title">SETTINGS</div>
              <div class="menu-description">Configure system parameters</div>
            </div>
            <div class="menu-arrow">→</div>
          </div>

          <button class="launch-button" onClick={() => navigate("/mission")}>
            <svg class="launch-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
            LAUNCH MISSION
          </button>
        </div>

        <div class="version-info">
          <span class="version-dot"></span>
          VERSION 2.47.3 • SYSTEM ONLINE
        </div>
      </div>
    </div>
  );
}
