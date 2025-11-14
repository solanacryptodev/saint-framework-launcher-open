import { useNavigate } from "@solidjs/router";
import { createSignal } from "solid-js";
import { invoke } from "@tauri-apps/api/core";
import "./TitleScreen.css";

export default function TitleScreen() {
  const navigate = useNavigate();
  const [playerData, setPlayerData] = createSignal<any>(null);
  const [isLoading, setIsLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);

  const handleNewGame = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Initialize the game and get all node IDs
      const responseString = await invoke<string>('initialize_new_game');
      const nodeIds = JSON.parse(responseString);
      console.log("Game initialized with node IDs:", nodeIds);
      
      // Fetch tavern node data using the actual tavernId
      const tavernData = await invoke<any>('get_node', { nodeId: nodeIds.tavernId });
      console.log("Tavern node data:", tavernData);
      
      setPlayerData(tavernData);
    } catch (err) {
      console.error("Error initializing game:", err);
      setError(err as string);
    } finally {
      setIsLoading(false);
    }
  };

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
          <div class="menu-item" onClick={handleNewGame} style={{ cursor: "pointer" }}>
            <div class="menu-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            </div>
            <div class="menu-text">
              <div class="menu-title">NEW GAME</div>
              <div class="menu-description">
                {isLoading() ? "Initializing..." : "Begin a new mission"}
              </div>
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

        {/* Display player metadata when available */}
        {playerData() && (
          <div style={{
            "margin-top": "20px",
            "padding": "20px",
            "background": "rgba(0, 255, 255, 0.1)",
            "border": "1px solid rgba(0, 255, 255, 0.3)",
            "border-radius": "8px",
            "font-family": "monospace",
            "color": "#00ffff",
          }}>
            <h3 style={{ "margin-bottom": "10px", "color": "#00ffff" }}>Player Data Loaded:</h3>
            <pre style={{ 
              "margin": "0",
              "white-space": "pre-wrap",
              "word-wrap": "break-word",
              "font-size": "12px"
            }}>
              {JSON.stringify(playerData(), null, 2)}
            </pre>
          </div>
        )}

        {/* Display errors if any */}
        {error() && (
          <div style={{
            "margin-top": "20px",
            "padding": "15px",
            "background": "rgba(255, 0, 0, 0.1)",
            "border": "1px solid rgba(255, 0, 0, 0.3)",
            "border-radius": "8px",
            "color": "#ff0000",
          }}>
            <strong>Error:</strong> {error()}
          </div>
        )}

        <div class="version-info">
          <span class="version-dot"></span>
          VERSION 2.47.3 • SYSTEM ONLINE
        </div>
      </div>
    </div>
  );
}
