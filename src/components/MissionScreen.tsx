import { For } from "solid-js";
import InfoContainer from "./SciFiContainers/InfoContainer";
import "./MissionScreen.css";

export default function MissionScreen() {
  const tacticalStatus = [
    { label: "Hull Integrity:", value: "94%" },
    { label: "Shield Power:", value: "87%" },
    { label: "Weapon Systems:", value: "Online" },
    { label: "Navigation:", value: "Locked" }
  ];

  const crewRoster = [
    { role: "Commander:", status: "Active" },
    { role: "Engineer:", status: "Ready" },
    { role: "Pilot:", status: "Standby" },
    { role: "Medic:", status: "Available" }
  ];

  const systemLogs = [
    "[12:34] Warp drive initialized",
    "[12:35] Scanning sector 7-Alpha",
    "[12:36] Unknown signal detected",
    "[12:37] Shields at maximum",
    "[12:38] Awaiting orders..."
  ];

  const sensorData = [
    { label: "Energy Signature:", value: "Unknown" },
    { label: "Distance:", value: "2.3 km" },
    { label: "Threat Level:", value: "Moderate" },
    { label: "Scan Progress:", value: "87%" }
  ];

  const objectives = [
    "Investigate energy source",
    "Secure research data",
    "Assess station status",
    "Report findings"
  ];

  const intelFeed = [
    {
      title: "Station Omega-7",
      description: "Research facility specializing in quantum mechanics and dimensional studies",
      type: "info"
    },
    {
      title: "Warning",
      description: "Last crew reported \"temporal anomalies\" before disappearing",
      type: "warning"
    },
    {
      title: "Alert",
      description: "Unknown life forms detected in lower decks",
      type: "alert"
    }
  ];

  const commandOptions = [
    "Dock immediately and begin investigation",
    "Perform detailed scans before approaching",
    "Hail the station on all frequencies",
    "Deploy reconnaissance drones first",
    "Request backup from Command"
  ];

  return (
    <div class="mission-screen">
      {/* Corner decorations */}
      <div class="corner-decoration top-left"></div>
      <div class="corner-decoration top-right"></div>
      <div class="corner-decoration bottom-left"></div>
      <div class="corner-decoration bottom-right"></div>

      {/* Header */}
      <div class="mission-header">
        <h1 class="mission-title">NEXUS COMMAND INTERFACE</h1>
        <div class="mission-status-bar">
          <span class="status-item">STATUS: ACTIVE</span>
          <span class="status-dot">•</span>
          <span class="status-item">MISSION: IN PROGRESS</span>
          <span class="status-dot">•</span>
          <span class="status-item">SYNC: 98.7%</span>
        </div>
      </div>

      {/* Main Content Grid */}
      <div class="mission-grid">
        {/* Left Column */}
        <div class="mission-column left-column">
          <InfoContainer title="TACTICAL STATUS">
            <div class="data-list">
              <For each={tacticalStatus}>
                {(item) => (
                  <div class="data-item">
                    <span class="data-label">{item.label}</span>
                    <span class="data-value">{item.value}</span>
                  </div>
                )}
              </For>
            </div>
          </InfoContainer>

          <InfoContainer title="CREW ROSTER">
            <div class="data-list">
              <For each={crewRoster}>
                {(item) => (
                  <div class="data-item">
                    <span class="data-label">{item.role}</span>
                    <span class="data-value">{item.status}</span>
                  </div>
                )}
              </For>
            </div>
          </InfoContainer>

          <InfoContainer title="SYSTEM LOGS">
            <div class="log-list">
              <For each={systemLogs}>
                {(log) => (
                  <div class="log-entry">{log}</div>
                )}
              </For>
            </div>
          </InfoContainer>
        </div>

        {/* Center Column */}
        <div class="mission-column center-column">
          <InfoContainer title="MISSION BRIEFING" class="mission-briefing">
            <div class="briefing-content">
              <p>
                The deep space scanner has detected an anomalous energy signature emanating 
                from the abandoned research station Omega-7. Long-range sensors indicate the 
                station's power core is still active despite being offline for over a decade.
              </p>
              <p>
                Your mission is to investigate the station and determine the source of the energy 
                readings. Be advised: the last transmission from Omega-7 mentioned 
                "unprecedented discoveries" before all communication ceased.
              </p>
              <p>
                As you approach the station, your tactical officer reports multiple hull breaches 
                and signs of a struggle. The docking bay appears intact, but life support readings 
                are inconclusive.
              </p>
            </div>
          </InfoContainer>
        </div>

        {/* Right Column */}
        <div class="mission-column right-column">
          <InfoContainer title="SENSOR DATA">
            <div class="data-list">
              <For each={sensorData}>
                {(item) => (
                  <div class="data-item">
                    <span class="data-label">{item.label}</span>
                    <span class="data-value">{item.value}</span>
                  </div>
                )}
              </For>
            </div>
          </InfoContainer>

          <InfoContainer title="OBJECTIVES">
            <div class="objectives-list">
              <For each={objectives}>
                {(objective) => (
                  <div class="objective-item">
                    <span class="objective-bullet">•</span>
                    <span class="objective-text">{objective}</span>
                  </div>
                )}
              </For>
            </div>
          </InfoContainer>

          <InfoContainer title="INTEL FEED">
            <div class="intel-list">
              <For each={intelFeed}>
                {(intel) => (
                  <div class={`intel-item intel-${intel.type}`}>
                    <div class="intel-title">{intel.title}</div>
                    <div class="intel-description">{intel.description}</div>
                  </div>
                )}
              </For>
            </div>
          </InfoContainer>
        </div>
      </div>

      {/* Command Options */}
      <div class="command-section">
        <InfoContainer title="COMMAND OPTIONS:">
          <div class="command-options">
            <For each={commandOptions}>
              {(option) => (
                <button class="command-option-btn">{option}</button>
              )}
            </For>
          </div>
        </InfoContainer>
      </div>
    </div>
  );
}
