import { Router, Route } from "@solidjs/router";
import TitleScreen from "./components/TitleScreen";
import MissionScreen from "./components/MissionScreen";
import "./App.css";

function App() {
  return (
    <Router>
      <Route path="/" component={TitleScreen} />
      <Route path="/mission" component={MissionScreen} />
    </Router>
  );
}

export default App;
