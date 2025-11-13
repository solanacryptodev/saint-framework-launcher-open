export default function SciFiContainer1() {
  return (
    <svg width="350" height="450" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 350 450" style="background-color:#0a0e17">
      <defs>
        {/* Blue gradient for container */}
        <linearGradient id="containerFill" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#0a2a60" stop-opacity="0.7"/>
          <stop offset="100%" stop-color="#0085ff" stop-opacity="0.85"/>
        </linearGradient>
        
        {/* Corner light glow effect */}
        <radialGradient id="lightGlow" cx="50%" cy="50%" r="75%">
          <stop offset="0%" stop-color="#fff" stop-opacity="0.9"/>
          <stop offset="50%" stop-color="#00ccff" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="#00ccff" stop-opacity="0"/>
        </radialGradient>
        
        {/* Tech border gradient */}
        <linearGradient id="borderGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#00ccff"/>
          <stop offset="100%" stop-color="#0085ff"/>
        </linearGradient>
        
        {/* Glowing effect filter */}
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>
      
      {/* Container background with blue gradient */}
      <rect x="30" y="30" width="290" height="390" rx="8" ry="8" fill="url(#containerFill)" opacity="0.9"/>
      
      {/* Main border outline */}
      <path d="M 40 40 
           L 290 40 
           L 300 50 
           L 300 390 
           L 290 400 
           L 40 400 
           L 30 390 
           L 30 50 
           Z" 
        fill="none" 
        stroke="url(#borderGradient)" 
        stroke-width="2" 
        stroke-linejoin="round"/>
      
      {/* Inner border details */}
      <path d="M 50 50 
           L 280 50 
           L 290 60 
           L 290 380 
           L 280 390 
           L 50 390 
           L 40 380 
           L 40 60 
           Z" 
        fill="none" 
        stroke="url(#borderGradient)" 
        stroke-width="1.5" 
        stroke-dasharray="12,8"
        stroke-linejoin="round"/>
      
      {/* Corner lights */}
      <circle cx="42" cy="42" r="6" fill="url(#lightGlow)" filter="url(#glow)"/>
      <circle cx="298" cy="42" r="6" fill="url(#lightGlow)" filter="url(#glow)"/>
      <circle cx="42" cy="398" r="6" fill="url(#lightGlow)" filter="url(#glow)"/>
      <circle cx="298" cy="398" r="6" fill="url(#lightGlow)" filter="url(#glow)"/>
      
      {/* Tech pattern elements */}
      <path d="M 60 65 H 100 M 60 75 H 100 M 60 85 H 100" 
        stroke="#00ccff" 
        stroke-width="0.8"
        stroke-linecap="round"/>
      <path d="M 290 65 V 100 M 290 75 V 100 M 290 85 V 100" 
        stroke="#00ccff" 
        stroke-width="0.8"
        stroke-linecap="round"/>
      <path d="M 285 380 H 245 M 285 370 H 245 M 285 360 H 245" 
        stroke="#00ccff" 
        stroke-width="0.8"
        stroke-linecap="round"/>
      <path d="M 40 380 V 340 M 40 370 V 340 M 40 360 V 340" 
        stroke="#00ccff" 
        stroke-width="0.8"
        stroke-linecap="round"/>
    </svg>
  );
}
