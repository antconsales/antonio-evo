# Antonio Desktop UI

Modern Electron + React desktop application for Antonio Local Orchestrator.

## Features

- Chat interface with message history
- Voice input (push-to-talk microphone)
- Voice output (text-to-speech)
- Settings panel
- Dark theme
- Connection status indicator

## Prerequisites

- Node.js 18+
- Antonio API server running on `http://localhost:8420`

## Development

```bash
# Install dependencies
npm install

# Start in development mode (Vite + Electron)
npm run dev

# Or start just the React app
npm run react-dev
```

## Production Build

```bash
# Build for current platform
npm run dist

# Build without packaging (for testing)
npm run build
```

## Project Structure

```
ui/
├── main.js           # Electron main process
├── preload.js        # IPC bridge (context isolation)
├── index.html        # HTML entry point
├── src/
│   ├── main.jsx      # React entry point
│   ├── App.jsx       # Main React component
│   ├── components/   # React components
│   │   ├── TitleBar.jsx
│   │   ├── Sidebar.jsx
│   │   ├── ChatArea.jsx
│   │   ├── MessageBubble.jsx
│   │   ├── VoiceButton.jsx
│   │   └── SettingsPanel.jsx
│   └── styles/
│       └── global.css
└── public/
    └── icon.svg
```

## API Communication

The UI communicates with the Antonio API server via the Electron IPC bridge:

```javascript
// Send a question
const result = await window.antonio.ask("Your question", { speak: true });

// Voice transcription
const result = await window.antonio.listen(audioBase64, { process: false });

// Health check
const result = await window.antonio.health();
```

## Running with Antonio

1. Start the Antonio API server:
   ```bash
   cd ..
   python -m src.api.server
   ```

2. In a new terminal, start the UI:
   ```bash
   cd ui
   npm run dev
   ```
