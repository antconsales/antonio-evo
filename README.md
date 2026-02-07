# Antonio Evo

**Your Local, Deterministic, Auditable Cognitive Runtime**

> Version 4.0 - Proto-AGI Research-Grade Edition

A desktop application that brings AI to your computer - offline, private, and under your control.

---

## Core Identity

**Antonio Evo is:**
- A local, deterministic, auditable cognitive runtime
- Designed to assist humans without replacing them

**Antonio Evo is NOT:**
- An autonomous agent
- A chatbot toy
- A decision-maker

**Core Principle:** `CODE DECIDES, MODELS DO NOT`

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Offline-First** | Works without internet using local AI models |
| **Voice Support** | Talk to Antonio, hear responses (Whisper + Piper) |
| **Privacy** | No data leaves your machine unless you allow it |
| **Evolutionary Memory** | Learns and remembers across sessions |
| **Dual Persona** | SOCIAL (friendly) / LOGIC (analytical) auto-switch |
| **External Escalation** | Complex tasks can use Claude/GPT (policy-controlled) |
| **API Server** | WebSocket + REST API for integration |
| **Auditable** | Every decision logged with SHA-256 hash chain |

---

## Quick Start

### Prerequisites

1. **Python 3.11+**
2. **Ollama** with Mistral model
3. **Node.js 18+** (for UI)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/antonio-local-orchestrator
cd antonio-local-orchestrator

# 2. Install Ollama and pull Mistral
# Windows: Download from https://ollama.com/download
# macOS/Linux:
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral

# 3. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Install UI dependencies
cd ui && npm install && cd ..
```

### Running

```bash
# Terminal 1: Start backend
python -m src.api.websocket_server

# Terminal 2: Start UI
cd ui && npm run dev
```

Open http://localhost:5173 in your browser.

### First Use

1. Type a message and press Enter
2. Antonio responds using local Mistral model
3. Attach files using the paperclip icon
4. Use voice with the microphone button

---

## Architecture

```
USER INPUT (Text/Voice/File)
        |
        v
+------------------------------------------+
|  FRONTEND (React/Electron)               |
|  - Chat UI, Avatar, Settings             |
|  - WebSocket Client                      |
+------------------+-----------------------+
                   | ws://localhost:8420
                   v
+------------------------------------------+
|  BACKEND (Python/FastAPI)                |
|                                          |
|  1. NORMALIZER -> Validate input         |
|  2. MEMORY -> BM25 context retrieval     |
|  3. CLASSIFIER -> Intent/domain          |
|  4. POLICY ENGINE -> Route decision      |
|  5. ROUTER -> Handler dispatch           |
|  6. HANDLER -> Process (sandboxed)       |
|  7. AUDIT -> SHA-256 hash chain          |
|                                          |
|  HANDLERS:                               |
|  - SOCIAL (Mistral, temp 0.7)            |
|  - LOGIC (Mistral, temp 0.3)             |
|  - Whisper (STT)                         |
|  - Piper (TTS)                           |
|  - CLIP (images)                         |
|  - External (Claude/GPT)                 |
+------------------------------------------+
                   |
                   v
+------------------------------------------+
|  MEMORY (SQLite + FTS5)                  |
|  - Neurons (with decay)                  |
|  - Sessions                              |
|  - Preferences                           |
+------------------------------------------+
```

---

## Configuration

### Environment Variables (.env)

```bash
# LLM
LLM_SERVER=http://localhost:11434
OLLAMA_MODEL=mistral
LLM_TIMEOUT=120

# Voice (optional)
ASR_SERVER=http://localhost:8803
TTS_SERVER=http://localhost:8804
WHISPER_MODEL=base
PIPER_VOICE=it_IT-riccardo-x_low

# External API (optional)
EXTERNAL_API_KEY=your-anthropic-key
```

### Policy Configuration (config/policy.json)

```json
{
  "rate_limits": {
    "requests_per_minute": 30,
    "external_calls_per_hour": 10
  },
  "allow_external_fallback": true,
  "blocked_patterns": ["hack", "exploit"]
}
```

---

## API Usage

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8420/ws/chat');

ws.send(JSON.stringify({
  type: 'chat',
  text: 'Hello Antonio!',
  session_id: 'my-session'
}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.text);
};
```

### REST

```bash
# Health check
curl http://localhost:8420/api/health

# Memory stats
curl http://localhost:8420/api/memory/stats

# UI state
curl http://localhost:8420/api/ui/state
```

---

## Project Structure

```
antonio-local-orchestrator/
├── src/
│   ├── api/                 # WebSocket + REST server
│   ├── handlers/            # Mistral, Whisper, TTS, CLIP
│   ├── memory/              # Neurons, storage, retrieval
│   ├── policy/              # Routing decisions
│   ├── router/              # Handler dispatch
│   ├── reasoning/           # Cognitive budget, simulation
│   └── sandbox/             # Process isolation
├── ui/                      # React frontend
├── config/                  # JSON configuration
├── prompts/                 # LLM system prompts
├── data/                    # SQLite database
└── logs/                    # Audit trail
```

---

## Security Model

1. **Sandboxing**: All handlers run with resource limits (CPU, memory, timeout)
2. **Policy Engine**: Code-based routing, LLMs never decide
3. **Audit Trail**: SHA-256 hash chain, tamper-evident
4. **External Gating**: API calls require explicit policy + justification
5. **Attachments**: Treated as UNTRUSTED INERT DATA - analyzed, never executed

---

## Philosophy

```
CODE DECIDES, MODELS DO NOT
LOCAL FIRST, EXTERNAL LAST
EXPLICIT OVER IMPLICIT
BORING IS GOOD
```

Antonio is built on the principle that **your AI assistant should be predictable and accountable**:

- Policy engine (code) decides routing, not LLMs
- External APIs are gated, justified, and cost-tracked
- Every decision is logged in a tamper-evident audit trail
- Sandboxing is mandatory, not optional

---

## Status

| Component | Status |
|-----------|--------|
| Core orchestrator | Ready |
| Policy engine | Ready |
| Local LLM (Mistral) | Ready |
| Evolutionary Memory | Ready |
| Dual Persona | Ready |
| WebSocket API | Ready |
| Voice (Whisper/Piper) | Ready |
| Desktop UI | Ready |
| External escalation | Ready |
| Proto-AGI features | v4.0 |

See [ROADMAP.md](ROADMAP.md) for full timeline and v4.0 Proto-AGI details.

---

## Contributing

Key principles:
1. Code decides, models suggest
2. Offline-first, external-gated
3. Boring is good
4. Every decision auditable

---

## License

MIT
