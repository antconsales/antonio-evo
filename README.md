<p align="center">
  <h1 align="center">Antonio Evo</h1>
  <p align="center"><strong>Local Cognitive Runtime — Your AI Thinks Locally, Learns Evolutionarily, and Never Phones Home</strong></p>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSL_1.1-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11+-green.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/node-18+-green.svg" alt="Node 18+">
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg" alt="Platform">
  <img src="https://img.shields.io/badge/version-4.0--proto--agi-orange.svg" alt="Version">
  <img src="https://img.shields.io/badge/LLM-Ollama%20%2B%20Mistral-purple.svg" alt="LLM">
</p>

<p align="center">
  <code>CODE DECIDES, MODELS DO NOT.</code>
</p>

---

<!-- ADD DEMO GIF HERE -->

## What is Antonio Evo?

Antonio Evo is a **local-first cognitive runtime** that runs AI on your machine with deterministic policy control, evolutionary memory, and full auditability. Unlike cloud-based AI assistants, Antonio Evo keeps your data local, makes routing decisions with code (not LLMs), and learns across sessions without retraining.

**It's not a chatbot. It's not an agent. It's a bounded, auditable cognitive runtime.**

---

## Why Antonio Evo?

| Feature | ChatGPT / Claude | GPT4All / Jan.ai | Open WebUI | LangChain | **Antonio Evo** |
|---------|:-----------------:|:-----------------:|:----------:|:---------:|:----------------:|
| **Routing** | LLM decides | LLM decides | LLM decides | LLM decides | **Code decides (deterministic)** |
| **Memory** | Cloud, limited | None | None | Plugin-based | **Local evolutionary neurons + decay** |
| **Learning** | No | No | No | No | **Proto-learning without retraining** |
| **Privacy** | Cloud | Local | Self-host | Varies | **100% local-first** |
| **Audit trail** | Basic logs | None | None | Basic | **SHA-256 hash chain** |
| **Personas** | Single | Single | Single | None | **SOCIAL/LOGIC auto-switch** |
| **Cognitive budget** | Unlimited | None | None | None | **Explicit bounds per profile** |
| **Simulation engine** | None | None | None | None | **Internal non-executing** |
| **Hardware aware** | No | No | No | No | **Profile-based adaptation** |
| **Voice** | Cloud STT/TTS | None | None | None | **Local Whisper + Piper** |

---

## Quick Start

```bash
# One-liner install (macOS/Linux)
curl -sSL https://raw.githubusercontent.com/antconsales/antonio-evo/main/install.sh | bash

# Or manual setup:
git clone https://github.com/antconsales/antonio-evo.git && cd antonio-evo
pip install -r requirements.txt && cd ui && npm install && cd ..
python -m src.api.websocket_server  # Terminal 1
cd ui && npm run dev                # Terminal 2
```

Open **http://localhost:5173** and start talking.

---

## Features

- **Offline-First** — Runs entirely on your machine with Ollama + Mistral. No internet required.
- **Deterministic Policy Engine** — Code decides routing, personas, and escalation. LLMs generate language, nothing more.
- **Evolutionary Memory** — Neurons with confidence decay, BM25 retrieval, and cross-session learning via SQLite + FTS5.
- **Dual Persona** — SOCIAL (warm, temp 0.7) and LOGIC (precise, temp 0.3) auto-switch based on intent classification.
- **Cognitive Budget System** — Explicit reasoning depth, context limits, and simulation budgets per hardware profile.
- **Internal Simulation Engine** — Non-executing hypothetical reasoning with mandatory disclosure framing.
- **Concept Graph** — World model with confidence-weighted nodes, causal chains, and path-finding.
- **Proto-Learning** — Learns from interactions without model retraining: confidence adjustment, abstraction refinement, failure-driven learning.
- **Runtime Profiles** — Hardware-aware adaptation: EVO-LITE (2GB) through EVO-HYBRID (full GPU + external APIs).
- **Voice I/O** — Local Whisper STT + Piper TTS. Talk to your AI, hear it respond. All local.
- **SHA-256 Audit Trail** — Every decision logged in a tamper-evident hash chain. Verify integrity at any time.
- **MCP Capabilities** — Approval-gated capabilities: file read/write, HTTP fetch, external LLM calls all require explicit consent.
- **External Escalation** — Complex tasks can fall back to Claude/GPT, but only with policy approval and cost tracking.
- **Desktop App** — React + Electron UI with animated avatar, mood system, and real-time WebSocket streaming.

---

## Architecture

```
USER INPUT (Text / Voice / File)
        │
        ▼
┌──────────────────────────────────────────┐
│  FRONTEND (React / Electron)             │
│  Chat UI  ·  Avatar  ·  Settings         │
│            WebSocket Client              │
└──────────────────┬───────────────────────┘
                   │ ws://localhost:8420
                   ▼
┌──────────────────────────────────────────┐
│  BACKEND (Python / FastAPI)              │
│                                          │
│  Runtime Profile ─► Cognitive Budget     │
│                                          │
│  Pipeline:                               │
│  1. Normalizer ─► Validate input         │
│  2. Memory ─► BM25 context retrieval     │
│  3. Classifier ─► Intent + domain        │
│  4. Policy Engine ─► Route decision      │
│  5. Router ─► Handler dispatch           │
│  6. Neuron Creator ─► Store learnings    │
│  7. Audit Logger ─► SHA-256 chain        │
│                                          │
│  Handlers:                               │
│  SOCIAL · LOGIC · Whisper · Piper        │
│  CLIP · External (gated)                 │
│                                          │
│  Reasoning:                              │
│  Simulation · ConceptGraph · ProtoLearner│
│                                          │
│  Memory: SQLite + FTS5 (neurons, decay)  │
└──────────────────────────────────────────┘
```

---

## Runtime Profiles

Antonio Evo adapts to your hardware automatically:

| Profile | RAM | Reasoning Depth | Context | Simulation | External |
|---------|-----|----------------|---------|------------|----------|
| **EVO-LITE** | 2-4 GB | 3 steps | 2048 tokens | Off | No |
| **EVO-STANDARD** | 4-8 GB | 5 steps | 4096 tokens | 3 sims | No |
| **EVO-FULL** | 8+ GB | 10 steps | 8192 tokens | 10 sims | No |
| **EVO-HYBRID** | 8+ GB + GPU | 10 steps | 8192 tokens | 10 sims | Yes |

---

## API Examples

### WebSocket (Real-time)

```javascript
const ws = new WebSocket('ws://localhost:8420/ws/chat');

ws.send(JSON.stringify({
  type: 'chat',
  text: 'Explain quantum entanglement simply',
  session_id: 'my-session'
}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.event === 'response') {
    console.log(data.text);
  }
};
```

### REST

```bash
# Health check
curl http://localhost:8420/api/health

# Ask a question
curl -X POST http://localhost:8420/api/ask \
  -H "Content-Type: application/json" \
  -d '{"text": "What is machine learning?", "session_id": "demo"}'

# Memory stats
curl http://localhost:8420/api/memory/stats

# Runtime profile
curl http://localhost:8420/api/profile/current
```

---

## Project Structure

```
antonio-evo/
├── src/
│   ├── main.py                    # Orchestrator entry point
│   ├── api/                       # WebSocket + REST server
│   ├── handlers/                  # Mistral, Whisper, TTS, CLIP, External
│   ├── memory/                    # Neurons, storage, retrieval, emotional
│   ├── reasoning/                 # Cognitive budget, simulation, concept graph
│   ├── learning/                  # Proto-learning, neuron creator
│   ├── policy/                    # Deterministic routing engine
│   ├── runtime/                   # Hardware-aware profiles
│   ├── services/                  # Personality, proactive, digital twin, wisdom
│   ├── tools/                     # Tool registry + executor (v5.0)
│   ├── mcp/                       # MCP capabilities framework
│   └── sandbox/                   # Process isolation
├── ui/                            # React + Electron frontend
├── config/                        # JSON configuration files
├── prompts/                       # LLM system prompts
├── tests/                         # Unit, integration, 48h automated
├── data/                          # SQLite database + knowledge base
└── logs/                          # Audit trail (SHA-256 chain)
```

---

## Configuration

### Environment Variables (`.env`)

```bash
LLM_SERVER=http://localhost:11434      # Ollama endpoint
OLLAMA_MODEL=mistral                   # Local model
LLM_TIMEOUT=120                        # Timeout in seconds

ASR_SERVER=http://localhost:8803       # Whisper STT (optional)
TTS_SERVER=http://localhost:8804       # Piper TTS (optional)

EXTERNAL_API_KEY=                      # Claude/GPT fallback (optional)
```

### Policy Configuration (`config/policy.json`)

Controls routing, rate limits, sandboxing, audit, and external escalation. All decisions are code-driven.

---

## Philosophy

```
CODE DECIDES, MODELS DO NOT.
LOCAL FIRST, EXTERNAL LAST.
EXPLICIT OVER IMPLICIT.
BORING IS GOOD.
```

- **Policy engine (code)** decides routing — not LLMs
- **External APIs** are gated, justified, and cost-tracked
- **Every decision** is logged in a tamper-evident audit trail
- **Sandboxing** is mandatory, not optional
- **Attachments** are treated as UNTRUSTED INERT DATA — analyzed, never executed

---

## Contributing

We welcome contributions. Please read our guidelines:

1. **Code decides, models suggest** — Keep policy logic in code, not prompts
2. **Offline-first** — Features must work without internet
3. **Boring is good** — Prefer simple, auditable solutions over clever ones
4. **Every decision auditable** — All new features must integrate with the audit trail
5. **No silent side effects** — Any action must be explicit and logged

### How to contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes following the principles above
4. Add tests (see `tests/` directory)
5. Submit a Pull Request

See [ROADMAP.md](ROADMAP.md) for planned features and current status.

---

## Security Model

| Layer | Protection |
|-------|-----------|
| **Sandboxing** | All handlers run with CPU, memory, and timeout limits |
| **Policy Engine** | Deterministic code-based routing |
| **Audit Trail** | SHA-256 hash chain, tamper-evident |
| **External Gating** | API calls require policy approval + justification |
| **Attachments** | Treated as untrusted inert data |
| **Plugins** | Whitelist-only with manifest validation |

---

## License

[Business Source License 1.1](LICENSE) — Free for non-commercial, educational, and personal use. Converts to Apache 2.0 on 2029-02-15.

---

<p align="center">
  <strong>Built with the belief that your AI should be predictable, accountable, and yours.</strong>
</p>
