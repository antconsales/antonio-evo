# Reddit Post — r/LocalLLaMA

## Title
I built a local cognitive runtime that treats LLMs as language generators, not decision-makers — deterministic routing, evolutionary memory, and SHA-256 audit trails

## Body

I've been working on **Antonio Evo**, an open-source local AI runtime that takes a fundamentally different approach from existing tools like Open WebUI, Jan.ai, or GPT4All.

**The core idea:** `CODE DECIDES, MODELS DO NOT.`

Most local AI tools let the LLM decide how to route requests, what to remember, and when to escalate. Antonio Evo inverts this: a deterministic policy engine (Python code) makes all routing decisions. The LLM (Mistral via Ollama) only generates language.

**What makes it different:**

- **Deterministic policy engine** — Code decides routing between SOCIAL (temp 0.7) and LOGIC (temp 0.3) personas. No prompt-based routing.
- **Evolutionary memory** — SQLite + FTS5 with "neurons" that have confidence scores and decay over time. BM25 retrieval. Learns across sessions without retraining.
- **Cognitive budget system** — Explicit bounds on reasoning depth, context tokens, and simulation budget. Adapts to your hardware (2GB laptop to full GPU).
- **Internal simulation engine** — Non-executing hypothetical reasoning with mandatory disclosure framing.
- **SHA-256 audit trail** — Every decision logged in a tamper-evident hash chain. You can verify integrity at any time.
- **Voice I/O** — Local Whisper STT + Piper TTS. All on-device.
- **Proto-learning** — Confidence adjustment, abstraction refinement, and failure-driven learning without model retraining.

**Tech stack:** Python (FastAPI) + React/Electron + Ollama + SQLite + Whisper + Piper

**100% offline-first.** No data leaves your machine unless you explicitly enable external API fallback (policy-gated, cost-tracked).

**Quick start:**
```
git clone https://github.com/antconsales/antonio-evo
cd antonio-evo && pip install -r requirements.txt
python -m src.api.websocket_server
```

GitHub: https://github.com/antconsales/antonio-evo

Looking for feedback on the architecture, especially the deterministic routing approach and the evolutionary memory system. Also interested in collaborators for Phase 12 (packaging: .dmg, .AppImage, .exe installers).

Licensed under BSL 1.1 (converts to Apache 2.0 in 2029).
