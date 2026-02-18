# Reddit Post — r/selfhosted

## Title
Antonio Evo: Self-hosted AI runtime with evolutionary memory, SHA-256 audit trails, and zero cloud dependencies

## Body

I built **Antonio Evo**, a self-hosted AI assistant that runs 100% on your hardware with no cloud dependencies whatsoever.

**Why this exists:** I wanted a local AI assistant that I could actually trust. Not "trust the provider's privacy policy" trust — but "I can read the code and verify nothing leaves my machine" trust.

**What it does:**
- Runs Mistral 7B locally via Ollama
- Deterministic policy engine — code decides routing, not the LLM
- Evolutionary memory with SQLite (neurons with confidence decay)
- SHA-256 tamper-evident audit trail for every decision
- Voice I/O with local Whisper (STT) and Piper (TTS)
- Desktop UI (React + Electron) with animated avatar
- Adapts to your hardware automatically (2GB to 32GB+)

**What it doesn't do:**
- No telemetry
- No phone-home
- No cloud dependencies (external API fallback is optional and policy-gated)
- No data leaves your machine unless you explicitly enable it

**Self-hosting details:**
- Backend: Python/FastAPI on port 8420
- Frontend: React/Vite on port 5173 (or Electron desktop app)
- Memory: SQLite file in `data/`
- Logs: JSONL audit trail in `logs/`
- Config: JSON files in `config/`

No Docker required (though a docker-compose is included). Works on macOS, Linux, and Windows.

**Quick start:**
```
git clone https://github.com/antconsales/antonio-evo
cd antonio-evo
./install.sh  # checks deps, installs everything, pulls Mistral
```

GitHub: https://github.com/antconsales/antonio-evo

Requirements: Python 3.11+, Node.js 18+, Ollama, ~6GB disk for Mistral model.

BSL 1.1 license (free for non-commercial use, converts to Apache 2.0 in 2029).
