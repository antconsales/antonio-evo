# X/Twitter Thread

## Tweet 1 (Hook)
I built an AI runtime where CODE decides and MODELS don't.

No cloud. No data leaves your machine. Every decision logged in a SHA-256 hash chain.

Meet Antonio Evo — a local cognitive runtime with evolutionary memory.

🧵 Thread ↓

## Tweet 2 (Problem)
The problem with most local AI tools:

The LLM decides routing, memory, and escalation.

That makes your system:
- Unpredictable
- Unauditable
- Impossible to debug

Antonio Evo inverts this completely.

## Tweet 3 (Solution)
How it works:

→ Deterministic policy engine (Python) decides routing
→ Dual persona: SOCIAL (warm) / LOGIC (precise)
→ LLM only generates language
→ Every decision logged with SHA-256 hash chain

Code decides. Models suggest.

## Tweet 4 (Memory)
The memory system is wild:

→ "Neurons" with confidence scores
→ Confidence decays over time (like biological memory)
→ BM25 retrieval via SQLite + FTS5
→ Learns across sessions WITHOUT retraining
→ Failure-driven learning adjusts confidence

No vector DB needed.

## Tweet 5 (Hardware)
It adapts to your hardware automatically:

→ EVO-LITE (2GB): 3 reasoning steps, 2K context
→ EVO-STANDARD (4-8GB): 5 steps, 4K context
→ EVO-FULL (8GB+): 10 steps, 8K context
→ EVO-HYBRID: Full GPU + external API fallback

Same code. Different constraints.

## Tweet 6 (Stack)
Tech stack:
- Backend: Python + FastAPI
- Frontend: React + Electron
- LLM: Ollama + Mistral (local)
- Memory: SQLite + FTS5
- Voice: Whisper STT + Piper TTS
- Audit: SHA-256 hash chain

100% offline-first. Privacy by design.

## Tweet 7 (CTA)
Antonio Evo is open source (BSL 1.1 → Apache 2.0 in 2029).

Try it:
```
git clone https://github.com/antconsales/antonio-evo
pip install -r requirements.txt
python -m src.api.websocket_server
```

Star if you believe AI should be predictable and accountable.

GitHub: https://github.com/antconsales/antonio-evo
