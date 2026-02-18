# Hacker News — Show HN

## Title
Show HN: Antonio Evo – Local AI runtime where code decides, models don't

## URL
https://github.com/antconsales/antonio-evo

## Comment (post as first comment)

Hi HN,

I built Antonio Evo because I was frustrated with how existing local AI tools delegate critical decisions to LLMs. In most setups, the model decides routing, escalation, and memory — making the system unpredictable and unauditable.

Antonio Evo inverts this: a deterministic policy engine (plain Python) makes all routing decisions. The LLM (Mistral via Ollama) only generates language. Every decision is logged in a SHA-256 hash chain you can verify.

Key design decisions:

1. **Code decides, models do not** — The policy engine routes between SOCIAL (temp 0.7) and LOGIC (temp 0.3) personas based on intent classification. The LLM never sees routing logic.

2. **Evolutionary memory** — Instead of vector embeddings, I use SQLite + FTS5 with BM25 scoring. "Neurons" have confidence scores that decay over time. The system learns across sessions without retraining.

3. **Cognitive budgets** — Each hardware profile (EVO-LITE through EVO-HYBRID) has explicit bounds on reasoning depth, context tokens, and simulation count. A 2GB Raspberry Pi gets different constraints than a 32GB workstation.

4. **Proto-learning** — The system adjusts confidence, refines abstractions, and learns from failures — all without touching model weights. Learning boundaries are immutable.

5. **Audit trail** — SHA-256 hash chain. Tamper-evident. Every decision, every routing choice, every external API call is logged with justification.

Tech: Python/FastAPI backend, React/Electron frontend, Ollama for inference, SQLite for memory, Whisper for STT, Piper for TTS. All local.

I'm a solo developer working on this as a side project. The code isn't perfect, but the architecture is intentional. I'd love feedback on the deterministic routing approach and whether the evolutionary memory model makes sense.

GitHub: https://github.com/antconsales/antonio-evo
