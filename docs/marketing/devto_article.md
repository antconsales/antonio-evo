---
title: "Why I Built a Deterministic AI Runtime (And Why Your LLM Shouldn't Make Routing Decisions)"
published: false
description: "How Antonio Evo uses code-based policy engines instead of LLM-driven routing, with evolutionary memory and SHA-256 audit trails."
tags: ai, ollama, python, opensource
cover_image:
---

# Why I Built a Deterministic AI Runtime

## The Problem

Every local AI tool I tried had the same fundamental flaw: **the LLM decides everything**.

Want to route a request to a different handler? Ask the LLM. Want to decide if a task is complex enough for an external API? Ask the LLM. Want to determine which persona to use? Ask the LLM.

This makes the system:
- **Unpredictable** — The same input can produce different routing decisions
- **Unauditable** — You can't explain why a particular handler was chosen
- **Uncontrollable** — There's no way to enforce hard limits without prompt engineering

I wanted something different. Something where I could look at the code and know exactly what would happen for any given input.

## The Solution: CODE DECIDES, MODELS DO NOT

Antonio Evo is a **local cognitive runtime** built on a simple axiom:

```
CODE DECIDES, MODELS DO NOT.
```

The LLM (Mistral via Ollama) generates language. That's all it does. Everything else — routing, memory, escalation, auditing — is handled by deterministic Python code.

### How Routing Works

```
User Input
    → Normalizer (validate, sanitize)
    → Memory Retriever (BM25 context lookup)
    → Classifier (intent + domain + complexity)
    → Policy Engine (deterministic routing rules)
    → Router (dispatch to handler)
    → Handler (SOCIAL or LOGIC persona)
    → Audit Logger (SHA-256 hash chain)
```

The Policy Engine is just Python code. No prompts. No "system messages that guide routing." Code.

When it decides between the SOCIAL persona (temperature 0.7, warm and conversational) and the LOGIC persona (temperature 0.3, precise and analytical), it uses the Classifier's output — which is itself deterministic.

### Evolutionary Memory

Instead of vector embeddings and RAG, Antonio Evo uses **neurons** — memory units with:

- **Confidence scores** (0.0 to 1.0)
- **Decay over time** (like biological memory)
- **Access counting** (frequently accessed memories stay strong)
- **BM25 retrieval** via SQLite + FTS5

This means:
- Memories that aren't accessed gradually fade
- Each access reinforces the memory (increases confidence)
- The system learns from failures by adjusting confidence downward
- No model retraining required

It's not true learning in the ML sense. It's **proto-learning** — the system adjusts its internal representations based on feedback, without modifying model weights.

### Cognitive Budget System

One of the most interesting features is the **cognitive budget**. Each request runs under explicit constraints:

| Profile | Reasoning Depth | Context | Simulation | External |
|---------|----------------|---------|------------|----------|
| EVO-LITE | 3 steps | 2048 tokens | Off | No |
| EVO-STANDARD | 5 steps | 4096 tokens | 3 sims | No |
| EVO-FULL | 10 steps | 8192 tokens | 10 sims | No |
| EVO-HYBRID | 10 steps | 8192 tokens | 10 sims | Yes |

The profile is selected automatically based on your hardware. A 2GB Raspberry Pi gets EVO-LITE. A 32GB workstation gets EVO-FULL. Same code, different constraints.

### SHA-256 Audit Trail

Every decision is logged in a **tamper-evident hash chain**:

```json
{
  "timestamp": "2026-02-18T10:00:00Z",
  "action": "route_to_handler",
  "handler": "SOCIAL",
  "reason": "intent=greeting, complexity=low",
  "hash": "a1b2c3...",
  "prev_hash": "d4e5f6..."
}
```

You can verify the entire chain at any time. If anyone (or anything) tampers with a log entry, the chain breaks.

## The Stack

- **Backend**: Python + FastAPI (async WebSocket + REST)
- **Frontend**: React + Electron (desktop app with animated avatar)
- **LLM**: Ollama + Mistral 7B (fully local)
- **Memory**: SQLite + FTS5 (no vector DB needed)
- **Voice**: Whisper STT + Piper TTS (all on-device)
- **Audit**: SHA-256 hash chain in JSONL format

Everything runs locally. No cloud dependencies. No telemetry. No data exfiltration.

## What I Learned

1. **Deterministic routing is simpler than it sounds.** A few hundred lines of Python policy code replaced thousands of lines of prompt engineering.

2. **BM25 is underrated.** For conversational memory retrieval, BM25 with FTS5 is fast, predictable, and doesn't require a vector database or embedding model.

3. **Cognitive budgets change how you think about AI.** When you explicitly bound reasoning depth, you stop asking "can the model do this?" and start asking "should the model try this given the constraints?"

4. **Audit trails should be mandatory.** Once you have a tamper-evident log of every decision, debugging becomes trivial.

## Try It

```bash
git clone https://github.com/antconsales/antonio-evo
cd antonio-evo
pip install -r requirements.txt
cd ui && npm install && cd ..
python -m src.api.websocket_server  # Terminal 1
cd ui && npm run dev                # Terminal 2
```

Open http://localhost:5173

## Links

- **GitHub**: [https://github.com/antconsales/antonio-evo](https://github.com/antconsales/antonio-evo)
- **License**: BSL 1.1 (free for non-commercial use, converts to Apache 2.0 in 2029)

I'm looking for feedback on the architecture and potential collaborators. If you believe AI should be predictable and accountable, give it a try.
