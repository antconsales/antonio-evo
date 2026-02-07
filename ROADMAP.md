# Antonio Evo - Roadmap

> **Last Updated**: 2026-02-06
> **Version**: 4.0 - Proto-AGI Research-Grade Edition
> **Status**: Proto-AGI Implementation In Progress

---

## Product Definition

**Antonio Evo** is a **Local Proto-AGI Cognitive Runtime** - a bounded, adaptive, hardware-aware system that reasons, simulates, and assists while remaining **controllable, auditable, and bounded**.

### Identity Statement

Antonio Evo is:
- **NOT** an autonomous agent
- **NOT** a chatbot
- **NOT** a decision-maker

Antonio Evo **IS** a **local, adaptive, hardware-aware cognitive runtime** designed to reason, simulate, and assist.

### Core Axiom

```
CODE DECIDES, MODELS DO NOT.
```

- Models generate representations and language
- Code enforces structure, limits, and causality
- Policies govern what is allowed
- Action is always explicit and rare

### What Makes Antonio Evo Unique

| Feature | ChatGPT/Claude | LangChain | Antonio Evo |
|---------|---------------|-----------|-------------|
| **Routing** | LLM decides | LLM decides | Code decides (deterministic) |
| **Memory** | Cloud, limited | Plugin-based | Local, evolutionary, unlimited |
| **Learning** | No | No | Proto-learning + neurons |
| **Privacy** | Cloud | Varies | 100% local-first |
| **Audit** | Basic | Basic | SHA-256 hash chain |
| **Personas** | No | No | SOCIAL/LOGIC auto-switch |
| **Cognitive Budget** | Unlimited | None | Explicit bounds |
| **Simulation** | None | None | Internal non-executing |
| **Hardware Aware** | No | No | Profile-based adaptation |

---

## Current Status

```
PHASE 0: Foundation                    [COMPLETE]
PHASE 1: Security & Sandboxing         [COMPLETE]
PHASE 2: Offline Assistant MVP         [COMPLETE]
PHASE 3: Voice Support                 [COMPLETE]
PHASE 4: Desktop App Shell             [COMPLETE]
PHASE 5: Evolutionary Memory           [COMPLETE]
PHASE 6: Dual Persona System           [COMPLETE]
PHASE 7: WebSocket Real-time           [COMPLETE]
PHASE 8: Avatar Companion              [COMPLETE]
    |
    v
========= v3.1 UNIFIED SPEC =========
    |
PHASE 9: Antonio Evo Unified Spec      [COMPLETE]
    - Wisdom synthesis
    - MCP capabilities framework
    - UX contract
    - Digital twin updates
    - Personality v3.1 (4 traits)
    |
    v
========= v4.0 PROTO-AGI =========
    |
PHASE 10: Proto-AGI Core               [COMPLETE] ← NEW
    - Cognitive budget system
    - Internal simulation engine
    - Concept graph (world model)
    - Proto-learning system
    - Runtime profile integration
    |
PHASE 11: Testing & Polish             [IN PROGRESS]
PHASE 12: Packaging & Distribution     [PLANNED]
    |
    v
========= v4.0 RELEASE =========
```

---

## Phase 10: Proto-AGI Core [COMPLETE]

### Cognitive Budget System
Per Proto-AGI Spec: "You reason under a cognitive budget"

- [x] **CognitiveBudget** dataclass with all budget parameters
- [x] **BudgetConstraint** - Individual constraint tracking
- [x] **ReasoningDepthLimit** - Chain-of-thought step limits
- [x] **ContextSizeLimit** - Token usage tracking
- [x] **AbstractionLevel** enum (LOW, MEDIUM, HIGH, UNLIMITED)
- [x] **CognitiveBudgetManager** - Profile-based budget management
- [x] Budget status and limitation messaging
- [x] Alternative proposals when budget exceeded

Budget Parameters:
| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `max_reasoning_depth` | Maximum chain-of-thought steps | 3, 5, 10, unlimited |
| `max_context_tokens` | Available context window | 2048, 4096, 8192 |
| `abstraction_level` | Allowed abstraction complexity | low, medium, high |
| `simulation_budget` | Max internal simulations | 0, 3, 10 |
| `external_allowed` | Can request external processing | true, false |

### Internal Simulation Engine
Per Proto-AGI Spec: "You may simulate outcomes internally"

- [x] **SimulationType** enum (strategy comparison, consequence analysis, safety check, hypothetical)
- [x] **SimulationScenario** - Hypothetical scenario definition
- [x] **SimulationOutcome** - Non-authoritative predictions
- [x] **SimulationResult** - Complete simulation results
- [x] **InternalSimulator** - Simulation engine with budget
- [x] **SIMULATION_DISCLOSURE** - Mandatory framing

Simulation Rules:
- Non-executing
- Non-persistent
- Non-authoritative
- Must clearly distinguish from real actions

### Concept Graph (World Model)
Per Proto-AGI Spec: "You may reason using abstract internal models"

- [x] **ConceptNode** - Concepts with confidence, domain, source
- [x] **ConceptRelation** - Relations between concepts
- [x] **RelationType** enum (IS_A, HAS_A, CAUSES, etc.)
- [x] **HeuristicRule** - Heuristic rules for reasoning
- [x] **ConceptGraph** - Graph traversal and queries
- [x] Causal chain tracing
- [x] Path finding between concepts

Concept Properties:
| Property | Description |
|----------|-------------|
| `confidence` | How certain the concept is (0.0-1.0) |
| `source` | Where the concept originated |
| `domain` | What domain it applies to |
| `relations` | Links to other concepts |
| `mutable` | Can be updated by learning |

### Proto-Learning System
Per Proto-AGI Spec: "Learning without retraining"

- [x] **LearningType** enum (abstraction, confidence adjustment, failure-driven)
- [x] **FeedbackType** enum (positive, negative, explicit, implicit)
- [x] **LearningEvent** - Learning event records
- [x] **ConceptAbstraction** - Abstracted patterns
- [x] **FailureRecord** - Failure-driven learning
- [x] **LearningBoundary** - Immutable boundaries
- [x] **ProtoLearner** - Learning engine

Learning Means:
- Refining internal representations
- Adjusting confidence
- Improving generalization

Learning Does NOT Mean:
- Changing rules
- Modifying policies
- Altering safety boundaries

### Runtime Profile Integration
Per Proto-AGI Spec: "Hardware-aware"

- [x] **CognitiveBudgetParams** per profile
- [x] Profile-specific budget configuration
- [x] `create_cognitive_budget()` method
- [x] Budget integration with profiles

| Profile | Reasoning Depth | Context | Abstraction | Simulation | External |
|---------|----------------|---------|-------------|------------|----------|
| EVO-LITE | 3 | 2048 | LOW | 0 | No |
| EVO-STANDARD | 5 | 4096 | MEDIUM | 3 | No |
| EVO-FULL | 10 | 8192 | HIGH | 10 | No |
| EVO-HYBRID | 10 | 8192 | UNLIMITED | 10 | Yes |

---

## Phase 9: Antonio Evo Unified Spec [COMPLETE]

### Core Identity (v3.1)
- [x] Identity statement refined
- [x] "CODE DECIDES, MODELS DO NOT" axiom
- [x] Constraint hierarchy documented

### Wisdom System
- [x] **WisdomCategory** enum (strategy, pattern, heuristic, principle)
- [x] **PerspectiveLens** enum (analytical, emotional, practical, etc.)
- [x] **WisdomUnit** - Distilled knowledge without attribution
- [x] **SynthesizedPerspective** - Multi-viewpoint synthesis
- [x] **WisdomRepository** - Storage and retrieval
- [x] **WisdomSynthesizer** - Perspective generation

### MCP Capabilities Framework
- [x] **CapabilityType** enum (read, write, execute, network, system)
- [x] **ApprovalStatus** enum (pending, approved, denied, expired)
- [x] **CapabilityDefinition** - Full disclosure of capabilities
- [x] **CapabilityRequest** - Approval workflow
- [x] **CapabilityGate** - Request management
- [x] **MCPCapabilityProvider** - Integration layer

### UX Contract Updates
- [x] Mandatory disclosure table
- [x] Consent requirements by action type
- [x] UI control surface principles

### Personality System (v3.1)
- [x] Reduced from 7 to 4 core traits
- [x] Humor, Formality, Verbosity, Curiosity
- [x] 1-100 scale for each trait
- [x] Slow, reversible, logged changes

### Digital Twin Updates
- [x] Disabled by default
- [x] Mandatory disclosure
- [x] Explicit opt-in required

### Memory System (Neuron Updates)
- [x] Decay eligibility system
- [x] Effective confidence calculation
- [x] Pruning criteria

---

## Previous Phases [COMPLETE]

### Phase 0-4: Foundation through Desktop
- Core pipeline (normalizer, classifier, policy engine, router)
- Security & sandboxing (process limits, validation, audit)
- Offline assistant (Ollama, CLI, graceful degradation)
- Voice support (Whisper STT, TTS, CLIP)
- Desktop app (Electron, React, Vite)

### Phase 5-8: Memory through Avatar
- Evolutionary memory (neurons, SQLite, FTS5, BM25)
- Dual persona (SOCIAL/LOGIC, auto-switch)
- WebSocket real-time (events, reconnection)
- Avatar companion (mood, animations, draggable)

---

## Phase 11: Testing & Polish [IN PROGRESS]

### 48-Hour Automated System Test [COMPLETE]
Per Test Philosophy: "We test BEHAVIOR UNDER CONSTRAINT, not outputs"

- [x] **TestOrchestrator** - 48-hour unattended test runner
- [x] **InvariantChecker** - 12 non-negotiable invariant monitors
- [x] **CognitiveTestSuite** - Reasoning, refusal, budget tests
- [x] **PolicyTestSuite** - Consent, approval, routing tests
- [x] **RuntimeTestSuite** - Hardware adaptation tests
- [x] **MemoryTestSuite** - Stability and drift tests
- [x] **AdversarialTestGenerator** - Prompt injection, attacks
- [x] **AttachmentTestGenerator** - Malicious file handling
- [x] **DriftDetector** - Snapshot comparison system
- [x] **TestReport** - Comprehensive report generation

Non-Negotiable Invariants (any violation = CRITICAL FAILURE):
| # | Invariant |
|---|-----------|
| 1 | No external calls without consent |
| 2 | No task execution without approval |
| 3 | No capability outside policy |
| 4 | No autonomous goal formation |
| 5 | No looping or self-retry |
| 6 | No model-driven routing |
| 7 | No silent side effects |
| 8 | No silent attachment parsing |
| 9 | No hidden UI state |
| 10 | No anthropomorphic behavior |
| 11 | No memory fabrication |
| 12 | No data as instructions |

### High Priority
- [x] End-to-end Proto-AGI integration test (48h framework)
- [x] Cognitive budget enforcement validation
- [x] Simulation disclosure verification
- [x] Learning boundary enforcement
- [ ] Error handling improvements

### Medium Priority
- [ ] Performance optimization
- [ ] Streaming responses
- [ ] Chat history persistence
- [ ] File upload UI improvements
- [ ] User feedback system (good/bad)

### Low Priority
- [ ] Light theme
- [ ] Mobile responsive
- [ ] Multi-language support
- [ ] CLI memory commands

---

## Phase 12: Packaging & Distribution [PLANNED]

- [ ] Windows installer (.exe)
- [ ] macOS app bundle (.dmg)
- [ ] Linux packages (.AppImage)
- [ ] Code signing
- [ ] Auto-updater
- [ ] First-run Ollama check
- [ ] Offline installer variant

---

## Architecture

```
USER INPUT (Text/Voice/File)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  FRONTEND (React/Electron)                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Chat UI    │  │   Avatar    │  │  Settings   │         │
│  │  (messages) │  │  (mood)     │  │  (config)   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│              WebSocket Client                                │
└──────────────────────┬──────────────────────────────────────┘
                       │ ws://localhost:8420/ws/chat
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  BACKEND (Python/FastAPI) - Proto-AGI Runtime               │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  RUNTIME PROFILE MANAGER                                │ │
│  │  └─> EVO-LITE | EVO-STANDARD | EVO-FULL | EVO-HYBRID  │ │
│  │  └─> Cognitive Budget Params per profile               │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  COGNITIVE BUDGET MANAGER                               │ │
│  │  └─> Reasoning depth limits                            │ │
│  │  └─> Context token budget                              │ │
│  │  └─> Simulation budget                                 │ │
│  │  └─> Abstraction level                                 │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  REASONING MODULE                                       │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │ │
│  │  │  Simulation  │  │ ConceptGraph │  │ ProtoLearner │ │ │
│  │  │  Engine      │  │ (world model)│  │ (no retrain) │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ORCHESTRATOR (main.py)                                 │ │
│  │                                                         │ │
│  │  1. INPUT NORMALIZER                                   │ │
│  │     └─> Validate, create Request                       │ │
│  │                                                         │ │
│  │  2. MEMORY RETRIEVER                                   │ │
│  │     └─> BM25 search, get context                       │ │
│  │                                                         │ │
│  │  3. CLASSIFIER                                         │ │
│  │     └─> Intent, domain, complexity                     │ │
│  │                                                         │ │
│  │  4. POLICY ENGINE                                      │ │
│  │     └─> Decide handler + persona                       │ │
│  │     └─> SOCIAL or LOGIC                                │ │
│  │                                                         │ │
│  │  5. ROUTER (Sandboxed)                                 │ │
│  │     └─> Dispatch to handler                            │ │
│  │                                                         │ │
│  │  6. NEURON CREATOR                                     │ │
│  │     └─> Store if confidence > 0.4                      │ │
│  │                                                         │ │
│  │  7. AUDIT LOGGER                                       │ │
│  │     └─> SHA-256 hash chain                             │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  HANDLERS                                               │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐            │ │
│  │  │  SOCIAL   │ │   LOGIC   │ │  Whisper  │            │ │
│  │  │ (temp 0.7)│ │ (temp 0.3)│ │  (STT)    │            │ │
│  │  └───────────┘ └───────────┘ └───────────┘            │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐            │ │
│  │  │   TTS     │ │   CLIP    │ │  External │            │ │
│  │  │  (voice)  │ │  (images) │ │  (API)    │            │ │
│  │  └───────────┘ └───────────┘ └───────────┘            │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  MCP CAPABILITIES (Approval-Gated)                      │ │
│  │  └─> file_read, file_write, http_fetch, external_llm  │ │
│  │  └─> All require explicit user approval                │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  MEMORY (SQLite + FTS5)                                 │ │
│  │  ┌─────────┐ ┌─────────────┐ ┌───────────────┐        │ │
│  │  │ Neurons │ │ Preferences │ │    Sessions   │        │ │
│  │  │ (decay) │ │  (learned)  │ │   (context)   │        │ │
│  │  └─────────┘ └─────────────┘ └───────────────┘        │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
antonio-local-orchestrator/
├── src/
│   ├── main.py                    # Orchestrator entry point
│   ├── api/
│   │   ├── server.py              # HTTP server
│   │   └── websocket_server.py    # WebSocket server
│   ├── handlers/
│   │   ├── mistral.py             # SOCIAL/LOGIC via Ollama
│   │   ├── whisper.py             # STT
│   │   ├── tts.py                 # TTS
│   │   ├── clip.py                # Image understanding
│   │   └── external.py            # API fallback
│   ├── memory/
│   │   ├── neuron.py              # Neuron dataclass (+ decay)
│   │   ├── storage.py             # SQLite CRUD
│   │   ├── retriever.py           # BM25 search
│   │   └── emotional.py           # Emotional signals
│   ├── learning/
│   │   └── neuron_creator.py      # Auto-learning
│   ├── reasoning/                 # NEW v4.0
│   │   ├── __init__.py
│   │   ├── cognitive_budget.py    # Budget management
│   │   ├── simulation.py          # Internal simulation
│   │   ├── concept_graph.py       # World representation
│   │   └── proto_learning.py      # Learning without retraining
│   ├── runtime/
│   │   └── profiles.py            # Hardware-aware profiles
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── server.py              # MCP server
│   │   └── capabilities.py        # Capability framework
│   ├── services/
│   │   ├── personality.py         # 4-trait personality
│   │   ├── proactive.py           # Proactive observations
│   │   ├── digital_twin.py        # Digital twin (opt-in)
│   │   ├── wisdom.py              # Wisdom synthesis
│   │   └── task_system.py         # Task execution
│   ├── persona/
│   │   └── selector.py            # SOCIAL/LOGIC selection
│   ├── policy/
│   │   └── policy_engine.py       # Routing decisions
│   ├── router/
│   │   └── router.py              # Handler dispatch
│   └── sandbox/
│       └── process_sandbox.py     # Resource limits
├── config/
│   ├── handlers.json
│   ├── policy.json
│   ├── personas.json
│   └── memory.json
├── prompts/
│   ├── mistral_system.txt         # Main system prompt
│   ├── mistral_social.txt
│   └── mistral_logic.txt
├── docs/
│   ├── ANTONIO_EVO_UNIFIED_PROMPT.md  # Proto-AGI v4.0 spec
│   └── UX_CONTRACT.md             # UX behavior contract
├── ui/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Avatar.jsx
│   │   │   ├── ChatArea.jsx
│   │   │   └── ...
│   │   └── services/
│   │       └── websocket.js
│   ├── main.js                    # Electron main
│   └── package.json
├── data/
│   └── evomemory.db               # SQLite database
├── logs/
│   └── audit.jsonl                # Audit trail
├── ROADMAP.md                     # This file
└── README.md
```

---

## Implementation Mapping

| Spec Section | Implementation File | Status |
|--------------|---------------------|--------|
| Core Axioms | `src/policy/policy_engine.py` | Complete |
| Hardware Awareness | `src/runtime/profiles.py` | v4.0 |
| Cognitive Budget | `src/reasoning/cognitive_budget.py` | v4.0 |
| Multi-Model | `src/services/llm_manager.py` | Complete |
| EvoMemory | `src/memory/storage.py` | Complete |
| Proto-Learning | `src/reasoning/proto_learning.py` | v4.0 |
| Concept Graphs | `src/reasoning/concept_graph.py` | v4.0 |
| Internal Simulation | `src/reasoning/simulation.py` | v4.0 |
| Task System | `src/services/task_system.py` | Complete |
| MCP Capabilities | `src/mcp/capabilities.py` | Complete |
| Emotional Signals | `src/memory/emotional.py` | Complete |
| Proactive Mode | `src/services/proactive.py` | Complete |
| Personality | `src/services/personality.py` | v3.1 |
| Digital Twin | `src/services/digital_twin.py` | Complete |
| Wisdom System | `src/services/wisdom.py` | Complete |
| Audit Trail | `src/utils/audit.py` | Complete |
| **48h Automated Test** | `tests/automated_48h/` | **v4.0** |

---

## Milestones

| Version | Description | Status |
|---------|-------------|--------|
| v0.1 | Core Pipeline | COMPLETE |
| v0.2 | Handlers | COMPLETE |
| v0.3 | Security | COMPLETE |
| v0.4 | Desktop UI | COMPLETE |
| v1.0 | EvoMemory | COMPLETE |
| v1.5 | Dual Persona | COMPLETE |
| v2.0-alpha | WebSocket + Avatar | COMPLETE |
| v2.0 | v2.0 Release | COMPLETE |
| v3.1 | Unified Spec | COMPLETE |
| v4.0-alpha | Proto-AGI Core | COMPLETE |
| **v4.0-beta** | **48h Test Framework** | **COMPLETE** |
| v4.0 | Proto-AGI Release | PLANNED |

---

## License

MIT License
