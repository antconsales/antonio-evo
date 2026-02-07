# Antonio Evo - API UI Contracts

Complete mapping of UI requirements to API endpoints for frontend integration.

## Global State Endpoint (Recommended for Initial Load)

```
GET /api/ui/state
```

Returns complete UI state in one call:

```json
{
  "success": true,
  "state": {
    "assistant_name": "Antonio Evo",
    "version": "2.0-evo",
    "session_id": "abc123",
    "profile": {
      "name": "evo-standard",
      "is_local_first": true
    },
    "connection": {
      "local_llm": true,
      "rag": false,
      "memory": true
    },
    "features": {
      "emotional_memory": true,
      "proactive_mode": true,
      "personality_evolution": true,
      "digital_twin": false
    },
    "pending": {
      "tasks": 0,
      "insights": 2
    },
    "consent": {
      "external_llm": {"allowed": false, "scope": "none"},
      "web_search": {"allowed": false, "scope": "none"},
      "data_collection": {"allowed": true, "scope": "always"}
    }
  }
}
```

---

## Top Bar / Status Indicators

### Runtime Profile

```
GET /api/profile
```

Response:
```json
{
  "active_profile": "evo-standard",
  "hardware": {
    "total_ram_gb": 32.0,
    "cpu_cores": 12,
    "has_gpu": true,
    "platform": "Windows"
  },
  "capabilities": {
    "local_llm_enabled": true,
    "external_llm_fallback": true,
    "image_generation": true,
    "proactive_mode": true
  }
}
```

### LLM Status

```
GET /api/llm/status
```

Response:
```json
{
  "enabled": true,
  "initialized": true,
  "endpoints": {
    "ollama:primary": {
      "status": "available",
      "model": "qwen2.5:14b",
      "is_local": true
    }
  }
}
```

---

## Chat Interface

### Send Message

```
POST /api/ask
Content-Type: application/json

{
  "text": "Hello Antonio",
  "session_id": "optional-session-id"
}
```

Response includes metadata for UI:
```json
{
  "success": true,
  "text": "Ciao! Come posso aiutarti?",
  "persona": "social",
  "mood": "friendly",
  "neuron_stored": true,
  "elapsed_ms": 1234
}
```

### WebSocket Real-time

```
ws://localhost:8420/ws/chat
```

Events:
- `connected` - Session started
- `thinking_start` - Processing began
- `thinking_end` - Processing complete
- `response` - Full response with metadata
- `mood_change` - Avatar mood update

---

## Task Cards (Approval System)

### Get Pending Tasks

```
GET /api/tasks/pending?limit=20
```

Response:
```json
{
  "success": true,
  "tasks": [
    {
      "id": "task-abc123",
      "task_type": "external_llm_query",
      "params": {"prompt": "..."},
      "status": "pending",
      "approval_level": "explicit",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### Approve/Reject Task

```
POST /api/tasks/approve
Content-Type: application/json

{
  "task_id": "task-abc123",
  "approved": true
}
```

### Task History

```
GET /api/tasks/history?limit=50
```

---

## Consent Management

### Get Consent State

```
GET /api/consent
```

Response:
```json
{
  "success": true,
  "consent": {
    "external_llm": {
      "allowed": false,
      "scope": "none"
    },
    "web_search": {
      "allowed": false,
      "scope": "none"
    },
    "data_collection": {
      "allowed": true,
      "scope": "always"
    }
  },
  "profile_allows_external": true
}
```

### Update Consent

```
POST /api/consent/update
Content-Type: application/json

{
  "category": "external_llm",
  "allowed": true,
  "scope": "session"
}
```

Scope values: `none`, `one-time`, `session`, `always`

### Check Consent Before Action

```
GET /api/consent/check/external_llm
```

Response:
```json
{
  "allowed": false,
  "scope": "none",
  "requires_prompt": false
}
```

---

## Memory Visibility

### Memory Stats

```
GET /api/memory/stats
```

Response:
```json
{
  "enabled": true,
  "total_neurons": 1234,
  "avg_confidence": 0.85,
  "total_accesses": 5678
}
```

### Search Memory

```
POST /api/memory/search
Content-Type: application/json

{
  "query": "search term",
  "limit": 5
}
```

---

## Emotional Context

### Current Stats

```
GET /api/emotional/stats
```

Response:
```json
{
  "enabled": true,
  "version": "2.1",
  "total_signals": 150,
  "weekly_distribution": {
    "neutral": 80,
    "happy": 30,
    "curious": 25
  }
}
```

### Session Emotions

```
GET /api/emotional/session/{session_id}
```

---

## Personality Traits

### Current Profile

```
GET /api/personality/profile
```

Response:
```json
{
  "success": true,
  "profile": {
    "traits": {
      "humor": 55,
      "formality": 45,
      "verbosity": 50,
      "empathy": 65,
      "curiosity": 70
    }
  }
}
```

### Provide Feedback

```
POST /api/personality/feedback
Content-Type: application/json

{
  "feedback_type": "positive"
}
```

Feedback types: `positive`, `negative`, `too_long`, `too_short`, `humor_good`

---

## Digital Twin

### Status

```
GET /api/twin/status
```

Response:
```json
{
  "enabled": true,
  "ready": false,
  "messages_analyzed": 25,
  "messages_needed": 50,
  "readiness": 0.5
}
```

### Reset Twin

```
POST /api/twin/reset
```

---

## Audit Trail

### Recent Entries

```
GET /api/audit/recent?limit=20
```

Response:
```json
{
  "success": true,
  "entries": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "request_text": "Hello...",
      "handler": "mistral",
      "persona": "social",
      "success": true,
      "elapsed_ms": 1234,
      "memory_stored": true,
      "external_call": false
    }
  ]
}
```

### External Calls Only

```
GET /api/audit/external?limit=50
```

---

## Health & Diagnostics

### Full Health Check

```
GET /api/health
```

Response includes all subsystem status:
```json
{
  "status": "ok",
  "version": "2.0-evo",
  "profile": {...},
  "memory": {...},
  "emotional_memory": {...},
  "proactive": {...},
  "personality": {...},
  "digital_twin": {...},
  "llm_manager": {...},
  "rag": {...}
}
```

---

## Image Generation (Z-Image)

```
POST /api/generate
Content-Type: application/json

{
  "prompt": "A sunset over mountains",
  "width": 512,
  "height": 512
}
```

Response:
```json
{
  "success": true,
  "image_path": "output/images/abc123.png",
  "image_url": "/images/abc123.png",
  "generation_time": 45.2
}
```

---

## UI Implementation Notes

1. **Top Bar**: Use `/api/ui/state` for initial load, then individual endpoints for updates
2. **Task Cards**: Poll `/api/tasks/pending` or use WebSocket events
3. **Consent Dialogs**: Always call `/api/consent/check/{category}` before external actions
4. **Audit Panel**: Use `/api/audit/recent` with pagination
5. **Memory Indicators**: Cache `/api/memory/stats` and refresh on neuron_stored events
6. **Profile Badge**: Update on `/api/profile/switch` success
