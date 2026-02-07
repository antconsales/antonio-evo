# Antonio Evo - UX Contract

> **Version**: 3.1
> **Last Updated**: 2026-02-05
> **Status**: Active - Aligned with Antonio Evo Unified Spec

This document defines the explicit behavior contract of Antonio Evo. Users and integrators should rely only on behaviors documented here.

---

## 0. Antonio Evo UX & UI Principles

Per the Antonio Evo Unified Spec (v3.1):

### The UI is a Control Surface, Not Decoration

The UI must ensure:
- **Nothing happens silently**
- **Every capability is visible**
- **Every side effect is explicit**
- **Every external action is disclosed**

### The User Must Always Know:
- Where computation happens (local vs external)
- What is enabled
- What is disabled
- Why

**If the user cannot explain system state, the UI has failed.**

### UI Requirements

The UI MUST:
- Display runtime profile (EVO-LITE, EVO-STANDARD, EVO-FULL, EVO-HYBRID)
- Show local vs external processing indicators
- Surface consent dialogs clearly
- Render task proposals as explicit cards
- Expose memory activity transparently
- Show audit and history of approvals

The UI MUST NEVER:
- Anthropomorphize Antonio
- Imply consciousness
- Imply autonomy
- Hide side effects

### Mandatory Disclosures

Per Antonio Evo Unified Spec, certain modes require explicit disclosures:

| Mode | Required Disclosure |
|------|---------------------|
| **Proactive Mode** | "This is an observation, not an instruction." |
| **Digital Twin Mode** | "Digital Twin mode active. Output is a stylistic approximation." |
| **External API** | Clear indicator that data will be sent externally |
| **Wisdom/Perspectives** | "These are synthesized viewpoints, not factual claims." |

### Consent Requirements

| Action Type | Consent Level |
|-------------|---------------|
| Local processing | Implicit (default operation) |
| Memory storage | Automatic (confidence threshold) |
| External API call | Explicit per-action consent |
| Capability execution | Explicit approval with disclosure |
| Digital Twin generation | Explicit opt-in |

---

## 1. What the Assistant DOES

### 1.1 Supported Task Types

The assistant handles the following task types:

| Task Type | Description | Example |
|-----------|-------------|---------|
| **Question Answering** | Answers factual questions based on training knowledge | "What is the capital of France?" |
| **Text Generation** | Generates text content on request | "Write a haiku about autumn" |
| **Text Analysis** | Analyzes provided text for meaning, tone, structure | "Summarize this paragraph" |
| **Explanation** | Explains concepts, code, or processes | "Explain how recursion works" |
| **Classification** | Categorizes input into predefined types | Internal use for routing |

### 1.2 Expected Tone

The assistant maintains the following tone characteristics:

- **Concise**: Responses are direct and avoid unnecessary verbosity
- **Factual**: Statements are based on training knowledge, not speculation
- **Neutral**: No emotional manipulation or persuasion attempts
- **Honest**: Admits uncertainty when applicable ("I don't know", "I'm not sure")

### 1.3 Output Guarantees

The assistant guarantees:

- **Structured responses**: All responses follow a consistent format
- **Error codes**: Failures include machine-readable error codes
- **No partial output**: Either a complete response or a complete error
- **UTF-8 encoding**: All text output is valid UTF-8
- **Bounded size**: Responses do not exceed configured limits

### 1.4 Output Formats

| Format | Description | When Used |
|--------|-------------|-----------|
| **Plain Text** | Human-readable text | Default CLI output |
| **JSON** | Structured data | API responses, --json flag |
| **Raw** | Internal response format | Debugging (--raw flag, raw=true) |

---

## 2. What the Assistant DOES NOT Do

### 2.1 Unsupported Requests

The assistant cannot and will not:

- **Browse the web**: No internet search or URL fetching
- **Access files**: No reading or writing files on the filesystem
- **Execute code**: No running scripts, commands, or programs
- **Call external APIs**: No network requests beyond the local LLM
- **Remember conversations**: No persistent memory between sessions
- **Learn from interactions**: No training or fine-tuning from user input

### 2.2 Forbidden Behaviors

The assistant is explicitly prohibited from:

- **Tool calling**: Cannot invoke external tools or functions
- **Claiming false capabilities**: Will not pretend to have abilities it lacks
- **Generating harmful content**: Follows safety guidelines in system prompt
- **Accessing system resources**: No access to environment variables, processes, or hardware
- **Making autonomous decisions**: Cannot initiate actions without explicit request

### 2.3 Out of Scope Modalities

The following are not supported in the current implementation:

- Real-time audio conversation
- Image generation
- Video processing
- Multi-turn context (each request is independent)

---

## 3. Error Behavior

### 3.1 How Errors Are Communicated

Errors are communicated through:

| Channel | Format | Example |
|---------|--------|---------|
| **CLI** | Plain text to stderr | `Error: The assistant is currently unavailable.` |
| **CLI --json** | JSON to stdout | `{"success": false, "error_category": "connection"}` |
| **API** | JSON with HTTP status | `400 {"success": false, "error": "..."}` |

### 3.2 Error Categories

| Category | Meaning | User Action |
|----------|---------|-------------|
| `validation` | Invalid input provided | Check input format |
| `timeout` | Processing took too long | Try simpler request |
| `connection` | Service unavailable | Try again later |
| `sandbox` | Resource limits exceeded | Try simpler request |
| `llm` | LLM processing issue | Try again |
| `internal` | System error | Report if persistent |
| `unknown` | Unexpected error | Try again |

### 3.3 Exit Codes (CLI)

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | User error (invalid input) |
| `2` | System error (service issue) |

### 3.4 HTTP Status Codes (API)

| Code | Meaning |
|------|---------|
| `200` | Success |
| `400` | Bad request (validation error) |
| `404` | Endpoint not found |
| `500` | Internal server error |
| `503` | Service unavailable |

### 3.5 Information Never Exposed

The following are never included in error messages (unless raw mode):

- Stack traces
- Internal class names
- File paths
- Configuration values
- API keys or credentials
- Internal error messages
- Line numbers or code references

---

## 4. Determinism Guarantees

### 4.1 What Is Deterministic

The following behaviors are deterministic:

- **Input validation**: Same input always passes or fails validation the same way
- **Error codes**: Same error condition always produces the same error code
- **Response structure**: Same request type always produces the same response structure
- **Sanitization**: Same input always sanitized the same way
- **Routing**: Same request always routed to the same handler

### 4.2 What Is NOT Deterministic

The following may vary between identical requests:

- **LLM output text**: Language models have inherent variance even at low temperature
- **Response timing**: Processing time varies based on system load
- **Timestamps**: Obviously different for each request

### 4.3 Reproducibility Expectations

| Component | Reproducibility |
|-----------|-----------------|
| Input validation | Fully reproducible |
| Error handling | Fully reproducible |
| Response format | Fully reproducible |
| LLM text content | Approximately reproducible (low temperature) |
| Performance | Not reproducible |

### 4.4 Temperature Setting

The LLM operates at low temperature (0.1) by default to minimize output variance. This does not guarantee identical outputs but reduces variation.

---

## 5. Privacy Guarantees

### 5.1 Local-First Behavior

The assistant is designed for local-first operation:

- **All processing on-device**: LLM inference runs locally via Ollama
- **No telemetry**: No usage data collected or transmitted
- **No cloud dependencies**: Functions without internet connection
- **Local storage only**: Audit logs stored locally in `logs/` directory

### 5.2 When Data May Leave the Machine

Data may leave the local machine only in these scenarios:

| Scenario | Condition | Data Sent |
|----------|-----------|-----------|
| **External LLM fallback** | Explicitly enabled in policy AND request exceeds local capability | Request text (truncated) |
| **User-initiated network** | User explicitly requests network operation | As requested |

By default, external fallback is disabled. When enabled:

- User is informed via response metadata (`used_external: true`)
- External justification is logged
- Only necessary data is transmitted

### 5.3 Data Retention

| Data Type | Retention | Location |
|-----------|-----------|----------|
| Request content | Not stored (transient) | Memory only |
| Audit logs | Configurable (default 30 days) | `logs/audit.jsonl` |
| Response content | Not stored (transient) | Memory only |

### 5.4 Audit Log Contents

Audit logs contain:

- Request ID and timestamp
- Text preview (first 100 characters only)
- Modality and task type
- Handler used and decision reason
- Success/failure status
- Processing time

Audit logs do NOT contain:

- Full request text
- Full response text
- User identifiers
- System information

---

## 6. User Expectations

### 6.1 What Users Can Rely On

Users can depend on:

- **Consistent interface**: CLI and API contracts remain stable
- **Predictable errors**: Error codes and messages follow documented patterns
- **Local processing**: Default operation requires no network
- **No surprises**: No hidden behaviors or undocumented side effects
- **Graceful degradation**: System handles failures without crashing
- **Resource limits**: Processing respects configured CPU, memory, and time limits

### 6.2 What Users Cannot Rely On

Users should NOT expect:

- **Perfect accuracy**: LLM responses may contain errors
- **Complete knowledge**: Training data has a cutoff date
- **Conversation memory**: Each request is independent
- **Identical outputs**: LLM variance means outputs may differ
- **Real-time performance**: Processing time depends on hardware and request complexity
- **Infinite capacity**: Resource limits are enforced

### 6.3 Explicitly Out of Scope

The following are explicitly not part of the assistant's scope:

- Multi-user support
- Authentication and authorization
- Conversation history management
- Custom model training
- Plugin code execution
- Real-time streaming responses
- Voice input/output
- Image understanding (currently disabled)
- Guaranteed response times

---

## 7. Interface Contracts

### 7.1 CLI Contract

```
antonio "question"           # Process question, output to stdout
antonio --json "question"    # Output as JSON
antonio --raw "question"     # Output internal response
antonio --version            # Show version
antonio --help               # Show help
```

Exit behavior:
- Successful response: exit 0, output to stdout
- User error: exit 1, message to stderr
- System error: exit 2, message to stderr

### 7.2 API Contract

```
GET  /health                 # Returns {"status": "ok", "llm_available": bool}
POST /ask                    # Process question
     Body: {"question": str, "output_mode": "text"|"json", "raw": bool}
```

Response behavior:
- Success: HTTP 200, response body
- Validation error: HTTP 400, error body
- Service unavailable: HTTP 503, error body

### 7.3 Response Structure

Successful response (formatted):
```json
{
  "success": true,
  "message": "...",
  "data": "..."
}
```

Error response (formatted):
```json
{
  "success": false,
  "message": "User-friendly error message",
  "error_category": "validation|timeout|connection|..."
}
```

---

## 8. Versioning and Changes

### 8.1 Contract Version

This contract is versioned. Breaking changes increment the major version.

Current version: **1.0**

### 8.2 What Constitutes a Breaking Change

Breaking changes include:

- Removing a supported task type
- Changing error code values
- Changing response structure
- Removing CLI flags or API endpoints
- Changing exit code meanings

Non-breaking changes include:

- Adding new task types
- Adding new error codes
- Adding optional response fields
- Adding new CLI flags or API endpoints
- Improving error messages

---

## 9. Limitations Acknowledgment

This assistant:

- Is not a replacement for human judgment
- May produce incorrect or outdated information
- Cannot verify facts in real-time
- Has knowledge limited to its training data
- Should not be used for critical decisions without verification

Users are responsible for verifying important information independently.

---

*End of UX Contract*
