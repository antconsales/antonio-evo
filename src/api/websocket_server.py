"""
Antonio Evo WebSocket API Server

Real-time communication with frontend (Chat UI + Avatar Companion).

Endpoints:
- WebSocket /ws/chat: Real-time chat with streaming and mood updates
- POST /api/ask: Single request (REST fallback)
- POST /api/upload: File upload
- GET /api/health: Health check
- GET /api/memory/stats: Memory statistics
- POST /api/memory/search: Search memory
- GET /api/emotional/*: Emotional memory (v2.1)
- GET /api/proactive/*: Proactive insights (v2.2)
- GET /api/personality/*: Personality evolution (v2.3)
- GET /api/twin/*: Digital twin (v3.0)
- GET /api/profile/*: Runtime profile and hardware info
- POST /api/generate: Z-Image generation
- POST /api/analyze-image: CLIP image analysis
- POST /api/rag/*: RAG document search
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, List  # List used for attachments
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")

# Import orchestrator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.main import Orchestrator


# ===================
# Pydantic Models
# ===================

if FASTAPI_AVAILABLE:
    class ChatMessage(BaseModel):
        """Incoming chat message."""
        text: str
        session_id: Optional[str] = None
        think: Optional[bool] = None

    class ListenRequest(BaseModel):
        """Audio transcription request."""
        audio: str  # base64 encoded audio
        format: str = "wav"
        process: bool = False  # if True, also process the transcribed text

    class ListenResponse(BaseModel):
        """Audio transcription response."""
        success: bool
        text: Optional[str] = None
        error: Optional[str] = None

    class ChatResponse(BaseModel):
        """Chat response."""
        success: bool
        text: Optional[str] = None
        output: Optional[Any] = None
        persona: Optional[str] = None
        mood: Optional[str] = None
        neuron_stored: bool = False
        neuron_id: Optional[str] = None
        elapsed_ms: int = 0
        error: Optional[str] = None

    class MemorySearchRequest(BaseModel):
        """Memory search request."""
        query: str
        limit: int = 5


# ===================
# WebSocket Events
# ===================

class WSEventType:
    """WebSocket event types."""
    # Client -> Server
    MESSAGE = "message"           # User sends a message
    PING = "ping"                 # Keep-alive ping

    # Server -> Client
    THINKING_START = "thinking_start"   # Processing started
    THINKING_END = "thinking_end"       # Processing finished
    RESPONSE = "response"               # Full response
    RESPONSE_CHUNK = "response_chunk"   # Streaming chunk (future)
    MOOD_CHANGE = "mood_change"         # Avatar mood should change
    ERROR = "error"                     # Error occurred
    PONG = "pong"                       # Keep-alive pong
    CONNECTED = "connected"             # Connection established

    # Tool System (v5.0)
    TOOL_ACTION_START = "tool_action_start"  # Tool execution starting
    TOOL_ACTION_END = "tool_action_end"      # Tool execution completed


def create_app() -> "FastAPI":
    """Create FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not available")

    app = FastAPI(
        title="Antonio Evo API",
        description="Local-first AI assistant with evolutionary memory",
        version="2.0-evo",
    )

    # Ensure output directories exist
    Path("output/images").mkdir(parents=True, exist_ok=True)

    # CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict this
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global orchestrator instance
    orchestrator = Orchestrator()

    # Active WebSocket connections
    active_connections: Dict[str, WebSocket] = {}

    # ===================
    # WebSocket Endpoint
    # ===================

    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """
        WebSocket endpoint for real-time chat.

        Protocol:
        1. Client connects
        2. Server sends "connected" event with session_id
        3. Client sends "message" events
        4. Server sends "thinking_start", then "response" or "error", then "thinking_end"
        5. Server may send "mood_change" events
        """
        await websocket.accept()

        # Create session
        session_id = orchestrator.start_session()
        active_connections[session_id] = websocket

        # Send connected event
        await send_event(websocket, WSEventType.CONNECTED, {
            "session_id": session_id,
            "version": "2.0-evo",
        })

        try:
            while True:
                # Receive message
                data = await websocket.receive_text()

                try:
                    message = json.loads(data)
                    event_type = message.get("type", WSEventType.MESSAGE)

                    if event_type == WSEventType.PING:
                        await send_event(websocket, WSEventType.PONG, {})
                        continue

                    if event_type == WSEventType.MESSAGE:
                        text = message.get("text", "")
                        if not text:
                            await send_event(websocket, WSEventType.ERROR, {
                                "error": "Empty message"
                            })
                            continue

                        # Extract attachments (v2.4)
                        # SECURITY: Attachments are UNTRUSTED INERT DATA
                        attachments = message.get("attachments", [])

                        # Process message
                        await process_chat_message(
                            websocket=websocket,
                            orchestrator=orchestrator,
                            text=text,
                            session_id=session_id,
                            attachments=attachments,
                        )

                except json.JSONDecodeError:
                    await send_event(websocket, WSEventType.ERROR, {
                        "error": "Invalid JSON"
                    })

        except WebSocketDisconnect:
            pass
        finally:
            # Cleanup
            orchestrator.end_session()
            if session_id in active_connections:
                del active_connections[session_id]

    async def process_chat_message(
        websocket: WebSocket,
        orchestrator: Orchestrator,
        text: str,
        session_id: str,
        attachments: List[Dict[str, Any]] = None,
    ):
        """Process a chat message and send events."""
        # Send thinking_start
        await send_event(websocket, WSEventType.THINKING_START, {
            "text": text,
            "has_attachments": bool(attachments),
        })

        try:
            # Build input dict with attachments (v2.4)
            # SECURITY: Attachments are passed as UNTRUSTED INERT DATA
            input_data = {
                "text": text,
                "source": "websocket",
            }
            if attachments:
                input_data["attachments"] = attachments

            # v5.0: Create tool callback bridge (async WS <- sync pipeline)
            loop = asyncio.get_event_loop()

            def tool_callback(event_type: str, data: dict):
                """Bridge sync tool events to async WebSocket."""
                try:
                    asyncio.run_coroutine_threadsafe(
                        send_event(websocket, event_type, data),
                        loop,
                    )
                except Exception:
                    pass  # Don't break pipeline if WS send fails

            # v6.0: Create chunk callback bridge for streaming tokens
            chunk_index = [0]  # mutable counter for closure

            def chunk_callback(token_text: str):
                """Bridge sync streaming tokens to async WebSocket."""
                try:
                    asyncio.run_coroutine_threadsafe(
                        send_event(websocket, WSEventType.RESPONSE_CHUNK, {
                            "text": token_text,
                            "index": chunk_index[0],
                        }),
                        loop,
                    )
                    chunk_index[0] += 1
                except Exception:
                    pass  # Don't break pipeline if WS send fails

            # Process in thread pool (orchestrator is sync)
            result = await loop.run_in_executor(
                None,
                lambda: orchestrator.process(input_data, tool_callback=tool_callback, chunk_callback=chunk_callback),
            )

            # Extract data
            success = result.get("success", False)
            raw_output = result.get("output") or result.get("text", "")
            decision = result.get("decision", {})
            meta = result.get("_meta", {})

            # Extract text from output (can be string or dict)
            # If output is empty but there's an error, use the error message
            output = _extract_response_text(raw_output)
            if not output and result.get("error"):
                output = f"Error: {result['error']}"

            persona = decision.get("persona", "social")
            mood = _determine_mood(result)

            # Send mood change if relevant
            await send_event(websocket, WSEventType.MOOD_CHANGE, {
                "mood": mood,
                "persona": persona,
            })

            # Get emotional context for response (v2.1)
            emotional = meta.get("emotional", {})

            # Get tools used from metadata (v5.0)
            tools_used = meta.get("tools_used", [])

            # Send response
            await send_event(websocket, WSEventType.RESPONSE, {
                "success": success,
                "text": str(output),
                "persona": persona,
                "mood": mood,
                "handler": decision.get("handler", "unknown"),
                "neuron_stored": meta.get("neuron_stored", False),
                "neuron_id": meta.get("neuron_id"),
                "elapsed_ms": result.get("elapsed_ms", 0),
                "error": result.get("error"),
                # Emotional context (v2.1)
                "user_emotion": emotional.get("user_state"),
                "tone_recommendation": emotional.get("tone_recommendation"),
                "emotional_trend": emotional.get("trend"),
                # Tool use (v5.0)
                "tools_used": tools_used,
            })

        except Exception as e:
            await send_event(websocket, WSEventType.ERROR, {
                "error": str(e),
            })

        finally:
            # Send thinking_end
            await send_event(websocket, WSEventType.THINKING_END, {})

    async def send_event(websocket: WebSocket, event_type: str, data: Dict[str, Any]):
        """Send a WebSocket event."""
        await websocket.send_text(json.dumps({
            "type": event_type,
            "timestamp": time.time(),
            **data,
        }))

    def _extract_response_text(output: Any) -> str:
        """Extract readable text from output (can be string, dict, or other)."""
        if output is None:
            return "No response"

        if isinstance(output, str):
            # Try to parse as JSON in case it's a JSON string
            try:
                parsed = json.loads(output)
                if isinstance(parsed, dict):
                    output = parsed
                else:
                    return output
            except (json.JSONDecodeError, TypeError):
                return output

        if isinstance(output, dict):
            # Priority order for extracting text
            for key in ['response', 'conclusion', 'text', 'output', 'answer', 'content', 'message']:
                if key in output and isinstance(output[key], str) and output[key].strip():
                    return output[key]

            # If 'analysis' exists but no conclusion, combine them
            if 'analysis' in output:
                analysis = output['analysis']
                conclusion = output.get('conclusion', '')
                if conclusion:
                    return conclusion
                if isinstance(analysis, str):
                    return analysis

            # Fallback: get first non-empty string value
            for value in output.values():
                if isinstance(value, str) and len(value) > 10:
                    return value

            # Last resort: pretty print
            return json.dumps(output, ensure_ascii=False, indent=2)

        # For any other type, convert to string
        return str(output)

    def _determine_mood(result: Dict[str, Any]) -> str:
        """Determine mood from response for avatar."""
        if not result.get("success"):
            return "cautious"

        # Check emotional context for tone-informed mood (v2.1)
        meta = result.get("_meta", {})
        emotional = meta.get("emotional", {})
        tone_rec = emotional.get("tone_recommendation", "")

        if tone_rec == "supportive":
            return "helpful"
        elif tone_rec == "patient":
            return "friendly"
        elif tone_rec == "enthusiastic":
            return "curious"
        elif tone_rec == "calming":
            return "neutral"
        elif tone_rec == "concise":
            return "analytical"

        decision = result.get("decision", {})
        persona = decision.get("persona", "social")

        if persona == "logic":
            return "analytical"

        # Check for helpful patterns
        text = str(result.get("output", "")).lower()
        if any(word in text for word in ["here's", "sure", "of course", "ecco", "certo"]):
            return "helpful"
        if any(word in text for word in ["interesting", "curious", "interessante"]):
            return "curious"

        return "friendly"

    # ===================
    # REST Endpoints
    # ===================

    @app.post("/api/ask", response_model=ChatResponse)
    async def ask(message: ChatMessage):
        """
        Single request endpoint (REST alternative to WebSocket).
        """
        if message.session_id:
            orchestrator.current_session_id = message.session_id
        else:
            orchestrator.start_session()

        try:
            # Build input dict with optional thinking metadata
            input_data = {
                "text": message.text,
                "source": "rest",
            }
            if message.think is not None:
                input_data["metadata"] = {"think": message.think}

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                orchestrator.process,
                input_data,
            )

            decision = result.get("decision", {})
            meta = result.get("_meta", {})

            raw_output = result.get("output") or result.get("text", "")
            return ChatResponse(
                success=result.get("success", False),
                text=_extract_response_text(raw_output),
                output=result.get("output"),
                persona=decision.get("persona"),
                mood=_determine_mood(result),
                neuron_stored=meta.get("neuron_stored", False),
                neuron_id=meta.get("neuron_id"),
                elapsed_ms=result.get("elapsed_ms", 0),
                error=result.get("error"),
            )
        except Exception as e:
            return ChatResponse(
                success=False,
                error=str(e),
            )

    @app.post("/api/listen", response_model=ListenResponse)
    async def listen(request: ListenRequest):
        """
        Transcribe audio to text using SpeechRecognition (Google API).

        Accepts base64-encoded audio (webm/wav).
        Returns transcribed text.
        """
        import base64
        import tempfile
        import subprocess
        import os

        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(request.audio)

            # Save to temp file (webm format from browser)
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
                f.write(audio_bytes)
                temp_webm = f.name

            # Convert webm to wav using ffmpeg (required for SpeechRecognition)
            temp_wav = temp_webm.replace('.webm', '.wav')
            ffmpeg_available = False

            # Try multiple ffmpeg paths
            ffmpeg_paths = [
                'ffmpeg',
                r'C:\Users\ant_1\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe',
                r'C:\Users\ant_1\AppData\Local\CapCut\Apps\8.0.1.3366\ffmpeg.exe',
                r'C:\ffmpeg\bin\ffmpeg.exe',
            ]

            for ffmpeg_bin in ffmpeg_paths:
                try:
                    convert_result = subprocess.run(
                        [ffmpeg_bin, '-y', '-i', temp_webm, '-ar', '16000', '-ac', '1', '-f', 'wav', temp_wav],
                        capture_output=True,
                        timeout=30
                    )
                    if convert_result.returncode == 0:
                        ffmpeg_available = True
                        break
                except FileNotFoundError:
                    continue

            if not ffmpeg_available:
                # Cleanup and return error
                if os.path.exists(temp_webm):
                    os.remove(temp_webm)
                return ListenResponse(
                    success=False,
                    error="ffmpeg not installed. Install it with: winget install ffmpeg"
                )

            # Use SpeechRecognition library
            try:
                import speech_recognition as sr
                recognizer = sr.Recognizer()

                with sr.AudioFile(temp_wav) as source:
                    audio_data = recognizer.record(source)

                # Try Google Speech Recognition (free, no API key needed)
                text = recognizer.recognize_google(audio_data, language="it-IT")

            except ImportError:
                return ListenResponse(
                    success=False,
                    error="SpeechRecognition not installed. Run: pip install SpeechRecognition"
                )
            except sr.UnknownValueError:
                return ListenResponse(
                    success=False,
                    error="Could not understand audio"
                )
            except sr.RequestError as e:
                return ListenResponse(
                    success=False,
                    error=f"Speech recognition service error: {e}"
                )

            # Cleanup
            if os.path.exists(temp_webm):
                os.remove(temp_webm)
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

            return ListenResponse(
                success=True,
                text=text
            )

        except Exception as e:
            return ListenResponse(
                success=False,
                error=str(e)
            )

    class TTSRequest(BaseModel):
        """TTS synthesis request."""
        text: str
        voice: Optional[str] = None

    @app.post("/api/tts")
    async def tts_synthesize(request: TTSRequest):
        """
        Synthesize speech from text using Piper TTS.

        Returns WAV audio as base64 or file path.
        Uses the local Piper binary in tools/piper/.
        """
        import base64
        import subprocess
        import os
        import hashlib

        text = request.text.strip()
        if not text:
            return {"success": False, "error": "No text provided"}

        # Strip emoji and non-ASCII chars that Piper/cp1252 can't handle on Windows
        import re
        text = re.sub(r'[^\x00-\x7F]+', '', text).strip()
        if not text:
            return {"success": False, "error": "No speakable text after removing special characters"}

        # Piper binary and voice paths
        project_root = str(Path(__file__).parent.parent.parent)
        piper_exe = os.path.join(project_root, "tools", "piper", "piper", "piper.exe")
        voice_model = os.path.join(
            project_root, "tools", "piper",
            request.voice or os.environ.get("PIPER_VOICE", "it_IT-paola-medium") + ".onnx"
        )

        # If voice doesn't end with .onnx, add it
        if not voice_model.endswith(".onnx"):
            voice_model += ".onnx"

        if not os.path.exists(piper_exe):
            return {"success": False, "error": f"Piper binary not found at {piper_exe}"}

        if not os.path.exists(voice_model):
            return {"success": False, "error": f"Voice model not found at {voice_model}"}

        # Output path
        output_dir = os.path.join(project_root, "output", "tts")
        os.makedirs(output_dir, exist_ok=True)
        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        output_path = os.path.join(output_dir, f"{text_hash}.wav")

        try:
            # Run Piper subprocess (use bytes input to avoid cp1252 encoding issues)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: subprocess.run(
                [piper_exe, "--model", voice_model, "--output_file", output_path],
                input=text.encode('utf-8'),
                capture_output=True,
                timeout=30,
            ))

            if result.returncode != 0:
                return {"success": False, "error": f"Piper error: {result.stderr}"}

            if not os.path.exists(output_path):
                return {"success": False, "error": "TTS output not created"}

            # Read and return as base64
            with open(output_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("ascii")

            return {
                "success": True,
                "audio": audio_data,
                "format": "wav",
                "text_length": len(text),
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "TTS timeout (30s)"}
        except FileNotFoundError:
            return {"success": False, "error": "Piper binary not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/api/health")
    async def health():
        """Health check endpoint."""
        return orchestrator.health_check()

    @app.get("/api/memory/stats")
    async def memory_stats():
        """Get memory statistics."""
        health = orchestrator.health_check()
        return health.get("memory", {"enabled": False})

    @app.post("/api/memory/search")
    async def memory_search(request: MemorySearchRequest):
        """Search memory for similar interactions."""
        results = orchestrator.memory_search(request.query, request.limit)
        return {"results": results}

    # ===================
    # Emotional Memory (v2.1)
    # ===================

    @app.get("/api/emotional/stats")
    async def emotional_stats():
        """
        Get emotional memory statistics.

        Returns distribution of detected emotions and analysis confidence.
        """
        if not orchestrator.emotional_memory:
            return {
                "enabled": False,
                "error": "Emotional memory not enabled"
            }

        try:
            stats = orchestrator.emotional_memory.get_stats()
            return {
                "enabled": True,
                "version": "2.1",
                **stats
            }
        except Exception as e:
            return {
                "enabled": True,
                "error": str(e)
            }

    @app.get("/api/emotional/recent")
    async def emotional_recent(hours: int = 24, limit: int = 20):
        """
        Get recent emotional signals.

        Args:
            hours: How many hours back to look (default 24)
            limit: Maximum signals to return (default 20)
        """
        if not orchestrator.emotional_memory:
            return {
                "success": False,
                "error": "Emotional memory not enabled"
            }

        try:
            signals = orchestrator.emotional_memory.get_recent_emotions(hours=hours, limit=limit)
            return {
                "success": True,
                "signals": [s.to_dict() for s in signals]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @app.get("/api/emotional/session/{session_id}")
    async def emotional_session(session_id: str, limit: int = 20):
        """
        Get emotional history for a specific session.
        """
        if not orchestrator.emotional_memory:
            return {
                "success": False,
                "error": "Emotional memory not enabled"
            }

        try:
            signals = orchestrator.emotional_memory.get_session_emotions(
                session_id=session_id,
                limit=limit
            )
            return {
                "success": True,
                "session_id": session_id,
                "signals": [s.to_dict() for s in signals]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    # ===================
    # Proactive Mode (v2.2)
    # ===================

    @app.get("/api/proactive/stats")
    async def proactive_stats():
        """
        Get proactive mode statistics.
        """
        if not orchestrator.pattern_analyzer:
            return {
                "enabled": False,
                "error": "Proactive mode not enabled"
            }

        try:
            stats = orchestrator.pattern_analyzer.get_stats()
            return stats
        except Exception as e:
            return {
                "enabled": True,
                "error": str(e)
            }

    @app.get("/api/proactive/insights")
    async def proactive_insights(limit: int = 5):
        """
        Get pending proactive insights.

        These are intelligent suggestions based on detected patterns.
        """
        if not orchestrator.pattern_analyzer:
            return {
                "success": False,
                "error": "Proactive mode not enabled"
            }

        try:
            insights = orchestrator.pattern_analyzer.get_pending_insights(limit=limit)
            return {
                "success": True,
                "insights": [i.to_dict() for i in insights]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @app.post("/api/proactive/insights/{insight_id}/dismiss")
    async def dismiss_insight(insight_id: str):
        """
        Dismiss an insight (mark as not useful).
        """
        if not orchestrator.pattern_analyzer:
            return {"success": False, "error": "Proactive mode not enabled"}

        try:
            orchestrator.pattern_analyzer.dismiss_insight(insight_id)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/api/proactive/patterns")
    async def proactive_patterns(days: int = 7):
        """
        Analyze and return detected patterns.
        """
        if not orchestrator.pattern_analyzer:
            return {
                "success": False,
                "error": "Proactive mode not enabled"
            }

        try:
            patterns = orchestrator.pattern_analyzer.analyze_patterns(days=days)
            return {
                "success": True,
                "patterns": [p.to_dict() for p in patterns]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    # ===================
    # Personality Evolution (v2.3)
    # ===================

    @app.get("/api/personality/profile")
    async def personality_profile():
        """
        Get current personality profile and traits.
        """
        if not orchestrator.personality_engine:
            return {
                "enabled": False,
                "error": "Personality evolution not enabled"
            }

        try:
            profile = orchestrator.personality_engine.get_profile()
            return {
                "success": True,
                "profile": profile.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @app.get("/api/personality/stats")
    async def personality_stats():
        """
        Get personality evolution statistics.
        """
        if not orchestrator.personality_engine:
            return {
                "enabled": False,
                "error": "Personality evolution not enabled"
            }

        try:
            stats = orchestrator.personality_engine.get_stats()
            return stats
        except Exception as e:
            return {
                "enabled": True,
                "error": str(e)
            }

    @app.get("/api/personality/history")
    async def personality_history(limit: int = 20):
        """
        Get trait evolution history.
        """
        if not orchestrator.personality_engine:
            return {
                "success": False,
                "error": "Personality evolution not enabled"
            }

        try:
            history = orchestrator.personality_engine.get_evolution_history(limit=limit)
            return {
                "success": True,
                "history": [h.to_dict() for h in history]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    class PersonalityFeedbackRequest(BaseModel):
        """Personality feedback request."""
        feedback_type: str  # "positive", "negative", "too_long", "too_short"
        context: Optional[Dict[str, Any]] = None

    @app.post("/api/personality/feedback")
    async def personality_feedback(request: PersonalityFeedbackRequest):
        """
        Provide feedback to evolve personality traits.

        Feedback types:
        - positive: Response was good
        - negative: Response was not helpful
        - too_long: Response was too verbose
        - too_short: Response needed more detail
        - humor_good: Humor was appreciated
        """
        if not orchestrator.personality_engine:
            return {
                "success": False,
                "error": "Personality evolution not enabled"
            }

        try:
            from src.services.personality import FeedbackSignal

            signal_map = {
                "positive": FeedbackSignal.EXPLICIT_POSITIVE,
                "negative": FeedbackSignal.EXPLICIT_NEGATIVE,
                "too_long": FeedbackSignal.TOO_LONG,
                "too_short": FeedbackSignal.TOO_SHORT,
                "humor_good": FeedbackSignal.HUMOR_APPRECIATED,
                "retry": FeedbackSignal.RETRY_REQUEST,
            }

            signal = signal_map.get(request.feedback_type)
            if not signal:
                return {
                    "success": False,
                    "error": f"Unknown feedback type: {request.feedback_type}"
                }

            evolutions = orchestrator.personality_engine.process_feedback(
                signal=signal,
                context=request.context or {}
            )

            return {
                "success": True,
                "evolutions": [e.to_dict() for e in evolutions]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    # ===================
    # Digital Twin (v3.0)
    # ===================

    @app.get("/api/twin/status")
    async def twin_status():
        """
        Get Digital Twin status and statistics.
        """
        if not orchestrator.digital_twin:
            return {
                "enabled": False,
                "error": "Digital Twin not enabled"
            }

        try:
            stats = orchestrator.digital_twin.get_stats()
            return stats
        except Exception as e:
            return {
                "enabled": True,
                "error": str(e)
            }

    @app.get("/api/twin/profile")
    async def twin_profile():
        """
        Get the learned user style profile.
        """
        if not orchestrator.digital_twin:
            return {
                "success": False,
                "error": "Digital Twin not enabled"
            }

        try:
            profile = orchestrator.digital_twin.get_profile()
            return {
                "success": True,
                "profile": profile.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @app.get("/api/twin/style-prompt")
    async def twin_style_prompt():
        """
        Get the style prompt for LLM integration.

        This prompt can be injected into system prompts to make
        responses match the user's communication style.
        """
        if not orchestrator.digital_twin:
            return {
                "success": False,
                "error": "Digital Twin not enabled"
            }

        try:
            prompt = orchestrator.digital_twin.generate_style_prompt()
            return {
                "success": True,
                "ready": orchestrator.digital_twin.is_ready(),
                "style_prompt": prompt if prompt else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @app.post("/api/twin/reset")
    async def twin_reset():
        """
        Reset the Digital Twin (clear all learned data).

        WARNING: This cannot be undone!
        """
        if not orchestrator.digital_twin:
            return {
                "success": False,
                "error": "Digital Twin not enabled"
            }

        try:
            orchestrator.digital_twin.reset()
            return {
                "success": True,
                "message": "Digital Twin has been reset"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @app.post("/api/upload")
    async def upload_file(file: UploadFile = File(...)):
        """
        Upload a file for processing.

        Currently supports text files. Future: images, audio.
        """
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Read file content
        content = await file.read()

        # Determine file type
        filename = file.filename.lower()

        if filename.endswith(('.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml')):
            # Text file - process as text
            try:
                text = content.decode('utf-8')
                # Truncate if too long
                if len(text) > 10000:
                    text = text[:10000] + "\n... (truncated)"

                return {
                    "success": True,
                    "type": "text",
                    "filename": file.filename,
                    "size": len(content),
                    "preview": text[:500] + "..." if len(text) > 500 else text,
                    "content": text,
                }
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Could not decode text file")

        elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            # Image file - save for later processing
            return {
                "success": True,
                "type": "image",
                "filename": file.filename,
                "size": len(content),
                "message": "Image received. Image processing coming soon.",
            }

        elif filename.endswith(('.wav', '.mp3', '.ogg', '.m4a')):
            # Audio file - save for later processing
            return {
                "success": True,
                "type": "audio",
                "filename": file.filename,
                "size": len(content),
                "message": "Audio received. Use /api/transcribe for speech-to-text.",
            }

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {filename}"
            )

    # ===================
    # Z-Image Generation
    # ===================

    class ImageGenerateRequest(BaseModel):
        """Image generation request."""
        prompt: str
        width: int = 512
        height: int = 512
        steps: int = 8
        seed: Optional[int] = None
        negative_prompt: str = ""

    class ImageGenerateResponse(BaseModel):
        """Image generation response."""
        success: bool
        image_path: Optional[str] = None
        image_url: Optional[str] = None
        prompt: str = ""
        generation_time: float = 0
        error: Optional[str] = None

    # Lazy-loaded Z-Image handler
    _zimage_handler = None

    def get_zimage_handler():
        nonlocal _zimage_handler
        if _zimage_handler is None:
            try:
                from ..handlers.zimage import ZImageHandler
                _zimage_handler = ZImageHandler({
                    "model_id": "Tongyi-MAI/Z-Image-Turbo",
                    "device": "cpu",
                    "width": 512,
                    "height": 512,
                    "steps": 8,
                    "output_dir": "output/images",
                })
            except ImportError:
                return None
        return _zimage_handler

    @app.post("/api/generate", response_model=ImageGenerateResponse)
    async def generate_image(request: ImageGenerateRequest):
        """
        Generate an image from a text prompt using Z-Image Turbo.
        """
        handler = get_zimage_handler()
        if handler is None:
            return ImageGenerateResponse(
                success=False,
                prompt=request.prompt,
                error="Z-Image not available. Install: pip install diffusers torch accelerate pillow"
            )

        try:
            from ..models.request import Request as AntonioRequest

            # Create request with metadata for generation params
            req = AntonioRequest(
                text=request.prompt,
                request_id=f"zimage-{uuid.uuid4().hex[:8]}",
                metadata={
                    "width": request.width,
                    "height": request.height,
                    "steps": request.steps,
                    "seed": request.seed,
                    "negative_prompt": request.negative_prompt,
                }
            )

            # Run in thread pool (generation is CPU intensive)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, handler.process, req)

            if result.success:
                return ImageGenerateResponse(
                    success=True,
                    image_path=result.output.get("image_path"),
                    image_url=f"/images/{Path(result.output.get('image_path', '')).name}",
                    prompt=request.prompt,
                    generation_time=result.output.get("generation_time_seconds", 0),
                )
            else:
                return ImageGenerateResponse(
                    success=False,
                    prompt=request.prompt,
                    error=result.error or "Generation failed"
                )

        except Exception as e:
            return ImageGenerateResponse(
                success=False,
                prompt=request.prompt,
                error=str(e)
            )

    @app.get("/images/{filename}")
    async def serve_image(filename: str):
        """Serve generated images."""
        from fastapi.responses import FileResponse
        # Use absolute path based on project root to avoid CWD issues
        project_root = Path(__file__).parent.parent.parent
        image_path = project_root / "output" / "images" / filename
        if image_path.exists():
            return FileResponse(str(image_path), media_type="image/png")
        # Fallback: try relative path
        fallback_path = Path("output/images") / filename
        if fallback_path.exists():
            return FileResponse(str(fallback_path), media_type="image/png")
        raise HTTPException(status_code=404, detail="Image not found")

    # ===================
    # CLIP Image Analysis
    # ===================

    class ImageAnalyzeRequest(BaseModel):
        """Image analysis request."""
        image_base64: Optional[str] = None
        image_path: Optional[str] = None
        question: Optional[str] = None

    class ImageAnalyzeResponse(BaseModel):
        """Image analysis response."""
        success: bool
        caption: Optional[str] = None
        confidence: float = 0
        all_scores: Optional[Dict[str, float]] = None
        error: Optional[str] = None

    _clip_handler = None

    def get_clip_handler():
        nonlocal _clip_handler
        if _clip_handler is None:
            try:
                from ..handlers.clip import CLIPHandler
                _clip_handler = CLIPHandler({"enabled": True})
            except ImportError:
                return None
        return _clip_handler

    @app.post("/api/analyze-image", response_model=ImageAnalyzeResponse)
    async def analyze_image(request: ImageAnalyzeRequest):
        """
        Analyze an image using CLIP.
        """
        handler = get_clip_handler()
        if handler is None:
            return ImageAnalyzeResponse(
                success=False,
                error="CLIP not available. Install: pip install transformers torch pillow"
            )

        try:
            import base64
            from ..models.request import Request as AntonioRequest

            # Decode image if base64
            image_bytes = None
            if request.image_base64:
                image_bytes = base64.b64decode(request.image_base64)

            req = AntonioRequest(
                text=request.question or "What is in this image?",
                request_id=f"clip-{uuid.uuid4().hex[:8]}",
                image_path=request.image_path,
                image_bytes=image_bytes,
            )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, handler.process, req)

            if result.success:
                return ImageAnalyzeResponse(
                    success=True,
                    caption=result.output.get("caption"),
                    confidence=result.output.get("confidence", 0),
                    all_scores=result.output.get("all_scores"),
                )
            else:
                return ImageAnalyzeResponse(
                    success=False,
                    error=result.error or "Analysis failed"
                )

        except Exception as e:
            return ImageAnalyzeResponse(
                success=False,
                error=str(e)
            )

    # ===================
    # RAG (Retrieval-Augmented Generation)
    # ===================

    class RAGSearchRequest(BaseModel):
        """RAG search request."""
        query: str
        limit: int = 5

    class RAGSearchResponse(BaseModel):
        """RAG search response."""
        success: bool
        results: List[Dict[str, Any]] = []
        error: Optional[str] = None

    class RAGIndexResponse(BaseModel):
        """RAG indexing response."""
        success: bool
        chunks_indexed: int = 0
        error: Optional[str] = None

    # v7.0: Use orchestrator.rag instead of separate client
    def _get_rag():
        """Get RAG client from orchestrator (v7.0 — single instance)."""
        return orchestrator.rag if orchestrator.rag and orchestrator.rag.is_available() else None

    @app.post("/api/rag/search", response_model=RAGSearchResponse)
    async def rag_search(request: RAGSearchRequest):
        """Search documents using semantic search (RAG)."""
        client = _get_rag()
        if client is None:
            return RAGSearchResponse(
                success=False,
                error="RAG not available. Install: pip install qdrant-client sentence-transformers"
            )

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, client.search, request.query, request.limit
            )

            return RAGSearchResponse(
                success=True,
                results=[
                    {"text": r.text, "source": r.source, "score": r.score, "metadata": r.metadata}
                    for r in results
                ]
            )
        except Exception as e:
            return RAGSearchResponse(success=False, error=str(e))

    @app.post("/api/rag/index", response_model=RAGIndexResponse)
    async def rag_index():
        """Index documents from the knowledge directory."""
        client = _get_rag()
        if client is None:
            return RAGIndexResponse(
                success=False,
                error="RAG not available. Install: pip install qdrant-client sentence-transformers"
            )

        try:
            loop = asyncio.get_event_loop()
            chunks_indexed = await loop.run_in_executor(None, client.index_documents)
            return RAGIndexResponse(success=True, chunks_indexed=chunks_indexed)
        except Exception as e:
            return RAGIndexResponse(success=False, error=str(e))

    @app.get("/api/rag/stats")
    async def rag_stats():
        """Get RAG statistics."""
        if not orchestrator.rag:
            return {"available": False, "error": "RAG not initialized"}
        return orchestrator.rag.get_stats()

    @app.post("/api/rag/upload")
    async def rag_upload(file: UploadFile = File(...)):
        """Upload a document to the knowledge base and index it (v7.0)."""
        client = _get_rag()
        if client is None:
            return {"success": False, "error": "RAG not available"}

        filename = file.filename or "unknown.txt"
        if not any(filename.lower().endswith(ext) for ext in ['.md', '.txt', '.markdown']):
            return {"success": False, "error": "Only .md and .txt files are supported"}

        content = await file.read()

        # Save to knowledge directory
        import os
        docs_path = client.docs_path
        os.makedirs(docs_path, exist_ok=True)

        file_path = os.path.join(docs_path, filename)
        with open(file_path, "wb") as f:
            f.write(content)

        # Re-index entire knowledge directory
        try:
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(None, client.index_documents)
            return {"success": True, "filename": filename, "chunks_indexed": chunks}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ===================
    # Runtime Profiles
    # ===================

    @app.get("/api/profile")
    async def get_profile():
        """
        Get current runtime profile and hardware info.
        """
        try:
            stats = orchestrator.profile_manager.get_stats()
            return {
                "success": True,
                **stats
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @app.get("/api/profile/capabilities")
    async def get_profile_capabilities():
        """
        Get capabilities for the active profile.
        """
        try:
            caps = orchestrator.profile_capabilities
            return {
                "success": True,
                "profile": orchestrator.active_profile.value,
                "capabilities": caps.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @app.get("/api/profile/hardware")
    async def get_hardware_info():
        """
        Get detected hardware information.
        """
        try:
            hw = orchestrator.profile_manager.hardware
            return {
                "success": True,
                "hardware": hw.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    class ProfileSwitchRequest(BaseModel):
        """Profile switch request."""
        profile: str  # evo-lite, evo-standard, evo-full, evo-hybrid

    @app.post("/api/profile/switch")
    async def switch_profile(request: ProfileSwitchRequest):
        """
        Switch to a different runtime profile.

        Note: This only changes the profile for this session.
        Some features may require restart to take effect.
        """
        try:
            success = orchestrator.profile_manager.switch_profile(request.profile)
            if success:
                # Update orchestrator references
                orchestrator.active_profile = orchestrator.profile_manager.get_active_profile()
                orchestrator.profile_capabilities = orchestrator.profile_manager.get_capabilities()
                return {
                    "success": True,
                    "new_profile": request.profile,
                    "message": "Profile switched. Some features may require restart."
                }
            else:
                return {
                    "success": False,
                    "error": f"Invalid profile: {request.profile}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @app.get("/api/profile/constraints")
    async def get_response_constraints():
        """
        Get response generation constraints for the active profile.
        """
        try:
            constraints = orchestrator.profile_manager.get_response_constraints()
            return {
                "success": True,
                "profile": orchestrator.active_profile.value,
                "constraints": constraints
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    # ===================
    # Multi-LLM Manager
    # ===================

    @app.get("/api/llm/status")
    async def llm_status():
        """
        Get Multi-LLM Manager status and available endpoints.
        """
        if not orchestrator.llm_manager:
            return {
                "enabled": False,
                "error": "LLM Manager not initialized"
            }

        try:
            stats = orchestrator.llm_manager.get_stats()
            return {
                "enabled": True,
                **stats
            }
        except Exception as e:
            return {
                "enabled": True,
                "error": str(e)
            }

    @app.get("/api/llm/models")
    async def llm_models():
        """
        Get list of available LLM models.
        """
        if not orchestrator.llm_manager:
            return {
                "success": False,
                "error": "LLM Manager not initialized"
            }

        try:
            models = orchestrator.llm_manager.get_available_models()
            return {
                "success": True,
                "models": models
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @app.post("/api/llm/check")
    async def llm_check_availability():
        """
        Check availability of all LLM endpoints.
        """
        if not orchestrator.llm_manager:
            return {
                "success": False,
                "error": "LLM Manager not initialized"
            }

        try:
            loop = asyncio.get_event_loop()
            availability = await loop.run_in_executor(
                None,
                orchestrator.llm_manager.check_availability
            )
            return {
                "success": True,
                "availability": {
                    k: v.value for k, v in availability.items()
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    # ===================
    # Webhook Management (n8n integration)
    # ===================

    @app.get("/api/webhooks")
    async def get_webhooks():
        """Get all configured webhooks."""
        try:
            from ..services.webhook_service import get_webhook_service
            service = get_webhook_service()
            return {"success": True, "webhooks": service.webhooks}
        except Exception as e:
            return {"success": False, "error": str(e)}

    class WebhookCreateRequest(BaseModel):
        """Webhook creation/update request."""
        name: str
        url: str
        trigger: str = "post_response"

    @app.post("/api/webhooks")
    async def create_webhook(request: WebhookCreateRequest):
        """Create a new webhook."""
        try:
            from ..services.webhook_service import get_webhook_service
            service = get_webhook_service()
            webhook = service.add_webhook(
                name=request.name,
                url=request.url,
                trigger=request.trigger,
            )
            return {"success": True, "webhook": webhook}
        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    class WebhookUpdateRequest(BaseModel):
        """Webhook update request."""
        name: Optional[str] = None
        url: Optional[str] = None
        trigger: Optional[str] = None
        enabled: Optional[bool] = None

    @app.put("/api/webhooks/{webhook_id}")
    async def update_webhook(webhook_id: str, request: WebhookUpdateRequest):
        """Update an existing webhook."""
        try:
            from ..services.webhook_service import get_webhook_service
            service = get_webhook_service()
            updates = {k: v for k, v in request.model_dump().items() if v is not None}
            webhook = service.update_webhook(webhook_id, updates)
            if webhook:
                return {"success": True, "webhook": webhook}
            return {"success": False, "error": "Webhook not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.delete("/api/webhooks/{webhook_id}")
    async def delete_webhook(webhook_id: str):
        """Delete a webhook."""
        try:
            from ..services.webhook_service import get_webhook_service
            service = get_webhook_service()
            if service.delete_webhook(webhook_id):
                return {"success": True}
            return {"success": False, "error": "Webhook not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.post("/api/webhooks/{webhook_id}/test")
    async def test_webhook(webhook_id: str):
        """Test a webhook connection."""
        try:
            from ..services.webhook_service import get_webhook_service
            service = get_webhook_service()
            result = service.test_webhook(webhook_id)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ===================
    # Web Search (Tavily)
    # ===================

    @app.get("/api/web-search/status")
    async def web_search_status():
        """Get web search service status."""
        try:
            ws = orchestrator.web_search_service
            if not ws:
                return {"available": False, "error": "Web search not initialized"}
            return {
                "available": ws.is_available(),
                "auto_search": ws.auto_search,
                "has_api_key": bool(ws.api_key),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    class WebSearchConfigRequest(BaseModel):
        """Web search configuration."""
        api_key: Optional[str] = None
        auto_search: Optional[bool] = None

    @app.post("/api/web-search/config")
    async def web_search_config(request: WebSearchConfigRequest):
        """Update web search configuration."""
        try:
            ws = orchestrator.web_search_service
            if not ws:
                return {"success": False, "error": "Web search not initialized"}

            if request.api_key is not None:
                # Save API key to data/api_keys.json
                keys_path = Path("data/api_keys.json")
                keys_path.parent.mkdir(parents=True, exist_ok=True)
                keys = []
                if keys_path.exists():
                    with open(keys_path, "r", encoding="utf-8") as f:
                        keys = json.load(f)

                # Update or add Tavily key
                found = False
                for key in keys:
                    if key.get("name", "").lower() == "tavily":
                        key["key"] = request.api_key
                        found = True
                        break
                if not found:
                    keys.append({
                        "id": f"tavily-{uuid.uuid4().hex[:8]}",
                        "name": "tavily",
                        "key": request.api_key,
                        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    })

                with open(keys_path, "w", encoding="utf-8") as f:
                    json.dump(keys, f, indent=2, ensure_ascii=False)

                # Update service
                ws._api_key = request.api_key

            if request.auto_search is not None:
                ws.auto_search = request.auto_search

            return {
                "success": True,
                "available": ws.is_available(),
                "auto_search": ws.auto_search,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    class WebSearchTestRequest(BaseModel):
        """Web search test request."""
        query: str = "What is the current date today?"

    @app.post("/api/web-search/test")
    async def web_search_test(request: WebSearchTestRequest):
        """Test web search with a query."""
        try:
            ws = orchestrator.web_search_service
            if not ws or not ws.is_available():
                return {"success": False, "error": "Web search not configured"}

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, ws.search, request.query)

            return {
                "success": result.success,
                "query": result.query,
                "answer": result.answer,
                "results_count": len(result.results),
                "results": [
                    {"title": r.get("title", ""), "url": r.get("url", "")}
                    for r in result.results[:3]
                ],
                "elapsed_ms": result.elapsed_ms,
                "error": result.error,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ===================
    # Audit Log Access
    # ===================

    @app.get("/api/audit/recent")
    async def get_audit_recent(limit: int = 20):
        """
        Get recent audit log entries.

        For UI transparency: "boring, reliable, and trustworthy".
        """
        try:
            entries = orchestrator.audit.get_recent(limit)
            # Sanitize for API response (entries are AuditEntry dataclasses)
            sanitized = []
            for entry in entries:
                payload = entry.payload if hasattr(entry, 'payload') else entry
                sanitized.append({
                    "timestamp": entry.timestamp_iso if hasattr(entry, 'timestamp_iso') else payload.get("timestamp_iso"),
                    "request_text": payload.get("request", {}).get("text_preview", "")[:100],
                    "handler": payload.get("decision", {}).get("handler"),
                    "persona": payload.get("decision", {}).get("persona"),
                    "success": payload.get("response", {}).get("success", False),
                    "elapsed_ms": payload.get("elapsed_ms", 0),
                    "memory_stored": bool(payload.get("memory_operation", {}).get("stored_neuron_id") if payload.get("memory_operation") else False),
                    "external_call": payload.get("decision", {}).get("handler") == "external",
                })
            return {
                "success": True,
                "entries": sanitized
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/api/audit/external")
    async def get_external_calls(limit: int = 50):
        """
        Get history of external API calls.

        Critical for UI consent transparency.
        """
        try:
            entries = orchestrator.audit.get_recent(limit * 2)  # Get more to filter
            external = []
            for entry in entries:
                payload = entry.payload if hasattr(entry, 'payload') else entry
                handler = payload.get("decision", {}).get("handler", "")
                if handler == "external" or "external" in handler.lower():
                    external.append({
                        "timestamp": entry.timestamp_iso if hasattr(entry, 'timestamp_iso') else payload.get("timestamp_iso"),
                        "provider": payload.get("decision", {}).get("provider", "unknown"),
                        "request_preview": payload.get("request", {}).get("text_preview", "")[:50],
                        "success": payload.get("response", {}).get("success", False),
                        "elapsed_ms": payload.get("elapsed_ms", 0),
                    })
                if len(external) >= limit:
                    break
            return {
                "success": True,
                "external_calls": external
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ===================
    # Consent Management
    # ===================

    # In-memory consent state (should be persisted in production)
    _consent_state = {
        "external_llm": {
            "allowed": False,
            "scope": "none",  # none, one-time, session, always
            "provider": None,
        },
        "web_search": {
            "allowed": False,
            "scope": "none",
        },
        "data_collection": {
            "allowed": True,  # Memory is local, default on
            "scope": "always",
        },
    }

    @app.get("/api/consent")
    async def get_consent_state():
        """
        Get current consent state.

        UI must display this in Settings/Governance panel.
        """
        return {
            "success": True,
            "consent": _consent_state,
            "profile_allows_external": orchestrator.profile_capabilities.external_llm_fallback,
        }

    class ConsentUpdateRequest(BaseModel):
        """Consent update request."""
        category: str  # external_llm, web_search, data_collection
        allowed: bool
        scope: str = "session"  # one-time, session, always

    @app.post("/api/consent/update")
    async def update_consent(request: ConsentUpdateRequest):
        """
        Update consent for a category.

        Used by UI Consent Dialog for explicit user permission.
        """
        if request.category not in _consent_state:
            return {
                "success": False,
                "error": f"Unknown consent category: {request.category}"
            }

        _consent_state[request.category] = {
            "allowed": request.allowed,
            "scope": request.scope,
        }

        return {
            "success": True,
            "category": request.category,
            "new_state": _consent_state[request.category]
        }

    @app.get("/api/consent/check/{category}")
    async def check_consent(category: str):
        """
        Check if a specific action is consented.

        Called before any external operation.
        """
        if category not in _consent_state:
            return {"allowed": False, "reason": "Unknown category"}

        state = _consent_state[category]
        return {
            "allowed": state["allowed"],
            "scope": state["scope"],
            "requires_prompt": state["scope"] == "one-time",
        }

    # ===================
    # UI Global State Endpoint
    # ===================

    @app.get("/api/ui/state")
    async def get_ui_global_state():
        """
        Get complete UI state in one call.

        Optimized for UI initial load and refresh.
        Contains everything needed for top bar and status indicators.
        """
        try:
            health = orchestrator.health_check()

            # Build UI-optimized state
            state = {
                "assistant_name": "Antonio Evo",
                "version": health.get("version", "2.0-evo"),
                "session_id": orchestrator.current_session_id,

                # Runtime profile
                "profile": {
                    "name": orchestrator.active_profile.value,
                    "is_local_first": not orchestrator.profile_capabilities.external_llm_primary,
                },

                # Connection status
                "connection": {
                    "local_llm": health.get("llm_manager", {}).get("initialized", False),
                    "rag": health.get("rag", {}).get("available", False),
                    "memory": health.get("memory", {}).get("enabled", False),
                },

                # Feature status for UI badges
                "features": {
                    "emotional_memory": health.get("emotional_memory", {}).get("enabled", False),
                    "proactive_mode": health.get("proactive", {}).get("enabled", False),
                    "personality_evolution": health.get("personality", {}).get("enabled", False),
                    "digital_twin": health.get("digital_twin", {}).get("enabled", False),
                },

                # Pending actions for UI notification badges
                "pending": {
                    "tasks": len([]),  # Would query pending tasks
                    "insights": 0,  # Would query pending insights
                },

                # Consent state
                "consent": _consent_state,
            }

            return {"success": True, "state": state}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ===================
    # API Key Management
    # ===================

    _api_keys_path = Path("data/api_keys.json")

    def _load_api_keys() -> list:
        """Load API keys from file."""
        try:
            if _api_keys_path.exists():
                with open(_api_keys_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return []

    def _save_api_keys(keys: list):
        """Save API keys to file."""
        _api_keys_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_api_keys_path, "w", encoding="utf-8") as f:
            json.dump(keys, f, indent=2, ensure_ascii=False)

    @app.get("/api/keys")
    async def get_api_keys():
        """Get all API keys (masked)."""
        keys = _load_api_keys()
        # Mask key values for security
        masked = []
        for k in keys:
            masked.append({
                **k,
                "key": k["key"][:8] + "..." + k["key"][-4:] if len(k.get("key", "")) > 12 else "***"
            })
        return {"success": True, "keys": masked}

    class CreateApiKeyRequest(BaseModel):
        """Create API key request."""
        id: str
        name: str
        key: str
        created: str
        lastUsed: Optional[str] = None

    @app.post("/api/keys")
    async def create_api_key(request: CreateApiKeyRequest):
        """Create a new API key."""
        keys = _load_api_keys()
        keys.append({
            "id": request.id,
            "name": request.name,
            "key": request.key,
            "created": request.created,
            "lastUsed": request.lastUsed,
        })
        _save_api_keys(keys)
        return {"success": True, "id": request.id}

    @app.delete("/api/keys/{key_id}")
    async def delete_api_key(key_id: str):
        """Delete an API key."""
        keys = _load_api_keys()
        keys = [k for k in keys if k.get("id") != key_id]
        _save_api_keys(keys)
        return {"success": True}

    # ===================
    # External APIs Management
    # ===================

    _external_apis_path = Path("data/external_apis.json")

    def _load_external_apis() -> dict:
        """Load external API configurations from file."""
        try:
            if _external_apis_path.exists():
                with open(_external_apis_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return {
            "openai": {"enabled": False, "apiKey": "", "model": "gpt-4"},
            "anthropic": {"enabled": False, "apiKey": "", "model": "claude-3-opus-20240229"},
            "google": {"enabled": False, "apiKey": "", "model": "gemini-pro"},
            "custom": [],
        }

    def _save_external_apis(apis: dict):
        """Save external API configurations to file."""
        _external_apis_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_external_apis_path, "w", encoding="utf-8") as f:
            json.dump(apis, f, indent=2, ensure_ascii=False)

    @app.get("/api/external-apis")
    async def get_external_apis():
        """Get external API configurations."""
        apis = _load_external_apis()
        # Mask API keys for security
        safe = {}
        for provider, config in apis.items():
            if provider == "custom":
                safe["custom"] = [
                    {**c, "apiKey": "***" if c.get("apiKey") else ""}
                    for c in config
                ]
            elif isinstance(config, dict):
                safe[provider] = {
                    **config,
                    "apiKey": "***" if config.get("apiKey") else "",
                }
            else:
                safe[provider] = config
        return {"success": True, "apis": safe}

    class ExternalApiUpdateRequest(BaseModel):
        """External API update request."""
        provider: str
        enabled: Optional[bool] = None
        apiKey: Optional[str] = None
        model: Optional[str] = None

    @app.post("/api/external-apis")
    async def update_external_api(request: ExternalApiUpdateRequest):
        """Update an external API configuration."""
        apis = _load_external_apis()
        if request.provider in apis and isinstance(apis[request.provider], dict):
            if request.enabled is not None:
                apis[request.provider]["enabled"] = request.enabled
            if request.apiKey is not None and request.apiKey != "***":
                apis[request.provider]["apiKey"] = request.apiKey
            if request.model is not None:
                apis[request.provider]["model"] = request.model
            _save_external_apis(apis)
            return {"success": True}
        return {"success": False, "error": f"Unknown provider: {request.provider}"}

    @app.get("/api/session")
    async def get_session():
        """Get current session info."""
        return {
            "session_id": orchestrator.current_session_id,
            "active": orchestrator.current_session_id is not None,
        }

    @app.post("/api/session/start")
    async def start_session():
        """Start a new session."""
        session_id = orchestrator.start_session()
        return {"session_id": session_id}

    @app.post("/api/session/end")
    async def end_session():
        """End current session."""
        orchestrator.end_session()
        return {"success": True}

    # ===================
    # Channels API (v6.0)
    # ===================

    @app.get("/api/channels")
    async def get_channels():
        """List all configured channels and their status."""
        channels_status = {}
        for name, channel in orchestrator.channels.items():
            if hasattr(channel, 'get_status'):
                channels_status[name] = channel.get_status()
            else:
                channels_status[name] = {"name": name, "running": True}
        return {"channels": channels_status}

    # ===================
    # Startup hook: init channels
    # ===================

    @app.on_event("startup")
    async def startup_init_channels():
        """Initialize messaging channels after server starts."""
        orchestrator.init_channels()

    return app


# ===================
# CLI Entry Point
# ===================

def main():
    """Run the WebSocket server."""
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI not installed")
        print("Run: pip install fastapi uvicorn python-multipart")
        return

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed")
        print("Run: pip install uvicorn")
        return

    app = create_app()

    print("=" * 50)
    print("  ANTONIO EVO - WebSocket API Server")
    print("=" * 50)
    print()
    print("Endpoints:")
    print("  WebSocket: ws://127.0.0.1:8420/ws/chat")
    print("  REST:      http://127.0.0.1:8420/api/...")
    print("  Docs:      http://127.0.0.1:8420/docs")
    print()

    uvicorn.run(app, host="127.0.0.1", port=8420)


# Module-level app instance for uvicorn
app = create_app() if FASTAPI_AVAILABLE else None


if __name__ == "__main__":
    main()
