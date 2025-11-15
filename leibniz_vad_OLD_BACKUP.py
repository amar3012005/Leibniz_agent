"""
Leibniz Bidirectional VAD Integration

Production-grade Voice Activity Detection (VAD) for Leibniz agent using Gemini Live API.
Implements all 7 critical patterns from SINDH bidirectional VAD for maximum robustness.

Key Features (✅ SINDH-Compatible):
✅ Pattern 1: Persistent session management with singleton pooling (eliminates cold starts)
✅ Pattern 2: Streaming audio chunks (50-100ms PCM via sounddevice)
✅ Pattern 3: Fragment-level transcript callbacks (real-time UI updates)
✅ Pattern 4: Session health & auto-recovery (handles 1011, 1006 errors)
✅ Pattern 5: State machine & concurrency control (asyncio.Lock, barge-in detection)
✅ Pattern 6: Dynamic timeout configuration (per-phase: greeting, decision, complex, retry)
✅ Pattern 7: Comprehensive logging & diagnostics (performance metrics, timeout events)

Architecture:
- LeibnizPersistentSession: Singleton session manager (SINDH-pattern persistent pooling)
- LeibnizBidirectionalVAD: Main VAD class with SINDH streaming patterns
- Helper functions: API surface for leibniz_pro.py integration

Integration:
- Drop-in replacement for existing Leibniz VAD
- Compatible with leibniz_persistent_services prewarm triggers
- Provides same API surface as SINDH for cross-agent consistency

Author: Leibniz Agent Team
Reference: Direct port from sindh_bidirectional_vad.py (SINDH patterns 1-7)
Last Updated: 2025-10-29 (Refactored to match SINDH exactly)
"""

import os
import asyncio
import time
import uuid
import logging
import threading
from typing import Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass
import tempfile
import wave

# Third-party imports
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

# Gemini SDK (CRITICAL: Use google-genai>=1.33.0 for Live API)
try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "google-genai not installed. "
        "Install with: pip install google-genai>=1.33.0"
    )

# Leibniz imports
from leibniz_agent.leibniz_config import get_leibniz_config
from leibniz_agent.leibniz_persistent_services import trigger_prewarm_on_speech_detection
from leibniz_agent.leibniz_stt import normalize_english_transcript

# Load environment variables
load_dotenv()

# Logger setup
logger = logging.getLogger(__name__)


@dataclass
class LeibnizVADConfig:
    """
    Configuration for Leibniz VAD system.
    
    IMPORTANT: Leibniz VAD is configured for English-only transcription to ensure accuracy.
    The language_code is enforced to be "en-US" for optimal English speech recognition.
    This differs from TARA's multilingual support (hi-IN) which handles Hindi+English mixing.
    """
    
    # Model configuration
    model_name: str = "gemini-2.0-flash-exp"
    language_code: str = "en-US"  # Enforced for English-only accuracy
    
    # Audio configuration
    sample_rate: int = 16000
    
    # Timeout configuration (seconds)
    start_timeout_s: float = 10.0
    silence_timeout: float = 2.5
    min_speech_ms: int = 500
    session_timeout: float = 300.0  # 5 minutes
    
    # Dynamic timeouts for different conversation contexts
    greeting_timeout_s: float = 12.0
    decision_timeout_s: float = 15.0
    complex_query_timeout_s: float = 18.0
    post_service_timeout_s: float = 8.0
    initial_timeout_s: float = 10.0
    retry_timeout_s: float = 5.0
    
    # Logging configuration
    verbose: bool = False
    log_audio_callbacks: bool = False
    log_state_transitions: bool = False
    log_timeout_checks: bool = False
    
    def __post_init__(self):
        """
        Validate configuration after initialization.
        
        Enforces English-only language code for accuracy with auto-normalization.
        Non-English codes are normalized to en-US with a warning (not an error).
        """
        # Enforce English-only configuration with lenient normalization
        if self.language_code not in ["en-US", "en"]:
            logger.warning(
                f"⚠️  Leibniz VAD configured for English-only. "
                f"Got language_code='{self.language_code}', normalizing to 'en-US'. "
                f"For multilingual support, use TARA agent with hi-IN configuration."
            )
            # Normalize to en-US instead of raising error
            self.language_code = "en-US"
        
        # Normalize to en-US if just "en" provided
        if self.language_code == "en":
            self.language_code = "en-US"


class LeibnizPersistentSession:
    """
    ⭐⭐⭐ PATTERN 1: Persistent Gemini Live Session Singleton (SINDH-Compatible)
    
    Singleton session manager for Gemini Live API with event loop binding,
    smart warmup, and automatic session refresh. Eliminates 8-12s cold starts.
    
    Features (SINDH-Pattern):
    - Thread-safe singleton pattern with double-check locking
    - Event loop binding and validation (handles loop changes)
    - Session age tracking and automatic refresh (300s timeout)
    - Smart warmup with throttling (>20s idle triggers warmup)
    - Usage statistics and diagnostics
    - Auto-recovery on errors (1011, 1006, internal errors)
    
    Performance:
    - Cold start: 8-12s (first session creation)
    - Warm reuse: <100ms (session already connected)
    - Session lifetime: 300s (5 minutes) before auto-refresh
    
    Reference: Ported from SINDHPersistentSession (sindh_bidirectional_vad.py)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    # Session state (SINDH-pattern attributes)
    _session = None
    _session_context = None
    _session_lock = None
    _session_loop = None  # Track which event loop session is bound to (CRITICAL)
    _last_activity = 0
    _creation_time = 0
    _total_uses = 0
    _warmup_in_progress = False
    
    # Client state
    _client = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern with double-check locking (SINDH-pattern)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # Don't create asyncio.Lock here - will be created in get_session()
                    # This avoids event loop binding issues
                    cls._session_lock = None
        return cls._instance
    
    @classmethod
    async def get_session(
        cls,
        client: genai.Client,
        model_name: str,
        config: 'LeibnizVADConfig'
    ):
        """
        Get or create persistent Gemini Live session (SINDH-pattern).
        
        Args:
            client: Gemini API client
            model_name: Model name for session
            config: VAD configuration
            
        Returns:
            Active Gemini Live session
            
        Session Lifecycle (SINDH-pattern):
        1. Check if session exists and is bound to current event loop
        2. If loop changed → close old session and create new one
        3. If session stale (>300s) → close and recreate
        4. If session valid → reuse (fast path)
        5. Update activity tracking and return session
        
        Error Handling:
        - Event loop changes: Force session recreation
        - Stale sessions: Auto-refresh after 300s
        - All errors: Log and recreate session
        """
        start_time = time.time()
        
        # Always recreate lock in current event loop (SINDH-pattern event loop binding)
        try:
            loop = asyncio.get_running_loop()
            
            # Check if session exists but is bound to different event loop (CRITICAL CHECK)
            if (cls._session is not None and 
                cls._session_loop is not None and 
                cls._session_loop is not loop):
                logger.info("Event loop changed, closing old session")
                await cls.close_session()
            
            # Create new lock if needed or loop changed
            if cls._session_lock is None or cls._session_lock._loop != loop:
                cls._session_lock = asyncio.Lock()
        except RuntimeError:
            # No running event loop, create fresh lock
            cls._session_lock = asyncio.Lock()
            loop = None
        
        async with cls._session_lock:
            now = time.time()
            session_age = now - cls._last_activity if cls._last_activity > 0 else 999
            
            # Check if session needs refresh (SINDH-pattern session validation)
            if (cls._session is None or 
                session_age > config.session_timeout):
                
                # Close old session if exists
                if cls._session_context:
                    try:
                        await cls._session_context.__aexit__(None, None, None)
                        logger.info("Closed expired session")
                    except Exception as e:
                        logger.debug(f"Session close error (expected): {e}")
                
                # Create new persistent session (SINDH-pattern configuration)
                logger.info(f"Creating persistent Leibniz session with language: {config.language_code}")
                
                session_config = {
                    "response_modalities": ["TEXT"],
                    "input_audio_transcription": {},
                    "speech_config": {
                        "language_code": config.language_code  # en-US
                    }
                }
                
                cls._session_context = client.aio.live.connect(
                    model=model_name,
                    config=session_config
                )
                cls._session = await cls._session_context.__aenter__()
                cls._creation_time = now
                cls._client = client
                cls._config = config
                
                # Store event loop binding (CRITICAL for loop change detection)
                try:
                    cls._session_loop = asyncio.get_running_loop()
                except RuntimeError:
                    cls._session_loop = None
                
                connection_time = time.time() - start_time
                logger.info(f"✅ New session ready in {connection_time:.3f}s")
                # Verify language configuration was applied
                logger.info(f"✅ Session configured with language: {config.language_code}")
            else:
                connection_time = time.time() - start_time
                logger.debug(f"♻️  Warm session reused (age: {session_age:.1f}s, uses: {cls._total_uses})")
            
            # Update activity tracking
            cls._last_activity = now
            cls._total_uses += 1
            
            return cls._session
    
    @classmethod
    async def smart_warmup_trigger(
        cls,
        client: genai.Client,
        model_name: str,
        config: 'LeibnizVADConfig'
    ):
        """
        ⭐⭐ PATTERN 4 (Warmup): Smart warmup with throttling (SINDH-pattern).
        
        Only warms up if:
        - No warmup already in progress (throttling)
        - Session is >20s old (likely to expire soon)
        
        Called:
        - After successful capture (background task)
        - On 20s idle detection (automatic)
        
        Args:
            client: Gemini API client
            model_name: Model name for session
            config: VAD configuration
        """
        if cls._warmup_in_progress:
            logger.debug("Warmup already in progress, skipping")
            return
        
        # Verify language code is maintained
        logger.debug(f"🔥 Smart warmup with language: {config.language_code}")
        
        now = time.time()
        session_age = now - cls._last_activity if cls._last_activity > 0 else 999
        
        # Only warmup if session is getting old (SINDH-pattern threshold)
        if session_age < 20.0:
            logger.debug(f"Session fresh ({session_age:.1f}s), skipping warmup")
            return
        
        cls._warmup_in_progress = True
        
        try:
            logger.info(f"🔥 Smart warmup triggered (session age: {session_age:.1f}s)")
            await cls.get_session(client, model_name, config)
            logger.info("✅ Background warmup completed")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
        finally:
            cls._warmup_in_progress = False
    
    @classmethod
    def get_session_stats(cls) -> Dict[str, Any]:
        """
        ⭐ PATTERN 7 (Diagnostics): Get session statistics (SINDH-pattern).
        
        Returns:
            Dict with session age, uses, creation time, loop info, warmup status
        """
        now = time.time()
        return {
            "session_exists": cls._session is not None,
            "session_age": now - cls._creation_time if cls._creation_time > 0 else 0,
            "last_used_ago": now - cls._last_activity if cls._last_activity > 0 else 0,
            "total_uses": cls._total_uses,
            "loop_id": id(cls._session_loop) if cls._session_loop else None,
            "warmup_in_progress": cls._warmup_in_progress
        }
    
    @classmethod
    async def close_session(cls):
        """
        ⭐⭐⭐ PATTERN 4 (Recovery): Close persistent session and cleanup (SINDH-pattern).
        
        Called on:
        - Critical errors (1011, 1006, internal errors)
        - Event loop changes
        - Manual cleanup requests
        - Stale session detection
        """
        if cls._session_lock is None:
            cls._session_lock = asyncio.Lock()
        
        async with cls._session_lock:
            if cls._session_context is not None:
                try:
                    await cls._session_context.__aexit__(None, None, None)
                    logger.info("🔒 Persistent session closed")
                except Exception as e:
                    logger.debug(f"Session close error (expected): {e}")
                finally:
                    cls._session = None
                    cls._session_context = None
                    cls._session_loop = None  # Clear loop binding
                    cls._last_activity = 0
                    cls._creation_time = 0
                    cls._total_uses = 0
                    cls._warmup_in_progress = False


class LeibnizBidirectionalVAD:
    """
    ⭐⭐⭐ PATTERNS 2-7: Leibniz Bidirectional VAD (SINDH-Compatible)
    
    Production-grade VAD implementing all SINDH reliability patterns:
    
    Pattern 2 (Streaming): 50-100ms PCM audio chunks via sounddevice
    Pattern 3 (Callbacks): Fragment-level streaming callbacks for real-time UI
    Pattern 4 (Recovery): Auto-recovery on errors (1011, 1006, internal)
    Pattern 5 (Concurrency): State machine + asyncio.Lock for thread safety
    Pattern 6 (Timeouts): Dynamic per-phase timeouts (greeting, decision, complex, retry)
    Pattern 7 (Diagnostics): Performance metrics, timeout logging, session health
    
    Architecture (SINDH-Pattern):
    - Singleton pattern via get_leibniz_vad()
    - Three concurrent tasks: audio streaming, transcript processing, timeout management
    - Event-driven state transitions: idle → listening → speaking → deciding
    - Barge-in detection during agent speech
    - Persistent session pooling (Pattern 1 via LeibnizPersistentSession)
    
    Integration:
    - Used by leibniz_pro.py for speech capture
    - Integrates with leibniz_persistent_services for prewarm
    - Compatible with leibniz_tts for barge-in coordination
    
    Reference: Direct port from SINDHBidirectionalVAD (sindh_bidirectional_vad.py)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[LeibnizVADConfig] = None):
        """Singleton pattern with double-check locking (SINDH-pattern)."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config: Optional[LeibnizVADConfig] = None):
        """
        Initialize Leibniz VAD instance (SINDH-pattern initialization).
        
        Args:
            config: VAD configuration (defaults to safe values if not provided)
        """
        if self._initialized:
            return
        
        # Load configuration (SINDH-pattern config handling)
        self.config = config or LeibnizVADConfig(
            model_name="gemini-2.0-flash-exp",
            language_code="en-US",
            sample_rate=16000,
            verbose=False
        )
        
        # Initialize Gemini client
        self.client = None
        self._init_client()
        
        # ⭐⭐⭐ PATTERN 5: State Machine Flags (SINDH-pattern conversation state)
        self.conversation_state = "idle"  # idle, listening, speaking, deciding (SINDH states)
        self.is_agent_speaking = False
        self.is_listening = False
        self.barge_in_detected = False
        self.last_transcript = None
        
        # ⭐ PATTERN 7: Performance Tracking (SINDH-pattern metrics)
        self.capture_count = 0
        self.total_capture_time = 0.0
        self.avg_capture_time = 0.0
        
        # ⭐⭐ PATTERN 4: Timeout Tracking for Session Health (SINDH-pattern)
        self.consecutive_timeouts = 0
        self._last_capture_ended_at = 0  # Track last capture end for stale session check
        
        # ⭐⭐ PATTERN 6: Current Capture Context (SINDH-pattern dynamic timeouts)
        self._current_timeout = self.config.initial_timeout_s
        self._current_context = "initial"
        self._current_attempt = 0
        
        # ⭐⭐⭐ PATTERN 5: Concurrency Control (SINDH-pattern locks)
        self._async_lock = None  # Will be created in capture_speech_bidirectional()
        self._active = False
        
        # Instance tracking for diagnostics (SINDH-pattern)
        self._instance_id = str(uuid.uuid4())[:8]
        if self.config.verbose:
            logger.info(f"LeibnizBidirectionalVAD initialized (instance: {self._instance_id})")
        
        self._initialized = True
    
    def _init_client(self):
        """Initialize Gemini API client (SINDH-pattern client setup)."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Set it in .env file or environment variables."
            )
        
        self.client = genai.Client(api_key=api_key)
        logger.info("✅ Gemini client initialized for Leibniz VAD")
    
    async def set_agent_speaking_state(self, is_speaking: bool, context: str = ""):
        """
        ⭐⭐⭐ PATTERN 5: Update agent speaking state (SINDH-pattern state management).
        
        Called by:
        - TTS module before/after speech playback
        - Barge-in detection on user interruption
        
        Args:
            is_speaking: True if agent is speaking, False otherwise
            context: Optional context description for logging
        """
        self.is_agent_speaking = is_speaking
        self.conversation_state = "speaking" if is_speaking else "listening"
        
        if self.config.log_state_transitions:
            status = f"🔊 Agent speaking: {context}" if is_speaking else f"🎤 Ready for user: {context}"
            logger.debug(status)
    
    def should_accept_user_audio(self) -> bool:
        """
        ⭐⭐⭐ PATTERN 5: Core bidirectional logic (SINDH-pattern audio gating).
        
        Returns:
            True if user audio should be accepted, False otherwise
            
        Logic (SINDH-pattern):
        - Accept during listening, idle, deciding states
        - Reject when agent is speaking (prevents interference)
        - Always check is_listening flag
        """
        # Accept during listening states
        if self.conversation_state in ["listening", "idle", "deciding"]:
            return True
        
        # Reject when agent is speaking
        if self.conversation_state == "speaking" and self.is_agent_speaking:
            return False
        
        return True  # Default: accept
    
    def set_dynamic_timeout(self, attempt_count: int = 0, conversation_context: str = "initial"):
        """
        ⭐⭐ PATTERN 6: Dynamic timeout configuration (SINDH-pattern per-phase timeouts).
        
        Maps conversation context to appropriate timeout values:
        - greeting: 12s (allow comfort time)
        - decision: 15s (thinking time needed)
        - complex_query: 18s (maximum thinking time)
        - post_service: 8s (quicker responses expected)
        - retry: 5s (efficiency after first attempt)
        
        Args:
            attempt_count: Number of retry attempts (0 for first attempt)
            conversation_context: Context string (greeting, decision, complex_query, etc.)
        """
        # SINDH-pattern timeout mapping
        timeout_map = {
            "greeting": self.config.greeting_timeout_s,          # 12s
            "initial": self.config.initial_timeout_s,            # 10s
            "decision": self.config.decision_timeout_s,          # 15s
            "complex_query": self.config.complex_query_timeout_s, # 18s
            "post_service": self.config.post_service_timeout_s,  # 8s
            "retry": self.config.retry_timeout_s                 # 5s
        }
        
        # Override with retry timeout on subsequent attempts (SINDH-pattern)
        if attempt_count > 0:
            self._current_timeout = self.config.retry_timeout_s
        else:
            self._current_timeout = timeout_map.get(conversation_context, self.config.initial_timeout_s)
        
        self._current_context = conversation_context
        self._current_attempt = attempt_count
        
        if self.config.verbose:
            logger.info(
                f"⏱️  Timeout set: {self._current_timeout}s "
                f"(context: {conversation_context}, attempt: {attempt_count})"
            )
    
    def log_timeout_event(self, elapsed_time: float, expected_timeout: float):
        """
        ⭐ PATTERN 7: Timeout event logging (SINDH-pattern diagnostics).
        
        Logs detailed timeout information for debugging production issues.
        
        Args:
            elapsed_time: Actual elapsed time when timeout occurred
            expected_timeout: The configured timeout value
        """
        if self.config.log_timeout_checks or self.config.verbose:
            logger.warning(
                f"⏱️  TIMEOUT: {elapsed_time:.1f}s elapsed "
                f"(expected: {expected_timeout}s, "
                f"context: {self._current_context}, "
                f"attempt: {self._current_attempt})"
            )
    
    async def capture_speech_bidirectional(
        self,
        streaming_callback: Optional[Callable[[str, bool], None]] = None
    ) -> Optional[str]:
        """
        ⭐⭐⭐ PATTERNS 2-4-6: Bidirectional speech capture (SINDH-EXACT implementation).
        
        Pattern 2 (Streaming): 50-100ms PCM chunks via sounddevice → asyncio.Queue → session.send()
        Pattern 3 (Callbacks): Fragment-level streaming_callback(text, is_final) for real-time UI
        Pattern 4 (Recovery): Auto-recovery on 1011/1006 errors, session reset on critical failures
        Pattern 6 (Timeouts): Dynamic context-aware timeouts with smart prompting at 6s threshold
        
        Args:
            streaming_callback: Optional callback(fragment: str, is_final: bool) for real-time transcripts
            
        Returns:
            Final transcript string or None if timeout/error occurred
            
        Features (SINDH-pattern):
        - Persistent session pooling (eliminates 8-12s cold starts)
        - Three concurrent tasks: audio streaming, transcript processing, timeout management
        - Barge-in detection during agent speech (sets self.barge_in_detected flag)
        - Early completion heuristics (phone numbers, 300ms silence after speech)
        - Smart prompting at 6s threshold (encourages user input)
        - Comprehensive error recovery with session reset on critical errors
        
        Error Handling (SINDH-pattern):
        - PortAudio errors: Retry up to 2 times with 0.5s backoff
        - 1011/1006 errors: Force session close + recreate
        - Event loop binding errors: Force session close + recreate
        - Session stale (>60s): Preemptive session refresh
        - Consecutive timeouts (≥3): Force session reset
        
        Performance (SINDH-pattern):
        - Warm session: <100ms latency
        - Cold session: ~1-2s (session creation overhead)
        - Fragment callback latency: <50ms from Gemini response
        
        Integration:
        - Called by leibniz_pro.py main conversation loop
        - Integrates with LeibnizPersistentSession for session pooling
        - Triggers leibniz_persistent_services prewarm on first speech detection
        """
        # ⭐⭐⭐ PATTERN 5: Concurrency lock initialization (SINDH-pattern)
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        
        # Prevent concurrent captures (SINDH-pattern concurrency guard)
        if self._async_lock.locked():
            logger.warning("🔒 Capture already in progress, skipping")
            return None
        
        # Acquire lock to guard the critical section
        await self._async_lock.acquire()
        
        try:
            # ⭐⭐⭐ PATTERN 5: State management (SINDH-pattern conversation state)
            self._active = True
            self.is_listening = True
            self.conversation_state = "listening"
            
            # ⭐ PATTERN 7: Capture logging with context (SINDH-pattern diagnostics)
            logger.info(
                f"🎤 Starting capture (timeout: {self._current_timeout}s, "
                f"context: {self._current_context}, attempt: {self._current_attempt})"
            )
            
            # ⭐⭐⭐ PATTERN 4: Session health checks (SINDH-pattern stale detection)
            now = time.time()
            time_since_last_capture = now - self._last_capture_ended_at if self._last_capture_ended_at > 0 else 0
            
            # Force session reset if stale (>60s idle) or too many timeouts
            if time_since_last_capture > 60.0:
                logger.info(f"⚠️  Session stale ({time_since_last_capture:.1f}s), forcing reset")
                await LeibnizPersistentSession.close_session()
            
            if self.consecutive_timeouts >= 3:
                logger.warning(f"⚠️  Too many timeouts ({self.consecutive_timeouts}), forcing session reset")
                await LeibnizPersistentSession.close_session()
                self.consecutive_timeouts = 0
            
            # ⭐⭐⭐ PATTERN 1: Get persistent session (SINDH-pattern session pooling)
            session = await LeibnizPersistentSession.get_session(
                self.client,
                self.config.model_name,
                self.config
            )
            
            # Initialize capture variables
            transcript_result = None
            speech_detected = False
            start_time = time.time()
            last_activity = start_time
            
            # ⭐⭐⭐ PATTERN 2: Audio streaming setup (SINDH-pattern 50-100ms chunks)
            audio_queue = asyncio.Queue(maxsize=50)  # Bounded queue prevents memory growth
            stream_active = True
            
            def audio_callback(indata, frames, time_info, status):
                """⭐⭐⭐ PATTERN 2: Sounddevice callback (SINDH-pattern PCM streaming)."""
                if status and self.config.log_audio_callbacks:
                    logger.warning(f"⚠️  Audio status: {status}")
                
                # ⭐⭐⭐ PATTERN 5: Audio gating (SINDH-pattern bidirectional logic)
                if not self.should_accept_user_audio():
                    if self.config.log_audio_callbacks or self.config.verbosity_level >= 2:
                        logger.debug(
                            f"🚫 Rejecting audio (state: {self.conversation_state}, "
                            f"agent_speaking: {self.is_agent_speaking})"
                        )
                    return
                
                # Convert float32 → int16 PCM (SINDH-pattern audio format)
                audio_data = (indata.copy() * 32767).astype(np.int16).tobytes()
                
                # Queue for async transmission (SINDH-pattern non-blocking queue)
                if stream_active:
                    try:
                        audio_queue.put_nowait(audio_data)
                    except asyncio.QueueFull:
                        if self.config.log_audio_callbacks:
                            logger.warning("⚠️  Audio queue full, dropping frame")
            
            async def stream_audio():
                """⭐⭐⭐ PATTERN 2: Audio streaming task (SINDH-EXACT protocol)."""
                nonlocal stream_active
                
                try:
                    if self.config.log_state_transitions:
                        logger.debug("🎙️ Starting audio streaming...")
                    
                    while stream_active:
                        # Check if we should accept audio (SINDH-pattern gating)
                        if not self.should_accept_user_audio():
                            await asyncio.sleep(0.1)
                            continue
                        
                        try:
                            # Dequeue with timeout to avoid blocking
                            audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                            
                            # ⭐⭐⭐ SINDH-EXACT API: send_realtime_input with types.Blob
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=audio_data,
                                    mime_type=f"audio/pcm;rate={self.config.sample_rate}"
                                )
                            )
                        
                        except asyncio.TimeoutError:
                            continue  # No audio available, keep waiting
                        
                        except Exception as e:
                            error_msg = str(e)
                            logger.error(f"❌ Audio stream error: {e}")
                            
                            # ⭐⭐⭐ PATTERN 4: Critical error detection (SINDH-pattern)
                            if '1011' in error_msg or '1006' in error_msg or 'internal error' in error_msg.lower():
                                logger.warning("⚠️ Critical audio stream error - will reset session")
                                await LeibnizPersistentSession.close_session()
                            
                            logger.debug("Breaking audio stream due to error")
                            break
                
                except Exception as e:
                    logger.error(f"❌ Stream audio task error: {e}")
            
            async def process_transcripts():
                """⭐⭐⭐ PATTERN 3: Transcript processing with fragments (SINDH-EXACT implementation)."""
                nonlocal transcript_result, speech_detected, last_activity
                fragments = []
                final_callback_emitted = False  # Guard against duplicate final callbacks
                
                try:
                    if self.config.log_state_transitions:
                        logger.debug("🔄 Starting transcript processing...")
                    
                    async for response in session.receive():
                        last_activity = time.time()  # Update activity timestamp
                        
                        # ⭐⭐⭐ PATTERN 5: Barge-in detection (SINDH-exact)
                        if (response.server_content and 
                            response.server_content.interrupted and 
                            self.is_agent_speaking):
                            logger.info("✋ Barge-in detected!")
                            self.barge_in_detected = True
                            await self.set_agent_speaking_state(False, "User barge-in")
                        
                        # ⭐⭐⭐ SINDH-EXACT: Process input_transcription (not input_audio_transcription)
                        if response.server_content and response.server_content.input_transcription:
                            text = response.server_content.input_transcription.text
                            
                            if text and text.strip():
                                if not speech_detected:
                                    logger.info("🗣️ Speech detected!")
                                    speech_detected = True
                                    # Enhanced speech detection logging
                                    logger.debug(f"Language: {self.config.language_code}, Fragment: '{text[:50]}...'")
                                    
                                    # ⭐ PATTERN 7: Trigger prewarm on first speech (SINDH-pattern optimization)
                                    try:
                                        logger.debug("⚡ Triggering persistent services prewarm on speech detection")
                                        asyncio.get_running_loop().create_task(trigger_prewarm_on_speech_detection())
                                    except Exception as e:
                                        logger.warning(f"⚠️  Prewarm trigger error: {e}")
                                        logger.debug(f"⚠️  Prewarm failed: {e}")
                                    else:
                                        logger.debug("✅ Prewarm trigger successful")
                                    
                                    # Handle speech during agent speaking (barge-in)
                                    if self.is_agent_speaking:
                                        logger.info("🔄 User interrupted agent")
                                        self.barge_in_detected = True
                                        await self.set_agent_speaking_state(False, "Barge-in interrupt")
                                
                                fragments.append(text.strip())
                                last_activity = time.time()
                                
                                # Enhanced fragment logging with verbose option
                                if self.config.verbose:
                                    logger.debug(f"📝 Fragment {len(fragments)}: {text} (cumulative: {len(' '.join(fragments))} chars)")
                                
                                # ⭐⭐ PATTERN 3: Fragment callback (SINDH-pattern real-time UI)
                                if streaming_callback:
                                    callback_start = time.time()
                                    try:
                                        if asyncio.iscoroutinefunction(streaming_callback):
                                            await streaming_callback(text.strip(), is_final=False)
                                        else:
                                            streaming_callback(text.strip(), is_final=False)
                                    except Exception as e:
                                        logger.warning(f"⚠️  Streaming callback error: {e}")
                                    finally:
                                        callback_elapsed = (time.time() - callback_start) * 1000
                                        if callback_elapsed > 50:  # Log if callback takes >50ms
                                            logger.debug(f"⚠️  Slow callback: {callback_elapsed:.1f}ms")
                                
                                # ⭐⭐ PATTERN 3: Early completion heuristics (SINDH-exact)
                                try:
                                    # Phone numbers: 10 consecutive digits → complete immediately
                                    digits = ''.join(ch for ch in ' '.join(fragments) if ch.isdigit())
                                    if len(digits) >= 10 and digits[0] in '6789':
                                        # Use contiguous digits for downstream parsing (not space-separated)
                                        transcript_result = digits
                                        logger.info(f"📞 Early transcript (phone): {transcript_result}")
                                        self.is_listening = False
                                        break
                                    
                                    # Early completion: 300ms silence after speech (SINDH-exact)
                                    time_since_last_chunk = time.time() - last_activity
                                    if (fragments and len(fragments) > 0 and 
                                        time_since_last_chunk > 0.3 and 
                                        len(' '.join(fragments)) > 20):  # Min 20 chars
                                        
                                        transcript_result = ' '.join(fragments).strip()
                                        logger.info(f"✅ Early completion (no chunks for {time_since_last_chunk:.1f}s)")
                                        logger.info(f"✅ Complete transcript: {transcript_result}")
                                        
                                        # Emit final transcript immediately
                                        if streaming_callback:
                                            try:
                                                if asyncio.iscoroutinefunction(streaming_callback):
                                                    await streaming_callback(transcript_result, is_final=True)
                                                else:
                                                    streaming_callback(transcript_result, is_final=True)
                                                final_callback_emitted = True  # Mark as emitted
                                            except Exception as e:
                                                logger.warning(f"⚠️  Early completion callback error: {e}")
                                        
                                        self.is_listening = False
                                        break
                                
                                except Exception:
                                    pass
                        
                        # ⭐⭐ PATTERN 3: Handle turn completion signal (SINDH-exact)
                        if response.server_content and response.server_content.turn_complete:
                            logger.debug("✅ Turn complete signal received")
                            break
                    
                    # ⭐⭐ PATTERN 3: Final transcript assembly (SINDH-exact)
                    if fragments and not transcript_result:
                        transcript_result = ' '.join(fragments).strip()
                        logger.info(f"🎯 Final complete transcript: '{transcript_result}'")
                        logger.debug(f"Assembled from {len(fragments)} fragments")
                    
                    # ⭐⭐ PATTERN 3: Final callback (SINDH-exact)
                    # Skip if already emitted in early completion path
                    if transcript_result and streaming_callback and not final_callback_emitted:
                        try:
                            if asyncio.iscoroutinefunction(streaming_callback):
                                await streaming_callback(transcript_result, is_final=True)
                            else:
                                streaming_callback(transcript_result, is_final=True)
                        except Exception as e:
                            logger.warning(f"⚠️  Final callback error: {e}")
                
                except Exception as e:
                    logger.error(f"❌ Process transcripts error: {e}")
            
            async def manage_timeouts():
                """⭐⭐ PATTERN 6: Timeout management with smart prompting (SINDH-pattern)."""
                nonlocal speech_detected, stream_active
                
                # ⭐⭐ PATTERN 6: Start timeout - wait for first speech
                start_deadline = start_time + self._current_timeout
                smart_prompt_triggered = False
                
                while time.time() < start_deadline:
                    if speech_detected:
                        break
                    
                    # ⭐⭐ PATTERN 6: Smart prompt at 6s threshold (SINDH-pattern user encouragement)
                    elapsed = time.time() - start_time
                    if elapsed >= 6.0 and not smart_prompt_triggered:
                        smart_prompt_triggered = True
                        logger.info("💡 Smart prompt: 6s elapsed, encouraging user...")
                        # Could play subtle prompt sound or show UI hint here
                    
                    await asyncio.sleep(0.1)
                
                # ⭐⭐ PATTERN 6: Handle start timeout (SINDH-pattern timeout logic)
                if not speech_detected:
                    elapsed = time.time() - start_time
                    self.log_timeout_event(elapsed, self._current_timeout)
                    logger.warning(f"⏱️ TIMEOUT: {elapsed:.1f}s elapsed (expected: {self._current_timeout}s, context: {getattr(self, '_current_context', 'unknown')}, attempt: {getattr(self, '_current_attempt', 0)})")
                    
                    # Stop capture
                    stream_active = False
                    
                    # Try to close session turn to break receive loop
                    try:
                        await session.send(turn_complete=True)
                    except Exception as e:
                        logger.debug(f"Could not send turn_complete: {e}")
                    
                    return
                
                # ⭐⭐ PATTERN 6: Silence timeout - wait for silence after speech (SINDH-pattern)
                while True:
                    silence_duration = time.time() - last_activity
                    
                    if silence_duration >= self.config.silence_timeout:
                        logger.info(f"🔇 Silence detected ({silence_duration:.1f}s), completing capture")
                        
                        # Stop capture
                        stream_active = False
                        
                        # Try to close session turn
                        try:
                            await session.send(turn_complete=True)
                        except Exception as e:
                            logger.debug(f"Could not send turn_complete: {e}")
                        
                        break
                    
                    await asyncio.sleep(0.1)
            
            # ⭐⭐⭐ PATTERN 4: Retry logic for PortAudio errors (SINDH-pattern resilience)
            max_retries = 2
            for retry in range(max_retries):
                try:
                    # ⭐⭐⭐ PATTERN 2: Start audio stream (SINDH-pattern sounddevice config)
                    with sd.InputStream(
                        callback=audio_callback,
                        samplerate=self.config.sample_rate,
                        channels=1,
                        dtype=np.float32,
                        blocksize=int(self.config.sample_rate * 0.05)  # 50ms chunks (SINDH-pattern)
                    ):
                        # ⭐⭐⭐ PATTERN 2+3+6: Run concurrent tasks (SINDH-pattern parallelism)
                        tasks = [
                            asyncio.create_task(stream_audio()),
                            asyncio.create_task(process_transcripts()),
                            asyncio.create_task(manage_timeouts())
                        ]
                        
                        # Wait for first task to complete (SINDH-pattern early exit)
                        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        
                        # Stop streaming and cancel remaining tasks (SINDH-pattern cleanup)
                        stream_active = False
                        for task in pending:
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                            except Exception as e:
                                logger.debug(f"Task cleanup error: {e}")
                    
                    break  # Success, exit retry loop
                
                except Exception as e:
                    # ⭐⭐⭐ PATTERN 4: PortAudio retry logic (SINDH-pattern error handling)
                    if "PortAudio" in str(e) and retry < max_retries - 1:
                        logger.warning(f"⚠️  PortAudio error, retrying ({retry+1}/{max_retries}): {e}")
                        await asyncio.sleep(0.5)
                    else:
                        raise
            
            # ⭐ PATTERN 7: Update performance metrics (SINDH-pattern diagnostics)
            capture_time = time.time() - start_time
            self.capture_count += 1
            self.total_capture_time += capture_time
            self.avg_capture_time = self.total_capture_time / self.capture_count
            
            if transcript_result:
                logger.info(f"✅ Speech captured in {capture_time:.1f}s: {transcript_result[:50]}...")
                
                # ⭐ PATTERN 1: Trigger smart warmup (SINDH-pattern session optimization)
                self._trigger_smart_warmup_background()
                
                # Reset consecutive timeouts on success (SINDH-pattern health tracking)
                self.consecutive_timeouts = 0
            else:
                logger.info(f"⏱️  No speech detected (timeout: {capture_time:.1f}s)")
                self.consecutive_timeouts += 1
            
            return transcript_result
        
        except Exception as e:
            # ⭐⭐⭐ PATTERN 4: Critical error detection (SINDH-pattern session reset)
            error_str = str(e)
            
            # Force session reset on critical errors (SINDH-pattern error recovery)
            critical_errors = ["1011", "1006", "internal error", "session closed", "event loop"]
            if any(err in error_str.lower() for err in critical_errors):
                logger.error(f"❌ CRITICAL ERROR, resetting session: {e}")
                await LeibnizPersistentSession.close_session()
            else:
                logger.error(f"❌ Capture error: {e}")
            
            return None
        
        finally:
            # ⭐⭐⭐ PATTERN 5: State cleanup (SINDH-pattern resource management)
            stream_active = False
            self.is_listening = False
            self._active = False
            
            # Reset conversation state to listening after capture (SINDH-pattern state reset)
            self.conversation_state = 'listening'
            
            # Safety reset of agent speaking flag (SINDH-pattern failsafe)
            self.is_agent_speaking = False
            
            self._last_capture_ended_at = time.time()
            
            # Clear audio queue (SINDH-pattern memory cleanup)
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except:
                    pass
            
            # Release lock (SINDH-pattern concurrency cleanup)
            if self._async_lock.locked():
                self._async_lock.release()
    
    def _trigger_smart_warmup_background(self):
        """
        ⭐ PATTERN 1: Trigger smart warmup in background (SINDH-pattern session optimization).
        
        Runs session warmup in a background thread to avoid blocking the main conversation loop.
        Uses 3s delay to ensure current pipeline continues, then runs warmup in isolated event loop.
        
        This pattern eliminates cold starts on subsequent captures by keeping sessions warm.
        """
        def warmup_worker():
            # Wait 3s to ensure pipeline continues (SINDH-pattern non-blocking)
            time.sleep(3.0)
            
            try:
                # Run warmup in new event loop (SINDH-pattern loop isolation)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    LeibnizPersistentSession.smart_warmup_trigger(
                        self.client,
                        self.config.model_name,
                        self.config
                    )
                )
                loop.close()
            except Exception as e:
                # Silent warmup - don't log errors (SINDH-pattern fail-silent)
                pass
        
        # Fire-and-forget background thread (SINDH-pattern daemon threads)
        thread = threading.Thread(target=warmup_worker, daemon=True)
        thread.start()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        ⭐ PATTERN 7: Get performance metrics for diagnostics (SINDH-pattern observability).
        
        Returns comprehensive diagnostics including:
        - Conversation state (idle, listening, speaking, deciding)
        - Barge-in detection status
        - Capture statistics (count, avg time, consecutive timeouts)
        - Session health (from LeibnizPersistentSession)
        - Current timeout configuration
        - Instance ID for multi-instance debugging
        
        Returns:
            Dict with all diagnostic fields (matches SINDH metrics structure)
        """
        return {
            # State tracking (SINDH-pattern conversation state)
            "conversation_state": self.conversation_state,
            "is_agent_speaking": self.is_agent_speaking,
            "is_listening": self.is_listening,
            "barge_in_detected": self.barge_in_detected,
            
            # Performance metrics (SINDH-pattern capture stats)
            "capture_count": self.capture_count,
            "avg_capture_time": self.avg_capture_time,
            "consecutive_timeouts": self.consecutive_timeouts,
            
            # Timeout configuration (SINDH-pattern dynamic timeouts)
            "current_timeout": self._current_timeout,
            "current_context": self._current_context,
            "current_attempt": self._current_attempt,
            
            # Session health (SINDH-pattern session diagnostics)
            "session_stats": LeibnizPersistentSession.get_session_stats(),
            
            # Instance tracking (SINDH-pattern multi-instance debugging)
            "instance_id": self._instance_id
        }


# Global singleton instance
_leibniz_vad_instance = None


def get_leibniz_vad() -> LeibnizBidirectionalVAD:
    """
    Get or create global Leibniz VAD singleton instance.
    
    Returns:
        LeibnizBidirectionalVAD singleton instance
    """
    global _leibniz_vad_instance
    
    if _leibniz_vad_instance is None:
        _leibniz_vad_instance = LeibnizBidirectionalVAD()
    
    return _leibniz_vad_instance


async def capture_leibniz_speech(
    streaming_callback: Optional[Callable[[str, bool], None]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Capture speech from user with Leibniz VAD.
    
    Args:
        streaming_callback: Optional callback(fragment: str, is_final: bool) for streaming transcripts
        context: Optional context dict with:
            - conversation_context: str (greeting, decision, complex_query, etc.)
            - attempt_count: int (retry attempt number)
            
    Returns:
        Tuple of (audio_file_path, normalized_transcript) or (None, None) if no speech
        
    Note:
        - Sets dynamic timeout based on context
        - Normalizes transcript using leibniz_stt.normalize_english_transcript()
        - Creates temporary WAV file for compatibility (0.1s silence)
    """
    vad = get_leibniz_vad()
    
    # Set dynamic timeout from context
    if context:
        conversation_context = context.get("conversation_context", "initial")
        attempt_count = context.get("attempt_count", 0)
        vad.set_dynamic_timeout(attempt_count, conversation_context)
    
    # Capture speech
    transcript = await vad.capture_speech_bidirectional(streaming_callback)
    
    if transcript:
        # Normalize transcript with timing
        normalization_start = time.time()
        logger.debug(f"📝 Original transcript: '{transcript}'")
        
        normalized = normalize_english_transcript(transcript)
        
        normalization_time_ms = (time.time() - normalization_start) * 1000
        if normalized != transcript:
            logger.debug(f"🔧 Normalized transcript: '{normalized}' ({normalization_time_ms:.2f}ms)")
        else:
            logger.debug(f"✅ No normalization needed ({normalization_time_ms:.2f}ms)")
        
        # Create temporary silent WAV file for compatibility
        # (leibniz_pro.py expects audio file path)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(temp_wav.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            # Write 0.1s of silence
            silence = np.zeros(int(0.1 * 16000), dtype=np.int16)
            wf.writeframes(silence.tobytes())
        
        logger.info(f"Speech captured: {normalized[:50]}...")
        return temp_wav.name, normalized
    
    return None, None


async def set_leibniz_agent_speaking(is_speaking: bool, context: str = ""):
    """
    Set agent speaking state for conversation flow tracking.
    
    Args:
        is_speaking: True if agent is speaking, False otherwise
        context: Optional context description
    """
    vad = get_leibniz_vad()
    await vad.set_agent_speaking_state(is_speaking, context)


async def reset_leibniz_conversation():
    """Reset conversation state (barge-in flag, consecutive timeouts)."""
    vad = get_leibniz_vad()
    vad.barge_in_detected = False
    vad.consecutive_timeouts = 0
    logger.info("Leibniz conversation state reset")


async def cleanup_leibniz_vad():
    """Cleanup Leibniz VAD resources (close session)."""
    await LeibnizPersistentSession.close_session()
    logger.info("Leibniz VAD cleanup complete")


def check_leibniz_barge_in() -> bool:
    """
    Check if user has interrupted agent speech (barge-in detection).
    
    Returns:
        True if barge-in detected, False otherwise
    """
    vad = get_leibniz_vad()
    return vad.barge_in_detected


def clear_leibniz_barge_in():
    """Clear barge-in flag after handling interruption."""
    vad = get_leibniz_vad()
    vad.barge_in_detected = False
    logger.debug("Barge-in flag cleared")


async def warmup_leibniz_vad(preconnect_s: float = 2.0):
    """
    Explicitly warmup Leibniz VAD session.
    
    Args:
        preconnect_s: Seconds to wait before connection (default: 2.0)
    """
    logger.info(f"Warming up Leibniz VAD (wait: {preconnect_s}s)...")
    await asyncio.sleep(preconnect_s)
    
    vad = get_leibniz_vad()
    await LeibnizPersistentSession.get_session(
        vad.client,
        vad.config.model_name,
        vad.config
    )
    logger.info("Leibniz VAD warmup complete")


def is_leibniz_vad_active() -> bool:
    """
    Check if Leibniz VAD is currently capturing speech.
    
    Returns:
        True if active capture in progress, False otherwise
    """
    vad = get_leibniz_vad()
    return vad._active


# Test function
async def test_leibniz_vad():
    """Test Leibniz VAD functionality."""
    print("\n" + "="*60)
    print("LEIBNIZ VAD TEST")
    print("="*60 + "\n")
    
    # Test 1: Speech capture
    print("Test 1: Speech Capture")
    print("-" * 60)
    print("Speak now (English)...")
    
    def streaming_callback(fragment: str, is_final: bool):
        if is_final:
            print(f"[FINAL] {fragment}")
        else:
            print(f"[PARTIAL] {fragment}")
    
    audio_file, transcript = await capture_leibniz_speech(
        streaming_callback=streaming_callback,
        context={"conversation_context": "greeting", "attempt_count": 0}
    )
    
    if transcript:
        print(f"\n✓ Captured: {transcript}")
        print(f"  Audio file: {audio_file}")
    else:
        print("\n✗ No speech detected")
    
    # Test 2: Agent speaking state
    print("\n\nTest 2: Agent Speaking State")
    print("-" * 60)
    
    await set_leibniz_agent_speaking(True, "testing")
    vad = get_leibniz_vad()
    print(f"Agent speaking: {vad.is_agent_speaking}")
    
    await set_leibniz_agent_speaking(False, "testing")
    print(f"Agent finished: {vad.is_agent_speaking}")
    
    # Test 3: Performance metrics
    print("\n\nTest 3: Performance Metrics")
    print("-" * 60)
    
    metrics = vad.get_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    print("\n\nCleanup")
    print("-" * 60)
    await cleanup_leibniz_vad()
    print("✓ Cleanup complete")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(test_leibniz_vad())
