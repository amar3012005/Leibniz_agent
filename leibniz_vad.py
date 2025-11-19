#!/usr/bin/env python3
"""
Leibniz Bidirectional VAD Integration Module
============================================

This module provides English-only VAD adapted from SINDH bidirectional VAD.
Direct port of sindh_bidirectional_vad.py with language changed from hi-IN to en-US.

Key Features:
 Drop-in compatibility with existing capture_leibniz_speech() function
 Persistent session management (eliminates 2-5s delays)
 Bidirectional conversation state tracking
 Smart warmup triggering
 Barge-in detection capabilities
 Dynamic timeout support (greeting/decision/complex/retry contexts)
 Singleton pattern for optimal resource usage
 Returns transcript only (NO temporary WAV file creation)

Integration: Matches SINDH pattern for cleaner API and better performance.

NOTE: This module implements per-turn VAD capture (blocking pattern).

For continuous background listening with real-time barge-in support,
use `leibniz_continuous_vad.py` instead.

Per-turn pattern:
- Pros: Simple, proven, reliable
- Cons: 1-2s audio stream startup per turn, no barge-in during TTS

Continuous pattern:
- Pros: <100ms latency per turn, real-time barge-in during TTS
- Cons: More complex, requires background task management

This module is kept for backward compatibility and as fallback.
Toggle via LEIBNIZ_ENABLE_CONTINUOUS_VAD environment variable.
"""

import warnings

# Emit deprecation warning if continuous VAD is available
try:
    from leibniz_continuous_vad import get_continuous_vad
    warnings.warn(
        "Per-turn VAD pattern is available but continuous VAD is recommended for better performance. "
        "Set LEIBNIZ_ENABLE_CONTINUOUS_VAD=true to enable.",
        DeprecationWarning,
        stacklevel=2
    )
except ImportError:
    pass  # Continuous VAD not available, no warning needed

import os
import asyncio
import time
import logging
import numpy as np
import sounddevice as sd
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from threading import Lock
from collections import deque
import uuid
from dotenv import load_dotenv

# Import audio source interface
try:
    from leibniz_webrtc_io import AudioSource
except ImportError:
    AudioSource = None  # Fallback if not available

# Import prewarm trigger for speech detection
from leibniz_agent.leibniz_stt import normalize_english_transcript

# Load environment variables
load_dotenv()

# Module-level logger
logger = logging.getLogger(__name__)

# Consolidate Gemini imports in single try-except (Comment 5)
try:
    from google import genai
    from google.genai import types
except ImportError as e:
    logger.error(f"google-genai library not installed: {e}. Run: pip install google-genai")
    genai = None
    types = None

@dataclass
class LeibnizVADConfig:
    """
    Enhanced configuration for Leibniz bidirectional VAD with granular timeout control
    
    Includes Gemini automatic activity detection (AAD) parameters for robust speech onset/offset detection
    """
    sample_rate: int = 16000
    model_name: str = "gemini-live-2.5-flash-preview"  # CRITICAL: Must use Live API model
    language_code: str = "en-US"  # English-only (vs SINDH's hi-IN)
    
    # Gemini VAD/AAD Configuration (NEW - Critical Fix #1)
    # Automatic Activity Detection parameters for robust speech boundary detection
    vad_prefix_padding_ms: int = 250  # Captures speech onset (200-300ms recommended)
    vad_silence_duration_ms: int = 450  # Silence before end-of-speech (400-500ms)
    # Valid enum values from Gemini API spec
    vad_start_sensitivity: str = "START_SENSITIVITY_HIGH"  # Quick speech detection
    vad_end_sensitivity: str = "END_SENSITIVITY_LOW"  # Don't cut off mid-thought (patient)
    
    # Increased from 3.0s to 4.5s to allow natural pauses (thinking time, breath pauses)
    silence_timeout: float = 4.5  # Wait 4.5 seconds of silence before ending turn
    # Progressive timeout strategy for natural conversation
    initial_timeout_s: float = 20.0     # First attempt - allow more time for complete sentences
    retry_timeout_s: float = 10.0       # Subsequent attempts - still generous
    max_timeout_s: float = 30.0         # Maximum timeout for complex responses
    start_timeout_s: float = 20.0       # Default timeout (will be dynamically adjusted)
    # Endpointing configuration (silence-based turn detection)
    ignore_turn_complete: bool = False   # Re-enabled: Use Gemini's turn_complete signal for proper conversation flow
    # Increased from 3.0s to 4.5s for more patient turn detection
    silence_before_finalize_s: float = 4.5  # Wait N seconds of silence before finalizing transcript
    # Comment 3: Context-aware silence thresholds - INCREASED for natural speech patterns
    # Greeting contexts: increased from 2.0s to 3.5s (allows formulation time)
    silence_before_finalize_greeting: float = 3.5  # More patient for greetings
    # Complex queries: increased from 3.5s to 5.0s (allows thinking pauses during complex questions)
    silence_before_finalize_complex: float = 5.0  # Very patient for complex queries
    # Enhanced granular timeout configuration for different conversation phases
    greeting_timeout_s: float = 25.0    # Initial greeting phase - allow complete introduction
    decision_timeout_s: float = 30.0    # Decision-making scenarios - plenty of thinking time
    complex_query_timeout_s: float = 35.0  # Complex RAG queries - maximum thinking time
    post_service_timeout_s: float = 20.0  # Post-service continuation - still generous
    # Smart prompting configuration
    smart_prompt_threshold_s: float = 6.0  # When to trigger smart prompts during silence
    smart_prompt_enabled: bool = True    # Enable/disable smart prompting system
    session_timeout: float = 600.0      # 10 minutes
    warmup_trigger_delay: float = 2.0   # Longer delay to avoid blocking
    barge_in_threshold: float = 0.5
    min_speech_ms: int = 250
    # Verbosity control for production vs debug mode
    verbose: bool = False                # General debug output (state diagnostics, queue status)
    log_audio_callbacks: bool = False    # Audio callback logging (creates massive spam)
    log_timeout_checks: bool = False     # Timeout check logging every 0.5s (high frequency)
    log_state_transitions: bool = False  # State transition messages (stream start/stop)
    
    def __post_init__(self):
        """Validate English-only configuration with auto-normalization and comprehensive parameter validation"""
        # Language validation
        if self.language_code not in ["en-US", "en"]:
            logger.warning(
                f" Leibniz VAD configured for English-only. "
                f"Got language_code='{self.language_code}', normalizing to 'en-US'. "
                f"For multilingual support, use TARA agent with hi-IN configuration."
            )
            self.language_code = "en-US"
        
        if self.language_code == "en":
            self.language_code = "en-US"
        
        # VAD sensitivity enum validation
        valid_start = ["START_SENSITIVITY_UNSPECIFIED", "START_SENSITIVITY_HIGH", "START_SENSITIVITY_LOW"]
        valid_end = ["END_SENSITIVITY_UNSPECIFIED", "END_SENSITIVITY_HIGH", "END_SENSITIVITY_LOW"]
        
        if self.vad_start_sensitivity not in valid_start:
            logger.warning(f"Invalid vad_start_sensitivity '{self.vad_start_sensitivity}', defaulting to START_SENSITIVITY_HIGH")
            self.vad_start_sensitivity = "START_SENSITIVITY_HIGH"
        if self.vad_end_sensitivity not in valid_end:
            logger.warning(f"Invalid vad_end_sensitivity '{self.vad_end_sensitivity}', defaulting to END_SENSITIVITY_LOW")
            self.vad_end_sensitivity = "END_SENSITIVITY_LOW"
        
        # Comprehensive parameter validation
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate all configuration parameters with reasonable ranges"""
        errors = []
        
        # Sample rate validation
        valid_sample_rates = [8000, 16000, 22050, 44100, 48000]
        if self.sample_rate not in valid_sample_rates:
            errors.append(f"sample_rate {self.sample_rate} not in valid rates {valid_sample_rates}")
        
        # Model name validation (basic check for Gemini models)
        if not self.model_name or not self.model_name.startswith("gemini"):
            errors.append(f"model_name '{self.model_name}' should start with 'gemini'")
        
        # VAD timing validation
        if not (0 <= self.vad_prefix_padding_ms <= 1000):
            errors.append(f"vad_prefix_padding_ms {self.vad_prefix_padding_ms} not in range 0-1000ms")
        if not (0 <= self.vad_silence_duration_ms <= 5000):
            errors.append(f"vad_silence_duration_ms {self.vad_silence_duration_ms} not in range 0-5000ms")
        
        # Timeout validation
        if self.silence_timeout <= 0:
            errors.append(f"silence_timeout {self.silence_timeout} must be positive")
        if self.initial_timeout_s <= 0:
            errors.append(f"initial_timeout_s {self.initial_timeout_s} must be positive")
        if self.retry_timeout_s <= 0:
            errors.append(f"retry_timeout_s {self.retry_timeout_s} must be positive")
        if self.max_timeout_s <= 0:
            errors.append(f"max_timeout_s {self.max_timeout_s} must be positive")
        
        # Context-specific timeout validation
        if self.greeting_timeout_s <= 0:
            errors.append(f"greeting_timeout_s {self.greeting_timeout_s} must be positive")
        if self.decision_timeout_s <= 0:
            errors.append(f"decision_timeout_s {self.decision_timeout_s} must be positive")
        if self.complex_query_timeout_s <= 0:
            errors.append(f"complex_query_timeout_s {self.complex_query_timeout_s} must be positive")
        if self.post_service_timeout_s <= 0:
            errors.append(f"post_service_timeout_s {self.post_service_timeout_s} must be positive")
        
        # Silence threshold validation
        if self.silence_before_finalize_s <= 0:
            errors.append(f"silence_before_finalize_s {self.silence_before_finalize_s} must be positive")
        if self.silence_before_finalize_greeting <= 0:
            errors.append(f"silence_before_finalize_greeting {self.silence_before_finalize_greeting} must be positive")
        if self.silence_before_finalize_complex <= 0:
            errors.append(f"silence_before_finalize_complex {self.silence_before_finalize_complex} must be positive")
        
        # Smart prompt validation
        if self.smart_prompt_threshold_s <= 0:
            errors.append(f"smart_prompt_threshold_s {self.smart_prompt_threshold_s} must be positive")
        
        # Session and timing validation
        if self.session_timeout <= 0:
            errors.append(f"session_timeout {self.session_timeout} must be positive")
        if self.warmup_trigger_delay < 0:
            errors.append(f"warmup_trigger_delay {self.warmup_trigger_delay} must be non-negative")
        
        # Barge-in and speech validation
        if not (0 <= self.barge_in_threshold <= 1):
            errors.append(f"barge_in_threshold {self.barge_in_threshold} not in range 0-1")
        if self.min_speech_ms <= 0:
            errors.append(f"min_speech_ms {self.min_speech_ms} must be positive")
        
        # Boolean validation (ensure they are actually booleans)
        bool_fields = ['verbose', 'log_audio_callbacks', 'log_timeout_checks', 
                      'log_state_transitions', 'ignore_turn_complete', 'smart_prompt_enabled']
        for field in bool_fields:
            value = getattr(self, field)
            if not isinstance(value, bool):
                errors.append(f"{field} {value} is not a boolean")
        
        # Report validation errors
        if errors:
            error_msg = f"Configuration validation failed: {'; '.join(errors)}"
            logger.error(f" {error_msg}")
            raise ValueError(error_msg)
        else:
            logger.debug(" Configuration validation passed")


class TranscriptBuffer:
    """
    Smart transcript fragment accumulator with word boundary detection (Critical Fix #2)
    
    Prevents incomplete words like "Al ice" by buffering partial words until complete.
    Implements deduplication to handle repeated fragments from Gemini Live.
    """
    
    def __init__(self):
        self.fragments: List[str] = []
        self.pending_partial: str = ""  # Buffered incomplete word from last fragment
        self.last_fragment: str = ""    # For deduplication
        self.total_fragments_received = 0
        self.duplicates_skipped = 0
        self.buffered_words_count = 0
    
    def add_fragment(self, text: str) -> str:
        """
        Add a transcript fragment with smart word boundary detection
        
        Args:
            text: Raw fragment from Gemini Live
            
        Returns:
            Complete text ready for display (buffered partial NOT included)
        """
        if not text or not text.strip():
            return ""
        
        self.total_fragments_received += 1
        
        # Deduplication: Skip exact duplicates
        if text == self.last_fragment:
            self.duplicates_skipped += 1
            logger.debug(f" Skipped duplicate fragment: '{text[:30]}...'")
            return ""
        
        self.last_fragment = text
        
        # Combine with any pending partial from previous fragment (NO space for word continuation)
        combined_text = (self.pending_partial + text).strip() if self.pending_partial else text
        
        # Check if fragment ends with complete word
        if self._ends_complete_word(combined_text):
            # Complete fragment - add to buffer and return
            self.fragments.append(combined_text)
            self.pending_partial = ""  # Clear buffered partial
            logger.debug(f" Complete fragment added: '{combined_text[:50]}...'")
            return combined_text
        else:
            # Incomplete word at end - buffer last word
            words = combined_text.rsplit(maxsplit=1)
            if len(words) == 2:
                complete_portion, partial_word = words
                self.fragments.append(complete_portion)
                self.pending_partial = partial_word
                self.buffered_words_count += 1
                logger.debug(
                    f" Buffered partial word: '{partial_word}' "
                    f"(complete portion: '{complete_portion[:40]}...')"
                )
                return complete_portion
            else:
                # Single incomplete word - buffer entirely
                self.pending_partial = combined_text
                self.buffered_words_count += 1
                logger.debug(f" Buffered single incomplete word: '{combined_text}'")
                return ""
    
    def _ends_complete_word(self, text: str) -> bool:
        """
        Check if text ends with a complete word (not mid-word fragment)
        
        Word boundary indicators:
        - Ends with punctuation (. , ! ? ; :)
        - Ends with whitespace
        - Last word is complete (not single letter, no trailing hyphen, etc.)
        """
        if not text:
            return False
        
        # Check if ends with punctuation or whitespace
        if text[-1] in ".,!?;: \t\n":
            return True
        
        # Get last word
        words = text.split()
        if not words:
            return False
        
        last_word = words[-1].strip()
        
        # Incomplete word patterns
        if len(last_word) == 1 and last_word.isalpha():
            # Single letter likely incomplete (except "I" or "a")
            return last_word.lower() in ["i", "a"]
        
        if last_word.endswith("-"):
            # Trailing hyphen indicates incomplete word
            return False
        
        # Check for common incomplete patterns
        incomplete_patterns = [
            r"^[A-Z]$",  # Single capital letter
            r"[a-z]-$",  # Word ending with hyphen
            r"^[a-z]{1,2}$"  # Very short words (likely partial)
        ]
        
        import re
        for pattern in incomplete_patterns:
            if re.match(pattern, last_word):
                return False
        
        # Assume complete if passed all checks
        return True
    
    def get_final_transcript(self) -> str:
        """
        Get complete final transcript including any buffered partial word
        
        Returns:
            Full transcript with all fragments joined
        """
        all_parts = self.fragments.copy()
        if self.pending_partial:
            all_parts.append(self.pending_partial)
            logger.debug(f" Including buffered partial in final: '{self.pending_partial}'")
        
        final = " ".join(all_parts).strip()
        
        logger.info(
            f" TranscriptBuffer stats - "
            f"Fragments: {len(self.fragments)}, "
            f"Received: {self.total_fragments_received}, "
            f"Duplicates: {self.duplicates_skipped}, "
            f"Buffered words: {self.buffered_words_count}, "
            f"Final length: {len(final)} chars"
        )
        
        return final


class RobustAudioStreamer:
    """
    Thread-safe audio streaming for Gemini Live with pre-buffering (Critical Fix #3)
    
    Fixes race conditions from asyncio.Queue + sync sounddevice callback.
    Uses threading.Queue for proper thread-safe operations.
    """
    
    def __init__(self, sample_rate: int = 16000):
        from queue import Queue  # Thread-safe queue
        
        self.sample_rate = sample_rate
        self.audio_queue = Queue()  # Thread-safe, NOT asyncio.Queue
        self.pre_buffer = deque(maxlen=int(sample_rate / 800))  # 1 second rolling window
        self.is_streaming = False
        self.error_event = asyncio.Event()
        self.dropped_chunks = 0
        self.total_chunks = 0
        self.stream = None
    
    def audio_callback(self, indata, frames, time_info, status):
        """Sounddevice callback - runs in audio thread"""
        if status:
            logger.debug(f" Audio callback status: {status}")
        try:
            # Convert float32 to PCM16
            audio_data = (indata.copy() * 32767).astype(np.int16).tobytes()
            
            # Always append to pre-buffer (rolling window of last 1s)
            self.pre_buffer.append(audio_data)
            
            # Try to enqueue for streaming (non-blocking)
            if self.is_streaming:
                try:
                    self.audio_queue.put_nowait(audio_data)
                    self.total_chunks += 1
                except:
                    # Queue full - drop chunk (expected during silence)
                    self.dropped_chunks += 1
        except Exception as e:
            logger.error(f" Audio callback error: {e}")
            self.error_event.set()
    
    async def stream_audio_to_session(self, session):
        """Stream audio to Gemini Live session with pre-buffer"""
        try:
            # Send pre-buffer first (captures speech onset)
            if self.pre_buffer:
                logger.debug(f" Sending pre-buffer ({len(self.pre_buffer)} chunks, ~1s audio)")
                for chunk in list(self.pre_buffer):
                    await session.send(data=chunk, mime_type="audio/pcm")
            
            # Stream real-time audio
            loop = asyncio.get_event_loop()
            while self.is_streaming:
                try:
                    # Use run_in_executor to await thread-safe queue
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(None, self.audio_queue.get, True, 0.1),
                        timeout=0.5
                    )
                    await session.send(data=chunk, mime_type="audio/pcm")
                except asyncio.TimeoutError:
                    # Expected during silence - continue
                    continue
                except Exception as e:
                    logger.error(f" Audio streaming error: {e}")
                    break
            
            # Send silence chunks to trigger turn_complete
            logger.debug(" Sending silence chunks to finalize turn")
            silence_chunk = np.zeros(800, dtype=np.int16).tobytes()
            for _ in range(5):  # Send 5 chunks (~250ms silence)
                await session.send(data=silence_chunk, mime_type="audio/pcm")
            
            if self.dropped_chunks > 0:
                logger.warning(
                    f" Dropped {self.dropped_chunks}/{self.total_chunks} audio chunks "
                    f"({self.dropped_chunks/max(self.total_chunks,1)*100:.1f}%)"
                )
        
        except Exception as e:
            logger.error(f" Audio stream task error: {e}")
            self.error_event.set()
    
    def start_stream(self):
        """Start sounddevice audio stream (returns context manager)"""
        import contextlib
        
        @contextlib.contextmanager
        def stream_context():
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=800,  # 50ms chunks
                callback=self.audio_callback
            )
            self.is_streaming = True
            self.stream.start()
            logger.debug(f" Audio stream started ({self.sample_rate}Hz, 50ms chunks)")
            try:
                yield self.stream
            finally:
                self.is_streaming = False
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
                logger.debug(" Audio stream stopped")
        
        return stream_context()


class LeibnizPersistentSession:
    """
    Singleton persistent session manager optimized for Leibniz Agent
    Eliminates session reinitialization delays
    
    Ported from SINDHPersistentSession with language set to en-US
    """
    _instance = None
    _lock = Lock()
    _session = None
    _session_context = None
    _session_lock = None
    _session_loop = None  # Track which event loop the session is bound to
    _last_activity = 0
    _creation_time = 0
    _total_uses = 0
    _warmup_in_progress = False
    _client = None
    _config = None
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._session_lock = None
            return cls._instance
    
    @classmethod
    async def get_session(cls, client, model_name, config: LeibnizVADConfig):
        """Get or create persistent session with performance optimization"""
        instance = cls()
        
        # Always recreate lock in current event loop to avoid event loop binding issues
        try:
            loop = asyncio.get_running_loop()
            
            # Check if session exists but is bound to a different event loop
            if (instance._session is not None and 
                instance._session_loop is not None and 
                instance._session_loop is not loop):
                print(" Leibniz: Event loop changed, closing old session")
                await cls.close_session()
            
            # Ensure session lock exists and is created in the current event loop
            if instance._session_lock is None or instance._session_loop is not loop:
                instance._session_lock = asyncio.Lock()
        except RuntimeError:
            # No running event loop, create fresh lock
            instance._session_lock = asyncio.Lock()
            loop = None
        
        start_time = time.time()
        
        async with instance._session_lock:
            now = time.time()
            session_age = now - instance._last_activity if instance._last_activity > 0 else 999
            
            # Check if session needs refresh
            if (instance._session is None or 
                session_age > config.session_timeout):
                
                # Close old session
                if instance._session_context:
                    try:
                        await instance._session_context.__aexit__(None, None, None)
                        print(" Leibniz: Closed expired session")
                    except:
                        pass
                
                # Create new persistent session with exponential backoff
                print(" Leibniz: Creating persistent bidirectional session with VAD config")
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # CRITICAL FIX #1: Add Gemini automatic activity detection (AAD) parameters
                        # This fixes speech onset detection and premature cutoffs
                        session_config = {
                            "response_modalities": ["TEXT"],
                            "input_audio_transcription": {}
                        }
                        
                        # Add VAD/AAD configuration if supported (Comment 6: Fixed key names)
                        if hasattr(config, 'vad_prefix_padding_ms'):
                            session_config["realtime_input_config"] = {
                                "automatic_activity_detection": {
                                    "prefix_padding_ms": config.vad_prefix_padding_ms,
                                    "silence_duration_ms": config.vad_silence_duration_ms,
                                    "start_of_speech_sensitivity": config.vad_start_sensitivity,
                                    "end_of_speech_sensitivity": config.vad_end_sensitivity
                                }
                            }
                            # Extract readable sensitivity names for logging
                            start_readable = config.vad_start_sensitivity.replace("START_SENSITIVITY_", "")
                            end_readable = config.vad_end_sensitivity.replace("END_SENSITIVITY_", "")
                            logger.info(
                                f" VAD config applied - "
                                f"prefix: {config.vad_prefix_padding_ms}ms, "
                                f"silence: {config.vad_silence_duration_ms}ms, "
                                f"start_sens: {start_readable}, "
                                f"end_sens: {end_readable}"
                            )
                        else:
                            logger.warning(" No VAD config found - using default Gemini settings")
                        
                        if config.language_code:
                            session_config["speech_config"] = {
                                "language_code": config.language_code
                            }
                        
                        instance._session_context = client.aio.live.connect(
                            model=model_name,
                            config=session_config
                        )
                        instance._session = await instance._session_context.__aenter__()
                        instance._creation_time = now
                        instance._client = client
                        instance._config = config
                        # Store the event loop this session is bound to
                        try:
                            instance._session_loop = asyncio.get_running_loop()
                        except RuntimeError:
                            instance._session_loop = None
                        
                        connection_time = time.time() - start_time
                        print(f"Session ready: {connection_time:.3f}s")
                        break  # Success, exit retry loop
                    
                    except Exception as e:
                        if attempt < max_retries - 1:
                            backoff_delay = 2 ** attempt  # 1s, 2s, 4s
                            await asyncio.sleep(backoff_delay)
                        else:
                            raise
            else:
                connection_time = time.time() - start_time
                logger.debug(f" Leibniz warm session reused in {connection_time:.3f}s")
            
            instance._last_activity = now
            instance._total_uses += 1
            
            return instance._session
    
    @classmethod
    async def smart_warmup_trigger(cls, client, model_name, config: LeibnizVADConfig):
        """Smart warmup after successful transcription (SINDH pattern)"""
        instance = cls()
        
        if instance._warmup_in_progress:
            return
        
        instance._warmup_in_progress = True
        try:
            now = time.time()
            session_age = now - instance._last_activity if instance._last_activity > 0 else 999
            
            # Only warmup if session is getting old (>20s since last use)
            if session_age > 20:
                await cls.get_session(client, model_name, config)
        except Exception as e:
            pass
        finally:
            instance._warmup_in_progress = False
    
    @classmethod
    def get_session_stats(cls):
        """Get session performance statistics"""
        instance = cls()
        now = time.time()
        
        return {
            "session_exists": instance._session is not None,
            "session_age": now - instance._creation_time if instance._creation_time > 0 else 0,
            "last_used_ago": now - instance._last_activity if instance._last_activity > 0 else 0,
            "total_uses": instance._total_uses,
            "warmup_in_progress": instance._warmup_in_progress
        }
    
    @classmethod
    async def close_session(cls):
        """Close persistent session"""
        instance = cls()
        if instance._session_context:
            try:
                await instance._session_context.__aexit__(None, None, None)
                logger.debug(" Leibniz: Session closed")
            except:
                pass
        instance._session = None
        instance._session_context = None
        instance._session_loop = None
        instance._warmup_in_progress = False


class LeibnizBidirectionalVAD:
    """
    Singleton bidirectional VAD optimized for Leibniz Agent
    Provides conversation state management and enhanced performance
    
    Ported from SINDHBidirectionalVAD with English-only configuration
    """
    _instance = None
    _lock = Lock()
    
    def __new__(cls, config: LeibnizVADConfig = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config: LeibnizVADConfig = None):
        if self._initialized:
            return
        
        self.config = config or LeibnizVADConfig()
        self.client = None
        
        # Instance tracking for singleton diagnosis
        self._instance_id = str(uuid.uuid4())[:8]
        if self.config.verbose:
            logger.debug(f" Leibniz: Instance created with ID: {self._instance_id}")
        
        # Bidirectional conversation state
        self.conversation_state = "idle"  # idle, listening, agent_speaking, processing
        self.is_agent_speaking = False
        self.is_listening = False
        self.barge_in_detected = False
        self.last_transcript = None
        
        # Performance tracking
        self.capture_count = 0
        self.total_capture_time = 0.0
        self.avg_capture_time = 0.0
        
        # Timeout tracking for session health
        self.consecutive_timeouts = 0
        self._last_capture_ended_at = None
        
        # Initialize client
        self._init_client()
        
        # Concurrency control
        self._async_lock = None
        self._active = False
        
        # Thread-safe speaking state (Critical Fix)
        self._speaking_lock = Lock()
        
        # Class-level transcript event system for compatibility
        if not hasattr(self.__class__, '_transcript_event'):
            self.__class__._transcript_event = None
        if not hasattr(self.__class__, '_transcript_value'):
            self.__class__._transcript_value = None
        
        self._initialized = True
    
    def _init_client(self):
        """Initialize Gemini client for Leibniz Agent"""
        # Early return if imports failed (Comment 5)
        if genai is None or types is None:
            logger.error(" Leibniz: Gemini SDK not available, cannot initialize client")
            return
        
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning(" Leibniz: No API key found")
            return
        
        try:
            self.client = genai.Client(api_key=api_key)
            logger.info(" Leibniz: Bidirectional VAD client initialized")
        except Exception as e:
            logger.error(f" Leibniz: Client init failed: {e}")
    
    async def set_agent_speaking_state(self, is_speaking: bool, context: str = ""):
        """
        Manage agent speaking state for bidirectional flow
        Call this before/after TTS in orchestration pipeline
        """
        with self._speaking_lock:
            self.is_agent_speaking = is_speaking
            self.conversation_state = "agent_speaking" if is_speaking else "idle"
        
        if context:
            status = f" Agent speaking - {context}" if is_speaking else f" Ready for user - {context}"
        else:
            status = " Agent speaking" if is_speaking else " Ready for user"
        
        logger.debug(status)
    
    def should_accept_user_audio(self) -> bool:
        """Core bidirectional logic - when to accept user input with enhanced logging and strict enforcement"""
        with self._speaking_lock:
            # Check all conditions
            agent_not_speaking = not self.is_agent_speaking
            state_listening = self.conversation_state in ["idle", "listening"]
            is_listening_active = self.is_listening

            # Determine if we should accept audio
            should_accept = agent_not_speaking and state_listening and is_listening_active

            # Enhanced logging for debugging microphone state issues
            if not should_accept:
                rejection_reasons = []
                if self.is_agent_speaking:
                    rejection_reasons.append("agent_speaking")
                if not state_listening:
                    rejection_reasons.append(f"state_{self.conversation_state}")
                if not is_listening_active:
                    rejection_reasons.append("not_listening")

                logger.debug(f" Audio rejected: {', '.join(rejection_reasons)} "
                           f"(agent_speaking={self.is_agent_speaking}, "
                           f"state={self.conversation_state}, listening={self.is_listening})")
            else:
                logger.debug(" Audio accepted: agent not speaking, in listening state")

            return should_accept
    
    def validate_audio_chunk(self, audio_data: np.ndarray) -> bool:
        """
        Validate audio chunk quality (Comment 10: Audio quality validation)
        
        Args:
            audio_data: Audio chunk as numpy array (float32, mono)
        
        Returns:
            True if chunk is valid, False if silent/clipped
        """
        try:
            # Check for silence (RMS < threshold)
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < 0.01:  # Very quiet audio
                return False
            
            # Detect clipping (>5% samples at max amplitude)
            max_amplitude = np.max(np.abs(audio_data))
            if max_amplitude > 0.95:
                clipped_samples = np.sum(np.abs(audio_data) > 0.95)
                clipping_ratio = clipped_samples / len(audio_data)
                if clipping_ratio > 0.05:  # >5% clipped
                    logger.warning(f" Audio clipping detected: {clipping_ratio*100:.1f}% samples clipped")
                    return False
            
            return True
        except Exception as e:
            logger.error(f" Audio validation error: {e}")
            return True  # Allow through on validation error
    
    def set_dynamic_timeout(self, attempt_count: int = 0, conversation_context: str = "initial") -> None:
        """
        Enhanced dynamic timeout based on conversation context and attempt count
        Comment 3: Also sets silence_before_finalize_s to context-appropriate value
        
        Args:
            attempt_count: Number of attempts (0 = first attempt)
            conversation_context: Contexts: 'greeting', 'decision', 'complex_query', 'post_service', etc.
        """
        # Store context for debugging
        self._current_context = conversation_context
        self._current_attempt = attempt_count
        
        if attempt_count == 0:
            # First attempt - context-aware timeout assignment with INCREASED thresholds for natural speech
            if conversation_context in ['greeting', 'initial']:
                self.config.start_timeout_s = self.config.greeting_timeout_s  # 25s
                self.config.silence_before_finalize_s = 3.5  # Increased from 2.0s - allow formulation time
                logger.debug(f" Applied greeting/initial: timeout={self.config.start_timeout_s}s, silence={self.config.silence_before_finalize_s}s")
            elif conversation_context in ['decision', 'service_selection']:
                self.config.start_timeout_s = self.config.decision_timeout_s  # 30s
                self.config.silence_before_finalize_s = 3.5  # Increased from 2.0s - allow thinking time
                logger.debug(f" Applied decision: timeout={self.config.start_timeout_s}s, silence={self.config.silence_before_finalize_s}s")
            elif conversation_context in ['complex_query', 'rag_query', 'complex']:
                self.config.start_timeout_s = self.config.complex_query_timeout_s  # 35s
                self.config.silence_before_finalize_s = 5.0  # Increased from 3.5s - very patient for complex queries
                logger.debug(f" Applied complex_query: timeout={self.config.start_timeout_s}s, silence={self.config.silence_before_finalize_s}s")
            elif conversation_context in ['post_service', 'continuation']:
                self.config.start_timeout_s = self.config.post_service_timeout_s  # 20s
                self.config.silence_before_finalize_s = 3.0  # Increased from 2.5s - medium-high threshold
                logger.debug(f" Applied post_service: timeout={self.config.start_timeout_s}s, silence={self.config.silence_before_finalize_s}s")
            else:
                self.config.start_timeout_s = self.config.initial_timeout_s  # 20s
                self.config.silence_before_finalize_s = 3.5  # Increased from 2.0s - more patient default
                logger.debug(f" Applied default: timeout={self.config.start_timeout_s}s, silence={self.config.silence_before_finalize_s}s")
        else:
            # Retry attempts - shorter timeout but STILL PATIENT (increased from 1.5s to 2.5s)
            self.config.start_timeout_s = self.config.retry_timeout_s  # 10s
            self.config.silence_before_finalize_s = 2.5  # Increased - be patient even on retries
            logger.debug(f" Applied retry (attempt {attempt_count}): timeout={self.config.start_timeout_s}s, silence={self.config.silence_before_finalize_s}s")
        
        logger.info(
            f" Leibniz: Dynamic timeout configured - "
            f"timeout: {self.config.start_timeout_s}s, "
            f"silence_finalize: {self.config.silence_before_finalize_s}s "
            f"(attempt: {attempt_count}, context: '{conversation_context}')"
        )
        logger.info(f" Leibniz: Dynamic timeout configured for {conversation_context} (attempt {attempt_count})")
    
    def log_timeout_event(self, elapsed_time: float, expected_timeout: float) -> None:
        """Enhanced timeout event logging"""
        context_info = getattr(self, '_current_context', 'unknown')
        attempt_info = getattr(self, '_current_attempt', 0)
        
        logger.warning(
            f"⏱ TIMEOUT: {elapsed_time:.1f}s elapsed "
            f"(expected: {expected_timeout:.1f}s, context: {context_info}, attempt: {attempt_info})"
        )
    
    def _trigger_smart_warmup_background(self):
        """Trigger background warmup task (non-blocking)"""
        async def warmup_task():
            await asyncio.sleep(self.config.warmup_trigger_delay)
            if self.client:
                await LeibnizPersistentSession.smart_warmup_trigger(
                    self.client,
                    self.config.model_name,
                    self.config
                )
        
        try:
            asyncio.create_task(warmup_task())
        except RuntimeError:
            # No event loop running, skip warmup
            pass
    
    async def capture_speech_bidirectional(self, streaming_callback: Optional[Callable[[str, bool], None]] = None, audio_source: Optional['AudioSource'] = None) -> Optional[str]:
        """
        Main bidirectional speech capture - adapted from SINDH
        
        Args:
            streaming_callback: Optional callback for streaming transcript fragments.
                              Called with (fragment: str, is_final: bool) for each fragment.
        
        Returns:
            Transcript string or None on timeout/error
        """
        if not self.client:
            logger.error(" Leibniz: No client available")
            return None
        
        # Always recreate lock in current event loop
        try:
            loop = asyncio.get_running_loop()
            # Ensure per-instance async lock is bound to the current event loop
            if getattr(self, '_async_lock', None) is None or getattr(self, '_async_lock_loop', None) is not loop:
                self._async_lock = asyncio.Lock()
                self._async_lock_loop = loop
        except RuntimeError:
            # No running loop - create a lock (will be re-bound when used)
            self._async_lock = asyncio.Lock()
            self._async_lock_loop = None
        
        # Prevent concurrent captures
        async with self._async_lock:
            if self._active:
                logger.warning(" Leibniz: VAD already active")
                return None
            self._active = True
        
        capture_start = time.time()
        final_callback_emitted = False  # Guard flag to prevent duplicate callbacks
        
        # PHASE 2 CHANGE 2.3: Add latency logging for capture
        logger.info(f" Speech capture started (timeout={self.config.start_timeout_s}s)")
        
        try:
            # Set listening state
            self.conversation_state = "listening"
            context_info = getattr(self, '_current_context', 'default')
            attempt_info = getattr(self, '_current_attempt', 0)
            
            logger.info(
                f" Listening (timeout={self.config.start_timeout_s}s, "
                f"context={context_info}, attempt={attempt_info})"
            )
            
            # Check for stale session (60+ seconds since last capture)
            if self._last_capture_ended_at is not None:
                time_since_last_capture = time.time() - self._last_capture_ended_at
                if time_since_last_capture > 60.0:
                    logger.warning(f" Leibniz: Stale session ({time_since_last_capture:.1f}s) - forcing reset")
                    await LeibnizPersistentSession.close_session()
            
            # Check for multiple consecutive timeouts
            if self.consecutive_timeouts >= 3:
                logger.warning(" Leibniz: Multiple timeouts detected, forcing session reset")
                await LeibnizPersistentSession.close_session()
                self.consecutive_timeouts = 0
            
            # Get persistent session
            session = await LeibnizPersistentSession.get_session(
                self.client,
                self.config.model_name,
                self.config
            )
            
            if not session:
                logger.error("Failed to get session")
                return None
            
            logger.info(f" Ready for input (timeout={self.config.start_timeout_s}s)")
            
            # Detailed VAD session logging (TARA pattern)
            print(f"Listening for speech: timeout={self.config.start_timeout_s}s, context={context_info}, attempt={attempt_info}, instance=N/A")
            print(f"Ready for next capture: timeout={self.config.start_timeout_s}s")
            
            # Initialize capture variables (Comment 8: Enhanced diagnostics)
            transcript_result = None
            speech_detected = False
            start_time = time.time()
            last_activity = 0.0  # Initialize to 0.0 instead of None to avoid comparison errors
            fragment_count = 0
            cumulative_length = 0
            transcript_buffer = TranscriptBuffer()  # CRITICAL FIX #3: Use smart buffer instead of naive list
            final_callback_emitted = False  # Track if final callback was sent
            
            # Comment 8: Additional diagnostic counters
            dropped_count = 0
            pre_buffer_chunks_sent = 0
            silence_durations = []  # Track silence gaps between fragments
            last_fragment_time = None
            
            self.is_listening = True
            self.barge_in_detected = False
            
            # Pre-buffer setup (TARA pattern - 1000ms rolling buffer)
            pre_buffer = deque(maxlen=20)  # 20 chunks × 50ms = 1000ms
            pre_buffer_sent = False
            
            # Audio streaming setup with backpressure (Comment 4)
            audio_queue = asyncio.Queue(maxsize=100)
            
            if audio_source is not None:
                # Use injected audio source
                async def audio_callback():
                    """Get audio from injected source"""
                    nonlocal dropped_count
                    try:
                        # Get frames from source (assuming 800 samples per chunk at 16kHz = 50ms)
                        frames = await audio_source.get_frames(800)
                        # Convert to int16 PCM
                        audio_data = (frames * 32767).astype(np.int16).tobytes()
                        
                        # Add to pre-buffer
                        pre_buffer.append(audio_data)
                        
                        # Queue for async send
                        try:
                            audio_queue.put_nowait(audio_data)
                        except asyncio.QueueFull:
                            dropped_count += 1
                            if self.config.log_audio_callbacks:
                                logger.debug(f"Audio queue full, dropped chunk (total drops: {dropped_count})")
                    except Exception as e:
                        logger.error(f"Audio source error: {e}")
                        self.error_event.set()
                
                # Start audio source task
                async def send_audio_task():
                    """Background task - streams audio to Gemini"""
                    nonlocal pre_buffer_sent, pre_buffer_chunks_sent
                    
                    # Wait for pre-buffer to reach ~20 chunks
                    wait_start = time.time()
                    while len(pre_buffer) < 20 and (time.time() - wait_start) < 0.75:
                        await asyncio.sleep(0.025)
                    
                    logger.debug(f"Pre-buffer filled to {len(pre_buffer)} chunks in {time.time() - wait_start:.3f}s")
                    
                    # Send pre-buffer first
                    if pre_buffer and not pre_buffer_sent:
                        logger.info(f" Sending pre-buffer: {len(pre_buffer)} chunks (~{len(pre_buffer)*50}ms)")
                        for buffered_chunk in pre_buffer:
                            try:
                                await session.send_realtime_input(
                                    audio=types.Blob(
                                        data=buffered_chunk,
                                        mime_type=f"audio/pcm;rate={self.config.sample_rate}"
                                    )
                                )
                                pre_buffer_chunks_sent += 1
                            except Exception as e:
                                logger.warning(f"Pre-buffer send error: {e}")
                                break
                        pre_buffer_sent = True
                        logger.info(" Pre-buffer sent successfully")
                    
                    # Stream real-time audio
                    audio_chunks_sent = 0
                    while self.is_listening:
                        try:
                            await audio_callback()  # Get from source
                            audio_chunk = await asyncio.wait_for(
                                audio_queue.get(), timeout=0.1
                            )
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=audio_chunk,
                                    mime_type=f"audio/pcm;rate={self.config.sample_rate}"
                                )
                            )
                            audio_chunks_sent += 1
                            if audio_chunks_sent % 50 == 0:
                                logger.debug(f" Sent {audio_chunks_sent} audio chunks ({audio_chunks_sent*50}ms)")
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.error(f"Audio send error: {e}")
                            break
                    
                    # Send extra silence to trigger turn_complete
                    silence_chunk = b'\x00' * (800 * 2)
                    for i in range(8):
                        try:
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=silence_chunk,
                                    mime_type=f"audio/pcm;rate={self.config.sample_rate}"
                                )
                            )
                        except Exception as e:
                            logger.debug(f"Silence chunk {i+1} send error: {e}")
                            break
                    logger.debug(" Sent 8 silence chunks (400ms) to trigger turn_complete")
                
                # Start streaming audio
                send_task = asyncio.create_task(send_audio_task())
                
            else:
                # Original sounddevice-based streaming
                def audio_callback(indata, frames, time_info, status):
                    """Sounddevice callback - sends audio to Gemini Live"""
                    nonlocal dropped_count
                    if status and self.config.log_audio_callbacks:
                        logger.debug(f"Audio status: {status}")
                    
                    # Convert to int16 PCM using flatten()
                    audio_data = (indata.flatten() * 32767).astype(np.int16).tobytes()
                    
                    # Add to pre-buffer
                    pre_buffer.append(audio_data)
                    
                    # Queue for async send
                    try:
                        audio_queue.put_nowait(audio_data)
                    except asyncio.QueueFull:
                        dropped_count += 1
                        if self.config.log_audio_callbacks:
                            logger.debug(f"Audio queue full, dropped chunk (total drops: {dropped_count})")
                    except RuntimeError:
                        pass
                
                # Start audio stream EARLY
                stream = sd.InputStream(
                    samplerate=self.config.sample_rate,
                    channels=1,
                    dtype=np.float32,
                    blocksize=800,
                    callback=audio_callback
                )
                
                async def send_audio_task():
                    """Background task - streams audio to Gemini"""
                    nonlocal pre_buffer_sent, pre_buffer_chunks_sent
                    
                    # Wait for pre-buffer to reach ~20 chunks
                    wait_start = time.time()
                    while len(pre_buffer) < 20 and (time.time() - wait_start) < 0.75:
                        await asyncio.sleep(0.025)
                    
                    logger.debug(f"Pre-buffer filled to {len(pre_buffer)} chunks in {time.time() - wait_start:.3f}s")
                    
                    # Send pre-buffer first
                    if pre_buffer and not pre_buffer_sent:
                        logger.info(f" Sending pre-buffer: {len(pre_buffer)} chunks (~{len(pre_buffer)*50}ms)")
                        for buffered_chunk in pre_buffer:
                            try:
                                await session.send_realtime_input(
                                    audio=types.Blob(
                                        data=buffered_chunk,
                                        mime_type=f"audio/pcm;rate={self.config.sample_rate}"
                                    )
                                )
                                pre_buffer_chunks_sent += 1
                            except Exception as e:
                                logger.warning(f"Pre-buffer send error: {e}")
                                break
                        pre_buffer_sent = True
                        logger.info(" Pre-buffer sent successfully")
                    
                    # Stream real-time audio
                    audio_chunks_sent = 0
                    while self.is_listening:
                        try:
                            audio_chunk = await asyncio.wait_for(
                                audio_queue.get(), timeout=0.1
                            )
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=audio_chunk,
                                    mime_type=f"audio/pcm;rate={self.config.sample_rate}"
                                )
                            )
                            audio_chunks_sent += 1
                            if audio_chunks_sent % 50 == 0:
                                logger.debug(f" Sent {audio_chunks_sent} audio chunks ({audio_chunks_sent*50}ms)")
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.error(f"Audio send error: {e}")
                            break
                    
                    # Send extra silence to trigger turn_complete
                    silence_chunk = b'\x00' * (800 * 2)
                    for i in range(8):
                        try:
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=silence_chunk,
                                    mime_type=f"audio/pcm;rate={self.config.sample_rate}"
                                )
                            )
                        except Exception as e:
                            logger.debug(f"Silence chunk {i+1} send error: {e}")
                            break
                    logger.debug(" Sent 8 silence chunks (400ms) to trigger turn_complete")
                
                # Start streaming audio
                with stream:
                    send_task = asyncio.create_task(send_audio_task())
                
                logger.info(" Starting to listen for Gemini responses...")
                response_count = 0
                
                try:
                    # Process Gemini Live responses (SINDH pattern - simple async for)
                    async for response in session.receive():
                        response_count += 1
                        logger.debug(f" Response #{response_count} received")
                        
                        # Debug: Log what we're receiving
                        if response:
                            if hasattr(response, 'server_content') and response.server_content:
                                logger.debug(f" Has server_content")
                                if hasattr(response.server_content, 'input_transcription') and response.server_content.input_transcription:
                                    logger.info(f" Found input_transcription!")
                                else:
                                    logger.debug(f" No input_transcription in server_content")
                        
                        # PHASE 2 CHANGE 2.2: Check timeout (single check per iteration)
                        elapsed = time.time() - start_time
                        if elapsed >= self.config.start_timeout_s:
                            self.log_timeout_event(elapsed, self.config.start_timeout_s)
                            self.consecutive_timeouts += 1
                            logger.info(f"⏱ Timeout after {response_count} responses")
                            break
                        
                        # Check silence-based finalization (if we have speech and enough silence)
                        if speech_detected and last_activity > 0:
                            silence_duration = time.time() - last_activity
                            if silence_duration >= self.config.silence_before_finalize_s:
                                # Only finalize if we have some transcript content
                                current_transcript = transcript_buffer.get_final_transcript()
                                if current_transcript:
                                    logger.info(f" {silence_duration:.1f}s silence after speech - finalizing transcript")
                                    break
                        
                        # Comment 10: Handle interruption signal
                        if response.server_content and hasattr(response.server_content, 'interrupted'):
                            if response.server_content.interrupted:
                                # Clear any buffered partial word
                                if transcript_buffer.pending_partial:
                                    logger.info(f" Interruption detected - clearing buffered partial: '{transcript_buffer.pending_partial}'")
                                    transcript_buffer.pending_partial = ""
                                logger.debug(" Gemini interrupted signal received - continuing to listen")
                                continue
                        
                        # Handle input transcription (SINDH pattern - spontaneous!)
                        if response.server_content and response.server_content.input_transcription:
                            text = response.server_content.input_transcription.text
                            
                            if text and text.strip():
                                if not speech_detected:
                                    speech_detected = True
                                    last_activity = time.time()  # Initialize last_activity (Comment 8)
                                    logger.info(f" Leibniz: Speech detected!")
                                    print(" Leibniz: Speech detected!")
                                    
                                    # Trigger RAG prewarm on first speech (fire-and-forget)
                                    try:
                                        print(" Pre-warming RAG models...")
                                        from leibniz_persistent_services import trigger_prewarm_on_speech_detection
                                        await trigger_prewarm_on_speech_detection()
                                        logger.info(" RAG prewarm triggered on speech detection")
                                        print(" RAG models pre-warmed successfully")
                                    except Exception as e:
                                        logger.warning(f" Prewarm trigger failed: {e}")
                                    
                                    # Handle speech during agent speaking (barge-in)
                                    with self._speaking_lock:
                                        if self.is_agent_speaking:
                                            logger.info(" Leibniz: User interrupted agent")
                                            self.barge_in_detected = True
                                            await self.set_agent_speaking_state(False, "Barge-in interrupt")
                                
                                # Accumulate fragments using TranscriptBuffer
                                complete_text = transcript_buffer.add_fragment(text.strip())
                                last_activity = time.time()
                                fragment_count += 1
                                cumulative_length += len(text)
                                
                                # Comment 8: Track silence duration between fragments
                                if last_fragment_time is not None:
                                    silence_gap = last_activity - last_fragment_time
                                    silence_durations.append(silence_gap)
                                last_fragment_time = last_activity
                                
                                logger.debug(f" Leibniz: {text}")
                                
                                # Invoke streaming callback for complete portion only (not buffered partials)
                                if streaming_callback and complete_text:
                                    callback_start = time.time()
                                    streaming_callback(complete_text, is_final=False)
                                    callback_time = time.time() - callback_start
                                    
                                    if callback_time > 0.05:
                                        logger.warning(
                                            f" Callback slow: {callback_time*1000:.1f}ms for fragment {fragment_count}"
                                        )
                                
                                # PHASE 2 CHANGE 2.1: Early completion detection (Comment 1 fix)
                                # Checks time since LAST fragment (works on next iteration)
                                # Guard against null last_activity (Comment 8)
                                if last_activity is not None:
                                    try:
                                        # Relaxed thresholds to allow natural pauses during speech
                                        time_since_last_chunk = time.time() - last_activity
                                        current_transcript = transcript_buffer.get_final_transcript()
                                        
                                        # Get current conversation context for adaptive thresholds
                                        context = getattr(self, '_current_context', 'initial')
                                        
                                        # Context-aware thresholds - RELAXED for natural speech patterns
                                        if context in ['complex_query', 'rag_query', 'complex']:
                                            # Complex queries: increased from 2.5s to 4.0s, reduced chars from 100 to 60
                                            min_silence = 4.0  # More patient - allows thinking pauses
                                            min_chars = 60     # Reduced - allow shorter complete utterances
                                            min_fragments = 3  # Keep at 3 but more flexible
                                        else:
                                            # Default: increased from 2.0s to 3.5s, reduced chars from 100 to 60
                                            min_silence = 3.5  # More patient for natural pauses
                                            min_chars = 60     # Reduced - allow shorter complete utterances
                                            min_fragments = 3  # Keep at 3
                                        
                                        # Maximum silence threshold to prevent indefinite waiting
                                        max_silence = 6.0
                                        
                                        # More flexible logic: OR between character count and fragment count
                                        # This allows either shorter complete sentences OR multiple fragments to trigger
                                        has_enough_content = (len(current_transcript) >= min_chars) or (fragment_count >= min_fragments)
                                        has_sufficient_silence = min_silence <= time_since_last_chunk <= max_silence
                                        
                                        if current_transcript and has_sufficient_silence and has_enough_content:
                                            # High confidence: speech ended, finalize immediately
                                            transcript_result = current_transcript
                                            logger.info(
                                                f" Leibniz: Early completion detected "
                                                f"(silence: {time_since_last_chunk:.1f}s [{min_silence}-{max_silence}s], "
                                                f"chars: {len(current_transcript)} [min: {min_chars}], "
                                                f"fragments: {fragment_count} [min: {min_fragments}], "
                                                f"context: {context}) "
                                                f"- Reason: {'chars' if len(current_transcript) >= min_chars else 'fragments'}"
                                            )
                                            print(f" Leibniz: Early completion detected (no chunks for {time_since_last_chunk:.1f}s)")
                                            print(f" Leibniz: Complete transcript: {transcript_result}")
                                            
                                            # Emit final callback
                                            if streaming_callback and not final_callback_emitted:
                                                streaming_callback(transcript_result, is_final=True)
                                                final_callback_emitted = True
                                            
                                            self.is_listening = False
                                            break
                                    except Exception:
                                        pass
                        
                        # Handle turn completion (Comment 5: Configurable - can be enabled/disabled)
                        if response.server_content and response.server_content.turn_complete:
                            if self.config.ignore_turn_complete:
                                logger.debug("⏭ Turn complete signal ignored (config.ignore_turn_complete=True)")
                                continue
                            
                            # Only ignore turn_complete if no speech detected yet
                            if speech_detected:
                                logger.info(f" Turn complete received after speech - finalizing transcript")
                                break
                            else:
                                logger.debug("⏳ Turn complete received but no speech yet - continuing to listen")
                                continue
                    
                finally:
                    # Stop audio streaming
                    self.is_listening = False
                    send_task.cancel()
                    try:
                        await send_task
                    except asyncio.CancelledError:
                        pass
            
            # Finalize transcript using TranscriptBuffer
            if not transcript_result:
                # Build transcript from buffer if early completion didn't fire
                transcript_result = transcript_buffer.get_final_transcript()
                if transcript_result:
                    logger.debug(" Built transcript from TranscriptBuffer (turn_complete path)")
            
            if transcript_result:
                # Reset timeout counter on success
                self.consecutive_timeouts = 0
                
                # Comment 9: Normalization removed here - done once in capture_leibniz_speech()
                # This prevents double normalization
                
                # PHASE 2 CHANGE 2.3: Add capture timing log + Comment 8 diagnostics
                capture_elapsed = time.time() - capture_start
                
                # Comment 8: Log concise summary with diagnostic metrics
                avg_silence = sum(silence_durations) / len(silence_durations) if silence_durations else 0.0
                logger.info(
                    f" Speech captured in {capture_elapsed*1000:.1f}ms: '{transcript_result[:50]}...' | "
                    f"Diagnostics: fragments={fragment_count}, chars={cumulative_length}, "
                    f"pre_buffer_sent={pre_buffer_chunks_sent}, dropped={dropped_count}, "
                    f"avg_silence={avg_silence:.2f}s, silence_gaps={len(silence_durations)}"
                )
                
                # Final callback if not already emitted
                if streaming_callback and not final_callback_emitted:
                    streaming_callback(transcript_result, is_final=True)
                
                logger.info(
                    f" Final transcript complete: '{transcript_result}' "
                    f"({fragment_count} fragments)"
                )
                
                # Update metrics
                capture_time = time.time() - capture_start
                self.capture_count += 1
                self.total_capture_time += capture_time
                self.avg_capture_time = self.total_capture_time / self.capture_count
                
                self.last_transcript = transcript_result
                self._last_capture_ended_at = time.time()
                
                # Session cleanup confirmation
                print(" Leibniz: Session cleaned up for next capture")
                
                # Trigger smart warmup
                self._trigger_smart_warmup_background()
            
            return transcript_result
            
        except Exception as e:
            logger.error(f" Capture error: {e}")
            
            # Force session reset on errors
            if "1011" in str(e) or "1006" in str(e) or "event loop" in str(e):
                logger.warning(" Forcing session reset due to error")
                await LeibnizPersistentSession.close_session()
            
            return None
            
        finally:
            self._active = False
            self.is_listening = False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get VAD performance statistics (Comment 8: Enhanced diagnostics)"""
        session_stats = LeibnizPersistentSession.get_session_stats()
        
        return {
            "conversation_state": self.conversation_state,
            "is_agent_speaking": self.is_agent_speaking,
            "barge_in_detected": self.barge_in_detected,
            "last_transcript": self.last_transcript,
            "capture_count": self.capture_count,
            "avg_capture_time": self.avg_capture_time,
            "consecutive_timeouts": self.consecutive_timeouts,
            "session_stats": session_stats,
            # Comment 8: Expose diagnostic counters
            "current_context": getattr(self, '_current_context', 'unknown'),
            "current_attempt": getattr(self, '_current_attempt', 0)
        }


# Global singleton instance
_leibniz_vad_instance = None

def get_leibniz_vad() -> LeibnizBidirectionalVAD:
    """Get singleton Leibniz VAD instance"""
    global _leibniz_vad_instance
    if _leibniz_vad_instance is None:
        _leibniz_vad_instance = LeibnizBidirectionalVAD()
    return _leibniz_vad_instance


# API Functions (matching SINDH pattern)

async def capture_leibniz_speech(
    streaming_callback: Optional[Callable[[str, bool], None]] = None,
    context: Optional[Dict[str, Any]] = None,
    audio_source: Optional['AudioSource'] = None
) -> Optional[str]:
    """
    Main API function for capturing English speech.
    
    CRITICAL: Returns transcript only (NO audio file) - matches SINDH pattern.
    
    Args:
        streaming_callback: Optional callback for real-time transcript fragments
        context: Optional context dict with 'conversation_context' and 'attempt_count' keys
    
    Returns:
        Normalized English transcript string or None on timeout/error
    """
    vad = get_leibniz_vad()
    
    # Set dynamic timeout from context if provided (Comment 7: Support both 'attempt' and 'attempt_count')
    if context:
        conversation_context = context.get('conversation_context', 'initial')
        # Prefer 'attempt_count' when present, fallback to 'attempt' for backward compatibility
        attempt = context.get('attempt_count', context.get('attempt', 0))
        vad.set_dynamic_timeout(attempt_count=attempt, conversation_context=conversation_context)
    
    # Capture speech bidirectionally
    transcript = await vad.capture_speech_bidirectional(streaming_callback=streaming_callback, audio_source=audio_source)
    
    # Apply normalization (Comment 9: Only normalize once here, not in capture_speech_bidirectional)
    if transcript:
        transcript = normalize_english_transcript(transcript)
    
    return transcript


async def set_leibniz_agent_speaking(is_speaking: bool, context: str = ""):
    """
    Set agent speaking state for bidirectional flow.

    Call this before/after TTS:
    await set_leibniz_agent_speaking(True, "Starting TTS")
    # ... TTS code ...
    await set_leibniz_agent_speaking(False, "TTS completed")
    
    SIMPLE SOLUTION: Turn off mic during TTS to prevent audio feedback.
    """
    vad = get_leibniz_vad()
    await vad.set_agent_speaking_state(is_speaking, context)

    # SIMPLE SOLUTION: Turn off mic during TTS to prevent audio feedback
    try:
        from leibniz_continuous_vad import get_continuous_vad
        continuous_vad = get_continuous_vad()

        if is_speaking:
            # Stop continuous VAD during TTS to prevent audio feedback
            await continuous_vad.stop_continuous_listening()
            logger.info(" Microphone disabled during TTS (preventing audio feedback)")
        else:
            # Restart continuous VAD after TTS
            await continuous_vad.start_continuous_listening()
            logger.info(" Microphone re-enabled after TTS")

    except ImportError:
        # Continuous VAD not available, skip control
        pass
    except Exception as e:
        logger.warning(f" Error controlling microphone during TTS: {e}")


async def reset_leibniz_conversation():
    """Reset conversation state (barge-in and timeout counters)"""
    vad = get_leibniz_vad()
    with vad._speaking_lock:
        vad.barge_in_detected = False
    vad.consecutive_timeouts = 0
    logger.debug(" Leibniz conversation state reset")


async def cleanup_leibniz_vad():
    """Cleanup Leibniz VAD resources"""
    try:
        await LeibnizPersistentSession.close_session()
        logger.info(" Leibniz VAD cleaned up")
    except Exception as e:
        logger.warning(f" Leibniz: Cleanup error: {e}")


def check_leibniz_barge_in() -> bool:
    """Check if barge-in was detected"""
    vad = get_leibniz_vad()
    with vad._speaking_lock:
        return vad.barge_in_detected


def clear_leibniz_barge_in():
    """Clear barge-in flag"""
    vad = get_leibniz_vad()
    with vad._speaking_lock:
        vad.barge_in_detected = False


async def warmup_leibniz_vad(preconnect_s: float = 0.0) -> Dict[str, Any]:
    """
    Warmup Leibniz VAD session (Comment 8: Updated signature to match tests)
    
    Args:
        preconnect_s: Optional delay before creating session (for timing tests)
    
    Returns:
        Dict with 'session_created': True on success or 'error': str on failure
    """
    try:
        vad = get_leibniz_vad()
        if vad.client:
            logger.info(" Warming up Leibniz VAD")
            
            # Optional delay before connection
            if preconnect_s > 0:
                await asyncio.sleep(preconnect_s)
            
            await LeibnizPersistentSession.get_session(
                vad.client,
                vad.config.model_name,
                vad.config
            )
            logger.info(" Leibniz VAD warmup completed")
            return {"session_created": True}
        else:
            error_msg = "No client available for warmup"
            logger.warning(f" {error_msg}")
            return {"error": error_msg}
    except Exception as e:
        error_msg = str(e)
        logger.error(f" Warmup failed: {error_msg}")
        return {"error": error_msg}


async def smart_warmup_leibniz_vad():
    """
    Smart warmup trigger for background pre-warming during TTS.
    Non-blocking fire-and-forget task.
    """
    vad = get_leibniz_vad()
    if vad.client:
        vad._trigger_smart_warmup_background()


def is_leibniz_vad_active() -> bool:
    """Check if VAD is currently active"""
    vad = get_leibniz_vad()
    return vad._active


def get_leibniz_vad_metrics() -> Dict[str, Any]:
    """Get comprehensive performance metrics"""
    vad = get_leibniz_vad()
    return vad.get_performance_metrics()


# Module test function
if __name__ == "__main__":
    async def test_leibniz_vad():
        """Test Leibniz bidirectional VAD system"""
        print(" Testing Leibniz Bidirectional VAD")
        print("=" * 40)
        
        vad = get_leibniz_vad()
        
        if not vad.client:
            print(" No API key - cannot test")
            return
        
        try:
            # Test speech capture
            print(" Testing speech capture...")
            
            def callback(fragment, is_final):
                print(f"{'FINAL' if is_final else 'Fragment'}: {fragment}")
            
            transcript = await capture_leibniz_speech(streaming_callback=callback)
            
            if transcript:
                print(f" Captured: {transcript}")
                
                # Test agent speaking state
                await set_leibniz_agent_speaking(True, "Testing agent state")
                await asyncio.sleep(1.0)
                await set_leibniz_agent_speaking(False, "Test complete")
                
                # Show metrics
                metrics = get_leibniz_vad_metrics()
                print(f" Metrics: {metrics}")
            else:
                print("⏰ No speech captured")
            
            print(" Leibniz VAD test completed!")
            
        finally:
            await cleanup_leibniz_vad()
    
    asyncio.run(test_leibniz_vad())
