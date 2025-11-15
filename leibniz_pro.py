#!/usr/bin/env python3
"""
Leibniz Pro - Professional Orchestration Module
================================================

Main conversation orchestration module for Leibniz University customer service agent.
English-only, friendly casual tone, integrated appointment booking.

Architecture:
- Message-Based Communication: Clean component integration using structured messages
- Dual Pre-Warming Strategy: Low latency via pre-warming during TTS + speech detection
- Barge-In Support: Natural conversation flow with interruption detection
- Streaming TTS: Progressive response playback for reduced perceived latency
- Persistent Services: Async queue-based processing with request deduplication
- Error Handling: Graceful fallbacks for all component failures

Dual Pre-warming Strategy (TARA Pattern):
1. During TTS (speak_friendly): warmup_leibniz_vad() + prewarm_leibniz_during_tts()
   - Prepares VAD session and RAG models for NEXT user input
   - Eliminates cold starts on subsequent captures
2. At Speech Detection (streaming callback in VAD): services.prewarm_rag()
   - Ensures models are warm for CURRENT query processing
   - Triggered on first speech fragment
This dual approach handles all conversation patterns: quick responses, long pauses, first queries

Message-Based Architecture (MCP Preparation):
- transcribe_and_classify() → Returns: (TranscriptMessage, IntentMessage)
- handle_rag_query() → Returns: RAGMessage
- speak_friendly() → Returns: TTSMessage (with audio path + duration)
Enables clean microservices migration without breaking existing code

Conversation Flow:
1. Initialization: Load all services, initialize persistent services manager
2. Greeting: Play intro audio or speak greeting
3. Main Loop (Max 5 attempts, TARA pattern):
   - ATTEMPT TRACKING: consecutive_no_input, last_interaction_type, conversation_active
   - CONTEXT DETERMINATION: Dynamic timeout (greeting=12s, decision=15s, complex=18s, post_service=8s)
   - CAPTURE & CLASSIFY: transcribe_and_classify() with streaming callback
   - INTENT ROUTING: Extract intent, should_use_rag, confidence
   - RESPONSE HANDLING: RAG_QUERY → RAG + streaming TTS, APPOINTMENT → FSM
   - SESSION CLEANUP: Clear queues, reset counters
4. Exit: Cleanup and goodbye message

Components:
- VAD: leibniz_vad.py (English-only Gemini Live)
- STT: leibniz_stt.py (integrated in VAD)
- Intent Parser: leibniz_intent_parser.py (via persistent services)
- RAG: leibniz_rag.py (context-aware retrieval)
- Appointment FSM: leibniz_appointment_fsm.py (slot filling)
- TTS: leibniz_tts.py (triple-provider: Gemini/Google/ElevenLabs)
- Persistent Services: leibniz_persistent_services.py (async processing)
- Messages: leibniz_messages.py (structured communication)
- Config: leibniz_config.py (friendly casual tone)

Key Differences from TARA:
- No Hindi features (number conversion, transliteration, script handling)
- No phone collection or normalization
- No worker registration or MongoDB operations
- No service routing (applied jobs, new jobs, personal info browser)
- Simpler conversation flow: greeting → intent → RAG/appointment → response
- English-only VAD (en-US)
- University-specific (Leibniz customer service)
- Friendly casual tone (not formal academic)

Usage:
    from leibniz_agent.leibniz_pro import main
    
    # Run agent
    asyncio.run(main())
    
    # Or run single session
    await initialize_leibniz_services()
    await run_conversation_session()
"""

import os
import sys
import asyncio
import logging
import time
import re  # FIX: Add missing re import for sentence splitting
import soundfile as sf  # For audio duration detection (smart warmup)
import hashlib  # For deduplication hash set (Comment 7)
from collections import deque  # For staging buffer (Comment 1 & 2)
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path

# Fix module imports when running script directly (not as module)
# Add parent directory to sys.path so 'leibniz_agent' package is importable
if __name__ == "__main__":
    # Get parent directory (SINDH-Orchestra-Complete)
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env.leibniz
from dotenv import load_dotenv
from pathlib import Path

# Get the leibniz_agent directory (where this file is located)
LEIBNIZ_DIR = Path(__file__).parent
ENV_FILE = LEIBNIZ_DIR / ".env.leibniz"

# Load Leibniz-specific environment variables
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
    logger.info(f"✅ Loaded environment from: {ENV_FILE}")
else:
    logger.warning(f"⚠️  .env.leibniz not found at: {ENV_FILE}")
    # Try loading default .env as fallback
    load_dotenv()

# Import pygame for audio playback (optional - Comment 4)
try:
    import pygame
    # Delay mixer initialization to avoid headless/CI crashes (Comment 4)
    # Initialize pygame mixer more robustly with retry logic
    pygame_init_success = False

    # Force stdout logging for pygame initialization visibility
    import sys
    original_stdout = sys.stdout

    print("🎵 INITIALIZING PYGAME MIXER FOR AUDIO PLAYBACK...", flush=True)
    print("🎵 This may take a few seconds on Windows systems...", flush=True)

    # Attempt 1: Preferred settings (high quality)
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame_init_success = True
        print("✅ PYGAME MIXER INITIALIZED: frequency=44100, channels=2, buffer=512", flush=True)
    except Exception as e:
        print(f"⚠️ PYGAME MIXER ATTEMPT 1 FAILED: {e}", flush=True)

    # Attempt 2: More compatible settings
    if not pygame_init_success:
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=1024)
            pygame_init_success = True
            print("✅ PYGAME MIXER INITIALIZED WITH FALLBACK: frequency=22050, channels=1, buffer=1024", flush=True)
        except Exception as e:
            print(f"⚠️ PYGAME MIXER ATTEMPT 2 FAILED: {e}", flush=True)

    # Attempt 3: Default settings
    if not pygame_init_success:
        try:
            pygame.mixer.init()
            pygame_init_success = True
            print("✅ PYGAME MIXER INITIALIZED WITH DEFAULTS", flush=True)
        except Exception as e:
            print(f"❌ PYGAME MIXER INITIALIZATION FAILED - AUDIO PLAYBACK WILL NOT WORK: {e}", flush=True)

    # Windows-specific diagnostics
    if pygame_init_success:
        try:
            # Get mixer info for diagnostics
            mixer_info = pygame.mixer.get_init()
            if mixer_info:
                freq, size, channels = mixer_info
                print(f"🎵 PYGAME MIXER INFO: {freq}Hz, {size}bit, {channels}ch", flush=True)
            else:
                print("⚠️ PYGAME MIXER INFO UNAVAILABLE", flush=True)
        except Exception as e:
            print(f"⚠️ PYGAME MIXER INFO ERROR: {e}", flush=True)

        # Test basic mixer functionality
        try:
            pygame.mixer.music.set_volume(0.8)  # Test volume control
            print("✅ PYGAME MIXER FUNCTIONALITY TEST PASSED", flush=True)
        except Exception as e:
            print(f"⚠️ PYGAME MIXER FUNCTIONALITY TEST FAILED: {e}", flush=True)
            pygame_init_success = False  # Mark as failed if basic functionality doesn't work

    PYGAME_AVAILABLE = pygame_init_success
    print(f"🎵 PYGAME_AVAILABLE = {PYGAME_AVAILABLE}", flush=True)

    # Force stdout flush to ensure visibility
    sys.stdout.flush()

except ImportError as e:
    print(f"⚠️ PYGAME NOT AVAILABLE - INTRO/BACKGROUND AUDIO DISABLED: {e}", flush=True)
    PYGAME_AVAILABLE = False

# Import sounddevice for audio fallback (optional)
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    print("✅ sounddevice available for audio fallback")
except ImportError:
    print("⚠️ sounddevice not available - audio fallback disabled")
    SOUNDDEVICE_AVAILABLE = False

# Import RAG infrastructure (TARA pattern)
try:
    from rag_cache import get_rag_cache_manager
    from rag_speculative import get_speculative_coordinator
    from rag_performance_tracker import get_performance_tracker, RAGPerformanceMetrics
    from rag_fallback_responses import get_fallback_manager
except ImportError as e:
    logger.warning(f"RAG infrastructure imports failed (optional): {e}")
    get_rag_cache_manager = None
    get_speculative_coordinator = None
    get_performance_tracker = None
    RAGPerformanceMetrics = None
    get_fallback_manager = None

# Import Leibniz components
try:
    from leibniz_agent.leibniz_config import get_leibniz_config, SEMANTIC_CONTEXT_DEBUG
    from leibniz_agent.leibniz_messages import (
        TranscriptMessage, IntentMessage, RAGMessage, TTSMessage
    )
    from leibniz_agent.leibniz_stt import get_leibniz_stt
    from leibniz_agent.leibniz_tts import get_leibniz_tts
    # Use Leibniz's native intent parser with Gemini 2.0 (NOT TARA's fast router!)
    from leibniz_agent.leibniz_intent_parser import get_leibniz_parser
    from leibniz_agent.leibniz_rag import get_leibniz_rag
    from leibniz_agent.leibniz_appointment_fsm import (
        create_appointment_fsm,
        format_appointment_for_submission,
        AppointmentState
    )
    from leibniz_agent.leibniz_persistent_services import (
        get_leibniz_services_manager,
        get_leibniz_service_status,
        prewarm_leibniz_during_tts  # Comment 3 - import real prewarm function
    )
    from leibniz_agent.leibniz_stt import normalize_english_transcript
    from leibniz_agent.leibniz_semantic_translator import (
        generate_semantic_context_for_fsm,
        get_semantic_translator
    )
    from leibniz_agent.leibniz_vad import (
        get_leibniz_vad,
        capture_leibniz_speech,
        set_leibniz_agent_speaking,
        reset_leibniz_conversation,
        cleanup_leibniz_vad,
        check_leibniz_barge_in,  # Comment 2
        clear_leibniz_barge_in,  # Comment 2
        smart_warmup_leibniz_vad  # NEW: Smart warmup for TTS pre-warming
    )
    # NEW: Continuous background VAD for barge-in support
    from leibniz_agent.leibniz_continuous_vad import (
        get_continuous_vad,
        start_leibniz_continuous_listening,
        stop_leibniz_continuous_listening,
        wait_for_leibniz_speech
    )
    from leibniz_agent.leibniz_dialogue_manager import get_leibniz_dialogue_manager
except ImportError as e:
    logger.warning(f"Leibniz components imports failed: {e}")
    # Set all Leibniz functions to None for graceful degradation
    get_leibniz_config = None
    SEMANTIC_CONTEXT_DEBUG = None
    TranscriptMessage = None
    IntentMessage = None
    TTSMessage = None
    get_leibniz_stt = None
    get_leibniz_tts = None
    get_leibniz_parser = None
    get_leibniz_rag = None
    create_appointment_fsm = None
    format_appointment_for_submission = None
    AppointmentState = None
    get_leibniz_services_manager = None
    get_leibniz_service_status = None
    prewarm_leibniz_during_tts = None
    normalize_english_transcript = None
    generate_semantic_context_for_fsm = None
    get_semantic_translator = None
    get_leibniz_vad = None
    capture_leibniz_speech = None
    set_leibniz_agent_speaking = None
    reset_leibniz_conversation = None
    cleanup_leibniz_vad = None
    check_leibniz_barge_in = None
    clear_leibniz_barge_in = None
    smart_warmup_leibniz_vad = None
    get_continuous_vad = None
    start_leibniz_continuous_listening = None
    stop_leibniz_continuous_listening = None
    wait_for_leibniz_speech = None
    get_leibniz_dialogue_manager = None

# ============================================================================
# Dialogue Management Helpers
# ============================================================================

def get_dialogue_text(dialogue_key: str, fallback_text: str = "") -> str:
    """
    Get dialogue text from dialogue manager with fallback.

    Args:
        dialogue_key: Key for the dialogue (e.g., 'greeting', 'error', or 'category.key')
        fallback_text: Fallback text if dialogue manager fails

    Returns:
        Dialogue text string
    """
    try:
        dialogue_manager = get_leibniz_dialogue_manager()

        # Parse dialogue_key - if it contains a dot, split into category.key
        if '.' in dialogue_key:
            category, key = dialogue_key.split('.', 1)
            text = dialogue_manager.get_dialogue(category, key)
        else:
            # For backward compatibility, try to infer category from key
            # Common categories: greetings, errors, prompts, appointments, thinking
            category_map = {
                'greeting': 'greetings',
                'intro': 'greetings',
                'welcome': 'greetings',
                'goodbye': 'farewells',
                'farewell': 'farewells',
                'exit': 'farewells',
                'error': 'errors',
                'timeout': 'errors',
                'no_input': 'errors',
                'technical': 'errors',
                'help_offer': 'prompts',
                'continue': 'prompts',
                'clarify': 'prompts',
                'confirmation': 'prompts',
                'appointment_confirm': 'appointments',
                'date_prompt': 'appointments',
                'time_prompt': 'appointments',
                'email_prompt': 'appointments',
                'name_prompt': 'appointments',
                'thinking': 'thinking',
                'thinking_indicator': 'thinking',
            }
            category = category_map.get(dialogue_key, 'prompts')
            text = dialogue_manager.get_dialogue(category, dialogue_key)

        return text if text else fallback_text
    except Exception as e:
        logger.warning(f"Failed to get dialogue text for '{dialogue_key}': {e}")
        return fallback_text

# Conditional imports for speculative coordinator and performance config
try:
    from rag_speculative import get_speculative_coordinator
    from performance_config import get_performance_config
    SPECULATIVE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Speculative coordinator not available: {e}")
    SPECULATIVE_AVAILABLE = False
    get_speculative_coordinator = None
    get_performance_config = None


# ============================================================================
# Paths and Globals
# ============================================================================

INTRO_AUDIO_PATH = os.path.join("leibniz_agent", "audio", "intro.wav")
VOICE_DIR = os.path.join("leibniz_agent", "voices")
DIALOGUE_ARCHIVE_DIR = os.path.join("leibniz_agent", "audio", "dialogues")
BACKGROUND_AUDIO_PATH = os.path.join("leibniz_agent", "audio", "background.wav")

# TARA-style dialogue cache names mapping (now uses dialogue manager)
def get_dialogue_cache_name(dialogue_key: str) -> Optional[str]:
    """
    Get cache name for dialogue key from dialogue manager.

    Args:
        dialogue_key: Key for the dialogue (e.g., 'greeting', 'greetings.intro')

    Returns:
        Cache name string or None if not found
    """
    try:
        dialogue_manager = get_leibniz_dialogue_manager()
        
        # Split dialogue_key into category and key parts
        if '.' in dialogue_key:
            category, key = dialogue_key.split('.', 1)
            dialogue_text = dialogue_manager.get_dialogue(category, key)
        else:
            # For backward compatibility, try to infer category from key
            category = dialogue_manager.get_dialogue_category(dialogue_key)
            dialogue_text = dialogue_manager.get_dialogue(category, dialogue_key)
        
        if dialogue_text:
            # Use MD5 hash of dialogue text as cache name for consistency
            import hashlib
            return hashlib.md5(dialogue_text.encode()).hexdigest()
        return None
    except Exception as e:
        logger.warning(f"Failed to get dialogue cache name for '{dialogue_key}': {e}")
        return None

# Legacy mapping for backward compatibility (deprecated - use get_dialogue_cache_name())
# DIALOGUE_CACHE_NAMES = {
#     'greeting': 'intro_greeting',
#     'intro': 'intro_greeting',
#     'welcome': 'intro_greeting',
#     'goodbye': 'outro_farewell',
#     'farewell': 'outro_farewell',
#     'exit': 'outro_farewell',
#     'help_offer': 'help_prompt',
#     'continue': 'continue_prompt',
#     'clarify': 'clarification_prompt',
#     'error': 'error_message',
#     'error_fallback': 'error_message',
#     'timeout': 'timeout_message',
#     'timeout_message': 'timeout_message',
#     'thinking': 'thinking_indicator',
#     'appointment_confirm': 'appointment_confirm',
#     'tts_error_fallback': 'tts_error_fallback',
# }

# Create dialogue archive directory
os.makedirs(DIALOGUE_ARCHIVE_DIR, exist_ok=True)

# Comment 3 FIX: Sentence normalization for deduplication
def normalize_sentence_for_hash(sentence: str) -> str:
    """
    Normalize sentence before hashing to prevent duplicate synthesis.
    Strips whitespace and normalizes terminal punctuation to single variant.
    
    Args:
        sentence: Raw sentence text
        
    Returns:
        Normalized sentence for consistent hashing
    """
    # Strip leading/trailing whitespace
    normalized = sentence.strip()
    
    # Normalize multiple terminal punctuation to single period
    # "Hello!!" -> "Hello."
    # "What??" -> "What."
    import re
    normalized = re.sub(r'([.!?])+\s*$', '.', normalized)
    
    # Normalize internal whitespace to single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized

# Global variables
background_player: Optional['BackgroundAudioPlayer'] = None
services_manager: Optional[Any] = None
leibniz_config: Optional[Any] = None
conversation_active: bool = False
# Comment 9: Removed global current_fsm - FSM state should be session-scoped, not global

# Continuous VAD state (for background listening mode)
_continuous_vad_enabled = False  # Toggle via environment variable
_continuous_vad_instance = None  # Singleton instance
_user_speech_ready = asyncio.Event()  # Event to signal main loop
_current_user_transcript = None  # Latest transcript from background listener
_current_user_intent = None  # Latest intent from callback


async def cancel_continuous_vad():
    """
    Unified cancellation path for continuous VAD shutdown.
    
    Can be called from signal handlers, exception handlers, or main shutdown.
    Ensures clean shutdown regardless of VAD state.
    """
    global _continuous_vad_enabled, _continuous_vad_instance
    
    if not _continuous_vad_enabled:
        return  # Already disabled
    
    logger.info("🔄 Cancelling continuous VAD...")
    
    try:
        # Stop the continuous listening
        if _continuous_vad_instance:
            await stop_leibniz_continuous_listening()
            logger.info("✅ Continuous VAD cancelled successfully")
        else:
            logger.debug("ℹ️ No continuous VAD instance to cancel")
    
    except Exception as e:
        logger.error(f"❌ Error during continuous VAD cancellation: {e}")
    
    finally:
        # Always reset state
        _continuous_vad_enabled = False
        _continuous_vad_instance = None
        _user_speech_ready.clear()
        logger.debug("🧹 Continuous VAD state reset")


# ============================================================================
# Background Audio Player Class
# ============================================================================

class BackgroundAudioPlayer:
    """Background audio player for ambient sound (optional)"""
    
    def __init__(self, audio_path: str, volume: float = 1.0):
        self.audio_path = audio_path
        self.volume = volume  # Increased from 0.3 to 0.5 for better audibility
        self._bg_channel = None
        self._bg_sound = None
        self.is_playing = False
        
        if not PYGAME_AVAILABLE:
            logger.warning("pygame not available - background audio disabled")
            return
    
    def _initialize_pygame(self):
        """Initialize pygame mixer with safe defaults"""
        try:
            # Pre-init with specific audio settings
            if not pygame.mixer.get_init():
                pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
                pygame.mixer.init()
            
            # Set number of channels (0 reserved for background)
            pygame.mixer.set_num_channels(8)
            logger.debug("✅ pygame mixer initialized for background audio")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize pygame mixer: {e}")
            return False
    
    def start(self):
        """Start playing background audio in loop"""
        if not PYGAME_AVAILABLE:
            return
        
        # Initialize pygame mixer
        if not self._initialize_pygame():
            return
        
        try:
            # Load the sound file
            if os.path.exists(self.audio_path):
                self._bg_sound = pygame.mixer.Sound(self.audio_path)
                self._bg_sound.set_volume(self.volume)
                logger.debug(f"✅ Background audio loaded: {self.audio_path}")
            else:
                # Comment 5: Auto-convert M4A to WAV if needed
                if not self.audio_path.endswith('.wav'):
                    converted_path = self._try_convert_to_wav(self.audio_path)
                    if converted_path:
                        self.audio_path = converted_path
                        self._bg_sound = pygame.mixer.Sound(self.audio_path)
                        self._bg_sound.set_volume(self.volume)
                        logger.debug(f"✅ Background audio loaded (converted): {self.audio_path}")
                    else:
                        logger.warning(f"Background audio not found: {self.audio_path}")
                        return
                else:
                    logger.warning(f"Background audio not found: {self.audio_path}")
                    return
            
            # Create dedicated channel (0) for background
            self._bg_channel = pygame.mixer.Channel(0)
            self._bg_channel.set_volume(self.volume)
            
            # Start playing on loop
            self._bg_channel.play(self._bg_sound, loops=-1)
            self.is_playing = True
            logger.debug("🎵 Background audio started on channel 0")
            
        except Exception as e:
            logger.error(f"Failed to start background audio: {e}")
    
    def _try_convert_to_wav(self, source_path: str) -> Optional[str]:
        """
        Comment 5: Try to convert M4A/other formats to WAV using pydub
        Returns: Path to converted WAV file or None if conversion failed
        """
        try:
            from pydub import AudioSegment
            
            # Target WAV path
            wav_path = "./leibniz_agent/audio/background.wav"
            
            logger.debug(f"Attempting to convert {source_path} to WAV...")
            
            # Load and convert
            audio = AudioSegment.from_file(source_path)
            audio.export(wav_path, format="wav")
            
            logger.debug(f"✅ Converted audio to: {wav_path}")
            return wav_path
            
        except ImportError:
            logger.warning("pydub not available - cannot auto-convert audio format")
            logger.info("Install with: pip install pydub")
            return None
        except Exception as e:
            logger.warning(f"Failed to convert audio: {e}")
            return None
    
    def get_current_volume(self) -> float:
        """Get current background audio volume"""
        if self._bg_channel and self.is_playing:
            try:
                return self._bg_channel.get_volume()
            except:
                return self.volume
        return self.volume
    
    def stop(self):
        """Stop background audio"""
        if self._bg_channel and self.is_playing:
            try:
                self._bg_channel.stop()
                self.is_playing = False
                logger.debug("🔇 Background audio stopped")
            except Exception as e:
                logger.error(f"Failed to stop background audio: {e}")
    
    def set_volume(self, volume: float):
        """Set background audio volume (0.0-1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        if self._bg_channel and self.is_playing:
            self._bg_channel.set_volume(self.volume)
        elif self._bg_sound:
            self._bg_sound.set_volume(self.volume)
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        self._bg_sound = None
        self._bg_channel = None


def start_background_audio() -> bool:
    """Start background audio with error handling"""
    global background_player
    
    # Check if enabled in config
    if not os.getenv("LEIBNIZ_ENABLE_BACKGROUND_AUDIO", "false").lower() == "true":
        return False
    
    if not PYGAME_AVAILABLE:
        return False
    
    try:
        audio_path = os.getenv("LEIBNIZ_BACKGROUND_AUDIO_PATH", BACKGROUND_AUDIO_PATH)
        # Use configurable volume from environment variable (default 30% instead of 100%)
        volume = float(os.getenv("LEIBNIZ_BACKGROUND_AUDIO_VOLUME", "0.3"))
        
        background_player = BackgroundAudioPlayer(audio_path, volume)
        background_player.start()
        
        # Log the actual volume being used
        logger.info(f"🔊 Background audio started at {volume:.1f} volume ({volume*100:.0f}%)")
        
        return True
    except Exception as e:
        logger.error(f"Failed to start background audio: {e}")
        return False


def stop_background_audio():
    """Stop background audio and cleanup"""
    global background_player
    
    if background_player:
        background_player.cleanup()
        background_player = None


# ============================================================================
# TTS Streaming Queue
# ============================================================================

# ============================================================================
# Global Queue Infrastructure for Streaming TTS (TARA pattern)
# ============================================================================

# Global queue for sentence-level streaming from RAG to TTS (bounded for backpressure)
_tts_streaming_queue = asyncio.Queue(maxsize=15)  # Bounded queue to prevent memory bloat
_streaming_active = False
_tts_consumer_task: Optional[asyncio.Task] = None  # Shared consumer task handle
_cancel_streaming = asyncio.Event()  # Cancellation signal for barge-in
_sentence_buffer = []  # Global sentence buffer for progressive streaming

# Comment 8: Sentinel tracking to prevent multiple sentinels
_sentinel_sent = False

# Comment 2 FIX: Sentinel debugging counters
_sentinels_sent_count = 0
_sentinels_received_count = 0

# Comment 7: In-flight synthesis deduplication
_synthesis_in_flight = set()  # Set of sentence hashes currently being synthesized

# Comment 6 FIX: Playback guard to prevent overlapping playback
_playing_now = False  # True during active pygame playback, False otherwise

# Global Leibniz parser instance (Gemini 2.0 based)
_leibniz_parser = None

# Comment 3 FIX: LRU cache for recently synthesized sentences (prevents immediate re-synth)
from collections import OrderedDict
_recent_synthesis_cache = OrderedDict()  # {normalized_hash: timestamp}
_recent_synthesis_cache_size = 64  # Size 32-64 as per comment
_recent_synthesis_ttl_seconds = 10.0  # 10 second TTL for deduplication

# VERIFICATION COMMENT 2: First-playback latency tracking
_first_enqueue_at: Optional[float] = None  # Timestamp when first sentence was enqueued

# Comment 12: Streaming mode tracking (log once per session)
_streaming_mode_logged = False


def check_recent_synthesis(sentence: str) -> tuple[bool, Optional[str]]:
    """
    Check if sentence was recently synthesized (within TTL window).
    Comment 3 FIX: LRU cache with 10s TTL to prevent duplicate synthesis.
    VERIFICATION COMMENT 1: Returns tuple (should_skip, cached_path) to enable playback of cached audio.
    
    Args:
        sentence: Sentence to check
        
    Returns:
        Tuple of (should_skip: bool, cached_path: Optional[str])
        - (False, cached_path): Found cached file, use it for playback
        - (True, None): Too recent without valid cache file, skip entirely
        - (False, None): Not recent, proceed with fresh synthesis
    """
    global _recent_synthesis_cache
    
    # Normalize before hashing
    normalized = normalize_sentence_for_hash(sentence)
    sentence_hash = hashlib.md5(normalized.encode()).hexdigest()
    
    current_time = time.time()
    
    # Check if in cache and not expired
    if sentence_hash in _recent_synthesis_cache:
        synth_time = _recent_synthesis_cache[sentence_hash]
        if (current_time - synth_time) < _recent_synthesis_ttl_seconds:
            # Move to end (LRU)
            _recent_synthesis_cache.move_to_end(sentence_hash)
            
            # VERIFICATION COMMENT 1: Look for cached audio file
            # Check both audio_archive and voices directories for cached files
            cached_paths = [
                f"./leibniz_agent/audio_archive/{sentence_hash}.wav",
                f"./leibniz_agent/voices/{sentence_hash}.wav",
            ]
            
            for cached_path in cached_paths:
                if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
                    logger.debug(f"✅ Found cached audio (age: {current_time - synth_time:.1f}s): '{sentence[:30]}...' at {cached_path}")
                    return (False, cached_path)  # Don't skip, use cached file
            
            # In cache but no valid file found - skip to avoid re-synthesis within TTL
            logger.debug(f"⚠️ Skipping recent synthesis (age: {current_time - synth_time:.1f}s, no cache file): '{sentence[:30]}...'")
            return (True, None)  # Skip - recently synthesized but no file
        else:
            # Expired - remove
            del _recent_synthesis_cache[sentence_hash]
    
    # Add to cache
    _recent_synthesis_cache[sentence_hash] = current_time
    
    # Maintain size limit (LRU eviction)
    while len(_recent_synthesis_cache) > _recent_synthesis_cache_size:
        _recent_synthesis_cache.popitem(last=False)  # Remove oldest
    
    return (False, None)  # Not recent - proceed with synthesis


# Comment 12: Streaming mode tracking (log once per session)
_streaming_mode_logged = False


async def clear_tts_queue():
    """
    Clear any residual items from the TTS streaming queue and reset sentence buffer.
    Comment 8: Does NOT push sentinel - only drains queue. Sentinel is producer responsibility.
    """
    global _tts_streaming_queue, _sentence_buffer, _cancel_streaming
    
    # Signal cancellation for any active consumer
    _cancel_streaming.set()
    
    while not _tts_streaming_queue.empty():
        try:
            _tts_streaming_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    
    # Comment 8: Do NOT push sentinel here - let producer manage sentinel
    # Clear sentence buffer
    _sentence_buffer = []
    
    # Reset sentence buffer to prevent cross-session corruption
    _sentence_buffer = []
    
    # Reset cancellation event
    _cancel_streaming.clear()
    
    # Silent for performance - logger.debug("🧹 TTS queue cleared and sentence buffer reset")


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences for streaming TTS (English-focused with TARA pattern)
    
    Args:
        text: Input text
        
    Returns:
        List of sentences with preserved punctuation
    """
    # Preserve abbreviations by protecting periods
    abbrev_map = {
        "Dr.": "Dr<PERIOD>",
        "Mr.": "Mr<PERIOD>",
        "Mrs.": "Mrs<PERIOD>",
        "Ms.": "Ms<PERIOD>",
        "Prof.": "Prof<PERIOD>",
        "Sr.": "Sr<PERIOD>",
        "Jr.": "Jr<PERIOD>",
        "etc.": "etc<PERIOD>",
        "vs.": "vs<PERIOD>",
        "e.g.": "e<PERIOD>g<PERIOD>",
        "i.e.": "i<PERIOD>e<PERIOD>",
    }
    
    # Replace abbreviations
    for abbrev, placeholder in abbrev_map.items():
        text = text.replace(abbrev, placeholder)
    
    # Split on sentence delimiters - TARA pattern
    # Keep punctuation with sentences
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    
    # Filter empty and very short fragments
    valid_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Restore abbreviations
        for abbrev, placeholder in abbrev_map.items():
            sentence = sentence.replace(placeholder, abbrev)
        
        if sentence and len(sentence) > 3:  # Minimum 3 chars to keep short meaningful sentences
            valid_sentences.append(sentence)
    
    # If no valid sentences found (no punctuation), split on length
    if not valid_sentences and text.strip():
        # Fallback: split long text into ~50 char chunks at word boundaries
        words = text.split()
        chunk = []
        chunk_len = 0
        for word in words:
            chunk.append(word)
            chunk_len += len(word) + 1
            if chunk_len >= 50:
                valid_sentences.append(' '.join(chunk))
                chunk = []
                chunk_len = 0
        if chunk:
            valid_sentences.append(' '.join(chunk))
    
    return valid_sentences


async def stream_rag_to_tts(rag_response: str, pace: float = 1.0, is_final: bool = True):
    """
    Split RAG response into sentences and stream to TTS queue with backpressure.
    Supports progressive streaming with sentence buffering for partial chunks.
    
    Args:
        rag_response: RAG response text (can be partial or complete)
        pace: Speech pace for TTS (1.0 = normal)
        is_final: If False, don't send end-of-stream signal (more chunks coming)
                  If True, send end-of-stream signal (this is the final chunk)
    """
    global _sentence_buffer, _tts_consumer_task
    
    # Comment 12: Assert consumer is running if partials are being enqueued
    streaming_enabled = os.getenv("LEIBNIZ_ENABLE_STREAMING_TTS", "true").lower() == "true"
    if streaming_enabled and (_tts_consumer_task is None or _tts_consumer_task.done()):
        logger.warning("⚠️ Streaming mode enabled but TTS consumer not running! Starting consumer...")
        await start_tts_consumer()
    
    try:
        # Accumulate text in buffer
        _sentence_buffer.append(rag_response)
        accumulated_text = ''.join(_sentence_buffer)
        
        # Split into sentences
        sentences = split_into_sentences(accumulated_text)
        
        # If not final, hold the last incomplete sentence for next chunk
        if not is_final and len(sentences) > 0:
            # Keep last sentence in buffer (might be incomplete)
            incomplete_sentence = sentences[-1]
            complete_sentences = sentences[:-1]
            _sentence_buffer = [incomplete_sentence]
        else:
            # Final chunk - send all sentences including last one
            complete_sentences = sentences
            _sentence_buffer = []  # Clear buffer
        
        # Queue complete sentences with backpressure handling
        for i, sentence in enumerate(complete_sentences, 1):
            try:
                # VERIFICATION COMMENT 2: Track first enqueue timestamp
                global _first_enqueue_at
                if _first_enqueue_at is None and i == 1:
                    _first_enqueue_at = time.time()
                    logger.debug(f"⏱️ First sentence enqueued at {_first_enqueue_at}")
                
                # Use put_nowait for non-blocking queue add
                _tts_streaming_queue.put_nowait((sentence, pace))
            except asyncio.QueueFull:
                # Backpressure: merge with last queued sentence if possible
                logger.warning(f"⚠️ TTS queue full, merging sentences to prevent bloat")
                try:
                    # Try to get last item and merge
                    last_item = _tts_streaming_queue.get_nowait()
                    if last_item is not None:
                        last_sentence, last_pace = last_item
                        merged = f"{last_sentence} {sentence}"
                        _tts_streaming_queue.put_nowait((merged, pace))
                    else:
                        # Was sentinel, put it back
                        _tts_streaming_queue.put_nowait(None)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    # Can't merge, drop this sentence
                    logger.warning(f"⚠️ Dropped sentence due to queue pressure")
        
        # Comment 8: Signal end of stream only if this is the final chunk and sentinel not sent
        if is_final and not _sentinel_sent:
            try:
                await _tts_streaming_queue.put(None)
                _sentinel_sent = True
                logger.debug("📍 Sentinel sent (stream_rag_to_tts)")
            except asyncio.QueueFull:
                # Force add sentinel by waiting
                await _tts_streaming_queue.put(None)
                _sentinel_sent = True
    
    except Exception as e:
        logger.error(f"❌ Error in stream_rag_to_tts: {e}")
        import traceback
        traceback.print_exc()
        
        # Clear sentence buffer to prevent corruption
        _sentence_buffer = []
        
        # Comment 8: Send end-of-stream sentinel ONLY if not already sent
        if not _sentinel_sent:
            try:
                await _tts_streaming_queue.put(None)
                _sentinel_sent = True
                logger.debug("📍 Sentinel sent (error recovery)")
            except Exception as queue_err:
                logger.error(f"❌ Failed to send error sentinel: {queue_err}")


async def start_tts_consumer() -> asyncio.Task:
    """
    Start TTS consumer task (idempotent).
    Returns shared task handle, creates new one only if absent/done.
    Comment 4 & 7: Reset all streaming state at stream start.
    """
    global _tts_consumer_task, _sentinel_sent, _sentence_buffer, _first_enqueue_at, _cancel_streaming, _streaming_active
    
    if _tts_consumer_task is None or _tts_consumer_task.done():
        # Comment 7: Reset ALL streaming state for new stream
        _sentinel_sent = False
        _sentence_buffer = []
        _first_enqueue_at = None
        _cancel_streaming.clear()
        
        # Create consumer task
        _tts_consumer_task = asyncio.create_task(consume_tts_streaming_queue())
        
        # CRITICAL FIX: Remove premature _streaming_active = True
        # Let the consumer set this when it actually starts running (line 912)
        # This prevents race condition where consumer exits immediately
    
    return _tts_consumer_task


async def speak_streaming(text: str, pace: float = 1.0) -> None:
    """
    Helper to stream text to TTS with sentence-level playback (Comment 4).
    Factored out to eliminate duplicate blocks across cache/speculative/persistent paths.
    
    Args:
        text: Text to speak
        pace: Speech pace (1.0 = normal)
    """
    # PHASE 3 CHANGE 3.1: Use shared consumer task instead of creating duplicate
    consumer_task = await start_tts_consumer()
    
    # Stream text to TTS queue sentence-by-sentence
    await stream_rag_to_tts(text, pace=pace, is_final=True)
    logger.debug(f"🎙️ Streaming {len(text)} chars sentence-by-sentence")
    
    # Wait for all sentences to be played
    await consumer_task


async def stop_tts_consumer():
    """
    Stop TTS consumer task gracefully.
    Cancels task, awaits completion, resets flags.
    """
    global _tts_consumer_task, _streaming_active, _cancel_streaming
    
    # Signal cancellation
    _cancel_streaming.set()
    
    if _tts_consumer_task and not _tts_consumer_task.done():
        _tts_consumer_task.cancel()
        try:
            await _tts_consumer_task
        except asyncio.CancelledError:
            pass
    
    _tts_consumer_task = None
    _streaming_active = False
    _cancel_streaming.clear()


async def finalize_tts_streaming():
    """
    Finalize TTS streaming: send sentinel, await consumer, clear queue, reset flags.
    Use in success and error paths to avoid double awaits.
    Comment 8: Track sentinel with _sentinel_sent flag to prevent duplicates.
    Comment 2 FIX: Track sentinel count for debugging
    CRITICAL FIX: Simplified to always await consumer, removed complex fast-path polling
    """
    global _tts_streaming_queue, _streaming_active, _tts_consumer_task, _sentinel_sent, _sentinels_sent_count
    
    logger.debug("🔄 Finalizing TTS streaming...")
    finalize_start = time.time()
    
    try:
        # Comment 8: Send sentinel ONLY if not already sent and consumer exists
        # Comment 2 FIX: Increment sentinel counter
        # FIX: Check _tts_consumer_task instead of _streaming_active to avoid race condition
        if (_tts_consumer_task and not _tts_consumer_task.done()) and not _sentinel_sent:
            try:
                await _tts_streaming_queue.put(None)
                _sentinel_sent = True
                _sentinels_sent_count += 1
                logger.debug(f"📍 Sentinel sent to TTS queue (total sent: {_sentinels_sent_count})")
            except asyncio.QueueFull:
                pass
        elif not _sentinel_sent and not _tts_streaming_queue.empty():
            # Edge case: queue has items but consumer not running - send sentinel anyway
            try:
                await _tts_streaming_queue.put(None)
                _sentinel_sent = True
                _sentinels_sent_count += 1
                logger.debug(f"📍 Sentinel sent to non-empty queue (no consumer)")
            except asyncio.QueueFull:
                pass
        
        # SIMPLIFIED FIX: Always await consumer if running, no complex fast-path
        # Consumer exits deterministically when it receives sentinel and finishes playback
        if _tts_consumer_task and not _tts_consumer_task.done():
            try:
                logger.debug("⏳ Awaiting TTS consumer completion...")
                await asyncio.wait_for(_tts_consumer_task, timeout=10.0)
                logger.debug("✅ TTS consumer completed successfully")
            except asyncio.TimeoutError:
                logger.warning("⚠️ TTS consumer timed out after 10s, forcing cancellation")
                _tts_consumer_task.cancel()
                try:
                    await _tts_consumer_task
                except asyncio.CancelledError:
                    pass
        
        # Comment 7: Always clear queue in finally
        await clear_tts_queue()
        
        finalize_duration = time.time() - finalize_start
        logger.debug(f"✅ TTS streaming finalized in {finalize_duration:.3f}s - Ready for next turn")
        
    finally:
        # Comment 7: Reset ALL flags including first_enqueue_at
        await clear_tts_queue()  # Ensure queue cleared even on early exit
        _tts_consumer_task = None
        _streaming_active = False
        _sentinel_sent = False
        _first_enqueue_at = None  # Comment 7: Reset for next stream


async def consume_tts_streaming_queue() -> bool:
    """
    Consumer: Read sentences from queue and speak them with barge-in support.
    Runs in parallel with RAG generation for perceived latency reduction.
    Monitors user speech and cancellation signals for early exit.
    
    Comment 1 & 2: 2-SLOT PIPELINE with staging deque for non-destructive peek:
    - Maintain current and next_future slots
    - Use local deque staging buffer to avoid premature queue removal
    - Synthesize N+1 while playing N without blocking
    - Never await synthesis before starting playback
    
    Returns:
        bool: True if all sentences played successfully, False if any failed
    """
    global _streaming_active, _sentinel_sent, _sentinels_received_count
    
    # Prevent multiple concurrent consumers
    if _streaming_active:
        return False
    
    _streaming_active = True
    # DIAGNOSTIC: Confirm consumer actually started
    logger.debug(f"🎵 TTS consumer started (streaming_active={_streaming_active})")
    
    sentence_count = 0
    sentences_queued = 0
    sentences_played = 0
    sentences_failed = 0
    agent_speaking_set = False  # Comment 9: Track if we've set speaking state
    
    # Comment 1 & 2: Staging buffer for non-destructive peek
    stage = deque()
    
    # Comment 1: 2-slot pipeline state
    current_result = None  # Audio result for current sentence
    next_future = None  # Background synthesis task for N+1
    
    # Comment 1 FIX: Disable idle-gap filler logic (contradicts acknowledgement removal)
    # Gate with environment variable - default disabled
    enable_idle_filler = os.getenv("LEIBNIZ_ENABLE_IDLE_FILLER", "false").lower() == "true"
    filler_emitted = False  # Guard to emit at most once per turn
    last_audio_started = time.time()  # Track time since last audio started
    filler_phrases = ["Okay—"]  # Ultra-short non-verbal cue (only if enabled)
    filler_threshold_ms = 999999 if not enable_idle_filler else 1000  # Effectively disabled by default
    
    try:
        # Comment 9: Set agent speaking state BEFORE first audio playback
        # (Will be set when we start playing first sentence)
        
        while True:
            # Check for cancellation before getting next item
            if _cancel_streaming.is_set():
                logger.debug("🛑 TTS consumer cancelled")
                # Comment 5: Clear agent speaking state if set
                if agent_speaking_set:
                    from leibniz_agent.leibniz_vad import get_leibniz_vad
                    vad = get_leibniz_vad()
                    await vad.set_agent_speaking_state(False, context="TTS cancelled")
                break
            
            # Check for user speech (barge-in detection)
            # Works with both per-turn and continuous VAD modes
            from leibniz_agent.leibniz_vad import check_leibniz_barge_in
            try:
                # Check existing barge-in flag (set by VAD during capture or continuous listener)
                if check_leibniz_barge_in():
                    logger.info("🛑 User started speaking (barge-in), stopping TTS queue")
                    # Comment 8: Don't push sentinel, just drain
                    await clear_tts_queue()
                    break
                
                # ENHANCEMENT: Also check continuous VAD event (if enabled)
                if _continuous_vad_enabled and _user_speech_ready.is_set():
                    logger.info("🛑 User speech detected (continuous VAD), stopping TTS queue")
                    # Set barge-in flag for consistency
                    vad = get_leibniz_vad()
                    vad.barge_in_detected = True
                    await clear_tts_queue()
                    break

            except Exception as vad_err:
                # VAD not available or error - continue without barge-in
                pass
            
            # Comment 2: Fill staging buffer from queue (ensure at least 1 item for current)
            while len(stage) < 2:
                try:
                    item = await asyncio.wait_for(_tts_streaming_queue.get(), timeout=0.5)
                    stage.append(item)
                    
                    # Comment 8: Don't prefetch sentinel
                    if item is None:
                        break
                except asyncio.TimeoutError:
                    break  # No more items yet, proceed with what we have
            
            # Comment 1 (NEW): Check if stage is empty for ≥800-1200ms, emit filler
            if not stage:
                elapsed_ms = (time.time() - last_audio_started) * 1000
                
                # Only emit filler if:
                # 1. Filler is enabled via env (default: disabled)
                # 2. Haven't emitted one yet this turn
                # 3. Enough time has passed (1000ms)
                # 4. Not cancelled
                # 5. sentence_count == 0 (BEFORE first real sentence, not after)
                # 6. Queue is not empty (sentinel not received yet)
                if (enable_idle_filler and
                    not filler_emitted and 
                    elapsed_ms >= filler_threshold_ms and 
                    not _cancel_streaming.is_set() and
                    sentence_count == 0 and  # FIX: Only BEFORE first sentence
                    not _tts_streaming_queue.empty()):  # FIX: Ensure sentinel not already queued
                    
                    # Emit ultra-short non-verbal filler ("Okay—")
                    filler_phrase = filler_phrases[0]  # Single phrase
                    logger.debug(f"⏳ Idle gap detected ({elapsed_ms:.0f}ms), emitting filler: '{filler_phrase}'")
                    
                    try:
                        _tts_streaming_queue.put_nowait((filler_phrase, 1.0))
                        filler_emitted = True  # Guard to avoid back-to-back fillers
                        stage.append((filler_phrase, 1.0))  # Add to stage immediately
                    except asyncio.QueueFull:
                        logger.warning("⚠️ Queue full, cannot emit filler")
                
                # Check cancellation and continue waiting
                if not stage and _cancel_streaming.is_set():
                    break
                elif not stage:
                    continue
            
            # Comment 1: Get current sentence from staging buffer
            current_item = stage.popleft()
            
            if current_item is None:
                # Comment 2 FIX: Sentinel received - increment counter
                _sentinels_received_count += 1
                # DIAGNOSTIC: Changed to info level for visibility
                logger.debug(f"🔚 Sentinel received, exiting consumer (played {sentences_played} sentences, total received: {_sentinels_received_count})")
                # Sentinel - end of stream
                break
            
            # Extract sentence and pace from tuple format
            sentence, pace = current_item
            
            sentence_count += 1
            sentences_queued += 1
            
            # Log full sentence when playback is about to start
            logger.info(f"🔊 PLAYING: {sentence}")
            
            # Comment 1 (NEW): Reset filler guard ONLY when real content starts playing
            # Do NOT reset when playing the filler itself (would cause infinite loop)
            if sentence not in filler_phrases:
                # This is real RAG content, not a filler
                filler_emitted = False  # Reset guard for next turn
                last_audio_started = time.time()  # Reset timer
            else:
                # This is a filler phrase - don't reset the guard
                logger.debug(f"   (Filler phrase detected - not resetting guard)")
            
            # Comment 9: Set agent speaking state on FIRST sentence (before playback starts)
            if not agent_speaking_set and sentence_count == 1:
                await set_leibniz_agent_speaking(True, "Streaming TTS - First Sentence")
                agent_speaking_set = True
                logger.debug("🔊 Agent speaking state set BEFORE first playback")
            
            # Comment 1: If we have prefetched future from previous iteration, await it now
            if next_future:
                try:
                    current_result = await next_future
                    logger.debug(f"✅ Using prefetched audio for: '{sentence[:30]}...'")
                except Exception as synth_error:
                    logger.error(f"❌ Prefetch synthesis error: {synth_error}")
                    current_result = None
                next_future = None
            else:
                current_result = None
            
            # Comment 2 & 1: Peek next sentence from stage (non-destructive) and start synthesis
            if stage:  # Has next sentence in staging buffer
                next_item = stage[0]  # Peek without removing
                
                if next_item is not None:  # Not sentinel
                    next_sentence, next_pace = next_item
                    
                    # VERIFICATION COMMENT 1: Check LRU cache and use cached path if available
                    should_skip, cached_path = check_recent_synthesis(next_sentence)
                    
                    if should_skip:
                        # Recently synthesized with no cache file - skip prefetch
                        next_future = None
                    elif cached_path:
                        # Found cached file - create synthetic result to use it
                        async def _return_cached(path):
                            return {'audio_path': path, 'success': True, 'cached': True, 'duration_ms': 0.0}
                        next_future = asyncio.create_task(_return_cached(cached_path))
                        logger.debug(f"🔄 Prefetching N+1 (cached): '{next_sentence[:30]}...'")
                    else:
                        # Comment 7: Check deduplication before synthesizing
                        # Comment 3 FIX: Normalize before hashing
                        normalized_next = normalize_sentence_for_hash(next_sentence)
                        sentence_hash = hashlib.md5(normalized_next.encode()).hexdigest()
                        
                        if sentence_hash not in _synthesis_in_flight:
                            _synthesis_in_flight.add(sentence_hash)
                            
                            # Comment 1: Start N+1 synthesis in background (don't await)
                            async def _synthesize_and_track(text, hash_key):
                                try:
                                    result = await _synthesize_only(text, emotion="helpful")
                                    return result
                                finally:
                                    _synthesis_in_flight.discard(hash_key)
                            
                            next_future = asyncio.create_task(_synthesize_and_track(next_sentence, sentence_hash))
                            logger.debug(f"🔄 Prefetching N+1: '{next_sentence[:30]}...'")
                        else:
                            logger.debug(f"⚠️ Skipping duplicate in-flight synthesis for: '{next_sentence[:30]}...'")
                            next_future = None
                else:
                    next_future = None  # Next is sentinel, don't prefetch
            
            # Comment 1: Play current sentence WITHOUT blocking on synthesis
            # Playback starts immediately if we have prefetched audio, else synthesize now
            try:
                if current_result:
                    # Comment 7: Use prefetched result, do NOT call speak_friendly (would duplicate synthesis)
                    result = current_result
                    logger.debug(f"🎵 Playing prefetched audio: '{sentence[:30]}...'")
                else:
                    # First sentence or prefetch failed - synthesize now
                    logger.debug(f"🎵 Synthesizing current (no prefetch): '{sentence[:30]}...'")
                    
                    # VERIFICATION COMMENT 1: Check LRU cache and use cached path if available
                    should_skip, cached_path = check_recent_synthesis(sentence)
                    
                    if should_skip:
                        # Recently synthesized with no cache file - skip playback
                        result = None
                    elif cached_path:
                        # Found cached file - use it directly
                        result = {'audio_path': cached_path, 'success': True, 'cached': True, 'duration_ms': 0.0}
                        logger.debug(f"🎵 Using cached audio: '{sentence[:30]}...'")
                    else:
                        # Comment 7: Check deduplication
                        # Comment 3 FIX: Normalize before hashing
                        normalized_sentence = normalize_sentence_for_hash(sentence)
                        sentence_hash = hashlib.md5(normalized_sentence.encode()).hexdigest()
                        
                        if sentence_hash not in _synthesis_in_flight:
                            _synthesis_in_flight.add(sentence_hash)
                            try:
                                result = await _synthesize_only(sentence, emotion="helpful")
                            finally:
                                _synthesis_in_flight.discard(sentence_hash)
                        else:
                            logger.warning(f"⚠️ Duplicate synthesis blocked: '{sentence[:30]}...'")
                            result = None
                
                # Comment 10: Wrap pygame playback in asyncio.to_thread
                audio_path = None
                duration_ms = 0.0
                
                if result:
                    if isinstance(result, dict):
                        # Dictionary format from TTS
                        audio_path = result.get('audio_path') or result.get('audio_file') or result.get('file')
                        duration_ms = result.get('duration_ms') or result.get('duration', 0.0) * 1000
                    elif hasattr(result, 'audio_path'):
                        # TTSMessage object format
                        audio_path = result.audio_path
                        duration_ms = result.duration_ms if hasattr(result, 'duration_ms') else 0.0
                
                if audio_path and PYGAME_AVAILABLE:
                    # DIAGNOSTIC: Log TTS playback guard conditions
                    logger.debug(f"🎵 TTS Playback Guard Check (streaming):")
                    logger.debug(f"   PYGAME_AVAILABLE = {PYGAME_AVAILABLE}")
                    logger.debug(f"   audio_path = '{audio_path}'")
                    sentences_played += 1
                    
                    # Comment 6 FIX: Check playing_now guard to prevent overlapping playback
                    global _playing_now
                    if _playing_now:
                        logger.warning(f"⚠️ Skipping playback (already playing): '{sentence[:30]}...'")
                        continue
                    
                    # Calculate audio duration for smart warmup scheduling
                    audio_duration = duration_ms / 1000.0 if duration_ms else len(sentence) * 0.05
                    
                    try:
                        # Comment 5: Set agent speaking state BEFORE first playback
                        if not agent_speaking_set:
                            from leibniz_agent.leibniz_vad import get_leibniz_vad
                            vad = get_leibniz_vad()
                            await vad.set_agent_speaking_state(True, context="TTS streaming playback")
                            agent_speaking_set = True
                            logger.debug("🔊 Agent speaking state set (first sentence)")
                            
                            # IMMEDIATE WARMUP: Start VAD warmup right when agent starts speaking
                            # This runs in parallel with TTS playback so user can speak immediately after
                            async def immediate_warmup():
                                try:
                                    await smart_warmup_leibniz_vad()
                                    logger.debug("🔥 Immediate VAD warmup completed (parallel with TTS)")
                                except Exception as e:
                                    logger.debug(f"⚠️ Immediate warmup failed: {e}")
                            
                            asyncio.create_task(immediate_warmup())
                        
                        # Comment 6 FIX: Set playing_now=True before play()
                        _playing_now = True
                        
                        # VERIFICATION COMMENT 2: Log first-playback latency
                        global _first_enqueue_at
                        if _first_enqueue_at is not None and sentences_played == 1:
                            first_play_latency_ms = (time.time() - _first_enqueue_at) * 1000
                            logger.info(f"⏱️ First playback started {first_play_latency_ms:.1f}ms after first enqueue")
                            _first_enqueue_at = None  # Reset to avoid duplicate logs
                        
                        # Comment 10: Use sounddevice with device=12 for consistent audio playback
                        # Load audio data with soundfile and play with sounddevice
                        audio_data, sample_rate = await asyncio.to_thread(sf.read, audio_path)
                        await asyncio.to_thread(sd.play, audio_data, sample_rate, device=12)
                        await asyncio.to_thread(sd.wait)  # Wait for playback to complete
                        
                        logger.debug("✅ TTS streaming playback completed with sounddevice")
                    
                    finally:
                        # Comment 6 FIX: Always set playing_now=False after unload (even on error)
                        _playing_now = False
                    
                    # Removed old delayed warmup - now using immediate warmup when TTS starts

                else:
                    sentences_failed += 1
                    logger.debug(f"⚠️ TTS playback failed for sentence: {sentence[:50]}...")
            except Exception as speak_err:
                sentences_failed += 1
                logger.error(f"❌ TTS speak error: {speak_err}")
                # SOUNDEVICE FALLBACK: Try sounddevice if pygame fails
                logger.warning("🔄 Pygame playback failed - attempting sounddevice fallback")
                try:
                    # Load audio data with soundfile (sf already imported at module level)
                    audio_data, sample_rate = await asyncio.to_thread(sf.read, audio_path)
                    
                    # Play with sounddevice (blocking call in thread)
                    await asyncio.to_thread(sd.play, audio_data, sample_rate, device=12)
                    await asyncio.to_thread(sd.wait)  # Wait for playback to complete
                    
                    logger.info("✅ Sounddevice fallback playback successful")
                    
                except Exception as sd_error:
                    logger.error(f"❌ Sounddevice fallback also failed: {sd_error}")
                    # Continue without audio - no further fallback possible
            
            # Check for barge-in after each sentence
            if _cancel_streaming.is_set():
                logger.debug("🛑 Cancelled during playback")
                break
            
    except asyncio.CancelledError:
        logger.debug("🛑 TTS consumer task cancelled")
        raise
    except Exception as e:
        logger.error(f"❌ TTS streaming error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _streaming_active = False
        
        # Comment 5 & 9: Clear agent speaking state in finally block (always executed)
        if agent_speaking_set:
            from leibniz_agent.leibniz_vad import get_leibniz_vad
            vad = get_leibniz_vad()
            await vad.set_agent_speaking_state(False, context="TTS consumer finished")
        
        # Log playback summary
        logger.info(f"📊 TTS Streaming Complete: {sentences_queued} queued, {sentences_played} played, {sentences_failed} failed")
        logger.info("✅ TTS consumer finished - Ready for next conversation turn")
        
        # Drain remaining queue items and staging buffer
        while not _tts_streaming_queue.empty():
            try:
                _tts_streaming_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        stage.clear()
        
        return sentences_failed == 0


async def _synthesize_only(text: str, emotion: str = "helpful") -> Optional[Any]:
    """
    Synthesize audio without playing (for prefetch pattern).
    
    Args:
        text: Text to synthesize
        emotion: Emotion/tone for synthesis
        
    Returns:
        TTSMessage with synthesis result or None on error
    """
    try:
        from leibniz_agent.leibniz_tts import get_leibniz_tts
        tts = get_leibniz_tts()
        
        result = await tts.synthesize_to_file(
            text=text,
            emotion=emotion,
            cache_name=None
        )
        return result
    except Exception as e:
        logger.error(f"❌ Synthesis-only error: {e}")
        return None


# ============================================================================
# Audio Archiving
# ============================================================================

async def archive_dialogue_audio(
    audio_file: Optional[str],
    session_id: str,
    turn_number: int,
    text: str,
    dialogue_type: str = "unknown"
) -> Optional[str]:
    """
    Archive dialogue audio to organized directory structure with descriptive file names.
    
    Args:
        audio_file: Path to audio file to archive, or None to retrieve from TTS cache
        session_id: Session identifier
        turn_number: Turn number in conversation
        text: Text that was synthesized
        dialogue_type: Type of dialogue (rag, greeting, error, etc.)
        
    Returns:
        Path to archived file, or None if archiving failed
    """
    try:
        import shutil
        import json
        import hashlib
        
        # If no audio_file provided, synthesize the text to get the audio file
        if audio_file is None:
            try:
                logger.debug(f"Synthesizing text for archiving: {text[:50]}...")
                tts = get_leibniz_tts()
                
                # Create a temporary file path for archiving
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(
                    prefix=f"archive_{session_id}_{turn_number}_",
                    suffix=".wav",
                    delete=False
                )
                temp_audio_path = temp_file.name
                temp_file.close()
                
                # Synthesize with explicit output file path
                result = await tts.synthesize_to_file(
                    text=text,
                    outfile=temp_audio_path,  # Provide explicit output path
                    emotion='helpful',  # Default emotion
                    cache_name=None  # Don't use dialogue cache for archiving
                )
                
                if result and result.get('success'):
                    audio_file = result.get('audio_file') or result.get('file')
                    if audio_file and os.path.exists(audio_file):
                        logger.debug(f"Synthesized audio for archiving: {audio_file}")
                    else:
                        logger.error(f"Synthesis succeeded but audio file not found: {audio_file}")
                        # Clean up temp file on error
                        if temp_audio_path and os.path.exists(temp_audio_path):
                            try:
                                os.unlink(temp_audio_path)
                            except:
                                pass
                        return None
                else:
                    logger.error(f"Failed to synthesize text for archiving: {result}")
                    # Clean up temp file on error
                    if temp_audio_path and os.path.exists(temp_audio_path):
                        try:
                            os.unlink(temp_audio_path)
                        except:
                            pass
                    return None
                    
            except Exception as e:
                logger.error(f"Failed to synthesize audio for archiving: {e}")
                # Clean up temp file on exception
                if 'temp_audio_path' in locals() and temp_audio_path and os.path.exists(temp_audio_path):
                    try:
                        os.unlink(temp_audio_path)
                    except:
                        pass
                return None
        
        # Create descriptive filename based on text content
        # Use first 50 chars of text, sanitized for filename
        text_preview = text[:50].strip()
        # Remove special characters that aren't safe for filenames
        safe_text = "".join(c for c in text_preview if c.isalnum() or c in (' ', '-', '_')).rstrip()
        if not safe_text:
            safe_text = "dialogue"
        
        # Create filename with turn number, text preview, and hash for uniqueness
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        filename = f"turn_{turn_number:03d}_{safe_text}_{text_hash}.wav"
        # Replace spaces with underscores and limit length (remove .wav since it's already there)
        filename = filename.replace(' ', '_')[:100]
        
        # Create organized directory structure: dialogues/session_{id}/{type}/
        session_dir = os.path.join(DIALOGUE_ARCHIVE_DIR, f"session_{session_id}", dialogue_type)
        os.makedirs(session_dir, exist_ok=True)
        
        # Archive audio file
        archive_path = os.path.join(session_dir, filename)
        shutil.copy(audio_file, archive_path)
        
        # Save metadata
        metadata_filename = filename.replace('.wav', '.json')
        metadata_path = os.path.join(session_dir, metadata_filename)
        
        metadata = {
            "session_id": session_id,
            "turn_number": turn_number,
            "dialogue_type": dialogue_type,
            "text": text,
            "audio_file": filename,
            "original_audio_path": audio_file,
            "timestamp": time.time(),
            "text_length": len(text),
            "text_hash": text_hash
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 Archived dialogue audio: {archive_path} (type: {dialogue_type})")
        return archive_path
        
    except Exception as e:
        logger.error(f"Failed to archive dialogue audio: {e}")
        return None


# ============================================================================
# Async Speak Function
# ============================================================================

async def speak_friendly(
    text: str = "",
    emotion: str = "helpful",
    cache_name: Optional[str] = None,
    dialogue_key: Optional[str] = None,
    enable_streaming: bool = False,
    session_id: Optional[str] = None,
    turn_number: Optional[int] = None
) -> TTSMessage:
    """
    Speak text with friendly casual tone
    
    Args:
        text: Text to speak
        emotion: Emotion for voice modulation (helpful, excited, calm)
        cache_name: Optional cache key for frequently used phrases
        dialogue_key: Optional dialogue key for cache name lookup from dialogue manager
        enable_streaming: Enable streaming TTS for long responses
        session_id: Optional session identifier for dialogue archiving
        turn_number: Optional turn number for dialogue archiving
        
    Returns:
        TTSMessage with synthesis results or None if TTS unavailable
    """
    try:
        # Comment 5: Wrap get_leibniz_tts() in try/except for graceful degradation
        try:
            tts = get_leibniz_tts()
        except Exception as tts_init_error:
            # Check if running without TTS is allowed
            mock_mode = os.getenv('MOCK_TTS', 'false').lower() == 'true'
            allow_no_tts = os.getenv('ALLOW_NO_TTS', 'false').lower() == 'true'
            
            if mock_mode or allow_no_tts:
                logger.warning(f"TTS initialization failed: {tts_init_error}")
                logger.info("🔇 Running in text-only mode (no audio output)")
                # Print text instead of speaking
                print(f"\n[AGENT]: {text}\n")
                print("🎤 Ready for user - TTS complete")
                return TTSMessage(
                    text=text,
                    audio_path=None,
                    duration_ms=0.0
                )
            else:
                # Re-raise if not allowed to run without TTS
                raise
        
        # Handle dialogue_key: get text from dialogue manager if text is empty
        if not text and dialogue_key:
            try:
                dialogue_manager = get_leibniz_dialogue_manager()
                # Split dialogue_key into category and key parts
                if '.' in dialogue_key:
                    category, key = dialogue_key.split('.', 1)
                    text = dialogue_manager.get_dialogue(category, key)
                else:
                    # Fallback: assume it's just a key and try to get category
                    category = dialogue_manager.get_dialogue_category(dialogue_key)
                    text = dialogue_manager.get_dialogue(category, dialogue_key)
                
                if not text:
                    logger.warning(f"Dialogue key '{dialogue_key}' not found in dialogue manager")
                    text = ""  # Ensure text is not None for downstream processing
            except Exception as e:
                logger.error(f"Failed to get dialogue text for key '{dialogue_key}': {e}")
                text = ""  # Ensure text is not None for downstream processing
        
        # Comment 5: Handle tts=None case (should not happen after above check, but defensive)
        if tts is None:
            logger.warning("TTS instance is None - running in text-only mode")
            print(f"\n[AGENT]: {text}\n")
            print("🎤 Ready for user - TTS complete")
            return TTSMessage(
                text=text,
                audio_path=None,
                duration_ms=0.0
            )
        
        # Check config flags (Comment 7)
        streaming_enabled = os.getenv("LEIBNIZ_ENABLE_STREAMING_TTS", "true").lower() == "true"
        barge_in_enabled = os.getenv("LEIBNIZ_ENABLE_BARGE_IN", "true").lower() == "true"
        
        # Override enable_streaming if disabled in config
        if not streaming_enabled:
            enable_streaming = False
        
        # Apply emotion modulation from config (if available)
        if leibniz_config:
            # Future: Apply personality-based emotion mapping
            pass
        
        # Use dialogue manager for cache name inference
        if cache_name is None:
            if dialogue_key:
                # Use dialogue key to get cache name from dialogue manager
                cache_name = get_dialogue_cache_name(dialogue_key)
            else:
                # Fallback: Try to match dialogue keys from dialogue manager using text
                text_lower = text.lower().strip()
                dialogue_manager = get_leibniz_dialogue_manager()

                # Check for specific dialogue patterns
                if any(word in text_lower for word in ['hello', 'hi', 'hey', 'welcome', 'greetings']):
                    cache_name = get_dialogue_cache_name('greeting')
                elif any(word in text_lower for word in ['goodbye', 'bye', 'farewell', 'see you']):
                    cache_name = get_dialogue_cache_name('farewell')
                elif any(word in text_lower for word in ['error', 'sorry', 'issue', 'problem']):
                    cache_name = get_dialogue_cache_name('error')
                elif any(word in text_lower for word in ['timeout', 'waiting', 'still there']):
                    cache_name = get_dialogue_cache_name('timeout')
                elif any(word in text_lower for word in ['appointment', 'confirmed', 'scheduled']):
                    cache_name = get_dialogue_cache_name('appointment_confirm')
                elif any(word in text_lower for word in ['help you', 'can i help']):
                    cache_name = get_dialogue_cache_name('help_offer')
                elif any(word in text_lower for word in ['continue', 'next', 'go on']):
                    cache_name = get_dialogue_cache_name('continue_prompt')
                elif any(word in text_lower for word in ['clarify', 'explain', 'understand']):
                    cache_name = get_dialogue_cache_name('clarification_prompt')
                elif any(word in text_lower for word in ['thinking', 'processing', 'working']):
                    cache_name = get_dialogue_cache_name('thinking_indicator')
        
        # Set agent speaking state
        await set_leibniz_agent_speaking(True, context="TTS synthesis")
        
        # CRITICAL FIX: Explicitly ensure continuous VAD is stopped before TTS
        # This prevents the agent from transcribing its own speech
        try:
            from leibniz_agent.leibniz_continuous_vad import get_continuous_vad
            continuous_vad = get_continuous_vad()
            if continuous_vad.is_running:
                logger.info("🎤 Force-stopping continuous VAD before TTS to prevent feedback loop")
                await continuous_vad.stop_continuous_listening()
        except Exception as vad_stop_error:
            logger.warning(f"⚠️ Failed to stop continuous VAD before TTS: {vad_stop_error}")
        
        # Additional safety: Verify main VAD state is properly set
        vad = get_leibniz_vad()
        if vad and not vad.is_agent_speaking:
            logger.warning("⚠️ Main VAD is_agent_speaking flag not set - forcing update")
            await vad.set_agent_speaking_state(True, "TTS synthesis - forced")
        
        try:
            if enable_streaming:
                # NOTE: Direct streaming via speak_friendly is deprecated
                # Use stream_rag_to_tts() + consume_tts_streaming_queue() pattern instead
                # This legacy streaming path is kept for backward compatibility only
                
                sentences = split_into_sentences(text)
                
                audio_files = []
                total_duration = 0.0
                barge_in_occurred = False
                
                try:
                    for sentence in sentences:
                        # Check for barge-in before synthesis
                        if barge_in_enabled and check_leibniz_barge_in():
                            logger.info("⚡ Barge-in detected - stopping TTS synthesis")
                            barge_in_occurred = True
                            clear_leibniz_barge_in()
                            break
                        
                        # Synthesize sentence
                        result = await tts.synthesize_to_file(
                            text=sentence,
                            emotion=emotion,
                            cache_name=f"{cache_name}_{len(audio_files)}" if cache_name else None
                        )
                        
                        if result and result.get("success"):
                            audio_file = result.get('audio_file') or result.get('file')
                            duration = result.get("duration", 0.0)
                            
                            audio_files.append(audio_file)
                            total_duration += duration
                            
                            # Comment 6: Offload pygame operations to thread
                            if PYGAME_AVAILABLE and os.path.exists(audio_file):
                                # Comment 2: Duck background audio during TTS playback
                                original_bg_volume = None
                                try:
                                    # Keep background audio at full volume (no ducking)
                                    # Original ducking logic disabled per user request
                                    # if background_player and background_player.is_playing:
                                    #     original_bg_volume = background_player.get_current_volume()
                                    #     ducking_factor = float(os.getenv('LEIBNIZ_BACKGROUND_DUCKING_FACTOR', '0.5'))
                                    #     ducked_volume = max(0.0, original_bg_volume * ducking_factor)
                                    #     background_player._bg_channel.set_volume(ducked_volume)
                                    #     logger.debug(f"🔉 Background ducked: {original_bg_volume:.2f} → {ducked_volume:.2f}")
                                    
                                    # No ducking - background stays at full volume during speech
                                    original_bg_volume = None
                                    
                                    # Retry logic to avoid race with file flush
                                    load_success = False
                                    for retry in range(3):
                                        try:
                                            await asyncio.to_thread(pygame.mixer.music.load, audio_file)
                                            load_success = True
                                            break
                                        except Exception as load_err:
                                            if retry < 2:
                                                await asyncio.sleep(0.05 + retry * 0.05)  # 50ms, 100ms
                                            else:
                                                raise load_err
                                    
                                    if not load_success:
                                        raise RuntimeError(f"Failed to load audio file after 3 attempts: {audio_file}")
                                    
                                    await asyncio.to_thread(pygame.mixer.music.play)
                                    
                                    # Wait for playback to complete with barge-in polling
                                    while pygame.mixer.music.get_busy():
                                        await asyncio.to_thread(pygame.mixer.music.get_busy)
                                        
                                        # IMMEDIATE BARGE-IN: Check every 20ms for faster response
                                        if barge_in_enabled and check_leibniz_barge_in():
                                            pygame.mixer.music.stop()
                                            logger.info("⚡ IMMEDIATE BARGE-IN: TTS playback stopped - user interrupted")
                                            barge_in_occurred = True
                                            clear_leibniz_barge_in()
                                            break
                                        
                                        # Reduced sleep for more responsive barge-in detection
                                        await asyncio.sleep(0.02)  # 20ms instead of 100ms
                                    
                                    # FIX 3: Release pygame audio device explicitly
                                    pygame.mixer.music.stop()
                                    try:
                                        pygame.mixer.music.unload()  # Release audio file
                                    except:
                                        pass
                                    await asyncio.sleep(0.3)  # Give OS time to release device (Windows needs longer)
                                    logger.debug("🎤 Audio device released for microphone")
                                    
                                    if barge_in_occurred:
                                        break
                                    
                                except Exception as e:
                                    logger.error(f"Playback error: {e}")
                                finally:
                                    # Restore background audio volume
                                    if original_bg_volume is not None and background_player and background_player.is_playing:
                                        background_player._bg_channel.set_volume(original_bg_volume)
                                        logger.debug(f"🔊 Background restored: {original_bg_volume:.2f}")
                            
                            # Removed old delayed warmup - using immediate warmup when TTS starts
                    
                except Exception as stream_error:
                    logger.error(f"Streaming TTS error: {stream_error}")
                
                print("🎤 Ready for user - TTS complete")
                
                # Comment 1: Archive streaming dialogue audio if enabled (concatenated or first file)
                if os.getenv('LEIBNIZ_ENABLE_DIALOGUE_ARCHIVE', 'false').lower() == 'true':
                    if session_id is not None and turn_number is not None and audio_files:
                        # Archive first sentence or concatenated audio
                        asyncio.create_task(
                            archive_dialogue_audio(
                                audio_file=audio_files[0],
                                session_id=session_id,
                                turn_number=turn_number,
                                text=text,
                                dialogue_type="streaming"
                            )
                        )
                
                # Create TTSMessage
                return TTSMessage(
                    text=text,
                    audio_path=audio_files[0] if audio_files else None,
                    duration_ms=total_duration * 1000
                )
            else:
                # File-based TTS
                result = await tts.synthesize_to_file(
                    text=text,
                    emotion=emotion,
                    cache_name=cache_name
                )
                
                if result and result.get("success"):
                    # Handle both file-based and raw bytes results
                    audio_file = result.get('audio_file') or result.get('file')
                    duration = result.get("duration", 0.0)
                    audio_bytes = result.get('audio_bytes')
                    sample_rate = result.get('sample_rate', 24000)
                    is_temporary = result.get('is_temporary', False)
                    
                    # Comment 6: Unconditional fallback chain - prioritize sounddevice with device 12
                    # Always attempt playback, starting with sounddevice (device 12) as primary
                    
                    # Track playback success/failure
                    playback_successful = False
                    playback_method = "none"

                    # Try sounddevice first (primary method with device 12)
                    if audio_bytes and SOUNDDEVICE_AVAILABLE:
                        # Play raw audio bytes directly
                        try:
                            print("▶️ TTS playback starting (sounddevice device 12 - raw bytes)...", flush=True)
                            # Convert bytes to numpy array (assuming int16 PCM)
                            import numpy as np
                            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                            
                            # Play with sounddevice on device 12 (Realtek primary)
                            await asyncio.to_thread(sd.play, audio_data, sample_rate, device=12)
                            await asyncio.to_thread(sd.wait)  # Wait for playback to complete

                            logger.info("✅ TTS playback completed with sounddevice (device 12) - raw bytes")
                            playback_successful = True
                            playback_method = "sounddevice_raw"

                        except Exception as e:
                            logger.error(f"❌ Sounddevice raw bytes playback failed: {e}")
                            print(f"⚠️ Sounddevice raw bytes playback failed: {e}", flush=True)
                            # Continue to file-based fallback

                    if not playback_successful and os.path.exists(audio_file) and SOUNDDEVICE_AVAILABLE:
                        try:
                            print("▶️ TTS playback starting (sounddevice device 12)...", flush=True)
                            # Load audio data with soundfile
                            audio_data, sample_rate = await asyncio.to_thread(sf.read, audio_file)

                            # Play with sounddevice on device 12 (Realtek primary)
                            await asyncio.to_thread(sd.play, audio_data, sample_rate, device=12)
                            await asyncio.to_thread(sd.wait)  # Wait for playback to complete

                            logger.info("✅ TTS playback completed with sounddevice (device 12)")
                            playback_successful = True
                            playback_method = "sounddevice"

                        except Exception as e:
                            logger.error(f"❌ Sounddevice playback failed: {e}")
                            print(f"⚠️ Sounddevice playback failed: {e}", flush=True)
                            # Continue to pygame fallback

                    # Fallback to pygame if sounddevice failed or file doesn't exist
                    if not playback_successful and PYGAME_AVAILABLE and os.path.exists(audio_file):
                        try:
                            print("🔄 Attempting pygame fallback...", flush=True)
                            await asyncio.to_thread(pygame.mixer.music.load, audio_file)
                            await asyncio.to_thread(pygame.mixer.music.play)
                            
                            # IMMEDIATE WARMUP: Start VAD warmup right when agent starts speaking
                            # This runs in parallel with TTS playback so user can speak immediately after
                            async def immediate_warmup():
                                try:
                                    await smart_warmup_leibniz_vad()
                                    logger.debug("🔥 Immediate VAD warmup completed (parallel with TTS)")
                                except Exception as e:
                                    logger.debug(f"⚠️ Immediate warmup failed: {e}")

                            asyncio.create_task(immediate_warmup())

                            # Wait for playback with barge-in detection and timeout
                            playback_start = time.time()
                            max_playback_time = duration + 10.0  # Allow extra time for playback

                            while pygame.mixer.music.get_busy() and (time.time() - playback_start) < max_playback_time:
                                # IMMEDIATE BARGE-IN: Check every 20ms for faster response
                                if barge_in_enabled and check_leibniz_barge_in():
                                    pygame.mixer.music.stop()
                                    logger.info("⚡ IMMEDIATE BARGE-IN: TTS playback stopped - user interrupted during file playback")
                                    clear_leibniz_barge_in()
                                    break
                                
                                # Reduced sleep for more responsive barge-in detection
                                await asyncio.sleep(0.02)  # 20ms instead of 100ms

                            # Check if we timed out
                            if pygame.mixer.music.get_busy():
                                logger.warning(f"⚠️ TTS playback timeout after {max_playback_time:.1f}s - forcing stop")
                                pygame.mixer.music.stop()

                            logger.info("✅ TTS playback completed with pygame")
                            print("✅ Pygame fallback successful", flush=True)
                            playback_successful = True
                            playback_method = "pygame"

                        except Exception as e:
                            logger.error(f"❌ Pygame fallback also failed: {e}")
                            print(f"❌ Pygame fallback failed: {e}", flush=True)
                            playback_method = "failed"

                    # Clean up temporary file if needed
                    if is_temporary and audio_file and os.path.exists(audio_file):
                        try:
                            os.unlink(audio_file)
                            logger.debug(f"🗑️ Cleaned up temporary TTS file: {audio_file}")
                        except Exception as cleanup_error:
                            logger.warning(f"⚠️ Failed to cleanup temporary TTS file {audio_file}: {cleanup_error}")

                    # Log final playback status
                    if playback_successful:
                        print(f"✅ TTS playback completed successfully ({playback_method})", flush=True)
                    else:
                        print("❌ TTS playback failed - no audio output", flush=True)
                        logger.error("TTS playback failed completely - synthesis succeeded but no audio output")
                    
                    print("🎤 Ready for user - TTS complete")
                    
                    # Comment 1: Archive dialogue audio if enabled
                    if os.getenv('LEIBNIZ_ENABLE_DIALOGUE_ARCHIVE', 'false').lower() == 'true':
                        if session_id is not None and turn_number is not None and audio_file and os.path.exists(audio_file):
                            asyncio.create_task(
                                archive_dialogue_audio(
                                    audio_file=audio_file,
                                    session_id=session_id,
                                    turn_number=turn_number,
                                    text=text,
                                    dialogue_type="rag"
                                )
                            )
                    
                    # Create TTSMessage
                    return TTSMessage(
                        text=text,
                        audio_path=audio_file if audio_file and os.path.exists(audio_file) else None,
                        duration_ms=duration * 1000  # Convert seconds to milliseconds
                    )
                else:
                    # Fallback: return empty message
                    return TTSMessage(
                        text=text,
                        audio_path=None,
                        duration_ms=0.0
                    )
        finally:
            # Clear agent speaking state
            await set_leibniz_agent_speaking(False, context="TTS complete")
            print("✅ Agent speaking flag cleared")
        
    except Exception as e:
        logger.error(f"Speak error: {e}", exc_info=True)
        
        # Return error message
        return TTSMessage(
            text=text,
            audio_path=None,
            duration_ms=0.0
        )


# Comment 3: Local prewarm function removed - using imported prewarm_leibniz_during_tts from persistent_services


# ============================================================================
# Audio Capture and Transcription
# ============================================================================

async def capture_and_transcribe(
    streaming_callback: Optional[Callable] = None,
    context: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Capture audio and transcribe with Gemini Live VAD (SINDH Pattern)

    Args:
        streaming_callback: Optional callback(fragment: str, is_final: bool) for real-time transcript display
        context: Optional conversation context dict with keys:
            - conversation_context: str (greeting, decision, complex_query, post_service, retry, initial)
            - attempt_count: int (number of retry attempts for dynamic timeout)
            
    Returns:
        Transcript string only (NO audio file) - matches SINDH/TARA pattern
        
    Note:
        - Returns transcript directly (no temporary WAV file creation)
        - Transcript is already normalized by VAD module
        - Streaming callback receives partial transcripts during capture
        - Dynamic timeout is set based on context before capture
    """
    try:
        # Enhanced logging before capture
        logger.info("🎧 Listening for your input...")
        print("🎤 Starting audio capture with parallel processing...")
        
        # Capture speech with VAD (returns transcript only)
        transcript = await capture_leibniz_speech(
            streaming_callback=streaming_callback,
            context=context
        )
        
        if transcript:
            # Log successful capture
            logger.info("✅ Speech captured successfully")
            print("✅ Speech captured successfully")
            print(f"📝 Original: '{transcript}'")
            
            # Transcript already normalized by VAD module
            return transcript
        else:
            return None
        
    except Exception as e:
        logger.error(f"Capture error: {e}", exc_info=True)
        return None


async def transcribe_and_classify(
    streaming_callback: Optional[Callable] = None,
    context: Optional[Dict[str, Any]] = None
) -> Tuple[TranscriptMessage, IntentMessage]:
    """
    Capture, transcribe, and classify speech (SINDH/TARA Pattern)

    Args:
        streaming_callback: Optional callback(fragment: str, is_final: bool) for real-time transcript display
        context: Optional conversation context dict for VAD (see capture_and_transcribe)
        
    Returns:
        Tuple of (TranscriptMessage, IntentMessage)
        
    Note:
        - Uses persistent services for fast intent classification if available
        - Falls back to direct parser if services unavailable
        - Streaming callback is passed through to VAD for real-time feedback
        - Context dict is used for dynamic timeout adjustment
        - Returns transcript only (no audio file) - matches SINDH/TARA pattern
    """
    try:
        # Streamlined logging (TARA pattern - less verbose in capture phase)
        logger.debug("🎤 Initiating speech capture...")

        # Capture and transcribe (single call, returns transcript only - SINDH pattern)
        transcript = await capture_and_transcribe(
            streaming_callback=streaming_callback,
            context=context
        )
        
        if not transcript:
            # Return empty messages immediately
            return (
                TranscriptMessage(
                    transcript="",
                    confidence=0.0
                ),
                IntentMessage(
                    intent="UNCLEAR",
                    confidence=0.0,
                    entities={},
                    reasoning="No speech captured"
                )
            )
        
        # Log successful transcript capture (TARA pattern)
        logger.info(f"📄 Transcript captured: '{transcript[:100]}{'...' if len(transcript) > 100 else ''}'")
        
        # Create TranscriptMessage
        transcript_msg = TranscriptMessage(
            transcript=transcript,
            confidence=1.0  # Gemini Live provides high-quality transcription
        )
        
        # STEP 1: Extract semantic context IMMEDIATELY (fast pattern-based, <5ms)
        # This runs BEFORE intent classification to provide enriched context
        from leibniz_agent.leibniz_semantic_extractor import extract_semantic_context
        
        context_gen_start = time.time()
        semantic_context = extract_semantic_context(transcript)
        context_gen_elapsed = time.time() - context_gen_start
        
        # Make timing visible in console (not just logs)
        print(f"⚡ Semantic extraction: {context_gen_elapsed*1000:.2f}ms - Goal: '{semantic_context['user_goal'][:60]}...'")
        if semantic_context['key_entities']:
            entities_str = ', '.join([f"{k}={v}" for k, v in semantic_context['key_entities'].items()])
            print(f"   📊 Entities: {entities_str}")
        logger.info(f"⚡ Semantic context extracted in {context_gen_elapsed*1000:.1f}ms: '{semantic_context['user_goal']}'")
        logger.debug(f"📊 Entities: {semantic_context['key_entities']}, Method: {semantic_context['extraction_method']}")
        
        # STEP 2: Classify intent with enriched semantic context (not raw transcript)
        # Intent classifier receives pre-processed context for better accuracy
        if services_manager and services_manager.intent_parser.parser:
            logger.debug("🔍 Classifying intent with semantic context...")
            classify_start = time.time()
            
            # Build enriched context for intent classifier
            enriched_context = {
                **(context or {}),
                'semantic_context': semantic_context,  # Pass pre-extracted context
                'user_goal': semantic_context['user_goal'],
                'key_entities': semantic_context['key_entities'],
                'extracted_meaning': semantic_context['extracted_meaning']
            }
            
            # Use Leibniz's intent parser with enriched context
            if _leibniz_parser:
                intent_result = await _leibniz_parser.classify_intent(
                    text=semantic_context['extracted_meaning'],  # Use normalized meaning, not raw transcript
                    context=enriched_context
                )
            else:
                # Fallback: Basic classification using semantic context
                logger.warning("No intent parser available, using fallback")
                intent_result = {
                    'intent': 'RAG_QUERY',
                    'confidence': 0.5,
                    'context': semantic_context  # Use pre-extracted context
                }
            
            classify_elapsed = time.time() - classify_start
            
            # Merge semantic context with intent result
            # Priority: semantic_context (fast extraction) > intent_result.context (LLM extraction)
            intent_context = {
                **semantic_context,  # Start with fast semantic context
                **intent_result.get('context', {})  # Override with LLM context if available
            }
            
            # Build enriched user context for RAG
            user_goal = intent_context.get('user_goal', '')
            extracted_meaning = intent_context.get('extracted_meaning', transcript)
            
            # Generate final user_context string for RAG
            if user_goal and user_goal != extracted_meaning:
                user_context_transcript = f"{user_goal}: {extracted_meaning}"
            else:
                user_context_transcript = extracted_meaning
            
            # PHASE 1 CHANGE 1.5: Add detailed latency logging with fast route info
            key_entities = intent_context.get('key_entities', {})
            logger.info(f"⚡ Intent classified in {classify_elapsed*1000:.1f}ms: {intent_result['intent']} (conf: {intent_result['confidence']:.2f})")
            if intent_result.get('fast_route'):
                logger.info(f"🚀 Fast route used (pattern match)")
            else:
                logger.info(f"🔄 LLM fallback used")
            logger.info(f"⏱️ Intent classification: {classify_elapsed*1000:.0f}ms, Context generation: {context_gen_elapsed*1000:.1f}ms")
            logger.debug(f"✨ Semantic context extracted: '{user_context_transcript[:80]}...'")
            
            # Show intent with context preview
            context_preview = f" | Context: '{user_context_transcript[:50]}...'" if user_context_transcript else ""
            logger.debug(f"📊 Full intent: {intent_result['intent']} (conf: {intent_result['confidence']:.2f}){context_preview}")
            
            # Create IntentMessage with context in entities field and user_context
            # Store timing breakdown for display in conversation loop
            intent_msg = IntentMessage(
                intent=intent_result.get("intent", "UNCLEAR"),
                confidence=intent_result.get("confidence", 0.0),
                entities=intent_context,  # Comment 2: Pass full context dict
                user_context=user_context_transcript,  # Enriched query for RAG (= semantic context)
                reasoning=intent_result.get("reasoning", ""),
                processing_time=classify_elapsed,  # Total time including context generation
                timing_breakdown={
                    'classification_ms': (classify_elapsed - context_gen_elapsed) * 1000,  # Intent classification only
                    'context_generation_ms': context_gen_elapsed * 1000  # Semantic context generation
                }
            )
        else:
            # Fallback: direct classification with Leibniz parser
            # STEP 1: Extract semantic context FIRST (fast, <5ms)
            from leibniz_agent.leibniz_semantic_extractor import extract_semantic_context
            
            context_gen_start = time.time()
            semantic_context = extract_semantic_context(transcript)
            context_gen_elapsed = time.time() - context_gen_start
            
            # Make timing visible in console (not just logs)
            print(f"⚡ Semantic extraction: {context_gen_elapsed*1000:.2f}ms - Goal: '{semantic_context['user_goal'][:60]}...'")
            if semantic_context['key_entities']:
                entities_str = ', '.join([f"{k}={v}" for k, v in semantic_context['key_entities'].items()])
                print(f"   📊 Entities: {entities_str}")
            logger.info(f"⚡ Semantic context extracted in {context_gen_elapsed*1000:.1f}ms (fallback): '{semantic_context['user_goal']}'")
            
            # STEP 2: Classify intent with enriched context
            logger.debug("🔍 Classifying intent with semantic context (fallback path)...")
            classify_start = time.time()
            
            # Build enriched context
            enriched_context = {
                **(context or {}),
                'semantic_context': semantic_context,
                'user_goal': semantic_context['user_goal'],
                'key_entities': semantic_context['key_entities'],
                'extracted_meaning': semantic_context['extracted_meaning']
            }
            
            # Use Leibniz's native parser with enriched context
            if _leibniz_parser:
                intent_result = await _leibniz_parser.classify_intent(
                    text=semantic_context['extracted_meaning'],  # Use normalized meaning
                    context=enriched_context
                )
            else:
                # Basic fallback using semantic context
                logger.warning("No intent parser available, using fallback")
                intent_result = {
                    'intent': 'RAG_QUERY',
                    'confidence': 0.5,
                    'context': semantic_context
                }
            
            classify_elapsed = time.time() - classify_start
            
            # Merge semantic context with intent result
            intent_context = {
                **semantic_context,
                **intent_result.get("context", {})
            }
            
            # Generate final user_context string
            user_goal = intent_context.get('user_goal', '')
            extracted_meaning = intent_context.get('extracted_meaning', transcript)
            
            if user_goal and user_goal != extracted_meaning:
                user_context_transcript = f"{user_goal}: {extracted_meaning}"
            else:
                user_context_transcript = extracted_meaning
            
            logger.info(f"⚡ Intent classified in {classify_elapsed:.3f}s (fallback): {intent_result['intent']} (conf: {intent_result.get('confidence', 0.0):.2f})")
            logger.info(f"✨ Semantic context generated in {context_gen_elapsed*1000:.1f}ms (part of classification)")
            
            intent_msg = IntentMessage(
                intent=intent_result.get("intent", "UNCLEAR"),
                confidence=intent_result.get("confidence", 0.0),
                entities=intent_context,  # Comment 2: Pass full context dict
                user_context=user_context_transcript,  # Enriched query for RAG (= semantic context)
                reasoning=intent_result.get("reasoning", ""),
                processing_time=classify_elapsed,
                timing_breakdown={
                    'classification_ms': (classify_elapsed - context_gen_elapsed) * 1000,
                    'context_generation_ms': context_gen_elapsed * 1000
                }
            )
        
        return transcript_msg, intent_msg
        
    except Exception as e:
        logger.error(f"Transcribe and classify error: {e}", exc_info=True)
        
        # Return error messages
        return (
            TranscriptMessage(transcript="", confidence=0.0),  # FIX: Use 'transcript' not 'text'
            IntentMessage(intent="UNCLEAR", confidence=0.0, entities={}, reasoning=str(e))
        )


async def handle_continuous_user_speech(transcript: str):
    """
    Callback invoked by continuous VAD when user speaks.
    
    Processes user speech from background listener, classifies intent,
    and signals main loop via event. Integrates with existing barge-in
    infrastructure for real-time TTS interruption.
    
    Args:
        transcript: User speech transcript from continuous VAD
    """
    global _current_user_transcript, _current_user_intent, _user_speech_ready
    
    try:
        logger.debug(f"👤 User (continuous): {transcript}")
        
        # CHECK IF AGENT IS CURRENTLY SPEAKING - prevent false barge-in during TTS playback
        from leibniz_agent.leibniz_vad import get_leibniz_vad
        vad = get_leibniz_vad()
        if vad and vad.is_agent_speaking:
            logger.debug("🎤 Agent is speaking - ignoring continuous VAD detection during TTS playback")
            return  # Ignore speech detection while agent is speaking
        
        # Extract semantic context
        from leibniz_agent.leibniz_semantic_extractor import extract_semantic_context
        
        context_gen_start = time.time()
        semantic_context = extract_semantic_context(transcript)
        context_gen_elapsed = time.time() - context_gen_start
        
        logger.debug(f"⚡ Semantic extraction: {context_gen_elapsed*1000:.2f}ms - Goal: '{semantic_context['user_goal'][:60]}...'")
        
        # Classify intent using existing path
        classify_start = time.time()
        
        if services_manager and services_manager.intent_parser.parser:
            # Build enriched context
            enriched_context = {
                'semantic_context': semantic_context,
                'user_goal': semantic_context['user_goal'],
                'key_entities': semantic_context['key_entities'],
                'extracted_meaning': semantic_context['extracted_meaning']
            }
            
            # Use Leibniz's intent parser
            if _leibniz_parser:
                intent_result = await _leibniz_parser.classify_intent(
                    text=semantic_context['extracted_meaning'],
                    context=enriched_context
                )
            else:
                intent_result = {
                    'intent': 'RAG_QUERY',
                    'confidence': 0.5,
                    'context': semantic_context
                }
        else:
            # Fallback
            intent_result = {
                'intent': 'RAG_QUERY',
                'confidence': 0.5,
                'context': semantic_context
            }
        
        classify_elapsed = time.time() - classify_start
        
        # Merge contexts
        intent_context = {
            **semantic_context,
            **intent_result.get('context', {})
        }
        
        # Generate user_context string
        user_goal = intent_context.get('user_goal', '')
        extracted_meaning = intent_context.get('extracted_meaning', transcript)
        
        if user_goal and user_goal != extracted_meaning:
            user_context_transcript = f"{user_goal}: {extracted_meaning}"
        else:
            user_context_transcript = extracted_meaning
        
        logger.debug(f"⚡ Intent classified (continuous): {intent_result['intent']} (conf: {intent_result.get('confidence', 0.0):.2f})")
        
        # Store results in global variables
        _current_user_transcript = transcript
        _current_user_intent = intent_result.get('intent', 'UNCLEAR')
        
        # BARGE-IN HANDLING: Check if agent is speaking
        # Note: _streaming_active and _cancel_streaming are module-level globals
        
        if _streaming_active:
            logger.debug("⚡ BARGE-IN: User spoke during TTS playback - stopping agent")
            
            # Set cancel streaming event
            _cancel_streaming.set()
            
            # Set barge-in flag for existing detection
            from leibniz_agent.leibniz_vad import get_leibniz_vad
            vad = get_leibniz_vad()
            vad.barge_in_detected = True
        else:
            logger.debug("BARGE-IN: User spoke but agent not speaking")
        
        # Signal main loop that intent classification is complete
        _user_speech_ready.set()
        
    except Exception as e:
        logger.error(f"❌ Error in continuous user speech handler: {e}")
        
        # Set UNCLEAR intent and still signal event
        _current_user_transcript = transcript
        _current_user_intent = 'UNCLEAR'
        _user_speech_ready.set()


# ============================================================================
# RAG Query Handling
# ============================================================================

def compute_rag_confidence(timing_breakdown: Optional[Dict[str, float]] = None, method: str = 'default') -> float:
    """
    Compute confidence score for RAG responses using a standardized approach (TARA pattern).
    
    This helper ensures consistent confidence scoring across cache hits, speculative hits,
    and fresh RAG executions. It prioritizes relevance_score from ensemble retrieval when
    available, otherwise uses method-specific defaults.
    
    Args:
        timing_breakdown: Optional timing dict containing 'relevance_score' from ensemble
        method: RAG method ('cache_hit', 'speculative_hit', 'persistent_rag', 'fallback', etc.)
    
    Returns:
        Confidence score clamped to [0.0, 1.0]
    """
    # Extract relevance score from timing breakdown if available
    relevance_score = None
    if timing_breakdown and isinstance(timing_breakdown, dict):
        relevance_score = timing_breakdown.get('relevance_score')
    
    if relevance_score is not None:
        # Derive confidence from ensemble relevance score with clamping
        # Map relevance (typically 0.0-1.0) to confidence range [0.5, 0.95]
        confidence = max(0.5, min(0.95, float(relevance_score)))
    else:
        # Use method-specific defaults when relevance not available
        if method == 'cache_hit':
            confidence = 0.85  # High confidence for cached responses
        elif method == 'speculative_hit':
            confidence = 0.82  # Slightly lower than cache (pre-computed, not user-specific)
        elif method == 'persistent_rag':
            confidence = 0.85  # High confidence for fresh RAG
        elif method in ('fallback_timeout', 'fallback_error', 'router_skip'):
            confidence = 0.3  # Low confidence for fallback responses
        else:
            confidence = 0.5  # Neutral default
    
    # Final clamp to [0.0, 1.0]
    return max(0.0, min(1.0, confidence))


async def retry_with_backoff(func, *args, max_retries: int = 2, initial_delay: float = 0.5, backoff_factor: float = 2.0, **kwargs):
    """
    Retry wrapper with exponential backoff for transient failures (TARA pattern).
    
    Args:
        func: Async function to retry
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts (default 2)
        initial_delay: Initial delay between retries in seconds (default 0.5s)
        backoff_factor: Exponential multiplier for delay (default 2.0)
        **kwargs: Keyword arguments for func
    
    Returns:
        Result from func if successful
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            return await func(*args, **kwargs)
        except (asyncio.TimeoutError, ConnectionError, Exception) as e:
            last_exception = e
            
            # Don't retry on cancellation
            if isinstance(e, asyncio.CancelledError):
                raise
            
            # Don't wait after last attempt
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
    
    # Raise last exception if all retries failed
    raise last_exception


def process_rag_for_natural_conversation(raw_answer: str, user_question: str, intent: str) -> str:
    """
    Enhanced RAG response processing with tone adjustment and quality checks (TARA pattern adapted for English).
    
    Args:
        raw_answer: Raw response from RAG system
        user_question: Original user question
        intent: Classified intent
        
    Returns:
        Cleaned, natural-sounding response
    """
    try:
        if not raw_answer or not raw_answer.strip():
            return "I don't have specific information on that. Can I help you with something else about the university?"
        
        # Clean up the response
        clean_response = raw_answer.strip()
        
        # Remove common RAG formatting artifacts
        clean_response = re.sub(r'^(Answer:|Response:|A:)', '', clean_response, flags=re.IGNORECASE).strip()
        clean_response = re.sub(r'\n+', ' ', clean_response)  # Replace newlines with spaces
        clean_response = re.sub(r'\s+', ' ', clean_response)  # Normalize whitespace
        
        # Detect "don't know" patterns and add helpful suggestions
        dont_know_patterns = ["don't know", "not sure", "no information", "can't find", "unclear"]
        if any(pattern in clean_response.lower() for pattern in dont_know_patterns):
            # Add helpful suggestion based on context
            if any(word in user_question.lower() for word in ['admission', 'apply', 'enroll']):
                clean_response += " You might want to visit the admissions office or check our website for the latest information."
            elif any(word in user_question.lower() for word in ['program', 'course', 'major', 'degree']):
                clean_response += " Would you like me to help you find information about our academic programs?"
            elif any(word in user_question.lower() for word in ['campus', 'facility', 'building', 'location']):
                clean_response += " You can also check the campus map or contact student services for more details."
            else:
                clean_response += " Feel free to ask about admissions, programs, campus facilities, or anything else!"
        
        # Validate response length
        if len(clean_response) < 20:
            return "I don't have detailed information on that. What else can I help you with regarding the university?"
        
        # Ensure response sounds conversational (add period if missing)
        if not clean_response.endswith(('.', '?', '!')):
            clean_response += '.'
        
        return clean_response
        
    except Exception as e:
        logger.error(f"Error processing RAG response: {e}")
        return "Sorry, I had trouble processing that information. What else can I help you with?"


async def handle_rag_query(
    text: str,
    context: Dict[str, Any],
    enable_streaming: bool = True,
    user_id: str = "anonymous",
    streaming_callback=None,
    extra_data: dict = None
) -> RAGMessage:
    """
    Comprehensive RAG query handler with TARA pattern (cache, speculative, adaptive timeout, fallbacks).
    
    Execution flow:
    1. Extract context and check router gating (should_use_rag flag)
    2. Check cache for instant response (1-5ms)
    3. Check speculative execution pre-computed results
    4. Execute persistent RAG with adaptive timeout
    5. Apply response quality processing
    6. Cache successful results
    7. Fallback to intent-specific responses on timeout/error
    
    Args:
        text: The user's question text
        context: Conversation context (includes intent, extracted_info, user_context)
        enable_streaming: Whether to enable streaming TTS (default True)
        user_id: User identifier for personalization (default "anonymous")
        streaming_callback: Optional callback for streaming updates
        extra_data: Additional context (user_context, conversation_history)
        
    Returns:
        RAGMessage with answer, sources, confidence, metadata (includes timing_breakdown, method)
    """
    start_time = time.time()
    logger.debug(f"📚 RAG query received: '{text}' (user_id={user_id}, streaming={enable_streaming})")
    
    # FIX: Add query deduplication to prevent multiple simultaneous RAG executions for same query
    # This prevents duplicate responses when multiple calls happen rapidly
    query_hash = hash(text.strip().lower())
    
    # Use a global deduplication lock (create if doesn't exist)
    global _rag_deduplication_locks
    if '_rag_deduplication_locks' not in globals():
        _rag_deduplication_locks = {}
    
    # Get or create lock for this specific query
    if query_hash not in _rag_deduplication_locks:
        _rag_deduplication_locks[query_hash] = asyncio.Lock()
    
    query_lock = _rag_deduplication_locks[query_hash]
    
    # Acquire lock to prevent duplicate executions
    async with query_lock:
        logger.debug(f"🔒 Acquired RAG deduplication lock for query hash: {query_hash}")
        
        # PHASE 3 CHANGE 3.1: Use shared consumer task instead of creating duplicate
        consumer_task = None
        if enable_streaming:
            consumer_task = await start_tts_consumer()
            logger.debug("🎵 TTS consumer task started for progressive playback")
        
        # === 1. Context Extraction ===
        intent_data = context.get('last_intent', {})
        intent_type = intent_data.get('intent', 'GENERAL_QUERY')
        extracted_info = intent_data.get('extracted_info', {})
        should_use_rag = intent_data.get('should_use_rag', True)
        
        # Extract user context from extra_data (optional)
        user_context = {}
        if extra_data:
            user_context = extra_data.get('user_context', {})
        
        # === Router Gating: Skip RAG if intent says not needed ===
        if not should_use_rag:
            logger.info(f"⚡ Router gating: Intent {intent_type} bypasses RAG")
            elapsed = time.time() - start_time
            
            # Use intent-specific fallback directly
            try:
                fallback_mgr = get_fallback_manager()
                fallback_answer = fallback_mgr.get_fallback(intent_type, context="router_skip")
            except:
                fallback_answer = "Let me help you with that directly."
            
            return RAGMessage(
                answer=fallback_answer,
                sources=[],
                confidence=compute_rag_confidence(method='router_skip'),
                processing_time_ms=elapsed * 1000,
                timing_breakdown={
                    'method': 'router_skip',
                    'should_use_rag': False
                }
            )
        
        # === Performance Tracking Initialization ===
        try:
            perf_tracker = get_performance_tracker()
            metrics = RAGPerformanceMetrics(
                query_text=text,
                user_id=user_id,
                intent=intent_type,
                timestamp=time.time()
            )
        except:
            perf_tracker = None
            metrics = None
        
        try:
            # === 2. Cache Check (1-5ms) ===
            cache_check_start = time.time()
            cache_mgr = get_rag_cache_manager()
            cached_response = cache_mgr.get_query_response(text, language='english')
            cache_check_time = (time.time() - cache_check_start) * 1000  # ms
            
            if cached_response:
                logger.info(f"⚡ Cache HIT in {cache_check_time:.1f}ms")
                elapsed = time.time() - start_time
                
                # Process cached response for natural conversation (same as fresh RAG)
                processed_answer = process_rag_for_natural_conversation(cached_response, text, intent_type)
                
                # Stream cached response to TTS queue if streaming enabled
                if enable_streaming:
                    try:
                        await stream_rag_to_tts(processed_answer, pace=1.0, is_final=True)
                        logger.debug(f"🎵 Cached response streamed to TTS queue: {len(processed_answer)} chars")
                    except Exception as stream_error:
                        logger.warning(f"Failed to stream cached response to TTS: {stream_error}")
                
                # Build RAG message from cache
                rag_message = RAGMessage(
                    answer=processed_answer,  # Use processed answer, not raw cache
                    sources=[],  # Sources not cached (could be added in future)
                    confidence=compute_rag_confidence(method='cache_hit'),
                    processing_time_ms=elapsed * 1000,
                    timing_breakdown={
                        'method': 'cache_hit',
                        'cache_check_ms': cache_check_time
                    }
                )
                
                # Record cache hit in performance tracker
                if perf_tracker and metrics:
                    metrics.method = 'cache_hit'
                    metrics.response_time = elapsed
                    metrics.success = True
                    perf_tracker.record_query(metrics)
                
                return rag_message
            
            logger.info(f"⚠️ Cache MISS ({cache_check_time:.1f}ms) - proceeding to retrieval")
            
            # === 3. Speculative Execution Check ===
            speculative_check_start = time.time()
            try:
                spec_coord = get_speculative_coordinator()
                speculative_result = spec_coord.get_best_result(text)
                speculative_check_time = (time.time() - speculative_check_start) * 1000  # ms
                
                if speculative_result:
                    # Compute accurate age from coordinator (Comment 3)
                    exec_id = spec_coord.find_execution_id_by_result(speculative_result)
                    if exec_id:
                        age_seconds = spec_coord.get_execution_age(exec_id)
                        # Clean up superseded executions
                        spec_coord.cancel_executions(except_execution_id=exec_id)
                    else:
                        age_seconds = 0.0
                    
                    logger.info(f"⚡ Speculative HIT in {speculative_check_time:.1f}ms (age: {age_seconds:.1f}s)")
                    elapsed = time.time() - start_time
                    
                    # Extract answer from speculative result (robust extraction - Comment 1)
                    spec_answer = ""
                    spec_sources = []
                    spec_timing = {}
                    
                    if isinstance(speculative_result, dict):
                        spec_answer = speculative_result.get('response', '') or speculative_result.get('answer', '')
                        spec_sources = speculative_result.get('sources', [])
                        spec_timing = speculative_result.get('timing_breakdown', {})
                    elif isinstance(speculative_result, tuple) and len(speculative_result) > 0:
                        spec_answer = str(speculative_result[0])
                        logger.debug(f"Speculative returned tuple, normalized to string: {len(spec_answer)} chars")
                    elif isinstance(speculative_result, str):
                        spec_answer = speculative_result
                        logger.debug(f"Speculative returned string: {len(spec_answer)} chars")
                    else:
                        logger.warning(f"Unexpected speculative result type: {type(speculative_result)}, attempting str() conversion")
                        spec_answer = str(speculative_result)
                    
                    # Process speculative response for natural conversation (same as fresh RAG)
                    processed_answer = process_rag_for_natural_conversation(spec_answer, text, intent_type)
                    
                    # Stream speculative response to TTS queue if streaming enabled
                    if enable_streaming:
                        try:
                            await stream_rag_to_tts(processed_answer, pace=1.0, is_final=True)
                            logger.debug(f"🎵 Speculative response streamed to TTS queue: {len(processed_answer)} chars")
                        except Exception as stream_error:
                            logger.warning(f"Failed to stream speculative response to TTS: {stream_error}")
                    
                    # Build RAG message
                    rag_message = RAGMessage(
                        answer=processed_answer,  # Use processed answer, not raw speculative
                        sources=spec_sources,
                        confidence=compute_rag_confidence(timing_breakdown=spec_timing, method='speculative_hit'),
                        processing_time_ms=elapsed * 1000,
                        timing_breakdown={
                            'method': 'speculative_hit',
                            'speculative_age_s': age_seconds,
                            **spec_timing
                        }
                    )
                    
                    # Cache the result for future queries
                    cache_mgr.cache_query_response(text, processed_answer, language='english')
                    
                    # Record speculative hit
                    if perf_tracker and metrics:
                        metrics.method = 'speculative_hit'
                        metrics.response_time = elapsed
                        metrics.success = True
                        perf_tracker.record_query(metrics)
                    
                    return rag_message
                
                logger.info(f"⚠️ Speculative MISS ({speculative_check_time:.1f}ms)")
            
            except Exception as e:
                logger.warning(f"Speculative check failed: {e}")
                speculative_check_time = 0
            
            # === 4. Persistent RAG with Adaptive Timeout ===
            # Comment 5 FIX: Removed verbose logging before RAG call to minimize latency
            
            # Comment 1.3: Fast-path optimization - minimize operations before RAG call
            # Only lightweight setup: adaptive timeout + callback definition
            # No heavy transforms, formatting, or processing on hot path
            
            # Determine adaptive timeout based on context (lightweight)
            base_timeout = 30.0  # Default for RAG queries
            
            # Adjust timeout based on intent complexity
            if intent_type in ('ADMISSION_QUERY', 'PROGRAM_INFO'):
                adaptive_timeout = 35.0  # More time for complex queries
            elif intent_type in ('GENERAL_QUERY', 'CAMPUS_INFO'):
                adaptive_timeout = 25.0  # Faster for simple queries
            else:
                adaptive_timeout = base_timeout
            
            # Get persistent services (singleton, fast)
            services = await get_leibniz_services_manager()
            
            # Comment 1.3: Start RAG call immediately - no formatting before this point
            rag_start = time.time()
            try:
                # Define synchronous streaming callback for thread-safe TTS delivery
                # Note: RAG runs in thread, so callback must be thread-safe
                # Comment 4: Remove _sentinel_sent guard, always push sentinel on is_final=True
                def rag_streaming_callback_sync(partial_text, is_final: bool = False):
                    """Thread-safe streaming callback - queues sentence strings for TTS"""
                    if enable_streaming:
                        # Use thread-safe queue operation
                        try:
                            # Queue sentence string only if non-empty
                            if partial_text.strip():
                                _tts_streaming_queue.put_nowait((partial_text, 1.0))
                                # DIAGNOSTIC: Log enqueued sentences with queue size
                                logger.debug(f"📝 Enqueued: '{partial_text[:50]}...' (queue size: {_tts_streaming_queue.qsize()})")
                            
                            # Comment 4: Always send sentinel on is_final=True (no guard)
                            # Consumer will handle multiple sentinels gracefully
                            if is_final:
                                _tts_streaming_queue.put_nowait(None)
                                # DIAGNOSTIC: Changed to info level for visibility
                                logger.debug(f"📍 Sentinel sent from callback (queue size: {_tts_streaming_queue.qsize()})")
                        except asyncio.QueueFull:
                            logger.warning(f"TTS queue full, dropping sentence: '{partial_text[:30]}...'")
                
                # Define synchronous wrapper for clean parameter passing (lightweight)
                def _call_rag():
                    return services.rag_system.rag_system.process_rag_query(
                        context=context,
                        query=text,
                        streaming_callback=rag_streaming_callback_sync if enable_streaming else None
                    )
                
                # Call RAG system in thread to avoid blocking event loop
                # Comment 1.3: Immediate execution - no preprocessing delay
                result = await asyncio.wait_for(
                    asyncio.to_thread(_call_rag),
                    timeout=adaptive_timeout
                )
                
                rag_time = (time.time() - rag_start) * 1000  # ms
                
                # Comment 5 FIX: Detailed logging AFTER RAG completes (not before)
                logger.info(f"✅ Persistent RAG completed in {rag_time:.1f}ms for: '{text[:50]}...'")
                
                # Extract result components (handle dict/tuple/string returns)
                raw_answer = ""
                sources = []
                timing_breakdown = {}
                
                if isinstance(result, dict):
                    raw_answer = result.get('response', '') or result.get('answer', '')
                    sources = result.get('sources', [])
                    timing_breakdown = result.get('timing_breakdown', {})
                elif isinstance(result, tuple) and len(result) > 0:
                    raw_answer = str(result[0])
                    logger.debug(f"RAG returned tuple, normalized to string: {len(raw_answer)} chars")
                elif isinstance(result, str):
                    raw_answer = result
                    logger.debug(f"RAG returned string: {len(raw_answer)} chars")
                else:
                    logger.warning(f"Unexpected RAG result type: {type(result)}, attempting str() conversion")
                    raw_answer = str(result)
                
                # === 5. Response Quality Processing ===
                processed_answer = process_rag_for_natural_conversation(raw_answer, text, intent_type)
                
                # Compute confidence from timing breakdown
                confidence = compute_rag_confidence(timing_breakdown=timing_breakdown, method='persistent_rag')
                
                elapsed = time.time() - start_time
                
                # Build RAG message
                rag_message = RAGMessage(
                    answer=processed_answer,
                    sources=sources,
                    confidence=confidence,
                    processing_time_ms=elapsed * 1000,
                    timing_breakdown={
                        'method': 'persistent_rag',
                        'rag_ms': rag_time,
                        'cache_check_ms': cache_check_time,
                        'speculative_check_ms': speculative_check_time,
                        **timing_breakdown
                    }
                )
                
                # === 6. Cache Successful Result ===
                try:
                    cache_mgr.cache_query_response(text, processed_answer, language='english')
                    logger.info(f"💾 Cached response for future queries")
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")
                
                # Record success in performance tracker
                if perf_tracker and metrics:
                    metrics.method = 'persistent_rag'
                    metrics.response_time = elapsed
                    metrics.success = True
                    metrics.confidence = confidence
                    perf_tracker.record_query(metrics)
                
                # Wait for TTS consumer to finish playing all sentences
                if consumer_task and enable_streaming:
                    try:
                        await consumer_task
                        logger.debug("✅ TTS consumer task completed")
                    except Exception as consumer_error:
                        logger.warning(f"TTS consumer task error: {consumer_error}")
                
                return rag_message
            
            except asyncio.TimeoutError:
                # === 7. Timeout Fallback ===
                logger.warning(f"⏱️ RAG timeout after {adaptive_timeout}s")
                elapsed = time.time() - start_time
                
                # Get intent-specific timeout fallback
                try:
                    fallback_mgr = get_fallback_manager()
                    fallback_answer = fallback_mgr.get_fallback(intent_type, context="timeout")
                except:
                    fallback_answer = "I'm taking a bit longer to find that information. Could you ask me something else while I look into it?"
                
                # Record timeout in performance tracker
                if perf_tracker and metrics:
                    metrics.method = 'fallback_timeout'
                    metrics.response_time = elapsed
                    metrics.success = False
                    metrics.error = 'timeout'
                    perf_tracker.record_query(metrics)
                
                return RAGMessage(
                    answer=fallback_answer,
                    sources=[],
                    confidence=compute_rag_confidence(method='fallback_timeout'),
                    processing_time_ms=elapsed * 1000,
                    timing_breakdown={
                        'method': 'fallback_timeout',
                        'timeout_s': adaptive_timeout
                    }
                )
            
            except Exception as rag_error:
                # === 7. Error Fallback ===
                logger.error(f"❌ Persistent RAG error: {rag_error}")
                elapsed = time.time() - start_time
                
                # Get intent-specific error fallback
                try:
                    fallback_mgr = get_fallback_manager()
                    fallback_answer = fallback_mgr.get_fallback(intent_type, context="error")
                except:
                    fallback_answer = "I'm having trouble accessing that information right now. Could you try asking in a different way?"
                
                # Record error in performance tracker
                if perf_tracker and metrics:
                    metrics.method = 'fallback_error'
                    metrics.response_time = elapsed
                    metrics.success = False
                    metrics.error = str(rag_error)
                    perf_tracker.record_query(metrics)
                
                return RAGMessage(
                    answer=fallback_answer,
                    sources=[],
                    confidence=compute_rag_confidence(method='fallback_error'),
                    processing_time_ms=elapsed * 1000,
                    timing_breakdown={
                        'method': 'fallback_error',
                        'error': str(rag_error)
                    }
                )
        
        except Exception as e:
            # === Critical Fallback (outermost exception handler) ===
            logger.error(f"❌ RAG handler critical failure: {e}", exc_info=True)
            elapsed = time.time() - start_time
            
            generic_fallback = "I'm sorry, I'm having some technical difficulties. Please try again or ask about something else."
            
            
            # Record critical failure
            if perf_tracker and metrics:
                metrics.method = 'fallback_critical'
                metrics.response_time = elapsed
                metrics.success = False
                metrics.error = str(e)
                perf_tracker.record_query(metrics)
            
            return RAGMessage(
                answer=generic_fallback,
                sources=[],
                confidence=0.1,
                processing_time_ms=elapsed * 1000,
                timing_breakdown={
                    'method': 'fallback_critical',
                    'error': str(e)
                }
            )

    # Cleanup: Remove old locks to prevent memory leaks (keep last 100)
    if len(_rag_deduplication_locks) > 100:
        # Remove oldest locks (simple FIFO cleanup)
        oldest_keys = list(_rag_deduplication_locks.keys())[:50]  # Remove 50 oldest
        for key in oldest_keys:
            del _rag_deduplication_locks[key]
        logger.debug(f"🧹 Cleaned up {len(oldest_keys)} old RAG deduplication locks")


# ============================================================================
# Appointment FSM Handling
# ============================================================================

async def handle_appointment_booking(
    initial_input: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Handle appointment booking conversation flow
    
    Args:
        initial_input: Optional initial user input (e.g., "I want to schedule an appointment")
        
    Returns:
        Booking data dictionary if completed, None if cancelled
    """
    # Comment 9: FSM state is now local to this function, not global
    
    # FIX: Pause continuous VAD to prevent session conflicts during appointment booking
    continuous_vad_was_running = False
    try:
        # Check if continuous VAD is enabled and running
        if _continuous_vad_enabled and _continuous_vad_instance and _continuous_vad_instance.is_running:
            logger.info("⏸️ Pausing continuous VAD during appointment booking to prevent session conflicts")
            await stop_leibniz_continuous_listening()
            continuous_vad_was_running = True
        
        # Create FSM instance (Comment 9: session-scoped, not global)
        fsm = create_appointment_fsm()
        
        # Process initial input if provided
        if initial_input:
            result = await fsm.process_input(initial_input)
        else:
            result = await fsm.process_input("")
        
        # Speak FSM response
        await speak_friendly(result['response'], emotion="helpful")
        
        # Enter FSM loop
        while True:
            # Check completion
            if result.get('complete', False):
                # Booking completed
                booking_data = format_appointment_for_submission(fsm.data)
                
                logger.info(f"✅ Appointment booking completed: {booking_data}")
                
                # Speak confirmation
                await speak_friendly(
                    dialogue_key='appointment_confirm',
                    emotion="excited"
                )
                
                return booking_data
            
            # Check cancellation (Comment 1: Compare against enum value string)
            if result.get('state') == AppointmentState.CANCELLED.value:
                logger.info("❌ Appointment booking cancelled by user")
                return None
            
            # Capture user input (Comment 1: transcript-only return, no audio_file)
            transcript = await capture_and_transcribe()
            
            # Comment 5: Pass empty transcripts to FSM for all confirmation states (including final confirm)
            # FSM handles empty responses by defaulting to 'yes' after max attempts for consistency
            if not transcript:
                # Check if we're in a confirmation state
                current_state = result.get('state', '')
                if 'confirm' in current_state:
                    # Pass empty string to FSM - it will handle empty response logic
                    transcript = ""
                else:
                    # Not in confirmation state - ask user to retry
                    await speak_friendly(
                        dialogue_key='timeout',
                        emotion="calm"
                    )
                    continue
            
            # MULTILINGUAL SUPPORT: Generate semantic context with translation
            # This enables slot filling from Hindi, Telugu, and other languages
            # Check if multilingual support is enabled (default: true)
            enable_multilingual = os.getenv("LEIBNIZ_APPOINTMENT_ENABLE_MULTILINGUAL", "true").lower() == "true"
            
            if transcript and enable_multilingual:
                try:
                    # Get current FSM state for context-aware translation
                    current_fsm_state = result.get('state', '')
                    
                    # Generate semantic context with translation
                    semantic_context = await generate_semantic_context_for_fsm(
                        user_input=transcript,
                        fsm_state=current_fsm_state
                    )
                    
                    # Use translated English text for FSM processing
                    translated_text = semantic_context['translated_text']
                    detected_lang = semantic_context['detected_language']
                    
                    # Log translation if non-English detected
                    if detected_lang != 'english':
                        logger.info(f"🌐 Multilingual input detected ({detected_lang}): '{transcript}' → '{translated_text}'")
                    
                    # Use translated text for FSM
                    fsm_input = translated_text
                    
                except Exception as e:
                    logger.warning(f"Semantic translation failed: {e}, using original transcript")
                    fsm_input = transcript
            elif transcript:
                # Multilingual disabled - use original transcript
                fsm_input = transcript
            else:
                # Empty transcript (confirmation state passthrough)
                fsm_input = ""
            
            # Process input with FSM
            result = await fsm.process_input(fsm_input)
            
            # Speak FSM response
            await speak_friendly(result['response'], emotion="helpful")
        
    except Exception as e:
        logger.error(f"Appointment booking error: {e}", exc_info=True)
        
        # Apologize and exit
        await speak_friendly(
            dialogue_key='errors.general',
            emotion="calm"
        )
        
        return None
    finally:
        # FIX: Resume continuous VAD if it was running before appointment booking
        if continuous_vad_was_running:
            try:
                logger.info("▶️ Resuming continuous VAD after appointment booking")
                await start_leibniz_continuous_listening()
            except Exception as e:
                logger.error(f"❌ Failed to resume continuous VAD: {e}")
        
        # Comment 9: No finally block needed - fsm is local variable, auto-cleaned


# ============================================================================
# Greeting and Introduction
# ============================================================================

async def play_natural_intro():
    """Play introduction audio or speak greeting"""
    # Check if intro audio enabled
    if os.getenv("LEIBNIZ_ENABLE_INTRO_AUDIO", "true").lower() == "true":
        intro_path = os.getenv("LEIBNIZ_INTRO_AUDIO_PATH", INTRO_AUDIO_PATH)
        
        if PYGAME_AVAILABLE and os.path.exists(intro_path):
            # Play intro audio file
            set_leibniz_agent_speaking(True)
            
            try:
                pygame.mixer.music.load(intro_path)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
                
                logger.info("✅ Intro audio played")
            except Exception as e:
                logger.error(f"Intro audio playback error: {e}")
            finally:
                set_leibniz_agent_speaking(False)
            
            return
    
    # Fallback: Speak greeting from intro_greeting.txt file
    intro_file = os.path.join(VOICE_DIR, "intro_greeting.txt")
    if os.path.exists(intro_file):
        try:
            with open(intro_file, 'r', encoding='utf-8') as f:
                greeting = f.read().strip()
            logger.info(f"✅ Loaded greeting from {intro_file}")
            
            # Try to speak the loaded greeting - if successful, return early
            try:
                print("\n🔊 AGENT SPEAKING...")
                await speak_friendly(
                    text=greeting,
                    emotion="helpful"
                )
                print("👂 AGENT LISTENING...")
                return
            except Exception as speak_error:
                logger.warning(f"Failed to speak loaded greeting: {speak_error}")
                # Fall through to fallback
            
        except Exception as e:
            logger.warning(f"Failed to read intro greeting file: {e}")
            # Fallback to dialogue manager
            greeting = get_dialogue_text('greeting', "Hello! I'm TARA, the receptionist at Leibniz University. Welcome! How may I assist you today?")
    else:
        logger.warning(f"Intro greeting file not found: {intro_file}")
        # Fallback to dialogue manager
        greeting = get_dialogue_text('greeting', "Hello! I'm TARA, the receptionist at Leibniz University. Welcome! How may I assist you today?")
    
    # Fallback: Speak greeting from dialogue manager
    print("\n🔊 AGENT SPEAKING...")
    await speak_friendly(
        text=greeting,
        emotion="helpful"
    )
    print("👂 AGENT LISTENING...")


# ============================================================================
# One-Time Initialization
# ============================================================================

async def initialize_leibniz_services():
    """One-time startup initialization"""
    global services_manager, leibniz_config, _streaming_mode_logged
    
    try:
        # Set event loop policy for Windows
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        logger.info("🚀 Initializing Leibniz Pro services...")
        print("=" * 70)
        print("🎓 LEIBNIZ PRO - University Customer Service Agent")
        print("🔥 Persistent Services | Gemini Live VAD | Dual Prewarm Strategy")
        print("=" * 70)
        
        # Comment 12: Log streaming mode ONCE per session
        streaming_enabled = os.getenv("LEIBNIZ_ENABLE_STREAMING_TTS", "true").lower() == "true"
        tts_provider = os.getenv("LEIBNIZ_TTS_PROVIDER", "auto")
        
        if not _streaming_mode_logged:
            logger.info(f"🎵 Streaming TTS Mode: {'ENABLED' if streaming_enabled else 'DISABLED'}")
            logger.info(f"🔊 TTS Provider: {tts_provider}")
            print(f"\n🎵 Streaming TTS: {'ENABLED' if streaming_enabled else 'DISABLED'} | Provider: {tts_provider}")
            _streaming_mode_logged = True
        
        # Load Leibniz config
        leibniz_config = get_leibniz_config()
        logger.info("✅ Configuration loaded")
        
        # Initialize persistent services manager with detailed logging
        print("\n🔥 Initializing Persistent Services for Parallel Processing...")
        init_start = time.time()
        services_manager = await get_leibniz_services_manager()
        init_elapsed = time.time() - init_start
        
        logger.info(f"✅ Persistent Services initialized in {init_elapsed:.2f}s")
        print(f"✅ Persistent Services ready in {init_elapsed:.2f}s")
        
        # Log detailed service status (TARA pattern)
        if services_manager:
            status = services_manager.get_service_status()
            intent_ready = status.get("intent_parser_ready", False)
            rag_ready = status.get("rag_system_ready", False)
            rag_stats = status.get("rag_system_stats", {})
            doc_count = rag_stats.get("vector_store_size", 0)
            
            logger.info(f"   🧠 Intent Parser: {'✅ Ready' if intent_ready else '❌ Failed'}")
            logger.info(f"   📚 RAG System: {'✅ Ready' if rag_ready else '❌ Failed'} ({doc_count} docs)")
            print(f"   🧠 Intent Parser: {'✅ Ready' if intent_ready else '❌ Failed'}")
            print(f"   📚 RAG System: {'✅ Ready' if rag_ready else '❌ Failed'} ({doc_count} docs)")
        else:
            logger.warning("   ⚠️  Services manager not available")
            print("   ⚠️  Services manager not available")
        
        # Initialize component singletons
        stt = get_leibniz_stt()
        logger.info("✅ STT ready")
        
        # Comment 5: Wrap TTS initialization with graceful fallback
        try:
            tts = get_leibniz_tts()
            logger.info("✅ TTS ready")
            tts_mode = "audio"
        except Exception as tts_error:
            mock_mode = os.getenv('MOCK_TTS', 'false').lower() == 'true'
            allow_no_tts = os.getenv('ALLOW_NO_TTS', 'false').lower() == 'true'
            
            if mock_mode or allow_no_tts:
                logger.warning(f"TTS initialization failed: {tts_error}")
                logger.info("🔇 Running without TTS (text-only mode)")
                tts_mode = "text-only"
            else:
                logger.error(f"TTS initialization failed and fallback not enabled: {tts_error}")
                raise
        
        # Initialize Leibniz Intent Parser (Gemini 2.0 based)
        try:
            parser = get_leibniz_parser()
            logger.info("✅ Leibniz Intent Parser initialized (Gemini 2.0)")
            print("✅ Leibniz Intent Parser ready (Gemini 2.0)")
        except Exception as parser_error:
            logger.warning(f"Intent parser initialization failed: {parser_error}, using fallback")
            parser = None
        
        # Store parser globally for transcribe_and_classify
        global _leibniz_parser
        _leibniz_parser = parser
        
        rag = get_leibniz_rag()
        logger.info("✅ RAG system ready")
        
        vad = get_leibniz_vad()
        
        # VERIFICATION COMMENT 3: Gate VAD verbose logging behind environment variables
        vad_verbose = os.getenv('LEIBNIZ_VAD_VERBOSE', 'false').lower() == 'true'
        vad_log_audio = os.getenv('LEIBNIZ_VAD_LOG_AUDIO', 'false').lower() == 'true'
        vad_log_state = os.getenv('LEIBNIZ_VAD_LOG_STATE', 'false').lower() == 'true'
        
        # Configure VAD verbose logging only if environment flags set
        try:
            vad.config.log_audio_callbacks = vad_log_audio
            vad.config.log_timeout_checks = vad_verbose
            vad.config.log_state_transitions = vad_log_state
            
            if vad_verbose or vad_log_audio or vad_log_state:
                logger.info(f"🔍 VAD verbose logging: audio={vad_log_audio}, state={vad_log_state}, verbose={vad_verbose}")
                print(f"🔍 VAD verbose logging ENABLED - audio={vad_log_audio}, state={vad_log_state}, verbose={vad_verbose}")
            else:
                logger.info("✅ VAD ready (verbose logging disabled)")
        except AttributeError:
            # Some configs might not have these attributes
            logger.info("✅ VAD ready")
            pass
        
        # Pre-warm VAD session (TARA pattern - create persistent session ahead of time)
        try:
            await smart_warmup_leibniz_vad()
        except Exception as e:
            pass  # Silent failure for VAD warmup
        
        # Optional startup audio test to verify playback system
        test_audio_on_startup = os.getenv("LEIBNIZ_TEST_AUDIO_ON_STARTUP", "false").lower() == "true"
        if test_audio_on_startup:
            logger.info("🔊 Testing audio playback system on startup...")
            print("\n🔊 Testing audio playback system...")
            try:
                # Test with a short diagnostic message
                test_message = "Audio system test successful. Leibniz agent is ready."
                await speak_friendly(
                    text=test_message,
                    emotion="professional",
                    enable_streaming=False  # Use non-streaming for immediate test
                )
                logger.info("✅ Startup audio test passed")
                print("✅ Audio playback test successful")
            except Exception as audio_test_error:
                logger.error(f"❌ Startup audio test failed: {audio_test_error}")
                print(f"❌ Audio playback test failed: {audio_test_error}")
                # Don't raise - allow system to continue with text-only mode
        
        # Optional: Start continuous background VAD for barge-in support
        continuous_vad_enabled = os.getenv("LEIBNIZ_ENABLE_CONTINUOUS_VAD", "true").lower() == "true"
        
        # Print readiness message
        logger.info("\n" + "="*60)
        logger.info("✅ Leibniz Pro ready for university customer service!")
        logger.info("="*60)
        logger.info("Service Status:")
        logger.info("  - Intent parser: Ready")
        logger.info("  - RAG system: Ready")
        logger.info(f"  - TTS: Ready ({tts_mode} mode)" if tts_mode == "audio" else f"  - TTS: {tts_mode.upper()} MODE")
        logger.info("  - STT: Ready (Gemini Live API)")
        logger.info("  - VAD: Ready (barge-in detection)")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Initialization error: {e}", exc_info=True)
        raise


# ============================================================================
# Main Conversation Loop
# ============================================================================

def is_exit_phrase(transcript: str) -> bool:
    """
    Check if transcript contains exit phrase using whole-word matching.
    
    Prevents false positives from substring matches (e.g., 'maybe' containing 'bye').
    Uses word tokenization for single-word exits and regex with word boundaries
    for multi-word phrases.
    
    Args:
        transcript: User's transcript to check
        
    Returns:
        True if transcript contains an exit phrase, False otherwise
    """
    import re
    
    if not transcript:
        return False
    
    transcript_lower = transcript.lower().strip()
    
    # Single-word exit keywords - use word boundary checks
    single_word_exits = ["bye", "goodbye", "exit", "quit", "stop"]
    
    # Multi-word exit phrases - use exact phrase matching with word boundaries
    multi_word_exits = [
        "that's all",
        "that is all",
        "thanks bye",
        "thank you bye",
        "no thanks",
        "i'm done",
        "i am done"
    ]
    
    # Check single-word exits with word boundaries
    for word in single_word_exits:
        # Use regex with word boundaries to match whole words only
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, transcript_lower):
            return True
    
    # Check multi-word phrases with word boundaries
    for phrase in multi_word_exits:
        # Use regex with word boundaries for entire phrase
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, transcript_lower):
            return True
    
    return False


async def run_conversation_session():
    """
    Run a single conversation session with bidirectional conversation loop.
    
    Uses TARA's proven for-loop structure with max_attempts (5) for automatic retry behavior.
    Implements conversation context tracking (last_interaction_type) for dynamic timeout adjustment.
    Includes VAD session health checks (reset after 3 consecutive timeouts) and automatic resets.
    Resets speculative coordinator between attempts to prevent stale executions.
    Adds inter-attempt pause (0.5s) for VAD stabilization.
    
    Flow:
        1. Play natural intro
        2. For each attempt (up to max_attempts=5):
            a. Determine conversation context based on attempt and last interaction
            b. Check VAD health (reset if >=3 consecutive timeouts)
            c. Reset speculative coordinator for fresh state
            d. Capture and classify user speech
            e. Route to appropriate handler (appointment, RAG, greeting, exit)
            f. Track interaction type for next iteration
            g. Pause 0.5s between attempts for VAD stabilization
        3. Log session metrics and cleanup
    """
    global conversation_active
    global _current_user_intent
    
    try:
        # Session initialization
        conversation_active = True
        await reset_leibniz_conversation()
        
        session_start = time.time()
        logger.info(f"🎬 Starting conversation session at {time.strftime('%H:%M:%S')}")
        
        # Natural introduction
        await play_natural_intro()
        
        # Start continuous VAD after intro (only for this session)
        continuous_vad_enabled = os.getenv("LEIBNIZ_ENABLE_CONTINUOUS_VAD", "true").lower() == "true"
        if continuous_vad_enabled:
            logger.debug("🎤 Starting continuous background VAD for this session...")
            try:
                # Get continuous VAD instance
                continuous_vad = get_continuous_vad()
                
                # Set callback to handle user speech
                continuous_vad.on_user_speech = handle_continuous_user_speech
                
                # Start background listening
                await start_leibniz_continuous_listening()
                
                # Store instance globally for this session
                global _continuous_vad_enabled, _continuous_vad_instance
                _continuous_vad_enabled = True
                _continuous_vad_instance = continuous_vad
                
                logger.debug("✅ Continuous VAD started - background listening active")
                print("✅ Continuous VAD enabled - user can interrupt anytime")
            except Exception as e:
                logger.error(f"❌ Failed to start continuous VAD: {e}")
                logger.debug("⚠️ Falling back to per-turn VAD mode")
                _continuous_vad_enabled = False
        
        # Initialize conversation tracking (outside loop to persist across attempts)
        consecutive_no_input = 0
        last_interaction_type = ""  # Track: greeting, rag_query, appointment, fallback
        max_attempts = 5  # TARA's proven configuration
        
        # Add session and turn tracking for dialogue archiving
        session_id = f"leibniz_session_{int(time.time())}"
        turn_number = 0
        
        # Main bidirectional conversation loop (TARA pattern: for-loop instead of while-loop)
        # Why for-loop: Provides explicit attempt boundaries, easier retry logic, clearer max limits
        for attempt in range(max_attempts):
            turn_start = time.time()
            turn_number += 1  # Increment turn number for dialogue archiving
            
            # CONTINUOUS VAD HEALTH CHECK: Verify continuous VAD is healthy before each attempt
            if _continuous_vad_enabled and _continuous_vad_instance:
                try:
                    # Check if continuous VAD is still running and healthy
                    if not _continuous_vad_instance.is_running:
                        logger.warning("⚠️ Continuous VAD not running - restarting")
                        await _continuous_vad_instance.start_continuous_listening()
                    
                    # Additional health check: verify no stuck state
                    metrics = _continuous_vad_instance.get_performance_metrics()
                    consecutive_timeouts = metrics.get('consecutive_timeouts', 0)
                    if consecutive_timeouts >= 3:
                        logger.warning(f"⚠️ Continuous VAD has {consecutive_timeouts} consecutive timeouts - resetting")
                        await _continuous_vad_instance.restart_listener()
                        
                except Exception as health_error:
                    logger.error(f"❌ Continuous VAD health check failed: {health_error}")
                    # Fall back to per-turn mode
                    logger.warning("🔄 Falling back to per-turn VAD mode due to health check failure")
                    _continuous_vad_enabled = False
            
            # Log attempt with clear boundaries
            logger.debug(f"\n🗣️ Conversation attempt {attempt + 1}/{max_attempts} (turn {turn_number})")
            
            try:
                # Step 1: Determine conversation context based on attempt and last interaction
                # Maps to VAD timeout adjustment for better capture behavior
                if attempt == 0:
                    current_context = "greeting"  # First attempt: user likely greeting
                elif last_interaction_type == "appointment":
                    # Comment 2: Use 'post_service' instead of 'post_appointment' (VAD-recognized context)
                    current_context = "post_service"  # After appointment: decision phase
                elif last_interaction_type == "rag_query":
                    current_context = "complex_query"  # After RAG: user processing answer
                else:
                    current_context = "decision"  # Default: decision-making phase
                
                # Log state information after current_context is computed
                logger.debug(f"📊 State: attempt={attempt+1}, no_input={consecutive_no_input}, last={last_interaction_type}, context={current_context}")
                
                # Step 2: Set dynamic timeout before capture (moved earlier to avoid duplicate VAD access)
                vad = get_leibniz_vad()
                if vad and hasattr(vad, 'set_dynamic_timeout'):
                    vad.set_dynamic_timeout(
                        attempt_count=attempt,
                        conversation_context=current_context
                    )
                    logger.debug(f"🕐 Dynamic timeout set: context={current_context}, attempt={attempt}")
                
                # Step 3: Reset speculative coordinator between attempts (if available)
                # Prevents stale speculative executions from previous turns
                if SPECULATIVE_AVAILABLE and get_speculative_coordinator and get_performance_config:
                    try:
                        coordinator = get_speculative_coordinator()
                        config = get_performance_config()
                        session_id = f"leibniz_attempt_{attempt}"
                        
                        coordinator.reset_session(session_id=session_id, keep_recent_seconds=5.0)
                        
                        if config.enable_speculative_logs:
                            logger.debug(f"🔄 Speculative coordinator reset for attempt {attempt + 1}")
                    except Exception as spec_error:
                        logger.debug(f"Speculative coordinator reset skipped: {spec_error}")
                
                # Step 4: Capture and classify user input
                print("\n🎤 Listening... SPEAK NOW!")
                
                # Start timing for capture duration
                capture_start_time = time.time()
                
                # Check if continuous VAD is enabled
                if _continuous_vad_enabled and _continuous_vad_instance:
                    # CONTINUOUS MODE: Wait for background listener event
                    logger.debug("🎧 Waiting for user speech (continuous VAD)...")
                    
                    # Determine timeout based on context
                    timeout_map = {
                        'greeting': 25.0,
                        'decision': 30.0,
                        'complex_query': 35.0,
                        'post_service': 20.0
                    }
                    timeout = timeout_map.get(current_context, 20.0)
                    
                    # Wait for user speech using helper (handles event internally)
                    from leibniz_agent.leibniz_continuous_vad import wait_for_leibniz_speech
                    transcript = await wait_for_leibniz_speech(timeout=timeout)
                    
                    if transcript:
                        # User spoke - wait for intent classification to complete
                        logger.debug("⏳ Waiting for intent classification to complete...")
                        
                        # Wait for the callback to finish processing (with timeout)
                        try:
                            await asyncio.wait_for(_user_speech_ready.wait(), timeout=5.0)
                            logger.debug("✅ Intent classification completed")
                        except asyncio.TimeoutError:
                            logger.debug("⏱️ Intent classification timeout - using UNCLEAR intent")
                            _current_user_intent = 'UNCLEAR'
                        
                        # Clear the event for next use
                        _user_speech_ready.clear()
                        
                        # Now safely access the intent
                        intent = _current_user_intent or 'UNCLEAR'
                        
                        # Build transcript and intent messages (same structure as transcribe_and_classify)
                        transcript_msg = TranscriptMessage(transcript=transcript, confidence=1.0)
                        intent_msg = IntentMessage(
                            intent=intent,
                            confidence=0.9,  # High confidence from background classification
                            entities={},  # Populated by callback
                            user_context=transcript,
                            reasoning="Continuous VAD"
                        )
                        
                        logger.debug(f"✅ User speech received (continuous): '{transcript}' → {intent}")
                    else:
                        # Timeout - no user speech
                        logger.debug(f"⏱️ Timeout waiting for user speech ({timeout}s)")
                        transcript_msg = TranscriptMessage(transcript="", confidence=0.0)
                        intent_msg = IntentMessage(intent="UNCLEAR", confidence=0.0, entities={}, user_context="", reasoning="Timeout")


                else:
                    # PER-TURN MODE: Use existing blocking capture (backward compatible)
                    logger.debug("🎤 Using per-turn VAD capture (continuous mode disabled)")
                    
                    # Define streaming callback for real-time transcript display (SINDH clean pattern)
                    speech_detected = [False]  # Mutable flag for closure
                    
                    def display_streaming_transcript(fragment: str, is_final: bool):
                        """Display partial transcripts in real-time during capture (clean output)"""
                        if fragment and fragment.strip():
                            # Show speech detection indicator once (no fragment display)
                            if not speech_detected[0]:
                                print("\n  🗣️  Speech detected!")
                                speech_detected[0] = True
                    
                    # Construct context dict for VAD
                    # Comment 4: Use attempt for conversation attempt count
                    vad_context = {
                        "conversation_context": current_context,
                        "attempt_count": attempt
                    }
                    
                    # Log VAD listening state entry with context
                    logger.debug(f"🎤 Entering VAD listening state (context: {current_context}, attempt: {attempt})")
                    
                    # Existing per-turn capture
                    transcript_msg, intent_msg = await transcribe_and_classify(
                        streaming_callback=display_streaming_transcript,
                        context=vad_context
                    )
                
                # Continue with existing code - transcript and intent processing
                capture_duration = time.time() - capture_start_time
                logger.debug(f"⏱️ VAD capture completed in {capture_duration:.2f}s")
                
                transcript = transcript_msg.transcript
                intent = intent_msg.intent
                context = intent_msg.entities
                confidence = intent_msg.confidence
                
                # Add transcript confirmation logging (TARA pattern)
                if transcript:
                    print(f"🎯 Final complete transcript: '{transcript}'")
                    logger.debug(f"📄 Transcript: '{transcript}'")
                    
                    # Display intent classification with timing
                    processing_time = getattr(intent_msg, 'processing_time', 0.0)
                    if processing_time > 0:
                        print(f"⚡ Intent classified in {processing_time:.3f}s: {intent} (conf: {confidence:.2f})")
                    else:
                        print(f"⚡ Intent classified: {intent} (conf: {confidence:.2f})")
                    
                    logger.debug(f"🎯 Intent: {intent} (confidence: {confidence:.2f})")
                    
                    # Add detailed timing breakdown (TARA pattern)
                    timing_breakdown = getattr(intent_msg, 'timing_breakdown', {})
                    if timing_breakdown:
                        classification_ms = timing_breakdown.get('classification_ms', 0)
                        context_gen_ms = timing_breakdown.get('context_generation_ms', 0)
                        if classification_ms > 0 or context_gen_ms > 0:
                            print(f"⏱️ Intent classification: {classification_ms:.0f}ms, Context generation: {context_gen_ms:.1f}ms")
                            logger.debug(f"Timing breakdown - Classification: {classification_ms:.0f}ms, Context: {context_gen_ms:.1f}ms")
                    
                    # Add semantic context logging (TARA pattern)
                    # NOTE: "Semantic context" = "Enriched context" = user_context (ALL THE SAME)
                    user_context = getattr(intent_msg, 'user_context', None)
                    if user_context and user_context != transcript:
                        print(f"✨ Semantic context extracted: '{user_context}'")
                        logger.debug(f"Semantic/Enriched context: '{user_context}'")
                
                # Step 6: Handle no-input with escalation backoff
                if not transcript:
                    consecutive_no_input += 1
                    logger.warning(f"⚠️ No speech captured (attempt {consecutive_no_input})")
                    
                    # Debug logging for timing visibility
                    logger.debug(f"⏱️ No-input timing: will speak prompt immediately, then wait for user response")
                    
                    # Max 5 attempts with escalating prompts
                    if consecutive_no_input >= 5:
                        logger.error("❌ Max no-input attempts (5) reached - exiting conversation")
                        await reset_leibniz_conversation()
                        await speak_friendly(
                            dialogue_key='errors.general',
                            emotion="professional"
                        )
                        break  # Exit conversation loop
                    
                    # Forced VAD reset after 2 consecutive timeouts (TARA pattern)
                    # Comment 6: Update warning message for clarity
                    if consecutive_no_input >= 2:
                        logger.warning("⚠️ Forcing immediate VAD reset (2+ consecutive timeouts)")
                        await reset_leibniz_conversation()
                        logger.info("✅ VAD session forcibly reset, ready for next capture")
                    
                    # Speak prompts immediately (no pre-speaking backoff delay)
                    # TARA pattern: speak immediately, then wait for user response
                    if consecutive_no_input == 1:
                        print("\n🔊 AGENT SPEAKING...")
                        await speak_friendly(
                            dialogue_key='timeout',
                            emotion="calm"
                        )
                        last_interaction_type = "fallback"
                        # No blocking sleep - VAD's configured timeout (20-30s) handles the waiting period
                        # Return to loop immediately to start listening
                        
                    elif consecutive_no_input == 2:
                        print("\n🔊 AGENT SPEAKING...")
                        await speak_friendly(
                            dialogue_key='timeout',
                            emotion="calm"
                        )
                        last_interaction_type = "fallback"
                        # No blocking sleep - VAD's configured timeout handles the waiting period
                        # Return to loop immediately to start listening
                        
                    else:  # 3-4 consecutive misses
                        print("\n🔊 AGENT SPEAKING...")
                        await speak_friendly(
                            dialogue_key='timeout',
                            emotion="calm"
                        )
                        last_interaction_type = "fallback"
                        # No blocking sleep - VAD's configured timeout handles the waiting period
                        # Return to loop immediately to start listening
                    
                    # Check if conversation should continue
                    if not conversation_active:
                        break
                    
                    # Continue to next attempt (no additional delay needed)
                    continue
                
                # Reset no-input counter on successful capture
                consecutive_no_input = 0
                
                # Step 7: Check for exit keywords before intent routing
                # Comment 1: Use whole-word matching to prevent false exits (e.g., 'maybe' contains 'bye')
                if is_exit_phrase(transcript):
                    logger.debug("👋 Exit keyword detected in transcript")
                    print("\n🔊 AGENT SPEAKING...")
                    await speak_friendly(
                        dialogue_key='farewells.exit',
                        emotion="calm"
                    )
                    conversation_active = False
                    break
                
                # Step 8: Route based on intent and track interaction type
                if intent == "APPOINTMENT_SCHEDULING":
                    logger.debug("📅 Routing to appointment booking FSM")
                    
                    booking_data = await handle_appointment_booking(initial_input=transcript)
                    
                    if booking_data:
                        logger.debug(f"✅ Booking completed: {booking_data}")
                    else:
                        logger.debug("❌ Booking cancelled or failed")
                        print("\n🔊 AGENT SPEAKING...")
                        await speak_friendly(
                            dialogue_key='prompts.continue',
                            emotion="helpful"
                        )
                    
                    last_interaction_type = "appointment"
                
                elif intent == "RAG_QUERY":
                    # Enhanced RAG processing logging (TARA pattern)
                    if len(transcript) > 100:
                        print(f"🚀 Processing RAG query: '{transcript[:100]}...'")
                    else:
                        print(f"🚀 Processing RAG query: '{transcript}'")
                    
                    logger.debug("📚 Routing to RAG system with context")
                    rag_start = time.time()
                    
                    # Use enriched user_context if available (TARA pattern)
                    query_text = intent_msg.user_context or transcript
                    
                    if intent_msg.user_context and intent_msg.user_context != transcript:
                        print(f"🔍 RAG processing (semantic context): '{query_text}'")
                        logger.info(f"✅ Using semantic context for RAG: '{query_text}'")
                    else:
                        print(f"🔍 RAG processing (raw transcript): '{query_text}'")
                        logger.info(f"⚠️ Using raw transcript for RAG (no semantic context): '{query_text}'")
                    
                    rag_msg = await handle_rag_query(
                        text=query_text,
                        context=context,
                        enable_streaming=False  # Use non-streaming for complete responses
                    )
                    
                    # Guard against None rag_msg (Comment 2)
                    if not rag_msg:
                        print("❌ RAG returned no result")
                        logger.error("RAG query returned None")
                        fallback_text = get_dialogue_text('error', "I'm sorry, I had trouble processing that question. Could you try rephrasing it?")
                        print(f"🔊 Speaking fallback response: {len(fallback_text)} chars")
                        print("\n🔊 AGENT SPEAKING...")
                        await speak_friendly(
                            dialogue_key='errors.general',
                            emotion="apologetic"
                        )
                        print("👂 AGENT LISTENING...")
                        last_interaction_type = "fallback"
                        continue
                    
                    rag_elapsed = time.time() - rag_start
                    
                    # Display RAG response content (user requested)
                    print(f"\n📄 RAG Response Generated:")
                    print(f"{'='*60}")
                    print(f"{rag_msg.answer}")
                    print(f"{'='*60}\n")
                    
                    # Enhanced RAG response logging (TARA pattern)
                    print(f"⚡ RAG response in {rag_elapsed:.2f}s: {len(rag_msg.answer)} chars, {len(rag_msg.sources)} sources")
                    logger.debug(f"✅ RAG response: {len(rag_msg.answer)} chars, {len(rag_msg.sources)} sources (in {rag_elapsed:.2f}s)")
                    logger.debug(f"📄 RAG answer content: {rag_msg.answer}")
                    
                    # Add timing breakdown if available (TARA pattern)
                    timing_breakdown = getattr(rag_msg, 'timing_breakdown', {})
                    if timing_breakdown:
                        ensemble_ms = timing_breakdown.get('ensemble_retrieval_ms', 0)
                        response_gen_ms = timing_breakdown.get('response_gen_ms', 0)
                        if ensemble_ms > 0 or response_gen_ms > 0:
                            logger.debug(f"⏱️ Ensemble retrieval: {ensemble_ms:.0f}ms, Response gen: {response_gen_ms:.0f}ms")
                    
                        logger.debug(f"📊 RAG confidence: {rag_msg.confidence:.2f}, method: {getattr(rag_msg, 'method', 'unknown')}")
                    
                    # Add character count to speaking log (TARA pattern)
                    print(f"🔊 Speaking RAG response: {len(rag_msg.answer)} chars")
                    
                    # Clean markdown formatting from RAG response for better TTS pronunciation
                    import re
                    clean_answer = re.sub(r'\*\*(.*?)\*\*', r'\1', rag_msg.answer)  # Remove **bold**
                    clean_answer = re.sub(r'\*(.*?)\*', r'\1', clean_answer)  # Remove *italic*
                    clean_answer = re.sub(r'`([^`]+)`', r'\1', clean_answer)  # Remove `code`
                    clean_answer = re.sub(r'#{1,6}\s+', '', clean_answer)  # Remove headers
                    clean_answer = re.sub(r'^\s*[-*+]\s+', '', clean_answer, flags=re.MULTILINE)  # Remove list markers
                    clean_answer = re.sub(r'\n\s*\n', '\n', clean_answer)  # Clean up extra newlines
                    
                    logger.debug(f"📝 Cleaned RAG response for TTS: {len(clean_answer)} chars (was {len(rag_msg.answer)})")
                    
                    # Use non-streaming TTS for complete RAG responses
                    print("\n🔊 AGENT SPEAKING...")
                    await speak_friendly(
                        text=clean_answer,
                        emotion="helpful",
                        enable_streaming=False
                    )
                    print("👂 AGENT LISTENING...")
                    
                    # Archive dialogue audio if enabled
                    if os.getenv('LEIBNIZ_ENABLE_DIALOGUE_ARCHIVE', 'false').lower() == 'true':
                        asyncio.create_task(
                            archive_dialogue_audio(
                                audio_file=None,  # Will be retrieved from TTS cache
                                session_id=session_id,
                                turn_number=turn_number,
                                text=clean_answer,  # Use cleaned text for archiving
                                dialogue_type="rag"
                            )
                        )
                    
                    last_interaction_type = "rag_query"
                
                elif intent == "GREETING":
                    logger.debug("👋 Greeting detected")
                    
                    greeting_text = get_dialogue_text('greeting', "Hello! How can I help you today?")
                    print(f"🔊 Speaking response: {len(greeting_text)} chars")
                    print("\n🔊 AGENT SPEAKING...")
                    await speak_friendly(
                        dialogue_key='greetings.intro',
                        emotion="helpful"
                    )
                    print("👂 AGENT LISTENING...")
                    last_interaction_type = "greeting"
                
                elif intent == "EXIT":
                    logger.debug("👋 Exit intent detected")
                    
                    exit_text = get_dialogue_text('farewell', "Thanks for chatting! Have a great day, and feel free to reach out anytime you need help.")
                    print(f"🔊 Speaking response: {len(exit_text)} chars")
                    print("\n🔊 AGENT SPEAKING...")
                    await speak_friendly(
                        dialogue_key='farewells.exit',
                        emotion="calm"
                    )
                    
                    conversation_active = False
                    break
                
                else:  # UNCLEAR or unknown
                    logger.debug("❓ Unclear intent")
                    
                    fallback_text = get_dialogue_text('clarification_prompt', "I didn't quite catch that. Could you rephrase your question? I can help with information about the university or schedule appointments.")
                    print(f"🔊 Speaking fallback response: {len(fallback_text)} chars")
                    print("\n🔊 AGENT SPEAKING...")
                    await speak_friendly(
                        dialogue_key='prompts.clarify',
                        emotion="helpful"
                    )
                    print("👂 AGENT LISTENING...")
                    last_interaction_type = "fallback"
                
                # Check if conversation should continue (external interrupt)
                if not conversation_active:
                    break
                
                # Log turn duration
                turn_duration = time.time() - turn_start
                logger.debug(f"⏱️ Attempt {attempt + 1} completed in {turn_duration:.2f}s")
                
                # Inter-attempt pause for VAD stabilization (TARA pattern)
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} error: {e}", exc_info=True)
                
                print("\n🔊 AGENT SPEAKING...")
                await speak_friendly(
                    dialogue_key='errors.general',
                    emotion="calm"
                )
                print("👂 AGENT LISTENING...")
                last_interaction_type = "fallback"
                
                # Don't break on errors - continue to next attempt
                await asyncio.sleep(0.5)
        
        # Session cleanup and metrics logging
                last_interaction_type = "fallback"
                
                # Don't break on errors - continue to next attempt
                await asyncio.sleep(0.5)
        
        # Session cleanup and metrics logging
        session_duration = time.time() - session_start
        logger.debug(f"\n✅ Session ended. Duration: {session_duration:.1f}s, Max attempts: {max_attempts}")
        
        # Enhanced session metrics logging (TARA pattern)
        logger.debug(f"📊 Session Metrics Summary:")
        logger.debug(f"  - Total conversation attempts: {max_attempts}")
        logger.debug(f"  - Consecutive timeouts: {consecutive_no_input}")
        logger.debug(f"  - Last interaction type: {last_interaction_type}")
        logger.debug(f"  - Session duration: {session_duration:.1f}s")
        
        # Log VAD performance metrics
        vad = get_leibniz_vad()
        if vad:
            metrics = vad.get_performance_metrics()
            session_stats = metrics.get('session_stats', {})
            
            logger.debug(f"Performance Metrics:")
            logger.debug(f"  - Total captures: {metrics.get('capture_count', 0)}")
            logger.debug(f"  - Average capture time: {metrics.get('avg_capture_time', 0.0):.2f}s")
            logger.debug(f"  - Consecutive timeouts: {metrics.get('consecutive_timeouts', 0)}")
            logger.debug(f"  - Barge-in detected: {metrics.get('barge_in_detected', False)}")
            logger.debug(f"  - Conversation state: {metrics.get('conversation_state', 'unknown')}")
            logger.debug(f"  - Session reuses: {session_stats.get('total_uses', 0)}")
            logger.debug(f"  - Session age: {session_stats.get('session_age', 0.0):.1f}s")
        
        # Log TTS provider statistics
        tts = get_leibniz_tts()
        if tts:
            provider_stats = tts.get_provider_stats()
            if provider_stats.get('provider_stats_enabled'):
                logger.debug(f"\n📊 TTS Provider Statistics:")
                for provider, stats in provider_stats['providers'].items():
                    total = stats['success'] + stats['failure']
                    if total > 0:
                        success_rate = (stats['success'] / total) * 100
                        logger.debug(f"  - {provider.capitalize()}: {stats['success']}/{total} ({success_rate:.1f}%)")
                        if stats['errors']:
                            error_summary = {}
                            for error_type, _ in stats['errors']:
                                error_summary[error_type] = error_summary.get(error_type, 0) + 1
                            logger.debug(f"    Errors: {error_summary}")
        
    except Exception as e:
        logger.error(f"Session error: {e}", exc_info=True)
    finally:
        conversation_active = False
        # Stop continuous VAD for this session
        if _continuous_vad_enabled and _continuous_vad_instance:
            try:
                logger.debug("⏸️ Stopping continuous VAD for session cleanup")
                await stop_leibniz_continuous_listening()
                _continuous_vad_enabled = False
                _continuous_vad_instance = None
            except Exception as e:
                logger.error(f"❌ Failed to stop continuous VAD: {e}")
        # Comment 9: No global current_fsm to clear - FSM is session-scoped


# ============================================================================
# Multi-Session Orchestrator
# ============================================================================

async def test_microphone():
    """
    Test microphone independently to verify hardware works.
    
    FIX 5: Independent microphone test to isolate audio capture issues
    from VAD/TTS state management problems.
    """
    try:
        import pyaudio
        import numpy as np
        
        print("\n🎤 MICROPHONE TEST")
        print("="*40)
        print("Speak now for 3 seconds...")
        
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                       input=True, frames_per_buffer=1024)
        
        max_volume = 0
        for _ in range(int(16000 / 1024 * 3)):  # 3 seconds
            data = stream.read(1024)
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            max_volume = max(max_volume, volume)
            print(f"  Volume: {volume:>6.0f}", end='\r')
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        print(f"\n  Max volume: {max_volume:.0f}")
        if max_volume < 100:
            print("  ⚠️ WARNING: Very low audio - check microphone!")
        else:
            print("  ✅ Microphone working!")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"  ❌ Microphone test failed: {e}")
        print("  Check if pyaudio and numpy are installed")
        print("="*40 + "\n")

async def main():
    """Main entry point with multi-session support"""
    try:
        # Start background audio FIRST (parallel, non-blocking)
        if start_background_audio():
            logger.info("✅ Background audio started")
        
        # One-time initialization
        await initialize_leibniz_services()
        
        # FIX 5: Test microphone independently before starting sessions
        await test_microphone()
        
        # Check readiness
        if services_manager:
            status = await get_leibniz_service_status()
            logger.info(f"Service status: {status}")
        
        # Print welcome message
        print("\n")
        print("╔" + "="*58 + "╗")
        print("║  🎓 Leibniz University Customer Service Agent - Ready!  ║")
        print("║" + " "*58 + "║")
        print("║  Press ENTER to start a new conversation session        ║")
        print("║  Press Ctrl+C to exit                                   ║")
        print("╚" + "="*58 + "╝")
        print("\n")
        
        # Enter-to-start loop
        while True:
            try:
                # Comment 8: Use asyncio.to_thread to avoid blocking event loop
                await asyncio.to_thread(input, "\nPress ENTER to start...")
                
                # Run conversation session
                await run_conversation_session()
                
                print("\n✅ Session complete. Ready for next session.\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
        
        # Graceful shutdown
        logger.info("Shutting down Leibniz Pro...")
        
        # Unified continuous VAD cancellation
        await cancel_continuous_vad()
        
        # Stop background audio
        stop_background_audio()
        
        # Print performance summary
        vad = get_leibniz_vad()
        if vad:
            metrics = vad.get_performance_metrics()
            session_stats = metrics.get('session_stats', {})
            
            print("\n" + "="*60)
            print("Performance Summary:")
            print(f"  Total captures: {metrics.get('capture_count', 0)}")
            print(f"  Average capture time: {metrics.get('avg_capture_time', 0.0):.2f}s")
            print(f"  Consecutive timeouts: {metrics.get('consecutive_timeouts', 0)}")
            print(f"  Barge-in detected: {metrics.get('barge_in_detected', False)}")
            print(f"  Session reuses: {session_stats.get('total_uses', 0)}")
            print(f"  Session age: {session_stats.get('session_age', 0.0):.1f}s")
            print("="*60 + "\n")
        
        # Cleanup VAD
        await cleanup_leibniz_vad()
        
        # Cleanup pygame
        if PYGAME_AVAILABLE:
            pygame.quit()
        
        logger.info("👋 Leibniz Pro shutdown complete. Goodbye!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Leibniz Pro interrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error in Leibniz Pro: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        print("Please check logs for details.")
