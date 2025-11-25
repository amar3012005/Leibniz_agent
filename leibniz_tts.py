#!/usr/bin/env python3
"""
Leibniz University Agent - Text-to-Speech Module
================================================

LemonFox TTS system for natural English speech synthesis with emotion-based voice modulation and comprehensive caching.

Key Features:
- **LemonFox Provider**: Cost-effective ($2.50/1M chars), fast, reliable, 8 voices, multilingual
- **Emotion-Based Modulation**: Adjust speed based on context (excited=1.2, calm=0.95, etc.)
- **Comprehensive Caching**: MD5-based caching with LRU cleanup, 30-day TTL
- **Two Synthesis Modes**: File-based (`synthesize_to_file()`) and streaming (`stream_tts()`)
- **Automatic Retry Logic**: Exponential backoff for transient errors
- **High-Quality Audio**: 24kHz sample rate for natural English speech

LemonFox TTS Features:
- **8 Voices**: sarah, heart, bella, michael, alloy, nova, echo, onyx
- **8+ Languages**: en-us, de-de, es-es, fr-fr, it-it, pl-pl, pt-br, nl-nl
- **Emotion Mapping**: Maps emotions to speed (excited=1.2, calm=0.95, etc.)
- **Cost**: $2.50 per 1M characters (~90% cheaper than alternatives)
- **Output**: WAV format, 24kHz mono, high quality
- **API Endpoint**: https://api.lemonfox.ai/v1/audio/speech

Usage Example - File Synthesis:
    ```python
    from leibniz_agent import leibniz_speak
    
    # Synthesize and play with emotion
    await leibniz_speak("Hello! How can I help you today?", emotion="helpful")
    ```

Usage Example - Streaming:
    ```python
    from leibniz_agent import leibniz_stream_speak
    
    # Stream synthesis with real-time playback
    await leibniz_stream_speak("This is a streaming test.", emotion="excited")
    ```

Usage Example - Custom Configuration:
    ```python
    from leibniz_agent import LeibnizTTS
    
    # Initialize with custom settings
    tts = LeibnizTTS()
    result = await tts.synthesize_to_file(
        text="Welcome to Leibniz University!",
        outfile="welcome.wav",
        emotion="helpful"
    )
    print(f"Synthesized: {result['file']}, Duration: {result['duration']:.2f}s")
    ```

Environment Variables:
- LEMONFOX_API_KEY: LemonFox API key (https://www.lemonfox.ai)
- LEIBNIZ_LEMONFOX_VOICE: Voice name (sarah, heart, bella, michael, alloy, nova, echo, onyx)
- LEIBNIZ_LEMONFOX_LANGUAGE: Language code (en-us, de-de, es-es, fr-fr, it-it, pl-pl, pt-br, nl-nl)
- LEIBNIZ_TTS_CACHE_DIR: Cache directory path
- LEIBNIZ_TTS_CACHE_ENABLED: Enable/disable caching (true/false)

Setup Instructions:

**LemonFox (Primary and Only Provider):**
1. Sign up at https://www.lemonfox.ai
2. Get API key from dashboard
3. Set LEMONFOX_API_KEY environment variable
4. Choose voice: sarah (default), heart, bella, michael, alloy, nova, echo, onyx
5. Set language: en-us (default), de-de, es-es, fr-fr, it-it, pl-pl, pt-br, nl-nl
6. No installation required - uses standard aiohttp library

**Recommended Setup:**
- Primary: LemonFox (cost-effective, fast, reliable)
- Cache: Enabled with 30-day TTL
- Use emotion-based speed modulation for natural speech
- Enable caching for common phrases (greetings, responses)
"""

import os
import json
import wave
import asyncio
import hashlib
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

# Try importing aiohttp for async HTTP (required for LemonFox)
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print(" aiohttp not available. Install with: pip install aiohttp")

# Import Leibniz config
from leibniz_config import get_leibniz_config

# Import VAD and STT for agent speaking state and prewarm
from leibniz_vad import get_leibniz_vad
from leibniz_stt import prewarm_during_tts

# Load environment variables
load_dotenv()


# Custom Exception Classes (Comment 4)
class RateLimitedError(Exception):
    """Raised when API rate limit is exceeded"""
    pass

class TransientTTSError(Exception):
    """Raised for temporary TTS errors that may succeed on retry"""
    pass

class NonRetryableTTSError(Exception):
    """Raised for permanent TTS errors that should not be retried"""
    pass


# Constants
DEFAULT_CACHE_DIR = os.path.abspath("./leibniz_agent/audio_archive")  # Central archive for all synthesized voices
DIALOGUE_CACHE_DIR = os.path.abspath("./leibniz_agent/voices")  # TARA-style dialogue cache
DEFAULT_SAMPLE_RATE = 24000  # Higher quality for English
DEFAULT_FORMAT = "wav"
CACHE_INDEX_FILE = "cache_index.json"
MAX_CACHE_SIZE = 500
CACHE_TTL_DAYS = 30

# Create audio archive and dialogue cache directories
os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
os.makedirs(DIALOGUE_CACHE_DIR, exist_ok=True)


# Constants
DEFAULT_CACHE_DIR = os.path.abspath("./leibniz_agent/audio_archive")  # Central archive for all synthesized voices
DIALOGUE_CACHE_DIR = os.path.abspath("./leibniz_agent/voices")  # TARA-style dialogue cache
DEFAULT_SAMPLE_RATE = 24000  # Higher quality for English
DEFAULT_FORMAT = "wav"
CACHE_INDEX_FILE = "cache_index.json"
MAX_CACHE_SIZE = 500
CACHE_TTL_DAYS = 30

# Create audio archive and dialogue cache directories
os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
os.makedirs(DIALOGUE_CACHE_DIR, exist_ok=True)


@dataclass
class LeibnizTTSConfig:
    """Configuration for Leibniz TTS with LemonFox provider only"""
    # Provider settings
    provider: str = "lemonfox"  # Only lemonfox supported
    
    # LemonFox TTS settings
    lemonfox_api_key: str = ""  # LemonFox API key (from LEMONFOX_API_KEY env var)
    lemonfox_voice: str = "sarah"  # Default voice (sarah, heart, bella, michael, alloy, nova, echo, onyx)
    lemonfox_language: str = "en-us"  # Default language (en-us, de-de, es-es, fr-fr, it-it, pl-pl, pt-br, nl-nl)
    
    # Audio settings
    language_code: str = "en-US"
    sample_rate: int = 24000
    
    # Cache settings
    enable_cache: bool = True
    cache_dir: str = DEFAULT_CACHE_DIR
    max_cache_size: int = MAX_CACHE_SIZE
    cache_ttl_days: int = CACHE_TTL_DAYS
    
    # Retry settings
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


class TTSCache:
    """
    TTS cache with MD5-based keys and LRU cleanup.
    Adapted from SINDH V2 audio_handler.py caching pattern.
    """
    
    def __init__(self, cache_dir: str, max_size: int = 500):
        """
        Initialize TTS cache.
        
        Args:
            cache_dir: Directory for cached audio files
            max_size: Maximum number of cached entries
        """
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.cache_index: Dict[str, Dict[str, Any]] = {}
        self.index_file = self.cache_dir / CACHE_INDEX_FILE
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cache index
        self._load_cache_index()
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def _load_cache_index(self):
        """Load cache index from JSON file"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.cache_index = json.load(f)
                print(f" Loaded TTS cache index: {len(self.cache_index)} entries")
            except Exception as e:
                print(f" Failed to load cache index: {e}")
                self.cache_index = {}
        else:
            self.cache_index = {}
    
    def _save_cache_index(self):
        """Save cache index to JSON file"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            print(f" Failed to save cache index: {e}")
    
    def get_cache_key(
        self,
        text: str,
        voice: str,
        language: str,
        provider: str,
        emotion: str = "neutral"
    ) -> str:
        """
        Generate MD5 cache key from synthesis parameters.
        
        Args:
            text: Text to synthesize
            voice: Voice name/ID
            language: Language code
            provider: Provider name
            emotion: Emotion type
            
        Returns:
            MD5 hash string
        """
        cache_string = f"{text}_{voice}_{language}_{provider}_{emotion}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get_cached_audio(
        self,
        text: str,
        voice: str,
        language: str,
        provider: str,
        emotion: str = "neutral"
    ) -> Optional[str]:
        """
        Get cached audio file if exists.
        
        Args:
            text: Text to synthesize
            voice: Voice name/ID
            language: Language code
            provider: Provider name
            emotion: Emotion type
            
        Returns:
            Path to cached file or None
        """
        cache_key = self.get_cache_key(text, voice, language, provider, emotion)
        
        # Check if in index
        if cache_key not in self.cache_index:
            self.misses += 1
            return None
        
        entry = self.cache_index[cache_key]
        
        # Comment 8: Check TTL enforcement - remove expired entries
        current_time = time.time()
        if 'created' in entry:
            age_days = (current_time - entry['created']) / (24 * 3600)
            if age_days > CACHE_TTL_DAYS:
                # Remove expired entry
                cache_file = self.cache_dir / f"{cache_key}.wav"
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except OSError:
                        pass  # Best effort cleanup
                
                del self.cache_index[cache_key]
                self._save_cache_index()
                self.misses += 1
                return None
        
        # Check if file exists
        cache_file = self.cache_dir / f"{cache_key}.wav"
        if not cache_file.exists():
            # Remove from index if file missing
            del self.cache_index[cache_key]
            self._save_cache_index()
            self.misses += 1
            return None
        
        # Update last accessed time
        self.cache_index[cache_key]['last_accessed'] = current_time
        self._save_cache_index()
        
        self.hits += 1
        return str(cache_file)
    
    def cache_audio(
        self,
        text: str,
        voice: str,
        language: str,
        provider: str,
        emotion: str,
        audio_file: str
    ) -> str:
        """
        Cache audio file.
        
        Args:
            text: Text to synthesize
            voice: Voice name/ID
            language: Language code
            provider: Provider name
            emotion: Emotion type
            audio_file: Path to audio file to cache
            
        Returns:
            Path to cached file
        """
        cache_key = self.get_cache_key(text, voice, language, provider, emotion)
        cache_file = self.cache_dir / f"{cache_key}.wav"
        
        # Copy file to cache
        import shutil
        shutil.copy2(audio_file, cache_file)
        
        # Update index
        self.cache_index[cache_key] = {
            'text': text[:100],  # Store first 100 chars
            'voice': voice,
            'language': language,
            'provider': provider,
            'emotion': emotion,
            'created': time.time(),
            'last_accessed': time.time(),
            'file_size': cache_file.stat().st_size
        }
        
        # Cleanup if over max size
        if len(self.cache_index) > self.max_size:
            self._cleanup_cache()
        
        # Save index
        self._save_cache_index()
        
        return str(cache_file)
    
    def _cleanup_cache(self):
        """Cleanup old cache entries using LRU"""
        # Sort by last accessed time
        sorted_entries = sorted(
            self.cache_index.items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        # Remove oldest entries
        entries_to_remove = len(sorted_entries) - self.max_size
        if entries_to_remove > 0:
            for cache_key, _ in sorted_entries[:entries_to_remove]:
                # Delete file
                cache_file = self.cache_dir / f"{cache_key}.wav"
                if cache_file.exists():
                    cache_file.unlink()
                
                # Remove from index
                del self.cache_index[cache_key]
            
            print(f" Cleaned up {entries_to_remove} old cache entries")
    
    def clear_cache(self):
        """Clear all cached audio files"""
        for cache_key in list(self.cache_index.keys()):
            cache_file = self.cache_dir / f"{cache_key}.wav"
            if cache_file.exists():
                cache_file.unlink()
        
        self.cache_index = {}
        self._save_cache_index()
        self.hits = 0
        self.misses = 0
        print(" Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache_index),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }


class LemonFoxTTSProvider:
    """
    LemonFox.ai TTS Provider with multilingual support.
    
    Provides high-quality natural voices with speed control for emotion modulation.
    Supports 8+ languages with OpenAI-compatible API.
    
    Key Features:
    - Natural-sounding voices (sarah, heart, bella, michael, etc.)
    - 8+ languages (en-us, de-de, es-es, fr-fr, it-it, pl-pl, pt-br, nl-nl)
    - Speed control (0.25-4.0) for emotion expression
    - WAV output format (24kHz mono)
    - Low cost ($2.50 per 1M characters, ~90% cheaper than ElevenLabs)
    - Async HTTP API with aiohttp
    
    API Documentation:
        https://www.lemonfox.ai/apis/text-to-speech
    """
    
    API_ENDPOINT = "https://api.lemonfox.ai/v1/audio/speech"
    
    def __init__(
        self,
        api_key: str,
        voice: str = "sarah",
        language: str = "en-us"
    ):
        """
        Initialize LemonFox TTS provider.
        
        Args:
            api_key: LemonFox API key
            voice: Voice name (sarah, heart, bella, michael, alloy, nova, echo, onyx)
            language: Language code (en-us, de-de, es-es, fr-fr, it-it, pl-pl, pt-br, nl-nl)
            
        Raises:
            ValueError: If API key is not set
            ImportError: If aiohttp is not installed
        """
        if not api_key:
            raise ValueError(
                "LemonFox API key not set. Set LEMONFOX_API_KEY environment variable."
            )
        
        # Check aiohttp availability
        try:
            import aiohttp
            self.aiohttp = aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp not installed. Install with: pip install aiohttp"
            )
        
        self.api_key = api_key
        self.voice = voice
        self.language = language
        self.session = None
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f" LemonFox TTS initialized (voice: {voice}, language: {language})"
        )
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _ensure_session(self):
        """Create aiohttp session if not exists."""
        if self.session is None or self.session.closed:
            self.session = self.aiohttp.ClientSession()
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        emotion: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Synthesize text to speech using LemonFox API.
        
        Args:
            text: Text to synthesize
            voice: Voice name (defaults to config voice)
            language: Language code (defaults to config language)
            emotion: Emotion for speed modulation
            **kwargs: Additional parameters (ignored)
            
        Returns:
            WAV audio bytes (24kHz mono)
            
        Raises:
            aiohttp.ClientError: On API request failure
            asyncio.TimeoutError: On request timeout
        """
        # Comment 3: Sanitize input text
        text = self._sanitize_text(text)
        
        await self._ensure_session()
        
        # Map emotion to speed
        speed = self._get_speed_from_emotion(emotion) if emotion else 1.0
        
        # Build request payload
        payload = {
            "input": text,
            "voice": voice or self.voice,
            "response_format": "wav",
            "speed": speed,
            "language": language or self.language
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.logger.debug(
            f" LemonFox TTS: {len(text)} chars, voice={payload['voice']}, "
            f"language={payload['language']}, speed={speed:.2f}"
        )
        
        try:
            async with self.session.post(
                self.API_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=self.aiohttp.ClientTimeout(total=30)
            ) as response:
                
                # Comment 4: Handle HTTP status codes with custom exceptions
                if response.status == 200:
                    # Success
                    pass
                elif response.status == 429:
                    # Rate limited
                    error_text = await response.text()
                    self.logger.error(f"Rate limited by LemonFox API: {error_text}")
                    raise RateLimitedError(f"LemonFox API rate limit exceeded: {error_text}")
                elif response.status in (500, 502, 503, 504):
                    # Server errors - transient
                    error_text = await response.text()
                    self.logger.warning(f"LemonFox API server error (status {response.status}): {error_text}")
                    raise TransientTTSError(f"LemonFox API server error (status {response.status}): {error_text}")
                elif response.status == 401:
                    # Unauthorized - invalid API key
                    error_text = await response.text()
                    self.logger.error(f"LemonFox API authentication failed: {error_text}")
                    raise NonRetryableTTSError(f"LemonFox API authentication failed: {error_text}")
                elif response.status == 400:
                    # Bad request - invalid parameters
                    error_text = await response.text()
                    self.logger.error(f"LemonFox API bad request: {error_text}")
                    raise NonRetryableTTSError(f"LemonFox API bad request: {error_text}")
                else:
                    # Other errors
                    error_text = await response.text()
                    self.logger.error(f"LemonFox API error (status {response.status}): {error_text}")
                    raise self.aiohttp.ClientError(
                        f"LemonFox API error (status {response.status}): {error_text}"
                    )
                
                wav_audio = await response.read()
                
                self.logger.debug(
                    f" LemonFox TTS: {len(text)} chars → {len(wav_audio)} bytes "
                    f"({len(wav_audio)/1024:.1f} KB)"
                )
                
                return wav_audio
                
        except asyncio.TimeoutError:
            self.logger.error("⏱ LemonFox API timeout (30s)")
            raise
        except self.aiohttp.ClientError as e:
            self.logger.error(f" LemonFox API error: {e}")
            raise
    
    async def stream_synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        emotion: Optional[str] = None,
        **kwargs
    ):
        """
        Stream synthesize (returns complete audio as single chunk).
        
        Note: LemonFox API doesn't support true streaming, so this calls
        synthesize() and yields the complete audio.
        
        Args:
            text: Text to synthesize
            voice: Voice name
            language: Language code
            emotion: Emotion for speed modulation
            **kwargs: Additional parameters
            
        Yields:
            Complete WAV audio bytes
        """
        self.logger.debug(" LemonFox streaming (single chunk)")
        
        audio_bytes = await self.synthesize(
            text=text,
            voice=voice,
            language=language,
            emotion=emotion,
            **kwargs
        )
        
        yield audio_bytes
    
    def get_available_voices(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available voices.
        
        Returns:
            Dict mapping voice IDs to voice metadata
        """
        voices = {
            "sarah": {
                "name": "Sarah",
                "language": "multi",
                "gender": "female",
                "description": "Soft, natural female voice"
            },
            "heart": {
                "name": "Heart",
                "language": "multi",
                "gender": "female",
                "description": "Warm, friendly female voice"
            },
            "bella": {
                "name": "Bella",
                "language": "multi",
                "gender": "female",
                "description": "Elegant, refined female voice"
            },
            "michael": {
                "name": "Michael",
                "language": "multi",
                "gender": "male",
                "description": "Professional, clear male voice"
            },
            "alloy": {
                "name": "Alloy",
                "language": "multi",
                "gender": "neutral",
                "description": "Neutral, balanced voice"
            },
            "nova": {
                "name": "Nova",
                "language": "multi",
                "gender": "female",
                "description": "Energetic female voice"
            },
            "echo": {
                "name": "Echo",
                "language": "multi",
                "gender": "male",
                "description": "Deep, resonant male voice"
            },
            "onyx": {
                "name": "Onyx",
                "language": "multi",
                "gender": "male",
                "description": "Strong, authoritative male voice"
            }
        }
        
        self.logger.info(f" LemonFox TTS: {len(voices)} voices available")
        return voices
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """
        Validate provider configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.api_key or len(self.api_key) < 10:
            return (
                False,
                "Invalid LemonFox API key (set LEMONFOX_API_KEY environment variable)"
            )
        
        return (True, None)
    
    def _get_speed_from_emotion(self, emotion: str) -> float:
        """
        Map emotion to speech speed.
        
        Args:
            emotion: Emotion string
            
        Returns:
            Speed value (0.25-4.0, 1.0 = normal)
        """
        emotion_speed_map = {
            "helpful": 1.0,
            "excited": 1.2,
            "calm": 0.95,
            "happy": 1.1,
            "neutral": 1.0,
            "professional": 1.0,
            "friendly": 1.05,
            "curious": 1.05,
            "empathetic": 0.95,
            "urgent": 1.25,
            "relaxed": 0.9
        }
        
        speed = emotion_speed_map.get(emotion.lower() if emotion else "neutral", 1.0)
        
        # Clamp to valid range
        return max(0.25, min(4.0, speed))
    
    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize input text for TTS synthesis.
        
        Args:
            text: Raw input text
            
        Returns:
            Sanitized text safe for TTS
        """
        import unicodedata
        
        # Normalize Unicode (NFKC - compatibility decomposition + canonical composition)
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters (except whitespace)
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\t\n\r')
        
        # Remove zero-width characters and other invisible characters
        invisible_chars = {'\u200B', '\u200C', '\u200D', '\u200E', '\u200F', '\uFEFF'}
        text = ''.join(char for char in text if char not in invisible_chars)
        
        # Enforce length limits (LemonFox API limit is around 5000 chars, be conservative)
        max_length = 4000
        if len(text) > max_length:
            text = text[:max_length]
            self.logger.warning(f"Text truncated to {max_length} characters")
        
        # Ensure minimum length
        text = text.strip()
        if not text:
            text = "Hello"  # Fallback for empty text
        
        return text
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.debug(" LemonFox session closed")


class MockTTSProvider:
    """
    Mock TTS provider for testing and development without real TTS backends.
    Generates silent or simple tone audio files.
    """
    
    def __init__(self):
        """Initialize mock TTS provider"""
        self.sample_rate = 24000
        print(" Mock TTS provider initialized (silent mode)")
    
    async def synthesize(
        self,
        text: str,
        language: str = "en-US",
        emotion: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Generate silent audio bytes.
        
        Args:
            text: Text to "synthesize" (determines duration based on length)
            language: Language code (ignored)
            emotion: Emotion (ignored)
            
        Returns:
            Silent audio bytes (PCM format, 24kHz, mono, 16-bit)
        """
        # Calculate duration: ~150 words per minute = 2.5 words per second
        # Assume 5 characters per word average
        word_count = max(1, len(text) / 5)
        duration_seconds = word_count / 2.5
        
        # Generate silent audio
        num_samples = int(self.sample_rate * duration_seconds)
        silence = np.zeros(num_samples, dtype=np.int16)
        
        return silence.tobytes()
    
    async def stream_synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        emotion: Optional[str] = None,
        **kwargs
    ):
        """
        Stream synthesize (yields chunks of silent audio).
        
        Args:
            text: Text to "synthesize"
            voice: Voice name (ignored)
            language: Language code (ignored)
            emotion: Emotion (ignored)
            **kwargs: Additional parameters
            
        Yields:
            Silent audio bytes
        """
        # Calculate duration
        word_count = max(1, len(text) / 5)
        duration_seconds = word_count / 2.5
        total_samples = int(self.sample_rate * duration_seconds)
        
        # Yield in 100ms chunks (typical for streaming)
        chunk_size = int(self.sample_rate * 0.1)
        samples_yielded = 0
        
        while samples_yielded < total_samples:
            current_chunk_size = min(chunk_size, total_samples - samples_yielded)
            silence = np.zeros(current_chunk_size, dtype=np.int16)
            yield silence.tobytes()
            samples_yielded += current_chunk_size
            # Simulate real-time latency
            await asyncio.sleep(0.1)


class LeibnizTTS:
    """
    Main TTS class for Leibniz University agent with LemonFox provider only.
    
    This class provides high-quality English speech synthesis using the LemonFox API,
    with comprehensive caching, emotion-based voice modulation, and robust error handling.
    
    Key Features:
    - **LemonFox Provider**: Cost-effective ($2.50/1M chars), fast, reliable, 8 voices, multilingual
    - **Emotion-Based Modulation**: Adjust speed based on context (excited=1.2, calm=0.95, etc.)
    - **Comprehensive Caching**: MD5-based caching with LRU cleanup, 30-day TTL
    - **Two Synthesis Modes**: File-based (`synthesize_to_file()`) and streaming (`stream_tts()`)
    - **Automatic Retry Logic**: Exponential backoff for transient errors
    - **High-Quality Audio**: 24kHz sample rate for natural English speech
    
    LemonFox TTS Features:
    - **8 Voices**: sarah, heart, bella, michael, alloy, nova, echo, onyx
    - **8+ Languages**: en-us, de-de, es-es, fr-fr, it-it, pl-pl, pt-br, nl-nl
    - **Emotion Mapping**: Maps emotions to speed (excited=1.2, calm=0.95, etc.)
    - **Cost**: $2.50 per 1M characters (~90% cheaper than alternatives)
    - **Output**: WAV format, 24kHz mono, high quality
    - **API Endpoint**: https://api.lemonfox.ai/v1/audio/speech
    
    Usage Examples:
        ```python
        # Initialize TTS
        tts = LeibnizTTS()
        
        # Basic synthesis
        result = await tts.synthesize_to_file("Hello world!", "output.wav")
        
        # Emotion-based synthesis
        result = await tts.synthesize_to_file("Welcome!", "welcome.wav", emotion="helpful")
        
        # Streaming synthesis
        audio_bytes = await tts.stream_tts("Streaming text", emotion="excited")
        
        # Get available voices
        voices = await get_available_voices()
        ```
    
    Environment Variables:
    - LEMONFOX_API_KEY: LemonFox API key (required)
    - LEIBNIZ_LEMONFOX_VOICE: Voice name (default: sarah)
    - LEIBNIZ_LEMONFOX_LANGUAGE: Language code (default: en-us)
    - LEIBNIZ_TTS_CACHE_DIR: Cache directory path
    - LEIBNIZ_TTS_CACHE_ENABLED: Enable/disable caching (true/false)
    
    Setup Instructions:
    1. Sign up at https://www.lemonfox.ai
    2. Get API key from dashboard
    3. Set LEMONFOX_API_KEY environment variable
    4. Choose voice: sarah (default), heart, bella, michael, alloy, nova, echo, onyx
    5. Set language: en-us (default), de-de, es-es, fr-fr, it-it, pl-pl, pt-br, nl-nl
    """
    
    # Class-level state for single-flight execution
    _active: bool = False
    _active_lock = asyncio.Lock()
    
    def __init__(self, config: Optional[LeibnizTTSConfig] = None):
        """
        Initialize Leibniz TTS module.
        
        Args:
            config: Optional TTS configuration
        """
        # Load config
        if config is None:
            leibniz_config = get_leibniz_config()
            voice_cfg = leibniz_config.voice
            tech_cfg = leibniz_config.technical
            
            # Comment 2: Read env overrides for TTS configuration
            provider_override = os.getenv('LEIBNIZ_TTS_PROVIDER')
            lemonfox_api_key_override = os.getenv('LEMONFOX_API_KEY')
            lemonfox_voice_override = os.getenv('LEIBNIZ_LEMONFOX_VOICE')
            lemonfox_language_override = os.getenv('LEIBNIZ_LEMONFOX_LANGUAGE')
            cache_enabled_override = os.getenv('LEIBNIZ_TTS_CACHE_ENABLED')
            cache_dir_override = os.getenv('LEIBNIZ_TTS_CACHE_DIR')
            cache_max_size_override = os.getenv('LEIBNIZ_TTS_CACHE_MAX_SIZE')
            timeout_override = os.getenv('LEIBNIZ_TTS_TIMEOUT')
            
            config = LeibnizTTSConfig(
                provider=provider_override or getattr(voice_cfg, 'tts_provider', 'lemonfox'),  # Comment 3: Default to lemonfox
                lemonfox_api_key=lemonfox_api_key_override or getattr(voice_cfg, 'lemonfox_api_key', ''),  # Comment 4: LemonFox API key
                lemonfox_voice=lemonfox_voice_override or getattr(voice_cfg, 'lemonfox_voice', 'sarah'),
                lemonfox_language=lemonfox_language_override or getattr(voice_cfg, 'lemonfox_language', 'en-us'),
                sample_rate=getattr(tech_cfg, 'tts_sample_rate', 24000),
                enable_cache=cache_enabled_override.lower() == 'true' if cache_enabled_override else getattr(tech_cfg, 'tts_cache_enabled', True),
                cache_dir=cache_dir_override or getattr(tech_cfg, 'tts_cache_dir', DEFAULT_CACHE_DIR),
                max_cache_size=int(cache_max_size_override) if cache_max_size_override else getattr(tech_cfg, 'tts_cache_max_size', 500),
                timeout=float(timeout_override) if timeout_override else getattr(tech_cfg, 'tts_timeout', 30.0),
                retry_attempts=getattr(tech_cfg, 'tts_retry_attempts', 3),
                retry_delay=getattr(tech_cfg, 'tts_retry_delay', 1.0)
            )
            
            # Get enable_fallback from tech config
            enable_fallback = True  # Always enabled for LemonFox-only
        else:
            # Config provided, use default fallback setting
            enable_fallback = True
        
        self.config = config
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache
        self.cache = TTSCache(config.cache_dir, config.max_cache_size) if config.enable_cache else None
        
        # Comment 7: Accumulate diagnostics for troubleshooting
        diagnostics = []
        diagnostics.append(f"Provider availability: LemonFox=True (aiohttp required)")
        
        # Check environment variables
        lemonfox_key_set = bool(os.getenv('LEMONFOX_API_KEY') or config.lemonfox_api_key)
        diagnostics.append(f"Environment keys: LEMONFOX_API_KEY={lemonfox_key_set}")
        
        # Initialize providers
        self.google_provider = None  # Removed - only LemonFox now
        self.lemonfox_provider = None
        self.mock_provider = MockTTSProvider() # Always available for fallback
        self.fallback_to_mock = False # State flag for automatic fallback
        
        # Initialize LemonFox (only provider now)
        try:
            api_key = os.getenv('LEMONFOX_API_KEY') or config.lemonfox_api_key
            # Comment 2: Validate LEMONFOX_API_KEY early
            if not api_key:
                diagnostics.append(" LemonFox init failed: LEMONFOX_API_KEY not set")
                print(" LEMONFOX_API_KEY not set - LemonFox TTS unavailable")
            else:
                self.lemonfox_provider = LemonFoxTTSProvider(
                    api_key=api_key,
                    voice=config.lemonfox_voice,
                    language=config.lemonfox_language
                )
                diagnostics.append(f" LemonFox TTS initialized (voice: {config.lemonfox_voice}, language: {config.lemonfox_language})")
        except Exception as e:
            diagnostics.append(f" LemonFox init failed: {e}")
            print(f" Failed to initialize LemonFox TTS: {e}")
        
        # Comment 1: Check for MOCK_TTS mode before raising error
        mock_mode = os.getenv('MOCK_TTS', 'false').lower() == 'true'
        
        # Validate at least one provider is available OR mock mode is enabled
        if not self.lemonfox_provider:
            if mock_mode:
                # Comment 1: Enable mock TTS provider
                self.lemonfox_provider = MockTTSProvider()
                diagnostics.append(" Mock TTS provider enabled (no real audio output)")
                print(" Running in MOCK_TTS mode - no real audio output")
            else:
                # Comment 7: Include diagnostics in error message
                error_msg = (
                    "LemonFox TTS provider not available.\n"
                    "Set LEMONFOX_API_KEY environment variable or enable MOCK_TTS=true for testing.\n\n"
                    "Setup:\n"
                    "  1. Sign up at https://www.lemonfox.ai\n"
                    "  2. Get API key from dashboard\n"
                    "  3. Set LEMONFOX_API_KEY environment variable\n\n"
                    "Diagnostics:\n" + "\n".join(f"  {d}" for d in diagnostics)
                )
                # Log diagnostics at ERROR level
                self.logger.error(f"TTS initialization failed:\n{error_msg}")
                raise ValueError(error_msg)
        
        # Performance metrics
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.provider_failures = 0
        
        # Log successful initialization
        init_msg = f" Leibniz TTS initialized (provider: {config.provider}, fallback: {enable_fallback}, cache: {config.enable_cache}"
        if mock_mode:
            init_msg += ", MOCK_MODE)"
        else:
            init_msg += ")"
        print(init_msg)
    
    def status(self) -> Dict[str, Any]:
        """
        Get TTS provider status
        
        Returns:
            Dict with provider availability and status
        """
        status = {
            "providers_available": [],
            "providers_failed": [],
            "primary_provider": self.config.provider,
            "any_provider_available": False
        }
        
        # Check each provider
        if self.lemonfox_provider:
            status["providers_available"].append("lemonfox")
        else:
            status["providers_failed"].append("lemonfox")
        
        if self.google_provider:
            status["providers_available"].append("google")
        else:
            status["providers_failed"].append("google")
        
        status["any_provider_available"] = len(status["providers_available"]) > 0
        
        return status
    
    async def synthesize_to_file(
        self,
        text: str,
        outfile: Optional[str] = None,
        emotion: str = "helpful",
        cache_name: Optional[str] = None,
        force_regenerate: bool = False,
        output_path: Optional[str] = None  # Comment 3: Alias for README compatibility
    ) -> Dict[str, Any]:
        """
        Synthesize text to audio file.
        
        Args:
            text: Text to synthesize
            outfile: Output file path (None = generate unique temp file)
            emotion: Emotion type (helpful, excited, calm, etc.)
            cache_name: Optional cache identifier for dialogue-level caching
            force_regenerate: Force regeneration even if cached
            output_path: Alias for outfile (for README compatibility)
            
        Returns:
            Dict with success, file, duration, cached, provider
        """
        # Comment 3: Normalize output_path to outfile
        if output_path is not None and outfile is None:
            outfile = output_path
        
        self.total_requests += 1
        start_time = time.time()
        
        # Comment 9: Initialize provider stats on first call
        if not hasattr(self, 'provider_stats'):
            self.provider_stats = {
                'google': {'success': 0, 'failure': 0, 'errors': []},
                'lemonfox': {'success': 0, 'failure': 0, 'errors': []}
            }
        
        # Validate input
        if not text or not text.strip():
            return {'success': False, 'error': 'Empty text'}
        
        text = text.strip()
        
        # Comment 5: Check dialogue caching feature flag (default false - DISABLED)
        dialogue_cache_enabled = os.getenv('LEIBNIZ_ENABLE_DIALOGUE_CACHE', 'false').lower() == 'true'
        
        # Initialize return_raw_bytes flag (fix for variable reference before assignment)
        return_raw_bytes = False
        
        # Determine output file path
        temp_file_path = None
        if cache_name and dialogue_cache_enabled:
            # TARA-style dialogue caching
            dialogue_wav = os.path.join(DIALOGUE_CACHE_DIR, f"{cache_name}.wav")
            dialogue_txt = os.path.join(DIALOGUE_CACHE_DIR, f"{cache_name}.txt")
            
            # Check for dialogue cache hit with content verification
            if os.path.exists(dialogue_wav) and os.path.exists(dialogue_txt) and not force_regenerate:
                try:
                    with open(dialogue_txt, 'r', encoding='utf-8') as f:
                        cached_text = f.read().strip()
                    
                    if cached_text == text:
                        # Content matches - use cached audio
                        if outfile:
                            import shutil
                            shutil.copy2(dialogue_wav, outfile)
                            final_path = outfile
                        else:
                            final_path = dialogue_wav
                        
                        duration = self.get_audio_duration(final_path)
                        self.cache_hits += 1
                        elapsed = time.time() - start_time
                        
                        self.logger.debug(f" Dialogue cache hit ({cache_name}): {duration:.2f}s audio in {elapsed:.3f}s")
                        
                        return {
                            'success': True,
                            'file': final_path,
                            'audio_file': final_path,
                            'duration': duration,
                            'cached': True,
                            'cache_type': 'dialogue',
                            'cache_name': cache_name,
                            'provider': 'cached',
                            'elapsed': elapsed
                        }
                except Exception as e:
                    print(f" Error reading dialogue cache: {e}")
            
            # Cache miss or content changed - synthesize and save to dialogue cache
            outfile = dialogue_wav
        
        elif outfile is None:
            # No cache name and no outfile - return raw audio bytes (no temp file)
            return_raw_bytes = True
            outfile = None  # Will be set after synthesis
        
        # Get TTS settings from config
        leibniz_config = get_leibniz_config()
        tts_settings = leibniz_config.get_tts_settings(emotion)
        
        # Apply emotion modulation
        pitch, speaking_rate = self._apply_emotion_modulation(emotion)
        
        # Determine voice and provider
        if self.fallback_to_mock:
            provider_order = [(self.mock_provider, 'mock', 'default')]
        elif self.config.provider == 'auto':
            # Only LemonFox available
            provider_order = [
                (self.lemonfox_provider, 'lemonfox', self.config.lemonfox_voice)
            ]
        elif self.config.provider == 'lemonfox':
            provider_order = [(self.lemonfox_provider, 'lemonfox', self.config.lemonfox_voice)]
        else:
            # Default to lemonfox if unknown provider
            provider_order = [(self.lemonfox_provider, 'lemonfox', self.config.lemonfox_voice)]
        
        # MD5 CACHE DISABLED - Force real-time synthesis every time
        # if self.cache and not force_regenerate and not cache_name:
        #     for provider_obj, provider_name, voice in provider_order:
        #         if provider_obj is None:
        #             continue
        #
        #         cached_file = self.cache.get_cached_audio(
        #             text, voice, self.config.language_code, provider_name, emotion
        #         )
        #
        #         if cached_file:
        #             # Copy to output file
        #             import shutil
        #             shutil.copy2(cached_file, outfile)
        #
        #             # Get duration
        #             duration = self.get_audio_duration(outfile)
        #
        #             self.cache_hits += 1
        #             elapsed = time.time() - start_time
        #
        #             print(f" Audio archive hit ({provider_name}): {duration:.2f}s audio in {elapsed:.3f}s")
        #             print(f"   Reusing: {os.path.basename(cached_file)}")
        #
        #             return {
        #                 'success': True,
        #                 'file': outfile,
        #                 'audio_file': outfile,  # Backward-compatible alias
        #                 'duration': duration,
        #                 'cached': True,
        #                 'cache_type': 'md5',
        #                 'provider': provider_name,
        #                 'elapsed': elapsed
        #             }
        
        # Cache miss - synthesize
        self.cache_misses += 1
        
        # Try each provider with retry logic
        last_error = None
        for provider_obj, provider_name, voice in provider_order:
            if provider_obj is None:
                continue
            
            for attempt in range(self.config.retry_attempts):
                try:
                    # print(f" Synthesizing with {provider_name} (attempt {attempt + 1}/{self.config.retry_attempts})...")
                    
                    # Synthesize based on provider (only LemonFox now)
                    if provider_name == 'lemonfox':
                        audio_bytes = await asyncio.wait_for(
                            provider_obj.synthesize(
                                text=text,
                                voice=voice,
                                language=self.config.lemonfox_language,
                                emotion=emotion
                            ),
                            timeout=self.config.timeout
                        )
                        # LemonFox outputs at 24kHz
                        effective_sample_rate = 24000
                    
                    # Handle raw bytes return vs file output
                    if return_raw_bytes:
                        # Return raw audio bytes directly (no file written)
                        estimated_duration = len(text) * 0.05
                        elapsed = time.time() - start_time
                        
                        return {
                            'success': True,
                            'audio_bytes': audio_bytes,
                            'sample_rate': effective_sample_rate,
                            'duration': estimated_duration,
                            'cached': False,
                            'cache_name': cache_name,
                            'provider': provider_name,
                            'is_temporary': False,  # No file to cleanup
                            'elapsed': elapsed
                        }
                    else:
                        # Convert to WAV file (offload to thread to prevent event loop blocking)
                        await asyncio.to_thread(
                            self._convert_to_wav,
                            audio_bytes,
                            effective_sample_rate,
                            outfile,
                            provider_name
                        )
                    
                    # Comment 5: Save dialogue cache content if cache_name provided and enabled
                    if cache_name and dialogue_cache_enabled:
                        dialogue_txt = os.path.join(DIALOGUE_CACHE_DIR, f"{cache_name}.txt")
                        # Comment 6: Move dialogue cache writes to background thread (non-blocking)
                        async def _save_dialogue_cache_background():
                            try:
                                await asyncio.to_thread(
                                    lambda: open(dialogue_txt, 'w', encoding='utf-8').write(text)
                                )
                            except Exception as e:
                                print(f" Error saving dialogue cache content: {e}")
                        
                        # Fire-and-forget background task
                        asyncio.create_task(_save_dialogue_cache_background())
                    
                    # BACKGROUND ARCHIVING DISABLED - No file saving
                    # async def _background_archiving():
                    #     """Background task: compute duration and cache audio (non-blocking)"""
                    #     archive_start = time.time()
                    #     try:
                    #         # Add delay to ensure pygame releases file handle
                    #         await asyncio.sleep(0.5)
                    #
                    #         # Get duration (can be slow for large files)
                    #         duration_start = time.time()
                    #         duration_actual = await asyncio.to_thread(self.get_audio_duration, outfile)
                    #         duration_elapsed = time.time() - duration_start
                    #
                    #         # Cache the result in MD5 cache (if no dialogue cache)
                    #         if self.cache and not cache_name:
                    #             cache_start = time.time()
                    #             cached_path = await asyncio.to_thread(
                    #                 self.cache.cache_audio,
                    #                 text, voice, self.config.language_code, provider_name, emotion, outfile
                    #             )
                    #             cache_elapsed = time.time() - cache_start
                    #
                    #             if cached_path:
                    #                 archive_total = time.time() - archive_start
                    #                 print(f" Saved to audio archive: {os.path.basename(cached_path)}")
                    #                 print(f"   ⏱ Archiving latency: {archive_total:.3f}s (duration: {duration_elapsed:.3f}s, cache: {cache_elapsed:.3f}s)")
                    #     except Exception as e:
                    #         print(f" Background archiving error: {e}")
                    #
                    # # Fire-and-forget archiving task (Comment 7)
                    # asyncio.create_task(_background_archiving())
                    
                    # Return immediately with estimated duration (50ms per char heuristic)
                    estimated_duration = len(text) * 0.05
                    elapsed = time.time() - start_time
                    cache_msg = f" (cached as {cache_name})" if cache_name else ""
                    
                    # Show sentence text in logs (truncate if long)
                    # text_preview = text if len(text) <= 60 else f"{text[:57]}..."
# #                     print(f" Synthesized: ~{estimated_duration:.2f}s audio in {elapsed:.3f}s ({provider_name}){cache_msg}")
#                     print(f"    Text: \"{text_preview}\"")
                    
                    # Comment 9: Track successful synthesis
                    self.provider_stats[provider_name]['success'] += 1
                    
                    return {
                        'success': True,
                        'file': outfile,
                        'audio_file': outfile,  # Backward-compatible alias
                        'audio_bytes': audio_bytes if 'audio_bytes' in locals() else None, # Return bytes for in-memory streaming
                        'sample_rate': effective_sample_rate if 'effective_sample_rate' in locals() else 24000,
                        'duration': estimated_duration,  # Estimated, actual computed in background
                        'cached': False,
                        'cache_name': cache_name,
                        'provider': provider_name,
                        'is_temporary': temp_file_path is not None,  # Flag for cleanup
                        'elapsed': elapsed
                    }
                
                except asyncio.TimeoutError as e:
                    last_error = e
                    # Comment 9: Track timeout errors
                    self.provider_stats[provider_name]['errors'].append(('TimeoutError', str(e)))
                    print(f"⏱ Timeout on attempt {attempt + 1} ({provider_name})")
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                
                except Exception as e:
                    last_error = e
                    
                    # Check for 401 Authentication Error from LemonFox
                    if "401" in str(e) or "authentication failed" in str(e).lower():
                        print(f"⚠️ LemonFox Authentication Failed: {e}")
                        print("⚠️ Switching to Mock TTS provider for this session.")
                        self.fallback_to_mock = True
                        
                        # Retry immediately with mock provider
                        try:
                            print("🔄 Retrying with Mock TTS...")
                            audio_bytes = await self.mock_provider.synthesize(
                                text=text,
                                language='en-US',
                                emotion=emotion
                            )
                            effective_sample_rate = self.mock_provider.sample_rate
                            provider_name = 'mock'
                            # Proceed to processing logic below...
                        except Exception as mock_err:
                            print(f"❌ Mock fallback also failed: {mock_err}")
                            continue
                    else:
                        # Standard retry logic
                        # Comment 9: Track exception types
                        self.provider_stats[provider_name]['errors'].append((type(e).__name__, str(e)))
                        print(f" Error on attempt {attempt + 1} ({provider_name}): {e}")
                        if attempt < self.config.retry_attempts - 1:
                            await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                        continue
            
            # Provider failed after all retries
            self.provider_failures += 1
            # Comment 9: Track provider failure
            if provider_name in self.provider_stats:
                self.provider_stats[provider_name]['failure'] += 1
            print(f" {provider_name} failed after {self.config.retry_attempts} attempts")
        
        # If we successfully fell back to mock inside the loop, handle the result
        if self.fallback_to_mock and 'audio_bytes' in locals():
             # Success path for mock fallback
             pass
        elif last_error and not self.fallback_to_mock:
             # All providers failed - cleanup temp file
             if temp_file_path and os.path.exists(temp_file_path):
                 try:
                     os.unlink(temp_file_path)
                 except:
                     pass
             
             # All providers failed
             return {
                 'success': False,
                 'error': f'All providers failed: {last_error}'
             }
    
    async def stream_tts(
        self,
        text: str,
        emotion: str = "helpful",
        play: bool = True,
        streaming_callback: Optional[Callable] = None
    ) -> bytes:
        """
        Stream TTS synthesis with real-time playback.
        
        Args:
            text: Text to synthesize
            emotion: Emotion type
            play: Play audio in real-time
            streaming_callback: Optional callback for each chunk
            
        Returns:
            Concatenated audio bytes
        """
        print(f" Streaming TTS: {text[:50]}...")
        
        # Check for fallback first
        if self.fallback_to_mock:
            provider = self.mock_provider
            provider_name = 'mock'
            voice = 'default'
        # Priority: Only LemonFox available now
        elif self.lemonfox_provider:
            provider = self.lemonfox_provider
            provider_name = 'lemonfox'
            voice = self.config.lemonfox_voice
        else:
            # If no provider, default to mock
            print("⚠️ No TTS provider available, using Mock TTS")
            provider = self.mock_provider
            provider_name = 'mock'
            voice = 'default'
            self.fallback_to_mock = True
        
        # Stream with provider
        audio_chunks = []
        
        if play:
            # Initialize audio stream
            stream = sd.RawOutputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype='int16',
                device=12
            )
            stream.start()
        
        try:
            # Generic streamer for any provider (mock or lemonfox)
            try:
                async for chunk in provider.stream_synthesize(
                    text=text,
                    voice=voice,
                    language=self.config.lemonfox_language,
                    emotion=emotion
                ):
                    audio_chunks.append(chunk)
                    
                    # Play chunk
                    if play and stream:
                        # Convert PCM bytes to numpy array
                        chunk_array = np.frombuffer(chunk, dtype=np.int16)
                        stream.write(chunk_array.tobytes())
                    
                    # Callback
                    if streaming_callback:
                        streaming_callback(chunk)
            
            except Exception as e:
                # Catch auth error during streaming and fallback
                if ("401" in str(e) or "authentication failed" in str(e).lower()) and provider_name == 'lemonfox':
                    print(f"⚠️ LemonFox Streaming Auth Failed: {e}")
                    print("⚠️ Switching to Mock TTS provider for future calls.")
                    self.fallback_to_mock = True
                    
                    # Try to recover stream with mock (restart)
                    print("🔄 Restarting stream with Mock TTS...")
                    async for chunk in self.mock_provider.stream_synthesize(text, voice='default', language='en-US', emotion=emotion):
                        audio_chunks.append(chunk)
                        if play and stream:
                            chunk_array = np.frombuffer(chunk, dtype=np.int16)
                            stream.write(chunk_array.tobytes())
                        if streaming_callback:
                            streaming_callback(chunk)
                else:
                    raise e
        
        finally:
            if play and stream:
                stream.stop()
                stream.close()
        
        # Concatenate chunks
        full_audio = b''.join(audio_chunks)
        print(f" Streaming complete: {len(full_audio)} bytes")
        
        return full_audio
    
    async def synthesize_with_emotion(
        self,
        text: str,
        emotion: str,
        outfile: Optional[str] = None  # Comment 4: Changed from "out.wav" to None
    ) -> Dict[str, Any]:
        """
        Convenience method for emotion-based synthesis.
        
        Args:
            text: Text to synthesize
            emotion: Emotion type
            outfile: Output file (None = generate unique temp file)
            
        Returns:
            Synthesis result
        """
        return await self.synthesize_to_file(text, outfile, emotion)
    
    async def play_audio_file(self, filepath: str) -> float:
        """
        Play audio file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Playback duration in seconds
        """
        # Read audio file
        audio_data, sample_rate = sf.read(filepath)
        
        # Calculate duration for prewarm timing
        duration = len(audio_data) / sample_rate
        
        # Comment 11: Remove VAD flag cleanup - belongs in VAD module
        # Trigger prewarm during TTS playback (fire-and-for-get)
        asyncio.create_task(prewarm_during_tts(duration))
        
        try:
            # Play with working device (device 12: Speakers 2 Realtek HD Audio output with SST)
            sd.play(audio_data, sample_rate, device=12)
            sd.wait()
            
            # Add delay to ensure pygame releases file handle
            await asyncio.sleep(0.2)
            
        finally:
            # VAD state management removed - handle in VAD module
            pass
        
        # Calculate duration
        duration = len(audio_data) / sample_rate
        return duration
    
    def get_audio_duration(self, filepath: str) -> float:
        """
        Get audio duration without loading full file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Duration in seconds
        """
        with wave.open(filepath, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
        return duration
    
    def _convert_to_wav(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        outfile: str,
        provider: str
    ):
        """
        Convert audio bytes to WAV format with robust Windows file handling and atomic operations.
        
        Args:
            audio_bytes: Raw audio bytes
            sample_rate: Sample rate
            outfile: Output file path
            provider: Provider name
        """
        import tempfile
        import shutil
        import time
        
        # Create parent directory if needed
        parent_dir = os.path.dirname(outfile)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        temp_path = None
        final_success = False
        
        # Get logger for this method
        logger = logging.getLogger(__name__)
        
        try:
            # Atomic file write: write to temp file first, then atomic rename
            with tempfile.NamedTemporaryFile(
                mode='wb',
                delete=False,
                dir=parent_dir or None,
                suffix='.wav.tmp',
                prefix='tts_'
            ) as temp_file:
                temp_path = temp_file.name
                
                if provider == 'google':
                    # Google returns LINEAR16 - wrap in WAV
                    with wave.open(temp_file, 'wb') as wf:
                        wf.setnchannels(1)  # Mono
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(sample_rate)
                        wf.writeframes(audio_bytes)
                
                else:  # lemonfox
                    # LemonFox returns WAV bytes directly - write as-is
                    temp_file.write(audio_bytes)
                
                # Ensure all data is written and flushed to disk
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Force write to disk
            
            # Robust atomic rename with multiple fallback strategies
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    # Strategy 1: Try os.replace() (atomic on Windows)
                    os.replace(temp_path, outfile)
                    final_success = True
                    break
                    
                except OSError as e:
                    error_code = getattr(e, 'winerror', None) or e.errno
                    
                    # Strategy 2: If file is locked, wait and retry with exponential backoff
                    if error_code == 32:  # ERROR_SHARING_VIOLATION (file in use)
                        if attempt < max_retries - 1:
                            wait_time = 0.1 * (2 ** attempt)  # 0.1s, 0.2s, 0.4s, 0.8s, 1.6s
                            logger.debug(f"File locked (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.1f}s: {e}")
                            time.sleep(wait_time)
                            continue
                        else:
                            # Final attempt: Force remove target and copy
                            logger.warning(f"Final attempt: force removing locked file {outfile}")
                            try:
                                os.unlink(outfile)  # Remove locked file
                                shutil.copy2(temp_path, outfile)  # Copy temp to target
                                final_success = True
                                logger.debug("Force copy succeeded")
                                break
                            except OSError as force_error:
                                logger.error(f"Force copy also failed: {force_error}")
                                raise force_error
                    
                    # Strategy 3: For other errors, try shutil.copy2 as fallback
                    elif attempt < max_retries - 1:
                        logger.warning(f"os.replace() failed (attempt {attempt + 1}/{max_retries}), trying copy: {e}")
                        try:
                            shutil.copy2(temp_path, outfile)
                            final_success = True
                            break
                        except OSError as copy_error:
                            logger.warning(f"Copy fallback failed: {copy_error}")
                            if attempt < max_retries - 1:
                                time.sleep(0.05)
                                continue
                            else:
                                raise copy_error
                    else:
                        # All strategies failed
                        raise e
        
        finally:
            # Cleanup temp file if it still exists
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass  # Best effort cleanup
            
            if not final_success:
                logger.error(f"Failed to write audio file after {max_retries} attempts: {outfile}")
                # Attempt a safe fallback: write to system temp dir and copy with permission fixups
                try:
                    import tempfile as _temp
                    import shutil as _shutil

                    fallback_tmp = _temp.NamedTemporaryFile(prefix='leibniz_tts_fallback_', suffix='.wav', delete=False)
                    fallback_tmp_path = fallback_tmp.name
                    fallback_tmp.close()

                    logger.info(f"Attempting fallback write to temp file: {fallback_tmp_path}")

                    # If we still have the temp content, try to write audio_bytes to fallback (we don't have audio_bytes here),
                    # so instead try copying any existing temp_path -> fallback_tmp_path (best-effort)
                    if temp_path and os.path.exists(temp_path):
                        try:
                            _shutil.copy2(temp_path, fallback_tmp_path)
                        except Exception:
                            # If copying temp_path failed, try to create an empty wav container to avoid crash
                            try:
                                with wave.open(fallback_tmp_path, 'wb') as _wf:
                                    _wf.setnchannels(1)
                                    _wf.setsampwidth(2)
                                    _wf.setframerate(sample_rate)
                                    _wf.writeframes(b'')
                            except Exception:
                                pass

                    # Ensure target directory exists and try to change permissions on target if exists
                    target_dir = os.path.dirname(outfile)
                    if target_dir and os.path.exists(target_dir):
                        try:
                            # Try to make target writable
                            os.chmod(target_dir, 0o777)
                        except Exception:
                            logger.debug(f"Could not chmod target dir: {target_dir}")

                    # Try copy fallback into final location
                    try:
                        _shutil.copy2(fallback_tmp_path, outfile)
                        logger.info(f"Fallback copy succeeded to {outfile}")
                        final_success = True
                    except PermissionError as perm_err:
                        logger.warning(f"Fallback copy permission error: {perm_err}")
                        # Try to relax permissions on existing outfile then copy
                        try:
                            if os.path.exists(outfile):
                                os.chmod(outfile, 0o666)
                                _shutil.copy2(fallback_tmp_path, outfile)
                                final_success = True
                        except Exception as e2:
                            logger.warning(f"Fallback copy after chmod failed: {e2}")
                    except Exception as e3:
                        logger.warning(f"Fallback copy to final location failed: {e3}")

                    # Cleanup fallback tmp file
                    try:
                        if os.path.exists(fallback_tmp_path):
                            os.unlink(fallback_tmp_path)
                    except Exception:
                        pass

                    if final_success:
                        return
                except Exception as fallback_exc:
                    logger.error(f"Fallback write also failed: {fallback_exc}")

                # If all fallbacks failed, raise an informative error including diagnostics
                raise RuntimeError(f"Could not write audio file: {outfile} (final_attempts={max_retries})")
    
    def _apply_emotion_modulation(self, emotion: str) -> tuple:
        """
        Apply emotion-based voice modulation.
        
        Args:
            emotion: Emotion type
            
        Returns:
            Tuple of (pitch, speaking_rate)
        """
        # Get Leibniz config emotion mappings
        leibniz_config = get_leibniz_config()
        
        # Default values
        pitch = 0.0
        speaking_rate = 1.0
        
        # Emotion-based adjustments
        emotion_map = {
            'excited': (0.15, 1.2),
            'happy': (0.10, 1.1),
            'calm': (0.0, 0.95),
            'helpful': (0.05, 1.0),
            'empathetic': (0.0, 0.95),
            'professional': (0.0, 1.0),
            'friendly': (0.08, 1.05),
            'neutral': (0.0, 1.0)
        }
        
        if emotion in emotion_map:
            pitch, speaking_rate = emotion_map[emotion]
        
        # Clamp values
        pitch = max(-20.0, min(20.0, pitch))
        speaking_rate = max(0.25, min(4.0, speaking_rate))
        
        return pitch, speaking_rate
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache:
            return {'cache_enabled': False}
        
        stats = self.cache.get_stats()
        stats['cache_enabled'] = True
        return stats
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get TTS provider statistics (Comment 9)"""
        if not hasattr(self, 'provider_stats'):
            return {'provider_stats_enabled': False}
        
        return {
            'provider_stats_enabled': True,
            'providers': self.provider_stats,
            'total_requests': self.total_requests,
            'provider_failures': self.provider_failures
        }
    
    def clear_cache(self):
        """Clear TTS cache"""
        if self.cache:
            self.cache.clear_cache()
    
    def _cleanup_temp_file(self, filepath: str) -> None:
        """
        Safely cleanup temporary file.
        
        Args:
            filepath: Path to temporary file to delete
        """
        if filepath and os.path.exists(filepath):
            try:
                os.unlink(filepath)
            except OSError as e:
                # Log but don't raise - cleanup is best-effort
                print(f" Failed to cleanup temp file {filepath}: {e}")
    
    async def warmup(self):
        """Pre-warm TTS providers"""
        test_text = "Hello, this is a test."
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            await self.synthesize_to_file(test_text, temp_path, force_regenerate=True)
            print(" TTS warmup complete")
        except Exception as e:
            print(f" TTS warmup failed: {e}")
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    async def close(self):
        """Close TTS providers and cleanup resources"""
        if self.lemonfox_provider:
            await self.lemonfox_provider.close()
        print(" LeibnizTTS resources closed")


# Global instance
_leibniz_tts: Optional[LeibnizTTS] = None


def get_leibniz_tts() -> LeibnizTTS:
    """Get global Leibniz TTS instance"""
    global _leibniz_tts
    if _leibniz_tts is None:
        _leibniz_tts = LeibnizTTS()
    return _leibniz_tts


# Convenience functions
async def leibniz_synthesize(
    text: str,
    outfile: Optional[str] = None,  # Comment 4: Changed from "out.wav" to None
    emotion: str = "helpful"
) -> Dict[str, Any]:
    """
    Convenience function for synthesis using global instance.
    
    Args:
        text: Text to synthesize
        outfile: Output file (None = generate unique temp file)
        emotion: Emotion type
        
    Returns:
        Synthesis result
    """
    tts = get_leibniz_tts()
    return await tts.synthesize_to_file(text, outfile, emotion)


async def leibniz_speak(
    text: str,
    emotion: str = "helpful",
    cache_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synthesize and play audio with automatic cleanup.
    
    Args:
        text: Text to speak
        emotion: Emotion type
        cache_name: Optional cache name for dialogue caching
        
    Returns:
        Synthesis result
    """
    import tempfile
    import time
    
    tts = get_leibniz_tts()
    temp_path = None
    
    try:
        # Generate unique temp file for playback
        temp_file = tempfile.NamedTemporaryFile(
            prefix=f"leibniz_speak_{int(time.time() * 1000)}_",
            suffix=".wav",
            delete=False
        )
        temp_path = temp_file.name
        temp_file.close()
        
        # Synthesize with optional cache name
        result = await tts.synthesize_to_file(
            text, 
            outfile=temp_path, 
            emotion=emotion,
            cache_name=cache_name
        )
        
        if result['success']:
            # Play
            await tts.play_audio_file(temp_path)
        
        return result
    
    finally:
        # Always cleanup temp file
        if temp_path:
            tts._cleanup_temp_file(temp_path)
    
    return result


async def leibniz_stream_speak(
    text: str,
    emotion: str = "helpful"
) -> bytes:
    """
    Stream synthesis with playback.
    
    Args:
        text: Text to speak
        emotion: Emotion type
        
    Returns:
        Audio bytes
    """
    tts = get_leibniz_tts()
    return await tts.stream_tts(text, emotion, play=True)


async def warmup_leibniz_tts():
    """Pre-warm TTS system"""
    tts = get_leibniz_tts()
    await tts.warmup()


async def cleanup_leibniz_tts():
    """Cleanup TTS resources"""
    global _leibniz_tts
    if _leibniz_tts is not None:
        await _leibniz_tts.close()
        if _leibniz_tts.cache:
            _leibniz_tts.cache._save_cache_index()
    print(" TTS cleanup complete")


async def get_available_voices(provider: str = "lemonfox") -> List[Dict[str, Any]]:
    """
    Get available voices from LemonFox provider.
    
    Args:
        provider: Provider name (only lemonfox supported now)
        
    Returns:
        List of voice info
    """
    tts = get_leibniz_tts()
    voices = []
    
    if provider in ['lemonfox', 'auto'] and tts.lemonfox_provider:
        lemonfox_voices = tts.lemonfox_provider.get_available_voices()
        for voice_id, voice_info in lemonfox_voices.items():
            voices.append({
                'voice_id': voice_id,
                'name': voice_info['name'],
                'language': voice_info['language'],
                'gender': voice_info['gender'],
                'provider': 'lemonfox'
            })
    
    return voices


# Test function
async def test_leibniz_tts():
    """Test Leibniz TTS functionality"""
    print("=" * 60)
    print(" Testing Leibniz TTS Module")
    print("=" * 60)
    
    # Test 1: Basic synthesis
    print("\n Test 1: Basic Synthesis")
    try:
        result = await leibniz_synthesize(
            "Hello! Welcome to Leibniz University.",
            "test_basic.wav",
            "helpful"
        )
        print(f"   Result: {result}")
        assert result['success'], "Synthesis failed"
        print(" Basic synthesis test passed")
    except Exception as e:
        print(f" Basic synthesis test failed: {e}")
    
    # Test 2: Emotion modulation
    print("\n Test 2: Emotion Modulation")
    try:
        emotions = ['excited', 'calm', 'helpful']
        for emotion in emotions:
            result = await leibniz_synthesize(
                f"This is a {emotion} message.",
                f"test_{emotion}.wav",
                emotion
            )
            print(f"   {emotion}: {result['duration']:.2f}s, cached: {result.get('cached', False)}")
        print(" Emotion modulation test passed")
    except Exception as e:
        print(f" Emotion modulation test failed: {e}")
    
    # Test 3: Caching
    print("\n Test 3: Caching")
    try:
        text = "This message should be cached."
        
        # First synthesis (should cache)
        result1 = await leibniz_synthesize(text, "test_cache1.wav")
        print(f"   First: cached={result1.get('cached', False)}, time={result1['elapsed']:.3f}s")
        
        # Second synthesis (should hit cache)
        result2 = await leibniz_synthesize(text, "test_cache2.wav")
        print(f"   Second: cached={result2.get('cached', False)}, time={result2['elapsed']:.3f}s")
        
        assert result2.get('cached', False), "Cache miss on second synthesis"
        print(" Caching test passed")
    except Exception as e:
        print(f" Caching test failed: {e}")
    
    # Test 4: Cache statistics
    print("\n Test 4: Cache Statistics")
    try:
        tts = get_leibniz_tts()
        stats = tts.get_cache_stats()
        print(f"   Stats: {stats}")
        print(" Cache statistics test passed")
    except Exception as e:
        print(f" Cache statistics test failed: {e}")
    
    # Test 5: Streaming (LemonFox only)
    print("\n Test 5: Streaming")
    try:
        tts = get_leibniz_tts()
        if tts.lemonfox_provider:
            audio = await tts.stream_tts(
                "This is a streaming test.",
                emotion="helpful",
                play=False
            )
            print(f"   Streamed: {len(audio)} bytes")
            print(" Streaming test passed")
        else:
            print("⏭ Skipping (LemonFox not available)")
    except Exception as e:
        print(f" Streaming test failed: {e}")
    
    # Test 6: Available voices (LemonFox only)
    print("\n Test 6: Available Voices")
    try:
        voices = await get_available_voices()
        print(f"   Found {len(voices)} voices")
        for voice in voices[:3]:
            print(f"   - {voice.get('name', voice.get('voice_id'))}: {voice['provider']}")
        print(" Available voices test passed")
    except Exception as e:
        print(f" Available voices test failed: {e}")
    
    # Cleanup
    print("\n Cleanup")
    await cleanup_leibniz_tts()
    
    print("\n" + "=" * 60)
    print(" Testing complete")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_leibniz_tts())
