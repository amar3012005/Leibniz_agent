"""
Gemini Live API Session Management for STT/VAD Microservice

Manages Gemini Live API session lifecycle with connection pooling and warmup.
Adapted from LeibnizPersistentSession and OptimizedGeminiConnection.

Reference:
    leibniz_agent/leibniz_vad.py (lines 98-256) - LeibnizPersistentSession
    leibniz_agent/leibniz_stt.py (lines 245-298) - OptimizedGeminiConnection
"""

import asyncio
import os
import time
import logging
from typing import Optional, Any, Dict

from google import genai
from google.genai import types

from leibniz_agent.services.stt_vad.config import VADConfig

logger = logging.getLogger(__name__)


class GeminiLiveSession:
    """
    Manages Gemini Live API session lifecycle with singleton pattern.
    
    Provides persistent session management with automatic expiry, warmup,
    and event loop handling for microservice environment.
    """
    
    # Class-level singleton state
    _client: Optional[genai.Client] = None
    _session: Optional[Any] = None
    _session_context: Optional[Any] = None
    _session_lock: Optional[asyncio.Lock] = None
    _session_loop: Optional[asyncio.AbstractEventLoop] = None
    
    # Session metadata
    _creation_time: float = 0.0
    _last_activity: float = 0.0
    _total_uses: int = 0
    
    # Configuration
    _config: Optional[VADConfig] = None
    
    @classmethod
    async def get_session(cls, config: VADConfig, force_new: bool = False) -> Any:
        """
        Create a NEW Gemini Live session for each capture.
        
        CRITICAL: Each WebSocket connection needs its own session because
        session.receive() can only have one active iterator at a time.
        
        Args:
            config: VAD configuration instance
            force_new: Ignored (always creates new session)
            
        Returns:
            NEW Gemini Live session ready for audio streaming
            
        Raises:
            ValueError: If GEMINI_API_KEY not set
        """
        # Ensure client exists
        if cls._client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable not set. "
                    "Set it in .env.leibniz or docker-compose environment."
                )
            cls._client = genai.Client(api_key=api_key)
            logger.info("✅ Gemini client initialized")
        
        # ALWAYS create new session for each capture (no pooling)
        # This is required because session.receive() is single-use
        logger.info("🌐 Creating new Gemini Live session")
        
        start_time = time.time()
        
        # Session configuration for English transcription
        session_config = {
            "response_modalities": ["TEXT"],  # TEXT only for transcription
            "input_audio_transcription": {}   # Enable user speech transcription
        }
        
        # Add language config
        if config.language_code:
            session_config["speech_config"] = {
                "language_code": config.language_code  # en-US
            }
        
        # Connect to Gemini Live (returns context manager) - STORE CONTEXT!
        cls._session_context = cls._client.aio.live.connect(
            model=config.model_name,
            config=session_config
        )
        cls._session = await cls._session_context.__aenter__()
        cls._session_loop = asyncio.get_running_loop()
        
        # Track stats
        cls._creation_time = time.time()
        cls._last_activity = time.time()
        cls._total_uses += 1
        cls._config = config
        
        connection_time = time.time() - start_time
        logger.info(f"✅ Gemini session ready in {connection_time:.3f}s")
        
        return cls._session
    
    @classmethod
    async def close_session(cls):
        """
        Gracefully close current session and cleanup resources.
        
        Note: With per-capture sessions, this is mainly for cleanup on shutdown.
        """
        # Cleanup singleton state
        if cls._session_context:
            try:
                await cls._session_context.__aexit__(None, None, None)
                logger.info("🔒 Gemini session closed gracefully")
            except Exception as e:
                logger.error(f"Error closing session: {e}")
            finally:
                cls._session = None
                cls._session_context = None
                cls._session_loop = None
                cls._creation_time = 0.0
                cls._last_activity = 0.0
    
    @classmethod
    def get_session_stats(cls) -> Dict[str, Any]:
        """
        Get session statistics for monitoring.
        
        Returns:
            dict: {
                "session_exists": bool,
                "session_age": float,
                "last_used_ago": float,
                "total_uses": int,
                "creation_time": float
            }
        """
        now = time.time()
        
        return {
            "session_exists": cls._session is not None,
            "session_age": now - cls._creation_time if cls._creation_time > 0 else 0.0,
            "last_used_ago": now - cls._last_activity if cls._last_activity > 0 else 0.0,
            "total_uses": cls._total_uses,
            "creation_time": cls._creation_time,
            "event_loop_bound": cls._session_loop is not None
        }
    
    @classmethod
    async def warmup_session(cls, config: VADConfig):
        """
        Pre-warm session for faster first request.
        
        Creates session in background without blocking.
        """
        try:
            logger.info("🔥 Triggering session warmup")
            await cls.get_session(config)
            logger.info("✅ Session warmup complete")
        except Exception as e:
            logger.error(f"❌ Session warmup failed: {e}")
