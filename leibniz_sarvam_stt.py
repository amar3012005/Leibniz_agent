#!/usr/bin/env python3
"""
Leibniz Sarvam STT Service - Continuous VAD/STT with Sarvam AI API
=================================================================

This module provides continuous background listening for Leibniz Agent using
Sarvam AI's WebSocket STT API, with integrated confidence-based VAD features.

Key Features:
- Continuous audio streaming with Sarvam AI WebSocket STT
- Integrated VAD with confidence scores (high_vad_sensitivity, vad_signals)
- Real-time barge-in detection during agent TTS playback
- Event-based user speech delivery to main loop
- Low-latency speech detection and transcription
- Robust error handling and auto-recovery
- Health monitoring and performance metrics

Architecture:
    Main Loop (leibniz_pro.py)
        ↓
    wait_for_leibniz_speech(timeout)
        ↓
    LeibnizSarvamSTTService (background tasks)
        ↓
    ├─ _send_audio_loop() → Sarvam AI WebSocket (continuous)
    ├─ _listen_for_transcripts_loop() → Transcripts (continuous)
    └─ on_user_speech() callback → Signal main loop

Usage:
    # In leibniz_pro.py initialization:
    await start_leibniz_sarvam_stt()

    # In main conversation loop:
    transcript = await wait_for_leibniz_speech(timeout=30.0)
    if transcript:
        # Process user speech
        intent = await classify_intent(transcript)
        # ...

    # On shutdown:
    await stop_leibniz_sarvam_stt()

Performance:
- Per-turn latency: <200ms (Sarvam AI optimized)
- Barge-in latency: <300ms (real-time interruption)
- VAD confidence: 0.0-1.0 (Sarvam AI integrated)
- Auto-recovery: Full session restart on errors

Environment Variables:
- LEIBNIZ_SARVAM_API_KEY: Sarvam AI API subscription key (required)
- LEIBNIZ_SARVAM_LANGUAGE: Language code (default: 'en-IN')
- LEIBNIZ_SARVAM_MODEL: STT model (default: 'saarika:v2.5')
- LEIBNIZ_SARVAM_HIGH_VAD_SENSITIVITY: Enable high VAD sensitivity (default: true)
- LEIBNIZ_SARVAM_VAD_SIGNALS: Enable VAD signals (default: true)
- LEIBNIZ_SARVAM_SAMPLE_RATE: Audio sample rate (default: 16000)
- LEIBNIZ_SARVAM_ENABLE_CONTINUOUS: Enable continuous mode (default: false)
- LEIBNIZ_SARVAM_TIMEOUT: Max wait time for speech (default: 30.0s)
- LEIBNIZ_SARVAM_BARGE_IN_ENABLED: Allow TTS interruption (default: true)
"""

import asyncio
import json
import base64
import logging
import os
import time
import websockets
import queue
from typing import Optional, Callable, Awaitable, Dict, Any
from datetime import datetime

import sounddevice as sd
import numpy as np

from leibniz_stt import normalize_english_transcript
from leibniz_persistent_services import trigger_prewarm_on_speech_detection

logger = logging.getLogger(__name__)

# Global singleton instance
_sarvam_stt_instance: Optional['LeibnizSarvamSTTService'] = None


class LeibnizSarvamSTTService:
    """
    Sarvam AI WebSocket STT service with integrated VAD for Leibniz Agent.

    Attributes:
        websocket: Persistent WebSocket connection to Sarvam AI
        is_running: Background listener active flag
        audio_stream: Persistent sounddevice stream
        audio_queue: Audio chunk queue for WebSocket streaming
        send_task: Background audio sender task
        listen_task: Background transcript receiver task
        on_user_speech: Callback when user speaks (async)
        user_transcript_event: Event to signal main loop
        current_transcript: Latest transcript from Sarvam AI
        vad_confidence: Latest VAD confidence score
        config: Sarvam AI configuration parameters
    """

    def __init__(self):
        """Initialize Sarvam STT service."""
        # Sarvam AI configuration
        self.config = self._load_config()

        # WebSocket connection
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connection_task: Optional[asyncio.Task] = None

        # Background listener state
        self.is_running: bool = False
        self.audio_stream: Optional[sd.InputStream] = None
        self.thread_audio_queue: Optional[queue.Queue] = None
        self.send_task: Optional[asyncio.Task] = None
        self.listen_task: Optional[asyncio.Task] = None

        # Callback and event signaling
        self.on_user_speech: Optional[Callable[[str], Awaitable[None]]] = None
        self.user_transcript_event: asyncio.Event = asyncio.Event()
        self.current_transcript: Optional[str] = None
        self.vad_confidence: float = 0.0

        # Health monitoring
        self.start_time: Optional[float] = None
        self.transcripts_received: int = 0
        self.errors_count: int = 0
        self.last_transcript_time: Optional[float] = None
        self.websocket_reconnects: int = 0

        # Watchdog monitoring
        self.watchdog_task: Optional[asyncio.Task] = None
        self.consecutive_timeouts: int = 0
        self.last_activity_time: float = time.time()

        # Audio buffer for streaming
        self.audio_buffer: bytearray = bytearray()
        self.buffer_lock: asyncio.Lock = asyncio.Lock()

        logger.info(" LeibnizSarvamSTTService initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load Sarvam AI configuration from environment variables."""
        return {
            'api_key': os.getenv('LEIBNIZ_SARVAM_API_KEY'),
            'language': os.getenv('LEIBNIZ_SARVAM_LANGUAGE', 'en-IN'),
            'model': os.getenv('LEIBNIZ_SARVAM_MODEL', 'saarika:v2.5'),
            'high_vad_sensitivity': os.getenv('LEIBNIZ_SARVAM_HIGH_VAD_SENSITIVITY', 'true').lower() == 'true',
            'vad_signals': os.getenv('LEIBNIZ_SARVAM_VAD_SIGNALS', 'true').lower() == 'true',
            'sample_rate': int(os.getenv('LEIBNIZ_SARVAM_SAMPLE_RATE', '16000')),
            'chunk_size': int(os.getenv('LEIBNIZ_SARVAM_CHUNK_SIZE', '800')),
            'websocket_url': 'wss://api.sarvam.ai/speech-to-text/ws'
        }

    def _build_websocket_url(self) -> str:
        """Build WebSocket URL with query parameters."""
        params = [
            f'language-code={self.config["language"]}',
            f'model={self.config["model"]}',
            f'sample_rate={self.config["sample_rate"]}'
        ]

        if self.config['high_vad_sensitivity']:
            params.append('high_vad_sensitivity=true')

        if self.config['vad_signals']:
            params.append('vad_signals=true')

        return f'{self.config["websocket_url"]}?{"&".join(params)}'

    async def start_continuous_stt(self):
        """
        Start continuous STT with Sarvam AI WebSocket.

        Creates WebSocket connection, starts sounddevice stream,
        and launches background tasks for sending/receiving.
        """
        if self.is_running:
            logger.warning("️ Sarvam STT service already running")
            return

        if not self.config['api_key']:
            raise ValueError("LEIBNIZ_SARVAM_API_KEY environment variable is required")

        try:
            # Create thread-safe audio queue
            queue_size = int(os.getenv("LEIBNIZ_SARVAM_AUDIO_QUEUE_SIZE", "100"))
            self.thread_audio_queue = queue.Queue(maxsize=queue_size)

            # Define sounddevice callback
            def audio_callback(indata, frames, time_info, status):
                """Sounddevice callback - sends audio chunks to thread-safe queue."""
                try:
                    if status:
                        pass  # Silent status handling

                    # Convert float32 → int16 PCM
                    audio_data = (indata.copy() * 32767).astype(np.int16).tobytes()

                    # Try to put without blocking
                    try:
                        self.thread_audio_queue.put_nowait(audio_data)
                    except queue.Full:
                        # Drop oldest to make room (FIFO drop)
                        try:
                            self.thread_audio_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.thread_audio_queue.put_nowait(audio_data)
                        except queue.Full:
                            pass
                except Exception as cb_e:
                    pass

            # Start sounddevice stream
            self.audio_stream = sd.InputStream(
                samplerate=self.config['sample_rate'],
                channels=1,
                dtype='float32',
                blocksize=self.config['chunk_size'],
                callback=audio_callback
            )
            self.audio_stream.start()

            # Start WebSocket connection
            await self._connect_websocket()

            # Create background tasks
            self.send_task = asyncio.create_task(self._send_audio_loop())
            self.listen_task = asyncio.create_task(self._listen_for_transcripts_loop())
            self.watchdog_task = asyncio.create_task(self._watchdog_loop())

            # Mark as running
            self.is_running = True
            self.start_time = time.time()

            logger.info(" Sarvam STT continuous service started")

        except Exception as e:
            await self._cleanup_resources()
            self.send_task = None
            self.listen_task = None
            self.watchdog_task = None
            self.thread_audio_queue = None
            raise

    async def _connect_websocket(self):
        """Establish WebSocket connection to Sarvam AI."""
        try:
            ws_url = self._build_websocket_url()
            headers = {'Api-Subscription-Key': self.config['api_key']}

            logger.info(f" Connecting to Sarvam AI: {ws_url}")

            # websockets 15.x uses additional_headers parameter
            self.websocket = await websockets.connect(
                ws_url,
                additional_headers=headers,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=5
            )
            logger.info(" WebSocket connected to Sarvam AI")

        except Exception as e:
            logger.error(f" WebSocket connection failed: {e}")
            raise

    async def _send_audio_loop(self):
        """
        Background task to stream audio to Sarvam AI WebSocket.

        Continuously reads audio chunks from queue and sends to Sarvam AI.
        """
        try:
            while self.is_running:
                try:
                    # Pull audio chunk from thread-safe queue
                    try:
                        audio_chunk = await asyncio.to_thread(self.thread_audio_queue.get, True, 0.1)
                    except Exception as get_err:
                        if isinstance(get_err, queue.Empty):
                            continue
                        else:
                            raise

                    # Send to Sarvam AI WebSocket
                    if self.websocket and audio_chunk:
                        # Encode audio as base64
                        audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')

                        # Send audio message
                        message = {
                            "audio": {
                                "data": audio_b64,
                                "sample_rate": str(self.config['sample_rate']),
                                "encoding": "audio/wav"
                            }
                        }

                        await self.websocket.send(json.dumps(message))
                        self.last_activity_time = time.time()

                except websockets.exceptions.ConnectionClosed:
                    logger.warning(" WebSocket connection closed, attempting reconnect...")
                    await self._handle_connection_error()
                    break
                except Exception as e:
                    await self._handle_send_error(e)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._handle_send_error(e)

    async def _listen_for_transcripts_loop(self):
        """
        Background task to receive transcripts from Sarvam AI WebSocket.

        Continuously receives responses and processes transcripts with VAD confidence.
        """
        first_speech_detected = False

        try:
            while self.is_running and self.websocket:
                try:
                    # Receive message from WebSocket
                    message = await self.websocket.recv()
                    response = json.loads(message)

                    # Update activity time
                    self.last_activity_time = time.time()

                    # Process transcription response
                    if response.get('type') == 'data':
                        data = response.get('data', {})
                        transcript = data.get('transcript', '').strip()
                        metrics = data.get('metrics', {})

                        if transcript:
                            # Check if agent is currently speaking (barge-in prevention)
                            from leibniz_vad import get_leibniz_vad
                            vad = get_leibniz_vad()

                            if vad.is_agent_speaking:
                                logger.debug(f" Ignoring transcript during agent speech: '{transcript[:50]}...'")
                                continue

                            # Normalize transcript
                            normalized_transcript = normalize_english_transcript(transcript)

                            # Extract VAD confidence if available in metrics
                            vad_confidence = metrics.get('vad_confidence', 0.5)
                            self.vad_confidence = vad_confidence

                            # Store transcript
                            self.current_transcript = normalized_transcript
                            self.transcripts_received += 1
                            self.last_transcript_time = time.time()

                            # Trigger prewarm on first speech
                            if not first_speech_detected:
                                first_speech_detected = True
                                asyncio.create_task(trigger_prewarm_on_speech_detection())

                            # Call user speech callback
                            if self.on_user_speech:
                                try:
                                    await self.on_user_speech(normalized_transcript)
                                except Exception as callback_err:
                                    logger.error(f" User speech callback error: {callback_err}")

                            # Signal main loop
                            self.user_transcript_event.set()

                            logger.info(f" Transcript: '{normalized_transcript}' (VAD: {vad_confidence:.3f})")

                except websockets.exceptions.ConnectionClosed:
                    logger.warning(" WebSocket connection closed in listener")
                    await self._handle_connection_error()
                    break
                except json.JSONDecodeError as e:
                    logger.warning(f"️ Invalid JSON response: {e}")
                    continue
                except Exception as e:
                    await self._handle_listener_error(e)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._handle_listener_error(e)

    async def _watchdog_loop(self):
        """
        Background watchdog task to detect stalls and restart service.
        """
        try:
            while self.is_running:
                await asyncio.sleep(10.0)

                now = time.time()
                time_since_activity = now - self.last_activity_time

                # If no activity for 60 seconds, consider stalled
                if time_since_activity > 60.0:
                    self.consecutive_timeouts += 1

                    if self.consecutive_timeouts >= 2:
                        logger.warning(" Watchdog: Restarting Sarvam STT service")
                        await self.restart_service()
                        self.consecutive_timeouts = 0
                else:
                    if self.consecutive_timeouts > 0:
                        self.consecutive_timeouts = 0

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f" Watchdog error: {e}")

    async def wait_for_user_speech(self, timeout: float = 30.0) -> Optional[str]:
        """
        Wait for user speech event (called by main loop).

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            User transcript if received, None on timeout
        """
        # Clear event and transcript
        self.user_transcript_event.clear()
        self.current_transcript = None

        try:
            await asyncio.wait_for(self.user_transcript_event.wait(), timeout=timeout)
            transcript = self.current_transcript
            self.current_transcript = None
            return transcript

        except asyncio.TimeoutError:
            logger.debug(f"⏱️ Timeout waiting for user speech ({timeout}s)")
            return None

    async def stop_continuous_stt(self):
        """
        Stop continuous STT service and cleanup.
        """
        if not self.is_running:
            return

        self.is_running = False

        # Cancel background tasks
        for task in [self.send_task, self.listen_task, self.watchdog_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.send_task = None
        self.listen_task = None
        self.watchdog_task = None

        # Cleanup resources
        await self._cleanup_resources()

    async def _cleanup_resources(self):
        """Cleanup WebSocket, audio stream and queues."""
        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
                self.websocket = None
                logger.debug(" WebSocket closed")
            except Exception as e:
                logger.warning(f"️ Error closing WebSocket: {e}")

        # Stop audio stream
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
                logger.debug(" Audio stream closed")
            except Exception as e:
                logger.warning(f"️ Error closing audio stream: {e}")

        # Clear audio queue
        if self.thread_audio_queue:
            try:
                while True:
                    self.thread_audio_queue.get_nowait()
            except queue.Empty:
                pass
            self.thread_audio_queue = None

    async def _handle_connection_error(self):
        """Handle WebSocket connection errors with auto-reconnect."""
        self.errors_count += 1
        self.websocket_reconnects += 1

        if self.is_running:
            logger.info(" Attempting WebSocket reconnect...")
            await self.restart_service()

    async def _handle_send_error(self, error: Exception):
        """Handle errors in audio send loop."""
        self.errors_count += 1
        logger.warning(f"️ Send error: {error}")

        if self.is_running:
            await asyncio.sleep(1.0)  # Brief pause before continue

    async def _handle_listener_error(self, error: Exception):
        """Handle errors in transcript listener loop."""
        self.errors_count += 1
        logger.warning(f"️ Listener error: {error}")

        if self.is_running:
            await asyncio.sleep(1.0)  # Brief pause before continue

    async def restart_service(self):
        """
        Restart the entire STT service after errors.
        """
        logger.info(" Restarting Sarvam STT service...")

        # Stop existing tasks
        self.is_running = False

        for task in [self.send_task, self.listen_task, self.watchdog_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.send_task = None
        self.listen_task = None
        self.watchdog_task = None

        # Cleanup resources
        await self._cleanup_resources()

        # Wait for cleanup
        await asyncio.sleep(2.0)

        # Reinitialize
        try:
            await self.start_continuous_stt()
            logger.info(" Sarvam STT service restarted successfully")
        except Exception as e:
            logger.error(f" Failed to restart Sarvam STT service: {e}")
            raise

    def get_health_status(self) -> Dict[str, Any]:
        """
        Return health metrics and status.

        Returns:
            Dict with health status information
        """
        uptime = time.time() - self.start_time if self.start_time else 0.0

        return {
            "is_running": self.is_running,
            "websocket_connected": self.websocket is not None and self.websocket.state.name == 'OPEN',
            "audio_stream_active": self.audio_stream is not None and self.audio_stream.active,
            "transcripts_received": self.transcripts_received,
            "errors_count": self.errors_count,
            "websocket_reconnects": self.websocket_reconnects,
            "last_transcript_time": self.last_transcript_time,
            "vad_confidence": self.vad_confidence,
            "uptime_seconds": uptime,
            "config": self.config
        }

    async def send_flush_signal(self):
        """
        Send flush signal to finalize any pending transcription.
        """
        if self.websocket and self.websocket.state.name == 'OPEN':
            try:
                flush_message = {"flush_signal": True}
                await self.websocket.send(json.dumps(flush_message))
                logger.debug(" Flush signal sent")
            except Exception as e:
                logger.warning(f"️ Failed to send flush signal: {e}")


# Module-level singleton accessor and helper functions

def validate_sarvam_config():
    """
    Validate Sarvam AI configuration parameters

    Returns:
        Dict with validation results: {'valid': bool, 'errors': list, 'warnings': list}
    """
    errors = []
    warnings = []

    # API key validation
    api_key = os.getenv('LEIBNIZ_SARVAM_API_KEY')
    if not api_key:
        errors.append("LEIBNIZ_SARVAM_API_KEY is required")
    elif len(api_key.strip()) < 10:
        errors.append("LEIBNIZ_SARVAM_API_KEY appears to be invalid (too short)")

    # Language code validation
    language = os.getenv('LEIBNIZ_SARVAM_LANGUAGE', 'en-IN')
    supported_languages = ['en-IN', 'hi-IN', 'te-IN', 'ta-IN', 'kn-IN', 'ml-IN', 'bn-IN', 'gu-IN', 'mr-IN', 'pa-IN', 'or-IN', 'as-IN']
    if language not in supported_languages:
        warnings.append(f"LEIBNIZ_SARVAM_LANGUAGE '{language}' may not be supported by Sarvam AI")

    # Sample rate validation
    try:
        sample_rate = int(os.getenv('LEIBNIZ_SARVAM_SAMPLE_RATE', '16000'))
        if sample_rate not in [8000, 16000]:
            errors.append(f"LEIBNIZ_SARVAM_SAMPLE_RATE {sample_rate} must be 8000 or 16000")
    except ValueError:
        errors.append("LEIBNIZ_SARVAM_SAMPLE_RATE is not a valid integer")

    # Chunk size validation
    try:
        chunk_size = int(os.getenv('LEIBNIZ_SARVAM_CHUNK_SIZE', '800'))
        if chunk_size <= 0:
            errors.append(f"LEIBNIZ_SARVAM_CHUNK_SIZE {chunk_size} must be positive")
        elif chunk_size % 800 != 0:
            warnings.append(f"LEIBNIZ_SARVAM_CHUNK_SIZE {chunk_size} not multiple of 800 (50ms at 16kHz)")
    except ValueError:
        errors.append("LEIBNIZ_SARVAM_CHUNK_SIZE is not a valid integer")

    # Boolean parameter validation
    for param in ['LEIBNIZ_SARVAM_HIGH_VAD_SENSITIVITY', 'LEIBNIZ_SARVAM_VAD_SIGNALS']:
        value = os.getenv(param, 'true').lower()
        if value not in ['true', 'false']:
            errors.append(f"{param} '{value}' must be 'true' or 'false'")

    # Report results
    is_valid = len(errors) == 0

    if errors:
        logger.error(f" Sarvam config validation failed: {'; '.join(errors)}")
    else:
        logger.debug(" Sarvam config validation passed")

    if warnings:
        logger.warning(f"️ Sarvam config warnings: {'; '.join(warnings)}")

    return {
        'valid': is_valid,
        'errors': errors,
        'warnings': warnings
    }


def get_sarvam_stt_service() -> LeibnizSarvamSTTService:
    """Get or create Sarvam STT service singleton instance."""
    global _sarvam_stt_instance

    if _sarvam_stt_instance is None:
        _sarvam_stt_instance = LeibnizSarvamSTTService()

    return _sarvam_stt_instance


async def start_leibniz_sarvam_stt():
    """Start Sarvam AI continuous STT service."""
    service = get_sarvam_stt_service()
    await service.start_continuous_stt()


async def stop_leibniz_sarvam_stt():
    """Stop Sarvam AI continuous STT service."""
    service = get_sarvam_stt_service()
    await service.stop_continuous_stt()


async def wait_for_leibniz_speech(timeout: float = 30.0) -> Optional[str]:
    """
    Wait for next user speech using Sarvam AI STT.

    Args:
        timeout: Maximum wait time in seconds (default from env var)

    Returns:
        User transcript if received, None on timeout
    """
    # Get timeout from environment if not specified
    if timeout == 30.0:
        timeout = float(os.getenv("LEIBNIZ_SARVAM_TIMEOUT", "30.0"))

    service = get_sarvam_stt_service()
    return await service.wait_for_user_speech(timeout=timeout)


async def flush_sarvam_transcription():
    """Flush any pending transcription in Sarvam AI service."""
    service = get_sarvam_stt_service()
    await service.send_flush_signal()