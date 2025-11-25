"""
FastRTC handler for Leibniz agent audio integration.

This module provides the LeibnizFastRTCHandler class that serves as the bridge between
FastRTC's WebSocket callbacks and the existing Leibniz conversation pipeline. The handler
manages bidirectional audio flow while maintaining clean separation of concerns - it only
handles audio I/O routing and coordination, leaving all conversation logic unchanged in
leibniz_pro.py.

Architecture Reference: docs/fastrtc-integration.md
"""

import numpy as np
import asyncio
import logging
from typing import Optional
from leibniz_fastrtc_adapters import FastRTCAudioSource, FastRTCAudioSink

logger = logging.getLogger(__name__)


class LeibnizFastRTCHandler:
    """
    FastRTC callback handler that bridges browser audio to the Leibniz conversation pipeline.

    This handler acts as a pure audio I/O wrapper around the existing Leibniz pipeline.
    It receives audio from the browser via FastRTC, routes it to the conversation pipeline
    via adapters, and streams TTS responses back to the browser. The conversation logic
    in leibniz_pro.py runs completely unchanged - this handler only manages audio routing
    and coordination signals.

    The handler implements the async __call__() method expected by FastRTC's ReplyOnPause,
    coordinating the four-step conversation flow:
    1. Receive browser audio → push to source adapter
    2. Signal main loop to process via asyncio.Event
    3. Wait for TTS response generation
    4. Stream response audio back to browser
    """

    def __init__(self):
        """
        Initialize the FastRTC handler with audio adapters and coordination primitives.

        Sets up the audio source (16kHz for VAD input) and sink (24kHz for TTS output),
        along with the event used to signal the main conversation loop.
        """
        # Audio adapters for bidirectional audio flow
        self.source = FastRTCAudioSource(sample_rate=16000)  # Browser input → VAD
        self.sink = FastRTCAudioSink(sample_rate=24000)      # TTS output → Browser

        # Coordination event for signaling main loop when user finishes speaking
        self.user_finished_speaking = asyncio.Event()

        # Event for signaling when user first starts speaking (mic pressed)
        self.user_started_speaking = asyncio.Event()

        # For potential future use (transcript tracking, etc.)
        self.current_transcript = None

        logger.info("Leibniz FastRTC handler initialized")

    def __call__(self, audio: tuple[int, np.ndarray]):
        """
        Main FastRTC callback invoked by ReplyOnPause when user pauses speaking.

        This method implements the complete conversation turn:
        1. Receives audio from browser and pushes it to the source adapter
        2. Signals the main conversation loop to process the audio
        3. Waits for TTS response generation via the sink adapter
        4. Streams the response audio back to the browser

        Args:
            audio: Tuple of (sample_rate: int, audio_array: np.ndarray) from FastRTC
                  The audio_array should be float32 normalized (-1.0 to 1.0)

        Yields:
            Tuples of (sample_rate: int, audio_chunk: np.ndarray) for browser playback
            Audio chunks are float32 normalized, typically 100ms at 24kHz (2400 samples)
        """
        try:
            # Step 1: Receive browser audio and push to source adapter (if provided)
            if audio is not None:
                sample_rate, audio_array = audio
                # Ensure mono audio by flattening
                audio_flat = audio_array.flatten()
                # Push audio directly to source (now synchronous)
                self.source.push_audio_from_fastrtc(audio_flat)
                logger.debug("Pushed browser audio to source buffer")

                # Step 2: Signal main conversation loop to process
                # The main loop will call capture_speech_bidirectional() with this source
                self.user_finished_speaking.set()

                # Signal that user has started speaking (for intro greeting)
                self.user_started_speaking.set()

                logger.debug("Signaled main loop to process audio")
            else:
                logger.debug("No input audio provided - this may be initialization or response streaming")

            # Step 3: Wait for TTS response generation
            # The main loop will call sink.write_audio() when TTS completes
            # No explicit wait needed - the streaming loop below handles it

            # Step 4: Stream response audio back to browser
            # Use the sink's streaming method to get properly chunked audio
            for sample_rate, audio_chunk in self.sink.stream_to_fastrtc():
                yield (sample_rate, audio_chunk)
                logger.debug(f"Yielded audio chunk: {len(audio_chunk)} samples at {sample_rate}Hz")

            logger.debug("Turn completed successfully")

        except Exception as e:
            import traceback
            logger.error(f"FastRTC handler error: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")

            # Yield silence on error to prevent browser from hanging
            yield (24000, np.zeros(1024, dtype=np.float32))

        finally:
            # Clear buffers to prevent audio carryover between conversation turns
            self.source.clear()
            self.sink.clear()
            self.user_finished_speaking.clear()
            logger.debug("Cleared buffers for next turn")