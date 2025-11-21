"""
Leibniz FastRTC Audio Adapters
==============================

This module provides audio adapter classes that bridge FastRTC WebSocket audio streams
with the existing Leibniz audio pipeline. These adapters handle format conversions and
provide thread-safe audio buffering between the browser (via FastRTC) and the VAD/TTS
components.

The FastRTCAudioSource implements the audio_source interface already supported by
leibniz_vad.py, allowing browser audio to be injected into the VAD pipeline.

The FastRTCAudioSink provides a new interface for streaming TTS output to the browser
through FastRTC WebSocket connections.

Audio Format Specifications:
- Input (Browser → VAD): 16000 Hz, float32 normalized (-1.0 to 1.0), mono
- Output (TTS → Browser): 24000 Hz, float32 normalized (-1.0 to 1.0), mono

References:
- docs/fastrtc-integration.md (lines 196-360)
- leibniz_vad.py (audio_source interface)
- leibniz_pro.py (TTS output handling)
"""

import asyncio
import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class FastRTCAudioSource:
    """
    Audio source adapter that receives browser audio from FastRTC and provides it
    to the VAD pipeline. Implements the audio_source interface expected by leibniz_vad.py.

    This adapter bridges the FastRTC WebSocket stream to the existing VAD audio capture
    loop, handling format conversions and providing thread-safe audio buffering.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the FastRTC audio source.

        Args:
            sample_rate: Audio sample rate in Hz (default: 16000 to match VAD requirements)
        """
        self.sample_rate = sample_rate
        self.audio_queue = asyncio.Queue(maxsize=100)  # Thread-safe audio buffering
        self.is_active = False

        logger.info(f"FastRTCAudioSource initialized with sample_rate={sample_rate}Hz")

    async def push_audio_from_fastrtc(self, audio_chunk: np.ndarray) -> None:
        """
        Inject browser audio from FastRTC into the source queue.

        Called by FastRTC handler to provide browser audio to the VAD pipeline.
        Audio should be float32 normalized (-1.0 to 1.0).

        Args:
            audio_chunk: Audio data as float32 numpy array
        """
        try:
            self.audio_queue.put_nowait(audio_chunk)
            self.is_active = True
            logger.debug(f"Pushed audio chunk of {len(audio_chunk)} samples to VAD queue")
        except asyncio.QueueFull:
            logger.warning("FastRTC audio queue full, dropping audio chunk")
        except Exception as e:
            logger.error(f"Error pushing audio to FastRTC source: {e}")

    async def get_frames(self, num_samples: int) -> np.ndarray:
        """
        Get audio frames for VAD processing.

        Interface method called by leibniz_vad.py to pull audio from the source.
        Returns float32 normalized audio as expected by the VAD pipeline.

        Args:
            num_samples: Number of samples to return (typically 800 = 50ms at 16kHz)

        Returns:
            Audio data as float32 numpy array, padded/truncated to num_samples
        """
        try:
            # Wait for audio with short timeout
            audio_chunk = await asyncio.wait_for(
                self.audio_queue.get(),
                timeout=0.1
            )

            # Handle audio chunk size
            if len(audio_chunk) < num_samples:
                # Pad with zeros if too short
                padding = np.zeros(num_samples - len(audio_chunk), dtype=np.float32)
                result = np.concatenate([audio_chunk, padding])
            elif len(audio_chunk) > num_samples:
                # Truncate if too long
                result = audio_chunk[:num_samples]
            else:
                result = audio_chunk

            logger.debug(f"Returned {len(result)} audio samples to VAD")
            return result

        except asyncio.TimeoutError:
            # Return silence when no audio available (expected during pauses)
            logger.debug(f"No audio available, returning {num_samples} samples of silence")
            return np.zeros(num_samples, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error getting frames from FastRTC source: {e}")
            return np.zeros(num_samples, dtype=np.float32)

    def clear(self) -> None:
        """Clear all buffered audio from the queue."""
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
            logger.debug("Cleared FastRTC audio source queue")
        except Exception as e:
            logger.warning(f"Error clearing FastRTC audio source: {e}")


class FastRTCAudioSink:
    """
    Audio sink adapter that receives TTS output and streams it to the browser.

    This adapter provides a new interface for streaming TTS audio through FastRTC
    WebSocket connections, handling format conversions and chunking for optimal
    real-time streaming.
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the FastRTC audio sink.

        Args:
            sample_rate: Audio sample rate in Hz (default: 24000 to match TTS output)
        """
        self.sample_rate = sample_rate
        from queue import Queue
        self.output_queue = Queue()  # Output audio buffering
        self.is_streaming = False

        logger.info(f"FastRTCAudioSink initialized with sample_rate={sample_rate}Hz")

    async def write_audio(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """
        Send TTS audio to the sink for streaming to browser.

        Called by modified speak_friendly() function to provide TTS output.
        Handles format conversion and queues audio for streaming.

        Args:
            audio_data: Audio data as numpy array (float32 or int16)
            sample_rate: Sample rate of the audio data
        """
        try:
            # Convert int16 to float32 if needed
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32767.0
            else:
                audio_float = audio_data.astype(np.float32)

            # Ensure we have float32 format
            if audio_float.dtype != np.float32:
                audio_float = audio_float.astype(np.float32)

            # Queue the audio for streaming (now sync since we use regular Queue)
            self.output_queue.put((sample_rate, audio_float))
            self.is_streaming = True

            logger.debug(f"Queued TTS audio: {len(audio_float)} samples at {sample_rate}Hz")

        except Exception as e:
            logger.error(f"Error writing audio to FastRTC sink: {e}")

    def stream_to_fastrtc(self):
        """
        Sync generator that yields audio chunks for FastRTC streaming.

        Called by FastRTC handler to pull audio for WebSocket transmission to browser.
        Yields audio in FastRTC-compatible chunks (100ms at 24kHz = 2400 samples).

        Yields:
            Tuple of (sample_rate: int, audio_chunk: np.ndarray)
        """
        logger.debug("Starting FastRTC audio streaming")

        try:
            while self.is_streaming or not self.output_queue.empty():
                try:
                    # Get audio from regular queue
                    sample_rate, audio_data = self.output_queue.get_nowait()

                    # Split into FastRTC-compatible chunks (100ms at 24kHz = 2400 samples)
                    chunk_size = 2400
                    total_samples = len(audio_data)

                    for start_idx in range(0, total_samples, chunk_size):
                        end_idx = min(start_idx + chunk_size, total_samples)
                        chunk = audio_data[start_idx:end_idx]

                        # Pad last chunk with zeros if needed
                        if len(chunk) < chunk_size:
                            padding = np.zeros(chunk_size - len(chunk), dtype=np.float32)
                            chunk = np.concatenate([chunk, padding])

                        logger.debug(f"Streaming audio chunk: {len(chunk)} samples at {sample_rate}Hz")
                        yield sample_rate, chunk

                except Exception as e:
                    logger.error(f"Error during FastRTC streaming: {e}")
                    break

        finally:
            self.is_streaming = False
            logger.debug("FastRTC audio streaming completed")

    def stop_streaming(self) -> None:
        """Stop the streaming process."""
        self.is_streaming = False
        logger.debug("FastRTC audio streaming stopped")

    def clear(self) -> None:
        """Clear all buffered audio from the output queue."""
        try:
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
            logger.debug("Cleared FastRTC audio sink queue")
        except Exception as e:
            logger.warning(f"Error clearing FastRTC audio sink: {e}")