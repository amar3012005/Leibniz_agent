#!/usr/bin/env python3
"""
Leibniz FastRTC WebRTC Wrapper
===============================

This module provides a complete FastRTC integration wrapper for the Leibniz
customer service pipeline. It bridges browser-based WebRTC audio streams with
the existing VAD/STT/TTS pipeline, enabling full browser-based conversations.

Key Features:
- Browser audio input via FastRTC (replaces host microphone)
- Browser audio output via FastRTC (replaces host speakers)
- Seamless integration with existing conversation pipeline
- RAG streaming TTS support
- Barge-in detection support
- Session management

Architecture:
- FastRTCStreamHandler: Manages bidirectional audio flow
- FastRTCAudioSource: Provides browser audio to VAD pipeline
- FastRTCAudioSink: Streams TTS output to browser
- Conversation session wrapper: Integrates with run_conversation_session()

Usage:
    from leibniz_fastrtc_wrapper import create_fastrtc_app
    
    app = create_fastrtc_app()
    app.launch(server_name="0.0.0.0", server_port=7866, share=True)
"""

import asyncio
import logging
import numpy as np
import time
import os

# Optimize TTS timeout to prevent blocking (5 seconds for greeting)
# This ensures if network is slow, we don't hang the WebRTC stream indefinitely
os.environ["LEIBNIZ_TTS_TIMEOUT"] = "5.0"

from typing import Optional, Dict, Any
from fastrtc import Stream, StreamHandler, AsyncStreamHandler, ReplyOnPause, wait_for_item
from fastrtc.utils import wait_for_item as wait_for_item_utils
import gradio as gr

# Import Leibniz components
from leibniz_state_machine import get_state_machine, AudioState
from leibniz_fastrtc_adapters import FastRTCAudioSource, FastRTCAudioSink
from leibniz_pro import (
    initialize_leibniz_services,
    run_conversation_session,
    get_leibniz_vad,
    set_leibniz_agent_speaking,
    clear_tts_queue,
    start_tts_consumer,
    stream_rag_to_tts,
    finalize_tts_streaming,
    _tts_streaming_queue,
    _streaming_active,
    _cancel_streaming,
    update_agent_speech_end_time,
    get_agent_speech_end_time
)

logger = logging.getLogger(__name__)


class LeibnizFastRTCStreamHandler(AsyncStreamHandler):
    """
    FastRTC AsyncStreamHandler that bridges browser WebRTC audio with the Leibniz
    conversation pipeline. Handles bidirectional audio flow:
    - Browser → VAD/STT (via FastRTCAudioSource)
    - TTS → Browser (via FastRTCAudioSink)
    
    Uses AsyncStreamHandler to properly handle async operations.
    """
    
    def __init__(self):
        super().__init__()
        
        # Audio adapters
        self.audio_source = FastRTCAudioSource(sample_rate=16000)
        self.audio_sink = FastRTCAudioSink(sample_rate=24000)
        
        # Conversation state
        self.conversation_task: Optional[asyncio.Task] = None
        self.is_conversation_active = False
        self.session_id = f"fastrtc_{int(time.time())}"

        # Reset streaming state for new session
        self._audio_emission_active = False
        self._audio_emission_start_time = 0
        self._audio_emission_end_time = 0
        
        # TTS streaming state
        self.tts_consumer_task: Optional[asyncio.Task] = None
        self._tts_streaming_enabled = True
        self._last_consumer_restart = 0
        
        # Pre-buffered audio chunks (matching test_tts_webrtc_stream.py pattern)
        # This allows emit() to return chunks immediately without processing
        self._audio_chunks: list = []
        self._chunk_index = 0
        
        logger.info("LeibnizFastRTCStreamHandler initialized")
    
    async def start_up(self):
        """Called when WebRTC stream starts - initialize conversation session."""
        logger.info("🚀 FastRTC stream started - initializing conversation session")
        
        # Start conversation session in background
        if not self.is_conversation_active:
            self.is_conversation_active = True
            
            # Get the running event loop (AsyncStreamHandler provides one)
            loop = asyncio.get_running_loop()
            
            # Start conversation session as background task
            self.conversation_task = loop.create_task(
                self._run_conversation_session()
            )
            
            # IMPORTANT: Pre-start TTS consumer so emit() can start pulling audio immediately
            # This matches test_tts_webrtc_stream.py pattern where streaming starts in start_up()
            if self._tts_streaming_enabled:
                try:
                    self.tts_consumer_task = loop.create_task(
                        self._consume_tts_for_fastrtc()
                    )
                    logger.info("🎵 Pre-started TTS consumer for FastRTC (ready to emit audio)")
                except Exception as e:
                    logger.error(f"Failed to pre-start TTS consumer: {e}")
            
            logger.info("✅ Conversation session started")
    
    async def receive(self, audio: tuple) -> None:
        """
        Receive audio from browser and push to VAD pipeline.
        
        According to FastRTC docs, receive() gets a tuple of (sample_rate, audio_array)
        where audio_array has shape (1, num_samples).
        
        Args:
            audio: Tuple of (sample_rate: int, audio_array: np.ndarray) from FastRTC
        """
        try:
            if audio is None:
                return
            
            # FastRTC provides (sample_rate, audio_array) tuple
            # audio_array has shape (1, num_samples) according to docs
            if isinstance(audio, tuple) and len(audio) == 2:
                sample_rate, audio_array = audio
            else:
                logger.warning(f"Unexpected audio format: {type(audio)}")
                return
            
            # Ensure numpy array
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array, dtype=np.float32)
            
            # Handle shape: FastRTC provides (1, num_samples), we need (num_samples,)
            if audio_array.ndim == 2:
                # Reshape from (1, num_samples) to (num_samples,)
                audio_array = audio_array.squeeze()
            elif audio_array.ndim > 2:
                # Flatten if needed
                audio_array = audio_array.flatten()
            
            # Ensure float32 format
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Normalize if needed (FastRTC may provide int16)
            max_val = np.max(np.abs(audio_array))
            if max_val > 1.0:
                audio_array = audio_array / 32767.0
            
            # Push to audio source for VAD pipeline (async push since we're in async context)
            await self.audio_source._async_push(audio_array)
            
            logger.debug(f"📥 Received audio: {len(audio_array)} samples at {sample_rate}Hz")
            
        except Exception as e:
            logger.error(f"Error receiving audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def emit(self):
        """
        Emit TTS audio chunks to browser.
        
        Uses fastrtc.utils.wait_for_item to efficiently wait for audio chunks
        from the async sink queue without blocking or busy-waiting.
        
        Returns:
            Tuple of (sample_rate: int, audio_chunk: np.ndarray)
        """
        # ROBUST STATE MANAGER: Only start TTS consumer when audio emission is complete
        current_time = time.time()

        # Check if we can start TTS consumer (audio fully played in browser)
        # Use emission end time + buffer to ensure audio is done playing in browser
        emission_buffer_time = 2.0  # 2 second buffer after emission ends (increased for safety)
        queue_size = self.audio_sink.output_queue.qsize()
        
        # Robust check: Only start consumer when:
        # 1. Streaming enabled
        # 2. No active consumer task
        # 3. No active audio emission
        # 4. Queue is empty (no pending chunks)
        # 5. Emission has ended AND buffer time has passed
        can_start_consumer = (
            self._tts_streaming_enabled and
            (self.tts_consumer_task is None or self.tts_consumer_task.done()) and
            not self._audio_emission_active and  # No active emission
            queue_size == 0 and  # Queue must be empty
            (self._audio_emission_end_time > 0) and  # We know when emission ended
            (current_time - self._audio_emission_end_time > emission_buffer_time)  # Buffer after emission ends
        )

        # AUDIO EMISSION TRACKING: Mark emission as active when we start getting chunks
        # Check queue size to detect active emission (more reliable than just empty check)
        if not self._audio_emission_active and queue_size > 0:
            self._audio_emission_active = True
            self._audio_emission_start_time = current_time
            logger.debug(f"🎵 Audio emission to browser started at {current_time} (queue size: {queue_size})")
            # Reset end time when new emission starts
            self._audio_emission_end_time = 0

        # Only start TTS consumer when audio emission is completely done
        if can_start_consumer:
            try:
                loop = asyncio.get_running_loop()
                self.tts_consumer_task = loop.create_task(self._consume_tts_for_fastrtc())
                self._last_consumer_restart = current_time
                logger.debug("🎵 Started TTS consumer for FastRTC (after audio emission complete)")
            except Exception as e:
                logger.error(f"Failed to start TTS consumer: {e}")

        # Use wait_for_item to wait for audio chunks from the async queue
        # This is the recommended pattern for AsyncStreamHandler

        try:
            # Wait for the next chunk from the sink queue
            # FastRTC expects (sample_rate, audio_array)
            item = await wait_for_item_utils(self.audio_sink.output_queue)

            # Handle case where wait_for_item returns None (e.g. queue empty/closed)
            if item is None:
                # Mark emission as complete when queue is empty
                if self._audio_emission_active:
                    self._audio_emission_active = False
                    self._audio_emission_end_time = time.time()
                    logger.debug(f"🎵 Audio emission to browser completed at {self._audio_emission_end_time}")
                    
                    # Transition to PLAYING state (buffered audio still playing)
                    state_machine = get_state_machine()
                    await state_machine.set_audio_state(AudioState.PLAYING, reason="Emission complete, playback continuing")
                    
                await asyncio.sleep(0.02)  # Prevent busy loop
                return (24000, np.zeros((1, 2400), dtype=np.int16))

            sample_rate, chunk = item

            # Mark emission as active when we receive actual audio data
            if not self._audio_emission_active:
                self._audio_emission_active = True
                self._audio_emission_start_time = time.time()
                self._audio_emission_end_time = 0  # Reset end time
                logger.debug(f"🎵 Audio emission to browser started (received chunk)")

            # Ensure chunk is (1, N) shape as required by FastRTC
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)
            
            # Log occasionally
            if not hasattr(self, '_chunk_count'):
                self._chunk_count = 0
            self._chunk_count += 1
            
            # Calculate chunk duration for VAD locking
            num_samples = chunk.shape[1]
            chunk_duration_seconds = num_samples / sample_rate
            
            # ROBUST TIMING: Update State Machine
            state_machine = get_state_machine()
            
            # Update state to EMITTING if not already
            # We do this here because this is where the chunks actually leave to the browser
            if state_machine.audio_state != AudioState.EMITTING:
                await state_machine.set_audio_state(AudioState.EMITTING, reason="Emitting to browser")

            # Update browser playback timing
            # This is the critical piece for echo cancellation
            await state_machine.update_browser_playback_end_time(chunk_duration_seconds)
            new_end_time = state_machine._agent_playback_end_time
            
            if self._chunk_count <= 5 or self._chunk_count % 10 == 0:
                logger.debug(f"📤 Emitting chunk {self._chunk_count}: {num_samples} samples ({chunk_duration_seconds*1000:.0f}ms) | VAD Locked until: {new_end_time:.2f}")
            
            return (sample_rate, chunk)
                
        except Exception as e:
            logger.error(f"Error waiting for item: {e}")
            # Return silence on error to keep stream alive, but wait a bit to prevent busy loop
            await asyncio.sleep(0.1)
            return (24000, np.zeros((1, 2400), dtype=np.int16))
                
        except Exception as e:
            logger.error(f"Error emitting audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await asyncio.sleep(0.1)
            return (24000, np.zeros((1, 2400), dtype=np.int16))
    
    async def _run_conversation_session(self):
        """Run conversation session with FastRTC audio adapters."""
        try:
            logger.info("🎙️ Starting conversation session with FastRTC audio")
            
            # Run conversation session with FastRTC adapters
            await run_conversation_session(
                audio_source=self.audio_source,
                audio_sink=self.audio_sink,
                skip_intro=False  # Play intro via FastRTC
            )
            
            logger.info("✅ Conversation session completed")
            
        except Exception as e:
            logger.error(f"Conversation session error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.is_conversation_active = False
            # Cleanup
            self.audio_source.clear()
            self.audio_sink.clear()
    
    async def _consume_tts_for_fastrtc(self):
        """
        Consume TTS streaming queue and route audio to FastRTC sink.
        
        Uses start_tts_consumer which accepts audio_sink parameter and internally
        calls consume_tts_streaming_queue with the audio_sink.
        """
        try:
            from leibniz_pro import start_tts_consumer
            import inspect
            
            logger.debug("🎵 Starting TTS consumer for FastRTC streaming")
            
            # Check signature to be safe
            sig = inspect.signature(start_tts_consumer)
            if 'audio_sink' in sig.parameters:
                # start_tts_consumer accepts audio_sink
                consumer_task = await start_tts_consumer(audio_sink=self.audio_sink)
            else:
                # Fallback for older version
                logger.warning("start_tts_consumer doesn't accept audio_sink, using global injection hack")
                # This is a hack: we might need to set a global variable in leibniz_pro if this happens
                # But for now let's just call it without sink and hope it picks up the sink from context 
                # if we modify leibniz_pro to look for it elsewhere, or just accept local playback.
                consumer_task = await start_tts_consumer()
            
            # Wait for the consumer task to complete
            # It will finish when the queue is empty and streaming is done
            try:
                await consumer_task
                logger.debug("🎵 TTS consumer completed for FastRTC")
            except asyncio.CancelledError:
                logger.debug("TTS consumer task was cancelled")
            
        except Exception as e:
            logger.error(f"TTS consumer error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def copy(self) -> 'LeibnizFastRTCStreamHandler':
        """Create a copy of this handler for FastRTC."""
        return LeibnizFastRTCStreamHandler()
    
    async def shutdown(self) -> None:
        """Cleanup resources when stream closes."""
        logger.info("🛑 FastRTC stream shutting down")
        
        # Cancel conversation task
        if self.conversation_task and not self.conversation_task.done():
            self.conversation_task.cancel()
            try:
                await self.conversation_task
            except asyncio.CancelledError:
                pass
        
        # Cancel TTS consumer
        if self.tts_consumer_task and not self.tts_consumer_task.done():
            self.tts_consumer_task.cancel()
            try:
                await self.tts_consumer_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup audio adapters
        self.audio_source.clear()
        self.audio_sink.clear()
        
        self.is_conversation_active = False


def create_fastrtc_app() -> gr.Interface:
    """
    Create a FastRTC Gradio app for browser-based conversation.
    
    Returns:
        Gradio Interface configured for WebRTC bidirectional audio
    """
    # Create handler instance
    handler = LeibnizFastRTCStreamHandler()
    
    # Create FastRTC stream
    stream = Stream(
        handler=handler,
        modality="audio",
        mode="send-receive",  # Bidirectional audio
        ui_args={
            "title": "Leibniz University Customer Service Agent",
            "description": "Speak with the Leibniz customer service agent via your browser. "
                         "Audio streams bidirectionally through WebRTC for real-time conversation."
        }
    )
    
    logger.info("✅ FastRTC app created")
    return stream.ui


async def initialize_and_create_app():
    """
    Initialize Leibniz services and create FastRTC app.
    
    This should be called before launching the app to ensure all services
    are ready for conversation.
    """
    logger.info("🔧 Initializing Leibniz services for FastRTC...")
    
    try:
        await initialize_leibniz_services()
        logger.info("✅ Leibniz services initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    return create_fastrtc_app()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Startup banner
    logger.info("=" * 60)
    logger.info("🎤 Leibniz FastRTC WebRTC Integration")
    logger.info("=" * 60)
    logger.info("📱 Browser UI: Will be shown when server starts")
    logger.info("🌐 Public HTTPS UI: Will be shown when server starts")
    logger.info("🎵 Bidirectional audio streaming via WebRTC")
    logger.info("💡 Full conversation pipeline with RAG support")
    logger.info("=" * 60)
    
    # Initialize and launch
    import asyncio
    import socket
    
    def find_free_port(start_port=7866, max_attempts=10):
        """Find an available port starting from start_port."""
        for i in range(max_attempts):
            port = start_port + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_attempts - 1}")
    
    try:
        # Find available port
        port = find_free_port(7866)
        logger.info(f"🚀 Using port {port} for FastRTC server")
        
        app = asyncio.run(initialize_and_create_app())
        
        # Launch with HTTPS support via Gradio share
        # This creates a secure tunnel that browsers will accept for microphone access
        logger.info(f"🌐 Launching FastRTC server on port {port}")
        logger.info(f"📱 Local access: http://localhost:{port}")
        logger.info(f"🔒 HTTPS URL will be provided below - USE THAT URL for microphone access")
        logger.info(f"⚠️  IMPORTANT: Use the HTTPS URL (not localhost) for microphone access!")
        
        # Launch with HTTPS tunnel via Gradio share (matching test_stt_webrtc_stream.py pattern)
        # CRITICAL: Use the HTTPS URL provided by Gradio, NOT localhost
        print("\n" + "="*70)
        print("⚠️  IMPORTANT: Media Device Access Instructions")
        print("="*70)
        print("1. Wait for the HTTPS URL below (starts with https://)")
        print("2. Copy and paste that HTTPS URL into your browser")
        print("3. DO NOT use http://localhost - browsers require HTTPS for microphone")
        print("4. When prompted, click 'Allow' for microphone access")
        print("="*70 + "\n")
        
        # Match test_stt_webrtc_stream.py launch pattern exactly for HTTPS
        # This creates HTTPS tunnel via Gradio share - REQUIRED for microphone access
        app.launch(
            server_name="0.0.0.0", 
            server_port=port, 
            show_error=True,
            share=True  # Create HTTPS tunnel via Gradio (matches test_stt_webrtc_stream.py)
        )
    except KeyboardInterrupt:
        logger.info("👋 Shutting down...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())

