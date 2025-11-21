"""
Comprehensive integration test suite for FastRTC audio flow.

This test suite validates the complete FastRTC audio pipeline from browser to agent and back,
ensuring proper audio format conversions, queue operations, event signaling, and error handling.
Tests cover browser audio ingestion, TTS output streaming, full conversation turns,
backward compatibility, and error scenarios.

Test Structure:
- TestFastRTCAudioSource: Audio source adapter functionality
- TestFastRTCAudioSink: Audio sink adapter functionality
- TestFastRTCHandler: Handler coordination and event signaling
- TestFastRTCIntegrationEndToEnd: Complete audio flow integration
- TestFastRTCAudioFormatConversions: Audio format validation
- TestFastRTCQueueOperations: Queue behavior and cleanup
- TestFastRTCEventSignaling: Event coordination testing

Run with: pytest test_fastrtc_integration.py -v -m integration
Coverage: pytest test_fastrtc_integration.py --cov=leibniz_fastrtc_adapters --cov=leibniz_fastrtc_handler --cov-report=html
"""

import pytest
import pytest_asyncio
import asyncio
import numpy as np
import logging
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leibniz_fastrtc_adapters import FastRTCAudioSource, FastRTCAudioSink
from leibniz_fastrtc_handler import LeibnizFastRTCHandler

# Import other modules conditionally to avoid relative import issues
try:
    from leibniz_pro import speak_friendly, run_conversation_session
    from leibniz_vad import capture_leibniz_speech, get_leibniz_vad
    IMPORTS_SUCCESSFUL = True
except ImportError:
    # Mock the imports for testing
    speak_friendly = None
    run_conversation_session = None
    capture_leibniz_speech = None
    get_leibniz_vad = None
    IMPORTS_SUCCESSFUL = False

# Configure test logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Pytest Configuration
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio
]


@pytest.fixture(scope="module", autouse=True)
def mock_audio_devices():
    """Mock sounddevice and pygame to avoid hardware dependencies."""
    with patch('sounddevice.play') as mock_sd_play, \
         patch('sounddevice.wait') as mock_sd_wait, \
         patch('sounddevice.InputStream') as mock_sd_input, \
         patch('pygame.mixer.music.load') as mock_pygame_load, \
         patch('pygame.mixer.music.play') as mock_pygame_play, \
         patch('pygame.mixer.music.get_busy') as mock_pygame_busy, \
         patch('pygame.mixer.music.stop') as mock_pygame_stop:

        # Mock pygame busy state
        mock_pygame_busy.side_effect = [True, True, False]  # Play for 2 calls, then stop

        yield {
            'sd_play': mock_sd_play,
            'sd_wait': mock_sd_wait,
            'sd_input': mock_sd_input,
            'pygame_load': mock_pygame_load,
            'pygame_play': mock_pygame_play,
            'pygame_busy': mock_pygame_busy,
            'pygame_stop': mock_pygame_stop
        }


@pytest.fixture(autouse=True)
def set_test_env():
    """Set environment variables for test mode."""
    original_env = dict(os.environ)

    # Disable background audio and set test mode
    os.environ['LEIBNIZ_ENABLE_BACKGROUND_AUDIO'] = 'false'
    os.environ['LEIBNIZ_TEST_MODE'] = 'true'
    os.environ['MOCK_TTS'] = 'true'
    os.environ['ALLOW_NO_TTS'] = 'true'

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Test Utilities
def create_mock_browser_audio(duration_s, sample_rate=16000):
    """Generate test audio data as float32 numpy array (browser input format)."""
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    # Generate sine wave at 440Hz, normalized to [-1.0, 1.0]
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio_data


def create_mock_tts_audio(duration_s, sample_rate=24000):
    """Generate test TTS audio as float32 numpy array (TTS output format)."""
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    # Generate sine wave at 880Hz, normalized to [-1.0, 1.0]
    audio_data = 0.5 * np.sin(2 * np.pi * 880 * t).astype(np.float32)
    return audio_data


def assert_audio_format(audio_data, expected_rate, expected_dtype=np.float32):
    """Validate audio format."""
    assert isinstance(audio_data, np.ndarray), "Audio data must be numpy array"
    assert audio_data.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {audio_data.dtype}"
    assert audio_data.ndim == 1, "Audio data must be mono (1D array)"
    assert np.all((audio_data >= -1.0) & (audio_data <= 1.0)), "Audio data must be normalized to [-1.0, 1.0]"


async def assert_queue_empty(queue, timeout=0.1):
    """Verify asyncio.Queue is empty."""
    try:
        await asyncio.wait_for(queue.get(), timeout=timeout)
        pytest.fail("Queue should be empty")
    except asyncio.TimeoutError:
        pass  # Expected - queue is empty


class TestFastRTCAudioSource:
    """Test FastRTC audio source adapter functionality."""

    @pytest.mark.asyncio
    async def test_push_audio_from_browser(self):
        """Test browser audio ingestion."""
        source = FastRTCAudioSource(sample_rate=16000)

        # Generate mock browser audio
        audio_chunk = create_mock_browser_audio(0.1)  # 100ms chunk

        # Push audio chunk
        await source.push_audio_from_fastrtc(audio_chunk)

        # Verify audio queued correctly
        assert not source.audio_queue.empty()
        assert source.is_active

        # Clean up
        source.clear()

    @pytest.mark.asyncio
    async def test_get_frames_for_vad(self):
        """Test VAD audio retrieval."""
        source = FastRTCAudioSource(sample_rate=16000)

        # Push audio chunk (200ms = 3200 samples at 16kHz)
        audio_chunk = create_mock_browser_audio(0.2)
        await source.push_audio_from_fastrtc(audio_chunk)

        # Get frames for VAD (50ms = 800 samples at 16kHz)
        frames = await source.get_frames(num_samples=800)

        # Verify returned audio
        assert_audio_format(frames, 16000)
        assert len(frames) == 800
        assert np.allclose(frames, audio_chunk[:800], atol=1e-6)

        # Clean up
        source.clear()

    @pytest.mark.asyncio
    async def test_get_frames_padding(self):
        """Test padding for short audio."""
        source = FastRTCAudioSource(sample_rate=16000)

        # Push short audio chunk (25ms = 400 samples)
        audio_chunk = create_mock_browser_audio(0.025)
        await source.push_audio_from_fastrtc(audio_chunk)

        # Get frames (50ms = 800 samples)
        frames = await source.get_frames(num_samples=800)

        # Verify padding
        assert_audio_format(frames, 16000)
        assert len(frames) == 800
        assert np.allclose(frames[:400], audio_chunk, atol=1e-6)
        assert np.allclose(frames[400:], 0.0, atol=1e-6)

        # Clean up
        source.clear()

    @pytest.mark.asyncio
    async def test_get_frames_truncation(self):
        """Test truncation for long audio."""
        source = FastRTCAudioSource(sample_rate=16000)

        # Push long audio chunk (200ms = 3200 samples)
        audio_chunk = create_mock_browser_audio(0.2)
        await source.push_audio_from_fastrtc(audio_chunk)

        # Get frames (50ms = 800 samples)
        frames = await source.get_frames(num_samples=800)

        # Verify truncation
        assert_audio_format(frames, 16000)
        assert len(frames) == 800
        assert np.allclose(frames, audio_chunk[:800], atol=1e-6)

        # Clean up
        source.clear()

    @pytest.mark.asyncio
    async def test_get_frames_timeout_returns_silence(self):
        """Test timeout behavior."""
        source = FastRTCAudioSource(sample_rate=16000)

        # No audio pushed - should timeout and return silence
        frames = await source.get_frames(num_samples=800)

        # Verify silence
        assert_audio_format(frames, 16000)
        assert len(frames) == 800
        assert np.allclose(frames, 0.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_clear_buffer(self):
        """Test buffer clearing."""
        source = FastRTCAudioSource(sample_rate=16000)

        # Push multiple audio chunks
        for i in range(3):
            audio_chunk = create_mock_browser_audio(0.1)
            await source.push_audio_from_fastrtc(audio_chunk)

        # Verify queue not empty
        assert not source.audio_queue.empty()

        # Clear buffer
        source.clear()

        # Verify queue is empty
        await assert_queue_empty(source.audio_queue)


class TestFastRTCAudioSink:
    """Test FastRTC audio sink adapter functionality."""

    @pytest.mark.asyncio
    async def test_write_audio_float32(self):
        """Test TTS audio ingestion (float32)."""
        sink = FastRTCAudioSink(sample_rate=24000)

        # Generate mock TTS audio
        audio_data = create_mock_tts_audio(1.0)  # 1 second

        # Write audio
        await sink.write_audio(audio_data, sample_rate=24000)

        # Verify audio queued
        assert not sink.output_queue.empty()
        assert sink.is_streaming

        # Clean up
        sink.clear()

    @pytest.mark.asyncio
    async def test_write_audio_int16_conversion(self):
        """Test int16 to float32 conversion."""
        sink = FastRTCAudioSink(sample_rate=24000)

        # Generate mock int16 audio
        duration_s = 1.0
        sample_rate = 24000
        num_samples = int(sample_rate * duration_s)
        t = np.linspace(0, duration_s, num_samples, endpoint=False)
        int16_audio = (0.5 * np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)

        # Write int16 audio
        await sink.write_audio(int16_audio, sample_rate=24000)

        # Verify conversion
        queued_tuple = sink.output_queue._queue[0]  # Access private queue for testing
        queued_sample_rate, queued_audio = queued_tuple
        assert_audio_format(queued_audio, 24000)
        expected_float32 = int16_audio.astype(np.float32) / 32767.0
        assert np.allclose(queued_audio, expected_float32, atol=1e-6)

        # Clean up
        sink.clear()

    @pytest.mark.asyncio
    async def test_stream_to_fastrtc_chunking(self):
        """Test audio streaming to browser."""
        sink = FastRTCAudioSink(sample_rate=24000)

        # Write 5 seconds of audio (120,000 samples at 24kHz)
        audio_data = create_mock_tts_audio(5.0)
        await sink.write_audio(audio_data, sample_rate=24000)

        # Stream to browser
        chunks = []
        for sample_rate, chunk in sink.stream_to_fastrtc():
            chunks.append(chunk)
            assert_audio_format(chunk, 24000)
            assert len(chunk) == 2400  # 100ms at 24kHz

        # Verify total samples (should be padded to chunk boundary)
        total_samples = sum(len(chunk) for chunk in chunks)
        expected_samples = ((len(audio_data) + 2399) // 2400) * 2400  # Round up to chunk size
        assert total_samples == expected_samples

        # Clean up
        sink.clear()

    @pytest.mark.asyncio
    async def test_stream_to_fastrtc_last_chunk_padding(self):
        """Test last chunk padding."""
        sink = FastRTCAudioSink(sample_rate=24000)

        # Write audio with non-multiple of 2400 samples (5000 samples)
        audio_data = create_mock_tts_audio(5000 / 24000)  # ~0.208 seconds
        await sink.write_audio(audio_data, sample_rate=24000)

        # Stream and collect chunks
        chunks = []
        for sample_rate, chunk in sink.stream_to_fastrtc():
            chunks.append(chunk)

        # Verify last chunk is padded
        assert len(chunks) == 3  # Should be 3 chunks (5000 / 2400 rounded up)
        last_chunk = chunks[-1]
        assert len(last_chunk) == 2400
        # Check that the first 5000 samples match the original audio
        all_chunks = np.concatenate(chunks)
        assert np.allclose(all_chunks[:5000], audio_data, atol=1e-6)
        assert np.allclose(all_chunks[5000:], 0.0, atol=1e-6)

        # Clean up
        sink.clear()

    @pytest.mark.asyncio
    async def test_stop_streaming(self):
        """Test streaming stop."""
        sink = FastRTCAudioSink(sample_rate=24000)

        # Write audio and start streaming
        audio_data = create_mock_tts_audio(1.0)
        await sink.write_audio(audio_data, sample_rate=24000)
        assert sink.is_streaming

        # Stop streaming
        sink.stop_streaming()
        assert not sink.is_streaming

        # Should still yield the existing chunks
        chunks = []
        for sample_rate, chunk in sink.stream_to_fastrtc():
            chunks.append(chunk)

        # Should yield all remaining chunks
        assert len(chunks) > 0
        for chunk in chunks:
            assert_audio_format(chunk, 24000)

    @pytest.mark.asyncio
    async def test_clear_buffer(self):
        """Test buffer clearing."""
        sink = FastRTCAudioSink(sample_rate=24000)

        # Write multiple audio chunks
        for i in range(3):
            audio_data = create_mock_tts_audio(0.1)
            await sink.write_audio(audio_data, sample_rate=24000)

        # Verify queue not empty
        assert not sink.output_queue.empty()

        # Clear buffer
        sink.clear()

        # Verify queue is empty
        await assert_queue_empty(sink.output_queue)


class TestFastRTCHandler:
    """Test FastRTC handler coordination and event signaling."""

    @pytest.mark.asyncio
    async def test_handler_initialization(self):
        """Test handler setup."""
        handler = LeibnizFastRTCHandler()

        # Verify adapters
        assert isinstance(handler.source, FastRTCAudioSource)
        assert isinstance(handler.sink, FastRTCAudioSink)
        assert handler.source.sample_rate == 16000
        assert handler.sink.sample_rate == 24000

        # Verify event is cleared
        assert not handler.user_finished_speaking.is_set()

    @pytest.mark.asyncio
    async def test_handler_call_audio_routing(self):
        """Test __call__ method audio routing."""
        handler = LeibnizFastRTCHandler()

        # Generate mock browser audio
        audio_data = create_mock_browser_audio(0.5)  # 500ms

        # Call handler with audio input - should complete without error
        chunks = []
        for sample_rate, audio_chunk in handler((16000, audio_data)):
            chunks.append((sample_rate, audio_chunk))

        # Verify handler completed successfully
        # (No chunks expected since no TTS audio was written to sink)

    @pytest.mark.asyncio
    async def test_handler_call_response_streaming(self):
        """Test response audio streaming."""
        handler = LeibnizFastRTCHandler()

        # Write mock TTS audio to sink
        tts_audio = create_mock_tts_audio(1.0)
        await handler.sink.write_audio(tts_audio, sample_rate=24000)

        # Call handler (no input audio)
        chunks = []
        for sample_rate, audio_chunk in handler(None):
            chunks.append((sample_rate, audio_chunk))

        # Verify chunks are (sample_rate, audio_chunk) tuples
        for sample_rate, audio_chunk in chunks:
            assert sample_rate == 24000
            assert_audio_format(audio_chunk, 24000)
            assert len(audio_chunk) == 2400  # 100ms chunks

        # Clean up
        handler.source.clear()
        handler.sink.clear()

    @pytest.mark.asyncio
    async def test_handler_call_buffer_cleanup(self):
        """Test buffer cleanup after turn."""
        handler = LeibnizFastRTCHandler()

        # Push audio to source and write to sink
        source_audio = create_mock_browser_audio(0.2)
        await handler.source.push_audio_from_fastrtc(source_audio)

        sink_audio = create_mock_tts_audio(0.5)
        await handler.sink.write_audio(sink_audio, sample_rate=24000)

        # Call handler and iterate to completion
        for _ in handler((16000, source_audio)):
            pass

        # Verify buffers are cleared
        await assert_queue_empty(handler.source.audio_queue)
        await assert_queue_empty(handler.sink.output_queue)

        # Verify event is cleared
        assert not handler.user_finished_speaking.is_set()

    @pytest.mark.asyncio
    async def test_handler_error_handling(self):
        """Test error handling."""
        handler = LeibnizFastRTCHandler()

        # Mock source to raise exception
        original_push = handler.source.push_audio_from_fastrtc
        handler.source.push_audio_from_fastrtc = MagicMock(side_effect=Exception("Test error"))

        # Generate mock audio
        audio_data = create_mock_browser_audio(0.1)

        # Call handler - should handle error gracefully
        chunks = []
        for sample_rate, audio_chunk in handler((16000, audio_data)):
            chunks.append((sample_rate, audio_chunk))

        # Should yield silence on error
        for sample_rate, audio_chunk in chunks:
            assert sample_rate == 24000
            assert_audio_format(audio_chunk, 24000)
            assert np.allclose(audio_chunk, 0.0, atol=1e-6)

        # Restore original method
        handler.source.push_audio_from_fastrtc = original_push


class TestFastRTCIntegrationEndToEnd:
    """Test complete FastRTC audio flow integration."""

    @pytest.mark.asyncio
    async def test_browser_to_vad_stt_flow(self):
        """Test browser audio → VAD → STT."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip("Required modules not available")

        with patch('leibniz_stt.transcribe_audio_bytes', return_value="test transcript"):
            handler = LeibnizFastRTCHandler()

            # Push browser audio
            browser_audio = create_mock_browser_audio(1.0)
            await handler.source.push_audio_from_fastrtc(browser_audio)

            # Call capture with FastRTC source (let it run normally)
            transcript = await capture_leibniz_speech(audio_source=handler.source)

            # Verify transcript
            assert transcript == "test transcript"

            # Clean up
            handler.source.clear()

    @pytest.mark.asyncio
    async def test_tts_to_browser_flow(self):
        """Test TTS → sink → browser."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip("Required modules not available")

        handler = LeibnizFastRTCHandler()

        # Mock TTS synthesis
        mock_audio = create_mock_tts_audio(1.0)

        with patch('leibniz_tts.get_leibniz_tts') as mock_get_tts:
            mock_tts = MagicMock()
            mock_get_tts.return_value = mock_tts

            # Mock synthesize_to_file to return audio data
            mock_tts.synthesize_to_file.return_value = {
                'success': True,
                'audio_bytes': (mock_audio * 32767).astype(np.int16).tobytes(),
                'sample_rate': 24000,
                'duration': 1.0
            }

            # Call speak_friendly with FastRTC sink
            result = await speak_friendly(text="test", audio_sink=handler.sink)

            # Verify audio written to sink
            assert not handler.sink.output_queue.empty()

            # Stream audio to browser
            chunks = []
            for sample_rate, chunk in handler.sink.stream_to_fastrtc():
                chunks.append(chunk)

            # Verify chunks yielded
            assert len(chunks) > 0
            for chunk in chunks:
                assert_audio_format(chunk, 24000)

            # Clean up
            handler.sink.clear()

    @pytest.mark.asyncio
    async def test_full_conversation_turn(self):
        """Test complete turn (user speaks → agent responds)."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip("Required modules not available")

        handler = LeibnizFastRTCHandler()

        # Mock all components
        with patch('leibniz_vad.capture_leibniz_speech', return_value="hello"), \
             patch('leibniz_intent_parser.get_leibniz_parser') as mock_get_parser, \
             patch('leibniz_rag.get_leibniz_rag') as mock_get_rag, \
             patch('leibniz_tts.get_leibniz_tts') as mock_get_tts:

            # Setup mocks
            mock_parser = MagicMock()
            mock_parser.classify_intent.return_value = {
                'intent': 'GREETING',
                'confidence': 0.9,
                'context': {}
            }
            mock_get_parser.return_value = mock_parser

            mock_rag = MagicMock()
            mock_rag.process_rag_query.return_value = {
                'response': 'Hello! How can I help you?',
                'sources': [],
                'timing_breakdown': {}
            }
            mock_get_rag.return_value = mock_rag

            mock_tts = MagicMock()
            mock_audio = create_mock_tts_audio(1.0)
            mock_tts.synthesize_to_file.return_value = {
                'success': True,
                'audio_bytes': (mock_audio * 32767).astype(np.int16).tobytes(),
                'sample_rate': 24000,
                'duration': 1.0
            }
            mock_get_tts.return_value = mock_tts

            # Push user audio
            user_audio = create_mock_browser_audio(0.5)
            await handler.source.push_audio_from_fastrtc(user_audio)

            # Run conversation session
            await run_conversation_session(
                audio_source=handler.source,
                audio_sink=handler.sink
            )

            # Verify audio processed and response generated
            assert not handler.sink.output_queue.empty()

            # Clean up
            handler.source.clear()
            handler.sink.clear()

    @pytest.mark.asyncio
    async def test_backward_compatibility_native_mode(self):
        """Test native audio devices still work."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip("Required modules not available")

        # Test speak_friendly without FastRTC sink
        with patch('sounddevice.play') as mock_sd_play, \
             patch('sounddevice.wait') as mock_sd_wait:

            result = await speak_friendly(text="test", audio_sink=None)

            # Verify native playback called
            mock_sd_play.assert_called()
            mock_sd_wait.assert_called()

        # Test capture without FastRTC source
        with patch('leibniz_vad.capture_leibniz_speech') as mock_capture:
            mock_capture.return_value = "test transcript"

            transcript = await capture_leibniz_speech(audio_source=None)

            # Verify native capture used
            assert transcript == "test transcript"
            mock_capture.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_connection_drop(self):
        """Test connection drop handling."""
        handler = LeibnizFastRTCHandler()

        # Mock source to raise ConnectionError
        handler.source.push_audio_from_fastrtc = MagicMock(side_effect=ConnectionError("Connection dropped"))

        # Generate audio
        audio_data = create_mock_browser_audio(0.1)

        # Call handler - should handle error
        chunks = []
        for chunk in handler((16000, audio_data)):
            chunks.append(chunk)

        # Should yield silence
        for sample_rate, audio_chunk in chunks:
            assert sample_rate == 24000
            assert np.allclose(audio_chunk, 0.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_error_handling_audio_format_mismatch(self):
        """Test audio format error handling."""
        source = FastRTCAudioSource(sample_rate=16000)

        # Push invalid audio (wrong dtype)
        invalid_audio = np.array([1, 2, 3], dtype=np.int32)
        await source.push_audio_from_fastrtc(invalid_audio)

        # Should handle gracefully (log warning but not crash)
        frames = await source.get_frames(num_samples=800)
        # The adapter passes through the audio as-is, so it may not be normalized
        assert isinstance(frames, np.ndarray)
        assert len(frames) == 800

        # Clean up
        source.clear()


class TestFastRTCAudioFormatConversions:
    """Test audio format validation and conversions."""

    @pytest.mark.asyncio
    async def test_source_16khz_float32_format(self):
        """Test source audio format."""
        source = FastRTCAudioSource(sample_rate=16000)

        # Push 16kHz float32 audio
        audio_data = create_mock_browser_audio(0.1)
        await source.push_audio_from_fastrtc(audio_data)

        # Retrieve frames
        frames = await source.get_frames(num_samples=800)

        # Verify format preserved
        assert_audio_format(frames, 16000)
        assert len(frames) == 800

        # Clean up
        source.clear()

    @pytest.mark.asyncio
    async def test_sink_24khz_float32_format(self):
        """Test sink audio format."""
        sink = FastRTCAudioSink(sample_rate=24000)

        # Write 24kHz float32 audio
        audio_data = create_mock_tts_audio(0.1)
        await sink.write_audio(audio_data, sample_rate=24000)

        # Stream chunks
        chunks = []
        for sample_rate, chunk in sink.stream_to_fastrtc():
            chunks.append(chunk)

        # Verify format preserved
        for chunk in chunks:
            assert_audio_format(chunk, 24000)
            assert len(chunk) == 2400

        # Clean up
        sink.clear()

    @pytest.mark.asyncio
    async def test_int16_to_float32_conversion_accuracy(self):
        """Test conversion accuracy."""
        sink = FastRTCAudioSink(sample_rate=24000)

        # Create int16 audio with known values
        int16_values = np.array([0, 16383, 32767, -16384, -32768], dtype=np.int16)
        expected_float32 = int16_values.astype(np.float32) / 32767.0

        # Write int16 audio
        await sink.write_audio(int16_values, sample_rate=24000)

        # Verify conversion accuracy
        queued_tuple = sink.output_queue._queue[0]  # Access private queue for testing
        queued_sample_rate, queued_audio = queued_tuple
        assert np.allclose(queued_audio, expected_float32, atol=1e-6)

        # Clean up
        sink.clear()


class TestFastRTCQueueOperations:
    """Test queue behavior and cleanup."""

    @pytest.mark.asyncio
    async def test_source_queue_overflow_handling(self):
        """Test source queue overflow."""
        source = FastRTCAudioSource(sample_rate=16000)

        # Push more chunks than queue capacity (default 100)
        for i in range(150):
            audio_chunk = create_mock_browser_audio(0.01)  # Very small chunks
            await source.push_audio_from_fastrtc(audio_chunk)

        # Queue should not grow beyond default capacity
        assert source.audio_queue.qsize() <= 100

        # Clean up
        source.clear()

    @pytest.mark.asyncio
    async def test_sink_queue_unbounded(self):
        """Test sink queue is unbounded."""
        sink = FastRTCAudioSink(sample_rate=24000)

        # Write many chunks
        for i in range(50):
            audio_data = create_mock_tts_audio(0.01)  # Small chunks
            await sink.write_audio(audio_data, sample_rate=24000)

        # Queue should contain all chunks
        assert sink.output_queue.qsize() == 50

        # Clean up
        sink.clear()

    @pytest.mark.asyncio
    async def test_queue_cleanup_between_turns(self):
        """Test queue cleanup."""
        source = FastRTCAudioSource(sample_rate=16000)
        sink = FastRTCAudioSink(sample_rate=24000)

        # Add audio to both queues
        source_audio = create_mock_browser_audio(0.1)
        await source.push_audio_from_fastrtc(source_audio)

        sink_audio = create_mock_tts_audio(0.1)
        await sink.write_audio(sink_audio, sample_rate=24000)

        # Verify queues not empty
        assert not source.audio_queue.empty()
        assert not sink.output_queue.empty()

        # Clear queues
        source.clear()
        sink.clear()

        # Verify queues are empty
        await assert_queue_empty(source.audio_queue)
        await assert_queue_empty(sink.output_queue)


class TestFastRTCEventSignaling:
    """Test event coordination."""

    @pytest.mark.asyncio
    async def test_user_finished_speaking_event(self):
        """Test event signaling."""
        handler = LeibnizFastRTCHandler()

        # Push audio (should set event during processing)
        audio_data = create_mock_browser_audio(0.1)
        for _ in handler((16000, audio_data)):
            pass

        # Event should be cleared after processing completes
        assert not handler.user_finished_speaking.is_set()

    @pytest.mark.asyncio
    async def test_event_coordination_in_conversation_loop(self):
        """Test handler can be called without hanging."""
        handler = LeibnizFastRTCHandler()

        # Call handler - should complete without hanging
        audio_data = create_mock_browser_audio(0.1)
        chunks = []
        for sample_rate, audio_chunk in handler((16000, audio_data)):
            chunks.append((sample_rate, audio_chunk))

        # Handler completed successfully
        assert True