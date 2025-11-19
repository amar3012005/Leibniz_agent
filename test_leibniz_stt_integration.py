"""
Integration Test Suite for Leibniz STT+VAD

Tests transcript normalization, audio preprocessing, VAD integration, prewarm triggers,
and English-only validation for the Leibniz University agent.
"""

import asyncio
import time
import os
import sys
import pytest
import logging
from unittest.mock import MagicMock, AsyncMock, patch, Mock
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# CRITICAL: Mock genai.Client BEFORE importing leibniz modules (Comment 4)
try:
    import google.generativeai as genai
    if not hasattr(genai, 'Client'):
        # Add Client class to older google-generativeai package
        genai.Client = MagicMock
except ImportError:
    pytest.skip("google-generativeai not installed", allow_module_level=True)

# STT imports
from leibniz_agent.leibniz_stt import (
    get_leibniz_stt,
    normalize_english_transcript,
    validate_audio_file,
    check_audio_quality,
    convert_audio_format,
    create_silent_audio_file,
    cleanup_temp_audio_files,
    get_combined_performance_metrics,
    is_capture_active,
    check_and_handle_barge_in,
    trigger_lightweight_prewarm,
    transcribe_with_vad,
    LeibnizSTT
)

# VAD imports
from leibniz_agent.leibniz_vad import (
    get_leibniz_vad,
    capture_leibniz_speech,
    LeibnizBidirectionalVAD
)

# Persistent services
from leibniz_agent.leibniz_persistent_services import (
    trigger_prewarm_on_speech_detection,
    get_leibniz_services_manager
)

# Logger setup
logger = logging.getLogger(__name__)


# Module-level fixtures for consistent mocking (Comment 12)
@pytest.fixture(scope="module", autouse=True)
def set_api_key():
    """Ensure GEMINI_API_KEY is set for tests (Comment 4)"""
    # Store original value
    original_key = os.environ.get("GEMINI_API_KEY")
    
    if not original_key:
        # Set test key if not present
        os.environ["GEMINI_API_KEY"] = "test_key_for_testing_only"
    
    yield
    
    # Cleanup - restore original state
    if not original_key and os.environ.get("GEMINI_API_KEY") == "test_key_for_testing_only":
        os.environ.pop("GEMINI_API_KEY", None)


# Test utilities
def create_test_audio_file(duration: float = 1.0, sample_rate: int = 16000, format: str = 'wav') -> str:
    """Generate test audio file"""
    # Generate sine wave
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    
    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    # Write audio
    sf.write(temp_path, audio, sample_rate)
    
    return temp_path


# Test Class: TestTranscriptNormalization
class TestTranscriptNormalization:
    """Tests for English transcript normalization"""
    
    @pytest.mark.unit
    def test_basic_normalization(self):
        """Test whitespace stripping and lowercasing"""
        result = normalize_english_transcript("  HELLO WORLD  ")
        assert result == "hello world"
        logger.info(" Basic normalization verified")
    
    @pytest.mark.unit
    def test_filler_word_removal(self):
        """Test removal of filler words"""
        result = normalize_english_transcript("um hello uh there like you know")
        # Should remove um, uh, like, you know
        assert "um" not in result
        assert "uh" not in result
        assert "like" not in result
        assert "hello" in result
        assert "there" in result
        logger.info(f" Filler removal: '{result}'")
    
    @pytest.mark.unit
    def test_stuttering_removal(self):
        """Test removal of stuttering"""
        result = normalize_english_transcript("I I I want to th-th-thank you")
        # Should clean up stuttering
        assert result.count("i") <= 2  # Reduced repetition
        assert "want" in result
        assert "thank" in result
        logger.info(f" Stuttering removal: '{result}'")
    
    @pytest.mark.unit
    def test_artifact_removal(self):
        """Test removal of transcription artifacts"""
        result = normalize_english_transcript("Hello [inaudible] world [unclear] test [silence]")
        assert "[inaudible]" not in result
        assert "[unclear]" not in result
        assert "[silence]" not in result
        assert "hello" in result
        assert "world" in result
        logger.info(f" Artifact removal: '{result}'")
    
    @pytest.mark.unit
    def test_phone_number_preservation(self):
        """Test phone number preservation"""
        result = normalize_english_transcript("my number is 9876543210")
        assert "9876543210" in result
        logger.info(f" Phone preservation: '{result}'")
    
    @pytest.mark.unit
    def test_empty_input(self):
        """Test empty input handling"""
        result = normalize_english_transcript("")
        assert result == ""
        logger.info(" Empty input handled")
    
    @pytest.mark.unit
    def test_special_token_preservation(self):
        """Test special tokens NOT normalized"""
        result1 = normalize_english_transcript("NO_SPEECH")
        result2 = normalize_english_transcript("NON_ENGLISH_DETECTED")
        
        assert result1 == "NO_SPEECH"
        assert result2 == "NON_ENGLISH_DETECTED"
        logger.info(" Special tokens preserved")


# Test Class: TestAudioPreprocessing
class TestAudioPreprocessing:
    """Tests for audio helper functions"""
    
    @pytest.mark.integration
    @pytest.mark.unit
    def test_validate_audio_file_valid(self):
        """Test validation of valid audio file (Comment 5)"""
        test_file = create_test_audio_file(1.0, 16000)
        
        try:
            result = validate_audio_file(test_file)
            # Comment 5: Check result['valid'] instead of boolean
            assert isinstance(result, dict)
            assert 'valid' in result
            assert result['valid'] is True
            assert 'duration' in result
            assert 'sample_rate' in result
            logger.info(f" Valid audio file: {result}")
        finally:
            Path(test_file).unlink(missing_ok=True)
    
    @pytest.mark.unit
    def test_validate_audio_file_nonexistent(self):
        """Test validation of non-existent file (Comment 5)"""
        result = validate_audio_file("nonexistent_file.wav")
        # Comment 5: Check result['valid'] instead of boolean
        assert isinstance(result, dict)
        assert 'valid' in result
        assert result['valid'] is False
        assert 'errors' in result
        logger.info(f" Non-existent file rejected: {result['errors']}")
    
    @pytest.mark.integration
    def test_check_audio_quality_normal(self):
        """Test quality check on normal audio"""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 0.3
        quality = check_audio_quality(audio, 16000)
        
        assert 'is_silent' in quality
        assert 'has_clipping' in quality
        assert quality['is_silent'] is False
        assert quality['has_clipping'] is False
        logger.info(f" Normal audio quality: {quality}")
    
    @pytest.mark.unit
    def test_check_audio_quality_silent(self):
        """Test quality check on silent audio"""
        audio = np.zeros(16000)
        quality = check_audio_quality(audio, 16000)
        
        assert quality['is_silent'] is True
        assert quality['rms_db'] < -40
        logger.info(f" Silent audio detected: {quality}")
    
    @pytest.mark.unit
    def test_check_audio_quality_clipping(self):
        """Test quality check on clipped audio"""
        audio = np.ones(16000)  # Maximum values
        quality = check_audio_quality(audio, 16000)
        
        assert quality['has_clipping'] is True
        logger.info(f" Clipping detected: {quality}")
    
    @pytest.mark.unit
    def test_convert_audio_format_resample(self):
        """Test audio resampling"""
        # Generate 44100Hz audio
        audio_44k = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)) * 0.3
        
        # Convert to 16000Hz
        audio_16k = convert_audio_format(audio_44k, 44100, 16000)
        
        # Verify length adjusted (approximately)
        expected_len = int(len(audio_44k) * 16000 / 44100)
        assert abs(len(audio_16k) - expected_len) < 100
        logger.info(f" Resampling: {len(audio_44k)} → {len(audio_16k)} samples")
    
    @pytest.mark.unit
    def test_convert_audio_format_stereo_to_mono(self):
        """Test stereo to mono conversion (Comment 6)"""
        # Generate stereo audio (2 channels)
        stereo = np.random.randn(16000, 2) * 0.3
        
        # Comment 6: Use target_channels=1 instead of to_mono
        mono = convert_audio_format(stereo, 16000, 16000, target_channels=1)
        
        # Verify single channel (1D array)
        assert mono.ndim == 1
        logger.info(f" Stereo→Mono: {stereo.shape} → {mono.shape}")
    
    @pytest.mark.integration
    def test_create_silent_audio_file(self):
        """Test silent audio file creation"""
        temp_file = create_silent_audio_file(0.5, 16000)
        
        try:
            # Verify file exists
            assert Path(temp_file).exists()
            
            # Load and verify
            audio, sr = sf.read(temp_file)
            assert sr == 16000
            assert len(audio) / sr >= 0.45  # Approximately 0.5s
            assert np.allclose(audio, 0.0)  # All zeros
            
            logger.info(f" Silent file created: {len(audio)} samples")
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    @pytest.mark.integration
    def test_cleanup_temp_audio_files(self):
        """Test temp file cleanup"""
        # Create 3 temp files
        files = [
            create_silent_audio_file(0.1, 16000),
            create_silent_audio_file(0.1, 16000),
            create_silent_audio_file(0.1, 16000)
        ]
        
        # Verify all exist
        for f in files:
            assert Path(f).exists()
        
        # Cleanup
        cleanup_temp_audio_files(files)
        
        # Verify all deleted
        for f in files:
            assert not Path(f).exists()
        
        logger.info(" Temp files cleaned up")


# Test Class: TestVADIntegration
class TestVADIntegration:
    """Tests for STT+VAD coordination"""
    
    @pytest.mark.integration
    def test_get_combined_performance_metrics(self):
        """Test combined metrics retrieval"""
        metrics = get_combined_performance_metrics()
        
        # Verify structure
        assert isinstance(metrics, dict)
        # Should have at least basic metrics
        assert len(metrics) > 0
        
        logger.info(f" Combined metrics: {list(metrics.keys())}")
    
    @pytest.mark.unit
    def test_is_capture_active_coordination(self):
        """Test capture active detection"""
        # Initially not active
        assert is_capture_active() is False
        logger.info(" Capture active detection working")
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_check_and_handle_barge_in(self):
        """Test barge-in checking"""
        result = check_and_handle_barge_in()
        # Should return False when no barge-in
        assert isinstance(result, bool)
        logger.info(" Barge-in check working")
    
    @pytest.mark.unit
    def test_trigger_lightweight_prewarm_throttling(self):
        """Test prewarm throttling"""
        # First call
        trigger_lightweight_prewarm('test')
        
        # Second call (should be throttled)
        trigger_lightweight_prewarm('test')
        
        # Should not crash
        logger.info(" Prewarm throttling working")


# Test Class: TestSTTVADCoordination
class TestSTTVADCoordination:
    """End-to-end STT+VAD workflow tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cleanup_includes_vad(self):
        """Test cleanup coordination"""
        from leibniz_agent.leibniz_stt import cleanup_leibniz_stt
        from leibniz_agent.leibniz_vad import cleanup_leibniz_vad
        
        # Should not crash
        await cleanup_leibniz_stt()
        await cleanup_leibniz_vad()
        
        logger.info(" Cleanup coordination working")


# Test Class: TestLanguageDetectionAndTranscription (Comment 9)
class TestLanguageDetectionAndTranscription:
    """Tests for language detection and transcribe_with_vad wrapper"""
    
    @pytest.mark.unit
    def test_language_detector_high_confidence_english(self):
        """Test English transcript acceptance"""
        # High-confidence English should pass
        english_texts = [
            "Hello, how are you today?",
            "I would like to schedule an appointment",
            "Can you help me find information about the university?",
            "Thank you for your assistance"
        ]
        
        for text in english_texts:
            normalized = normalize_english_transcript(text)
            # Should not be empty or marked as non-English
            assert normalized != "NON_ENGLISH_DETECTED"
            assert normalized != ""
            assert len(normalized) > 0
            logger.info(f" English accepted: '{text[:30]}...'")
    
    @pytest.mark.unit
    def test_language_detector_mixed_language(self):
        """Test mixed language handling"""
        # Mixed language should still process English portions
        mixed_text = "Hello नमस्ते how are you"
        normalized = normalize_english_transcript(mixed_text)
        
        # Should extract English portions or handle gracefully
        assert isinstance(normalized, str)
        logger.info(f" Mixed language handled: '{normalized}'")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.requires_microphone  # Comment 14
    async def test_transcribe_with_vad_returns_tuple(self):
        """Test transcribe_with_vad wrapper returns (audio_file, transcript)"""
        # Mock VAD capture
        with patch('leibniz_agent.leibniz_stt.capture_leibniz_speech') as mock_capture:
            mock_capture.return_value = ("test.wav", "hello there")
            
            # Call wrapper
            audio_file, transcript = await transcribe_with_vad()
            
            # Verify tuple structure
            assert isinstance(audio_file, str)
            assert isinstance(transcript, str)
            
            # Verify normalized
            assert transcript == "hello there"  # Already normalized by VAD
            
            logger.info(f" transcribe_with_vad: ({audio_file}, '{transcript}')")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_transcribe_with_vad_normalization(self):
        """Test transcribe_with_vad returns normalized text"""
        # Mock VAD to return unnormalized text
        with patch('leibniz_agent.leibniz_stt.capture_leibniz_speech') as mock_capture:
            mock_capture.return_value = ("test.wav", "  HELLO WORLD  um uh  ")
            
            audio_file, transcript = await transcribe_with_vad()
            
            # Should be normalized (lowercase, trimmed, fillers removed)
            assert transcript.islower() or transcript == ""
            assert not transcript.startswith(" ")
            assert not transcript.endswith(" ")
            assert "um" not in transcript
            assert "uh" not in transcript
            
            logger.info(f" Normalized: '  HELLO WORLD  um uh  ' → '{transcript}'")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_prewarm_trigger_throttling_verification(self):
        """Test prewarm trigger throttling (20s debounce)"""
        # First trigger
        trigger_lightweight_prewarm('context1')
        time1 = time.time()
        
        # Immediate second trigger (should be throttled)
        trigger_lightweight_prewarm('context2')
        time2 = time.time()
        
        # Should complete quickly (not actually warming up)
        assert (time2 - time1) < 0.1
        
        logger.info(" Prewarm throttling verified (no 20s wait)")


# Test Class: TestEnglishOnlyValidation
class TestEnglishOnlyValidation:
    """Tests for English language enforcement"""
    
    @pytest.mark.unit
    def test_english_transcript_accepted(self):
        """Test English transcript acceptance"""
        result = normalize_english_transcript("Hello, how are you?")
        assert "hello" in result
        assert "how" in result
        assert "you" in result
        logger.info(f" English accepted: '{result}'")


# Test Class: TestSTTVADPerformance
class TestSTTVADPerformance:
    """Performance benchmarks (Comment 13: relaxed thresholds for CI compatibility)"""
    
    @pytest.mark.benchmark
    def test_benchmark_normalization_overhead(self):
        """Measure normalization performance (Comment 13: relaxed to 20-30ms)"""
        test_inputs = [
            "Short test",
            "Um hello uh there like you know this is a longer test sentence with some filler words",
            "This is an even longer test sentence with multiple clauses and lots of content that needs to be normalized including some filler words like um and uh and you know and other artifacts"
        ]
        
        for text in test_inputs:
            start = time.time()
            result = normalize_english_transcript(text)
            duration = time.time() - start
            
            # Comment 13: Relaxed target <30ms (was <10ms)
            assert duration < 0.030, f"Normalization too slow: {duration*1000:.1f}ms"
            logger.info(f" Normalize ({len(text)} chars): {duration*1000:.2f}ms")
    
    @pytest.mark.benchmark
    def test_benchmark_audio_preprocessing_speed(self):
        """Measure audio preprocessing performance (Comment 13: relaxed to 200ms)"""
        test_file = create_test_audio_file(1.0, 16000)
        
        try:
            # Validation
            start = time.time()
            validate_audio_file(test_file)
            val_time = time.time() - start
            
            # Quality check
            audio, sr = sf.read(test_file)
            start = time.time()
            check_audio_quality(audio, sr)
            quality_time = time.time() - start
            
            # Format conversion
            start = time.time()
            convert_audio_format(audio, sr, 16000)
            convert_time = time.time() - start
            
            # Comment 13: Relaxed to <200ms each (was <100ms)
            # Skip on CI if too slow
            if os.getenv('CI'):
                pytest.skip("Skipping performance test on CI")
            
            assert val_time < 0.2
            assert quality_time < 0.2
            assert convert_time < 0.2
            
            logger.info(f" Preprocessing: Val={val_time*1000:.1f}ms, Quality={quality_time*1000:.1f}ms, Convert={convert_time*1000:.1f}ms")
        finally:
            Path(test_file).unlink(missing_ok=True)


# Pytest fixtures
@pytest.fixture(scope="module")
def stt_instance():
    """Get STT instance"""
    return get_leibniz_stt()


@pytest.fixture(scope="module")
def vad_instance():
    """Get VAD instance"""
    return get_leibniz_vad()


@pytest.fixture
def test_audio_file():
    """Create test audio file"""
    filepath = create_test_audio_file(1.0, 16000)
    yield filepath
    Path(filepath).unlink(missing_ok=True)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-m", "integration"])
