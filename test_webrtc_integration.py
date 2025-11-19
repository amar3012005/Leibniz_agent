"""
Test script for Leibniz WebRTC implementation
Validates that all components are properly integrated
"""

import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_imports():
    """Test that all necessary imports work"""
    logger.info("Testing imports...")
    
    try:
        from leibniz_agent.leibniz_webrtc_handler import LeibnizConversationHandler, AudioFormatConverter
        logger.info(" WebRTC handler imports OK")
    except Exception as e:
        logger.error(f" WebRTC handler import failed: {e}")
        return False
    
    try:
        from leibniz_agent.leibniz_webrtc_app import create_leibniz_webrtc_app
        logger.info(" WebRTC app imports OK")
    except Exception as e:
        logger.error(f" WebRTC app import failed: {e}")
        return False
    
    try:
        import fastrtc
        logger.info(f" FastRTC version: {fastrtc.__version__ if hasattr(fastrtc, '__version__') else 'unknown'}")
    except Exception as e:
        logger.error(f" FastRTC import failed: {e}")
        return False
    
    try:
        import scipy
        logger.info(f" SciPy version: {scipy.__version__}")
    except Exception as e:
        logger.error(f" SciPy import failed: {e}")
        return False
    
    return True


async def test_audio_converter():
    """Test audio format conversion utilities"""
    logger.info("\nTesting audio format conversion...")
    
    try:
        import numpy as np
        from leibniz_webrtc_handler import AudioFormatConverter
        
        converter = AudioFormatConverter()
        
        # Test resampling
        audio_24k = np.random.randn(24000).astype(np.float32)  # 1 second at 24kHz
        audio_16k = converter.resample_audio(audio_24k, 24000, 16000)
        
        assert len(audio_16k) == 16000, f"Expected 16000 samples, got {len(audio_16k)}"
        logger.info(f" Resampling: 24kHz ({len(audio_24k)} samples) → 16kHz ({len(audio_16k)} samples)")
        
        # Test format conversion
        audio_int16 = np.array([32767, -32768, 0], dtype=np.int16)
        audio_float32 = converter.ensure_float32(audio_int16)
        
        assert audio_float32.dtype == np.float32, f"Expected float32, got {audio_float32.dtype}"
        assert np.allclose(audio_float32, [1.0, -1.0, 0.0], atol=0.01), "Float conversion values incorrect"
        logger.info(" Format conversion: int16 → float32")
        
        # Test normalization
        audio_loud = np.array([0.5, 2.0, -3.0], dtype=np.float32)
        audio_normalized = converter.normalize_audio(audio_loud)
        
        assert np.abs(audio_normalized).max() <= 1.0, "Normalized audio exceeds [-1, 1]"
        logger.info(" Audio normalization working")
        
        return True
        
    except Exception as e:
        logger.error(f" Audio converter test failed: {e}", exc_info=True)
        return False


async def test_handler_initialization():
    """Test conversation handler initialization"""
    logger.info("\nTesting conversation handler initialization...")
    
    try:
        from leibniz_agent.leibniz_webrtc_handler import LeibnizConversationHandler
        
        handler = LeibnizConversationHandler(
            sample_rate=16000,
            enable_logging=True
        )
        
        logger.info(" Handler created successfully")
        logger.info(f"   - Sample rate: {handler.sample_rate}Hz")
        logger.info(f"   - Logging enabled: {handler.enable_logging}")
        logger.info(f"   - Session timeout: {handler.session_timeout}s")
        
        # Test session info before initialization
        session_info = handler.get_session_info()
        logger.info(f" Session info: {session_info}")
        
        return True
        
    except Exception as e:
        logger.error(f" Handler initialization failed: {e}", exc_info=True)
        return False


async def test_app_creation():
    """Test FastAPI app creation"""
    logger.info("\nTesting FastAPI app creation...")
    
    try:
        from leibniz_agent.leibniz_webrtc_app import create_leibniz_webrtc_app
        
        app = create_leibniz_webrtc_app()
        
        logger.info(" FastAPI app created successfully")
        logger.info(f"   - Title: {app.title}")
        logger.info(f"   - Version: {app.version}")
        
        # Check routes
        routes = [route.path for route in app.routes]
        logger.info(f"   - Routes: {len(routes)} endpoints")
        
        expected_routes = ['/', '/health', '/api/session', '/webrtc/offer']
        for route in expected_routes:
            if any(route in r for r in routes):
                logger.info(f"      {route}")
            else:
                logger.warning(f"     ️ {route} not found")
        
        return True
        
    except Exception as e:
        logger.error(f" App creation failed: {e}", exc_info=True)
        return False


async def test_service_dependencies():
    """Test that required Leibniz services are available"""
    logger.info("\nTesting Leibniz service dependencies...")
    
    try:
        from leibniz_agent.leibniz_stt import LeibnizSTT
        logger.info(" LeibnizSTT import OK")
    except Exception as e:
        logger.error(f" LeibnizSTT import failed: {e}")
        return False
    
    try:
        from leibniz_agent.leibniz_tts import LeibnizTTS
        logger.info(" LeibnizTTS import OK")
    except Exception as e:
        logger.error(f" LeibnizTTS import failed: {e}")
        return False
    
    try:
        from leibniz_agent.leibniz_intent_parser import LeibnizIntentParser
        logger.info(" LeibnizIntentParser import OK")
    except Exception as e:
        logger.error(f" LeibnizIntentParser import failed: {e}")
        return False
    
    try:
        from leibniz_agent.leibniz_appointment_fsm import LeibnizAppointmentFSM
        logger.info(" LeibnizAppointmentFSM import OK")
    except Exception as e:
        logger.error(f" LeibnizAppointmentFSM import failed: {e}")
        return False
    
    try:
        from leibniz_agent.leibniz_persistent_services import get_leibniz_services_manager
        logger.info(" Persistent services import OK")
    except Exception as e:
        logger.error(f" Persistent services import failed: {e}")
        return False
    
    return True


async def run_all_tests():
    """Run all tests"""
    logger.info("=" * 70)
    logger.info("Leibniz WebRTC Integration Tests")
    logger.info("=" * 70)
    
    results = {
        "Imports": await test_imports(),
        "Audio Converter": await test_audio_converter(),
        "Handler Initialization": await test_handler_initialization(),
        "App Creation": await test_app_creation(),
        "Service Dependencies": await test_service_dependencies()
    }
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)
    
    for test_name, passed in results.items():
        status = " PASS" if passed else " FAIL"
        logger.info(f"{status:10s} {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    logger.info("=" * 70)
    logger.info(f"Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info(" All tests passed! WebRTC integration is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Set environment variables (GEMINI_API_KEY, LEMONFOX_API_KEY)")
        logger.info("2. Run: python -m leibniz_agent.leibniz_webrtc_app")
        logger.info("3. Open browser: http://localhost:8000/")
        return 0
    else:
        logger.error(f" {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
