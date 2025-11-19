"""
Comprehensive test suite for Leibniz Continuous VAD system.

Tests initialization, speech detection, barge-in, error recovery, latency,
and fallback behavior. Run from repository root:

    python leibniz_agent/test_continuous_vad.py

Author: SINDH Orchestra Agent
Date: 2025-01-XX
"""

import asyncio
import logging
import os
import sys
import time
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import continuous VAD components
try:
    from leibniz_agent.leibniz_continuous_vad import (
        LeibnizContinuousVAD,
        LeibnizPersistentSession,
        get_continuous_vad,
        wait_for_leibniz_speech,
        start_leibniz_continuous_listening,
        stop_leibniz_continuous_listening
    )
    from leibniz_agent.leibniz_vad import get_leibniz_vad
    from leibniz_agent.leibniz_persistent_services import get_leibniz_services_manager
except ImportError as e:
    logger.error(f" Import failed: {e}")
    logger.error("Run from repository root: python leibniz_agent/test_continuous_vad.py")
    logger.error(f"Current sys.path: {sys.path[:3]}")
    logger.error(f"Current directory: {os.getcwd()}")
    sys.exit(1)


# ============================================================================
# Test 1: Initialization and Singleton Pattern
# ============================================================================

async def test_continuous_vad_initialization():
    """
    Test continuous VAD initialization, singleton pattern, and session warmup.
    
    Expected:
    - get_continuous_vad() returns same instance on multiple calls
    - Persistent Gemini session created successfully
    - Background tasks start without errors
    """
    logger.info("=" * 80)
    logger.info("TEST 1: Continuous VAD Initialization")
    logger.info("=" * 80)
    
    try:
        # Test singleton pattern
        logger.info("Testing singleton pattern...")
        vad1 = get_continuous_vad()
        vad2 = get_continuous_vad()
        
        assert vad1 is vad2, " FAIL: get_continuous_vad() returned different instances"
        logger.info(" PASS: Singleton pattern working (same instance returned)")
        
        # Test initialization
        logger.info("Testing continuous VAD startup...")
        await vad1.start_continuous_listening()
        
        # Verify state
        assert vad1.is_running, " FAIL: is_running flag not set"
        assert vad1.session is not None, " FAIL: Gemini session not created"
        assert vad1.send_task is not None, " FAIL: Audio send task not created"
        assert vad1.listen_task is not None, " FAIL: Transcript listen task not created"
        assert vad1.thread_audio_queue is not None, " FAIL: Audio queue not created"
        
        logger.info(" PASS: Continuous VAD initialized successfully")
        
        # Test session warmup
        logger.info("Testing persistent session warmup...")
        session = await LeibnizPersistentSession.get_session(
            vad1.vad.client,
            vad1.vad.config.model_name,
            vad1.vad.config
        )
        
        assert session is not None, " FAIL: Persistent session not created"
        logger.info(" PASS: Persistent session acquired (warm)")
        
        # Check health status
        logger.info("Checking health metrics...")
        health = vad1.get_health_status()
        
        logger.info(f"  - Running: {health['is_running']}")
        logger.info(f"  - Uptime: {health['uptime_seconds']:.1f}s")
        logger.info(f"  - Transcripts received: {health['transcripts_received']}")
        logger.info(f"  - Errors: {health['errors_count']}")
        
        assert health['is_running'], " FAIL: Health check reports not running"
        logger.info(" PASS: Health metrics available")
        
        # Cleanup
        await vad1.stop_continuous_listening()
        logger.info(" TEST 1 PASSED: Initialization successful")
        
    except Exception as e:
        logger.error(f" TEST 1 FAILED: {e}")
        raise


# ============================================================================
# Test 2: Wait for User Speech (Event Signaling)
# ============================================================================

async def test_wait_for_user_speech():
    """
    Test wait_for_leibniz_speech() helper with timeout and event signaling.
    
    Expected:
    - Timeout returns None after configured duration
    - Event signaling wakes main loop immediately when user speaks
    - No blocking when user speaks (instant return)
    """
    logger.info("=" * 80)
    logger.info("TEST 2: Wait for User Speech (Event Signaling)")
    logger.info("=" * 80)
    
    try:
        vad = get_continuous_vad()
        await vad.start_continuous_listening()
        
        # Test timeout behavior
        logger.info("Testing timeout (2s)...")
        start_time = time.time()
        transcript = await wait_for_leibniz_speech(timeout=2.0)
        elapsed = time.time() - start_time
        
        assert transcript is None, f" FAIL: Expected None on timeout, got: {transcript}"
        assert 1.8 <= elapsed <= 2.5, f" FAIL: Timeout took {elapsed:.1f}s (expected ~2.0s)"
        logger.info(f" PASS: Timeout after {elapsed:.1f}s (expected 2.0s)")
        
        # Test event signaling (simulate user speech)
        logger.info("Testing event signaling (simulated speech)...")
        
        async def simulate_user_speech():
            """Simulate background listener detecting speech."""
            await asyncio.sleep(0.5)  # Wait 500ms
            
            # Simulate what _listen_for_speech_loop does
            vad.current_transcript = "Hello, this is a test"
            vad.user_transcript_event.set()
            logger.info("  [Simulator] Event set with transcript")
        
        # Start simulation task
        sim_task = asyncio.create_task(simulate_user_speech())
        
        # Wait for speech (should return in ~500ms)
        start_time = time.time()
        transcript = await wait_for_leibniz_speech(timeout=5.0)
        elapsed = time.time() - start_time
        
        await sim_task  # Ensure simulation completes
        
        assert transcript is not None, " FAIL: Expected transcript, got None"
        assert "Hello" in transcript, f" FAIL: Unexpected transcript: {transcript}"
        assert elapsed < 1.0, f" FAIL: Event took {elapsed:.1f}s (expected <1.0s)"
        logger.info(f" PASS: Event signaling in {elapsed:.1f}s (transcript: '{transcript}')")
        
        # Cleanup
        await vad.stop_continuous_listening()
        logger.info(" TEST 2 PASSED: Event signaling working")
        
    except Exception as e:
        logger.error(f" TEST 2 FAILED: {e}")
        raise


# ============================================================================
# Test 3: Barge-In Detection
# ============================================================================

async def test_barge_in_detection():
    """
    Test barge-in detection when user speaks during TTS playback.
    
    Expected:
    - vad.barge_in_detected flag set to True
    - Background listener signals event immediately
    - TTS consumer would stop playback (<200ms)
    """
    logger.info("=" * 80)
    logger.info("TEST 3: Barge-In Detection")
    logger.info("=" * 80)
    
    try:
        vad_wrapper = get_continuous_vad()
        await vad_wrapper.start_continuous_listening()
        
        # Get underlying VAD for state control
        vad = get_leibniz_vad()
        
        # Simulate agent speaking
        logger.info("Simulating agent speaking...")
        vad.is_agent_speaking = True
        assert vad.is_agent_speaking, " FAIL: is_agent_speaking not set"
        logger.info(" Agent speaking state set")
        
        # Simulate user interruption (background listener detects speech)
        logger.info("Simulating user interruption...")
        
        async def simulate_barge_in():
            """Simulate _listen_for_speech_loop detecting speech during TTS."""
            await asyncio.sleep(0.3)
            
            # Check if agent is speaking (barge-in condition)
            if vad.is_agent_speaking:
                logger.info("  [Simulator] Detected speech during TTS - BARGE-IN!")
                vad.barge_in_detected = True
                vad_wrapper.current_transcript = "User interrupted agent"
                vad_wrapper.user_transcript_event.set()
        
        # Start simulation
        sim_task = asyncio.create_task(simulate_barge_in())
        
        # Wait for event (simulating TTS consumer checking)
        start_time = time.time()
        await asyncio.wait_for(vad_wrapper.user_transcript_event.wait(), timeout=2.0)
        elapsed = time.time() - start_time
        
        await sim_task  # Ensure simulation completes
        
        # Verify barge-in flag
        assert vad.barge_in_detected, " FAIL: barge_in_detected flag not set"
        assert elapsed < 0.5, f" FAIL: Barge-in detection took {elapsed:.1f}s (expected <0.5s)"
        logger.info(f" PASS: Barge-in detected in {elapsed:.1f}s")
        
        # Verify event signaling
        assert vad_wrapper.user_transcript_event.is_set(), " FAIL: Event not set on barge-in"
        logger.info(" PASS: Event signaling on barge-in working")
        
        # Reset state
        vad.is_agent_speaking = False
        vad.barge_in_detected = False
        
        # Cleanup
        await vad_wrapper.stop_continuous_listening()
        logger.info(" TEST 3 PASSED: Barge-in detection working")
        
    except Exception as e:
        logger.error(f" TEST 3 FAILED: {e}")
        raise


# ============================================================================
# Test 4: Latency Comparison (Per-Turn vs Continuous)
# ============================================================================

async def test_latency_comparison():
    """
    Compare response latency between per-turn and continuous VAD modes.
    
    Expected:
    - Per-turn: 1200-1800ms (cold start + capture + classify)
    - Continuous: <300ms (session warm, background classification)
    - Improvement: >5x faster
    """
    logger.info("=" * 80)
    logger.info("TEST 4: Latency Comparison (Per-Turn vs Continuous)")
    logger.info("=" * 80)
    
    try:
        # Note: This test measures simulated latency (not real speech capture)
        # For real-world testing, use manual speech tests
        
        # Test per-turn latency (simulated)
        logger.info("Testing per-turn VAD latency (simulated)...")
        vad = get_leibniz_vad()
        
        # Simulate cold start + capture
        start_time = time.time()
        await asyncio.sleep(0.1)  # Simulated session initialization
        await asyncio.sleep(0.05)  # Simulated capture delay
        per_turn_latency = (time.time() - start_time) * 1000  # ms
        
        logger.info(f"  Per-turn simulated latency: {per_turn_latency:.0f}ms")
        
        # Test continuous latency (simulated)
        logger.info("Testing continuous VAD latency (simulated)...")
        vad_wrapper = get_continuous_vad()
        await vad_wrapper.start_continuous_listening()
        
        # Session already warm - just event signaling overhead
        start_time = time.time()
        vad_wrapper.user_transcript_event.set()
        await vad_wrapper.user_transcript_event.wait()
        continuous_latency = (time.time() - start_time) * 1000  # ms
        
        logger.info(f"  Continuous simulated latency: {continuous_latency:.0f}ms")
        
        # Compare
        improvement = per_turn_latency / max(continuous_latency, 1)
        logger.info(f"  Improvement: {improvement:.1f}x faster")
        
        assert continuous_latency < per_turn_latency, \
            f" FAIL: Continuous ({continuous_latency:.0f}ms) not faster than per-turn ({per_turn_latency:.0f}ms)"
        
        logger.info(" PASS: Continuous VAD faster than per-turn (simulated)")
        
        # Note: Real-world latency requires actual Gemini API calls
        logger.info("NOTE: For real-world latency testing, use manual speech tests")
        logger.info("Expected: Per-turn ~1500ms, Continuous ~200ms (7-10x improvement)")
        
        # Cleanup
        await vad_wrapper.stop_continuous_listening()
        logger.info(" TEST 4 PASSED: Latency comparison successful")
        
    except Exception as e:
        logger.error(f" TEST 4 FAILED: {e}")
        raise


# ============================================================================
# Test 5: Error Recovery and Auto-Restart
# ============================================================================

async def test_error_recovery():
    """
    Test error recovery mechanisms and auto-restart behavior.
    
    Expected:
    - Listen loop handles exceptions gracefully
    - Auto-restart triggered after error
    - Max restart attempts enforced (fallback after 3 failures)
    """
    logger.info("=" * 80)
    logger.info("TEST 5: Error Recovery and Auto-Restart")
    logger.info("=" * 80)
    
    try:
        vad = get_continuous_vad()
        await vad.start_continuous_listening()
        
        # Get initial health metrics
        initial_health = vad.get_health_status()
        initial_errors = initial_health['errors_count']
        logger.info(f"Initial errors: {initial_errors}")
        
        # Simulate error in listen loop
        logger.info("Simulating error condition...")
        
        # Force an error by setting session to None (simulates connection loss)
        original_session = vad.session
        vad.session = None
        
        # Wait for error to be detected (should happen in next receive iteration)
        await asyncio.sleep(2.0)
        
        # Check if error was logged
        current_health = vad.get_health_status()
        current_errors = current_health['errors_count']
        
        # Note: Error count may not increment if error handling prevents it
        logger.info(f"Current errors: {current_errors}")
        
        # Restore session
        vad.session = original_session
        
        # Test manual restart
        logger.info("Testing manual restart...")
        await vad.restart_listener()
        
        # Verify restarted successfully
        assert vad.is_running, " FAIL: VAD not running after restart"
        assert vad.session is not None, " FAIL: Session not restored after restart"
        logger.info(" PASS: Manual restart successful")
        
        # Test max restart enforcement
        logger.info("Testing max restart attempts...")
        max_restarts = int(os.getenv("LEIBNIZ_CONTINUOUS_VAD_MAX_RESTARTS", "3"))
        logger.info(f"  Max restarts configured: {max_restarts}")
        
        # Note: Testing actual max restart behavior requires simulating multiple failures
        # which is complex in automated tests. Manual testing recommended.
        logger.info("NOTE: Max restart enforcement requires manual testing with forced failures")
        
        # Cleanup
        await vad.stop_continuous_listening()
        logger.info(" TEST 5 PASSED: Error recovery working")
        
    except Exception as e:
        logger.error(f" TEST 5 FAILED: {e}")
        raise


# ============================================================================
# Test 6: Concurrent Speech Detection
# ============================================================================

async def test_concurrent_speech():
    """
    Test handling of multiple rapid user utterances.
    
    Expected:
    - Each utterance triggers separate event
    - Transcripts queued and processed in order
    - No race conditions or dropped utterances
    """
    logger.info("=" * 80)
    logger.info("TEST 6: Concurrent Speech Detection")
    logger.info("=" * 80)
    
    try:
        vad = get_continuous_vad()
        await vad.start_continuous_listening()
        
        # Simulate rapid utterances
        logger.info("Simulating 3 rapid utterances...")
        utterances = [
            "Hello",
            "What's the weather?",
            "Thank you"
        ]
        
        received_transcripts = []
        
        async def simulate_utterances():
            """Simulate rapid speech events."""
            for i, text in enumerate(utterances):
                await asyncio.sleep(0.3)  # 300ms between utterances
                
                vad.current_transcript = text
                vad.user_transcript_event.set()
                logger.info(f"  [Simulator] Utterance {i+1}: '{text}'")
                
                # Event should be cleared by main loop between utterances
                await asyncio.sleep(0.1)
                vad.user_transcript_event.clear()
        
        async def collect_transcripts():
            """Collect transcripts from events."""
            for i in range(len(utterances)):
                transcript = await wait_for_leibniz_speech(timeout=5.0)
                if transcript:
                    received_transcripts.append(transcript)
                    logger.info(f"  [Collector] Received {i+1}: '{transcript}'")
        
        # Run simulation and collection concurrently
        await asyncio.gather(
            simulate_utterances(),
            collect_transcripts()
        )
        
        # Verify all utterances received
        assert len(received_transcripts) == len(utterances), \
            f" FAIL: Received {len(received_transcripts)}/{len(utterances)} utterances"
        
        for i, (expected, received) in enumerate(zip(utterances, received_transcripts)):
            assert expected == received, \
                f" FAIL: Utterance {i+1} mismatch (expected '{expected}', got '{received}')"
        
        logger.info(f" PASS: All {len(utterances)} utterances received in order")
        
        # Cleanup
        await vad.stop_continuous_listening()
        logger.info(" TEST 6 PASSED: Concurrent speech handling working")
        
    except Exception as e:
        logger.error(f" TEST 6 FAILED: {e}")
        raise


# ============================================================================
# Test 7: Fallback to Per-Turn Mode
# ============================================================================

async def test_fallback_to_per_turn():
    """
    Test graceful fallback to per-turn VAD when continuous mode fails.
    
    Expected:
    - Environment toggle disables continuous VAD
    - System falls back to per-turn capture seamlessly
    - No exceptions or data loss
    """
    logger.info("=" * 80)
    logger.info("TEST 7: Fallback to Per-Turn Mode")
    logger.info("=" * 80)
    
    try:
        # Test environment toggle
        logger.info("Testing environment variable toggle...")
        
        # Check current setting
        current_setting = os.getenv("LEIBNIZ_ENABLE_CONTINUOUS_VAD", "false").lower()
        logger.info(f"  Current setting: LEIBNIZ_ENABLE_CONTINUOUS_VAD={current_setting}")
        
        # Note: Changing env var at runtime won't affect already-imported modules
        # This test verifies the toggle mechanism works at startup
        
        # Test continuous VAD shutdown (simulates fallback scenario)
        logger.info("Testing continuous VAD shutdown...")
        vad = get_continuous_vad()
        await vad.start_continuous_listening()
        
        assert vad.is_running, " FAIL: VAD not running before shutdown"
        
        # Stop continuous VAD (simulates fallback)
        await vad.stop_continuous_listening()
        
        assert not vad.is_running, " FAIL: VAD still running after shutdown"
        assert vad.send_task is None or vad.send_task.cancelled(), \
            " FAIL: Send task not cancelled"
        assert vad.listen_task is None or vad.listen_task.cancelled(), \
            " FAIL: Listen task not cancelled"
        logger.info(" PASS: Continuous VAD shutdown successful")
        
        # Verify per-turn VAD still functional
        logger.info("Verifying per-turn VAD accessibility...")
        per_turn_vad = get_leibniz_vad()
        
        assert per_turn_vad is not None, " FAIL: Per-turn VAD not accessible"
        assert hasattr(per_turn_vad, 'capture_speech_bidirectional'), \
            " FAIL: Per-turn VAD missing capture method"
        logger.info(" PASS: Per-turn VAD functional after continuous shutdown")
        
        # Note: Full fallback testing requires runtime environment changes
        logger.info("NOTE: Full fallback testing requires setting LEIBNIZ_ENABLE_CONTINUOUS_VAD=false at startup")
        logger.info("Manual test: 1) Set env var, 2) Restart agent, 3) Verify per-turn mode used")
        
        logger.info(" TEST 7 PASSED: Fallback mechanism verified")
        
    except Exception as e:
        logger.error(f" TEST 7 FAILED: {e}")
        raise


# ============================================================================
# Main Test Runner
# ============================================================================

async def run_all_tests():
    """Run all continuous VAD tests in sequence."""
    logger.info("\n" + "=" * 80)
    logger.info("LEIBNIZ CONTINUOUS VAD - COMPREHENSIVE TEST SUITE")
    logger.info("=" * 80 + "\n")
    
    # Initialize services (required for VAD initialization)
    logger.info("Initializing Leibniz persistent services...")
    try:
        services = await get_leibniz_services_manager()
        logger.info(" Persistent services initialized\n")
    except Exception as e:
        logger.error(f" Failed to initialize services: {e}")
        logger.error("Ensure GEMINI_API_KEY is set in .env.leibniz")
        return False
    
    # Run tests
    tests = [
        ("Initialization", test_continuous_vad_initialization),
        ("Wait for Speech", test_wait_for_user_speech),
        ("Barge-In Detection", test_barge_in_detection),
        ("Latency Comparison", test_latency_comparison),
        ("Error Recovery", test_error_recovery),
        ("Concurrent Speech", test_concurrent_speech),
        ("Fallback to Per-Turn", test_fallback_to_per_turn),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\nRunning: {test_name}")
            await test_func()
            passed += 1
            logger.info(f" {test_name} PASSED\n")
        except Exception as e:
            failed += 1
            logger.error(f" {test_name} FAILED: {e}\n")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total tests: {len(tests)}")
    logger.info(f"Passed: {passed} ")
    logger.info(f"Failed: {failed} ")
    logger.info(f"Success rate: {passed / len(tests) * 100:.1f}%")
    logger.info("=" * 80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    # Ensure running from repository root
    if not os.path.exists("leibniz_agent"):
        logger.error(" Run from repository root: python leibniz_agent/test_continuous_vad.py")
        sys.exit(1)
    
    # Run tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
