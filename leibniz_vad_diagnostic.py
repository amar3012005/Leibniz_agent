#!/usr/bin/env python3
"""
Leibniz VAD/STT Diagnostic Test Script
======================================

Comprehensive test suite to diagnose speech detection issues.
Tests microphone, audio streaming, Gemini session, and real-time transcription.

Author: Diagnostic Suite
Version: 1.0
"""

import os
import sys
import asyncio
import time
import logging
from pathlib import Path

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging (minimal output to file only)
logging.basicConfig(
    level=logging.WARNING,  # Only warnings and errors
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vad_diagnostic.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

print("="*80)
print("LEIBNIZ VAD/STT DIAGNOSTIC TEST SUITE")
print("="*80)
print()

# =============================================================================
# TEST 1: Environment & Dependencies
# =============================================================================
async def test_environment():
    """Test 1: Verify environment and dependencies."""
    print("\n" + "="*80)
    print("TEST 1: Environment & Dependencies")
    print("="*80)

    tests_passed = 0
    tests_total = 5

    # Check .env file (in parent directory or current)
    print("\n[1.1] Checking .env file...")
    env_paths = [
        Path('.env'),
        Path('../.env'),
        Path('.env.leibniz'),
        Path('../.env.leibniz')
    ]
    env_found = False
    for env_path in env_paths:
        if env_path.exists():
            print(f"   .env file found: {env_path}")
            env_found = True
            tests_passed += 1
            break
    
    if not env_found:
        print("  ️  .env file NOT found in current or parent directory")
        print("     (Checking if GEMINI_API_KEY is in environment anyway...)")

    # Check GEMINI_API_KEY
    print("\n[1.2] Checking GEMINI_API_KEY...")
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"   GEMINI_API_KEY found: {api_key[:8]}...{api_key[-4:]}")
        tests_passed += 1
    else:
        print("   GEMINI_API_KEY NOT found")

    # Check google-genai (NOT google.generativeai)
    print("\n[1.3] Checking google-genai...")
    try:
        from google import genai
        print(f"   google-genai imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"   google-genai import failed: {e}")
        print(f"     Install with: pip install google-genai>=1.33.0")

    # Check pyaudio
    print("\n[1.4] Checking pyaudio...")
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        print(f"   PyAudio version: {pyaudio.__version__}")
        p.terminate()
        tests_passed += 1
    except Exception as e:
        print(f"   PyAudio check failed: {e}")

    # Check sounddevice
    print("\n[1.5] Checking sounddevice...")
    try:
        import sounddevice as sd
        print(f"   sounddevice available")
        tests_passed += 1
    except ImportError as e:
        print(f"   sounddevice import failed: {e}")

    print(f"\n Test 1 Result: {tests_passed}/{tests_total} checks passed")
    return tests_passed == tests_total


# =============================================================================
# TEST 2: Microphone Hardware
# =============================================================================
async def test_microphone_hardware():
    """Test 2: Test microphone hardware independently."""
    print("\n" + "="*80)
    print("TEST 2: Microphone Hardware Test")
    print("="*80)

    try:
        import pyaudio
        import numpy as np

        print("\n[2.1] Listing audio devices...")
        p = pyaudio.PyAudio()

        input_devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_devices.append((i, info))
                print(f"  [{i}] {info['name']}")
                print(f"      Input channels: {info['maxInputChannels']}")
                print(f"      Sample rate: {info['defaultSampleRate']}")

        if not input_devices:
            print("   No input devices found!")
            p.terminate()
            return False

        # Use default input device
        default_input = p.get_default_input_device_info()
        print(f"\n   Default input device: {default_input['name']}")

        # Test microphone for 5 seconds
        print("\n[2.2] Testing microphone (speak for 5 seconds)...")
        print("   Recording... SPEAK NOW!")

        SAMPLE_RATE = 16000
        CHUNK = 1024
        DURATION = 5

        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        max_volume = 0
        avg_volume = 0
        frame_count = 0
        volumes = []

        for i in range(0, int(SAMPLE_RATE / CHUNK * DURATION)):
            data = stream.read(CHUNK)
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            max_volume = max(max_volume, volume)
            avg_volume += volume
            frame_count += 1
            volumes.append(volume)

            # Real-time volume display
            bars = int(volume / 100)
            bar_display = "█" * bars
            print(f"  Volume: {volume:>6.0f} {bar_display}", end='\r')

        stream.stop_stream()
        stream.close()
        p.terminate()

        avg_volume = avg_volume / frame_count if frame_count > 0 else 0

        print("\n")
        print(f"   Max volume: {max_volume:.0f}")
        print(f"   Avg volume: {avg_volume:.0f}")

        # Analyze results
        if max_volume < 50:
            print("   CRITICAL: Volume too low! Check:")
            print("     - Microphone is connected")
            print("     - Microphone is not muted")
            print("     - Windows microphone permissions")
            print("     - Microphone volume in Windows settings")
            return False
        elif max_volume < 200:
            print("  ️  WARNING: Volume is low but usable")
            print("     Consider increasing microphone volume")
            return True
        else:
            print("   Microphone working well!")
            return True

    except Exception as e:
        print(f"   Microphone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 3: Gemini API Connection
# =============================================================================
async def test_gemini_connection():
    """Test 3: Test Gemini API connection and authentication."""
    print("\n" + "="*80)
    print("TEST 3: Gemini API Connection")
    print("="*80)

    try:
        from google import genai
        from google.genai import types
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')

        if not api_key:
            print("   GEMINI_API_KEY not found")
            return False

        print("\n[3.1] Initializing Gemini client...")
        client = genai.Client(api_key=api_key)
        print("   Client initialized")

        print("\n[3.2] Testing Live API session...")
        try:
            config = types.LiveConnectConfig(
                response_modalities=["TEXT"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Puck"
                        )
                    )
                )
            )
            
            # SINDH PATTERN: Use context manager protocol correctly
            session_context = client.aio.live.connect(
                model='models/gemini-2.0-flash-exp',
                config=config
            )
            session = await session_context.__aenter__()
            print("   Live API session created successfully")

            # Close session using SINDH pattern
            await session_context.__aexit__(None, None, None)
            print("   Session closed successfully")
            return True

        except Exception as e:
            print(f"   Live API session failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"   Gemini connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 4: Audio Streaming to Gemini
# =============================================================================
async def test_audio_streaming():
    """Test 4: Test audio streaming to Gemini with real-time transcript."""
    print("\n" + "="*80)
    print("TEST 4: Audio Streaming to Gemini (Real-Time STT)")
    print("="*80)

    try:
        from google import genai
        from google.genai import types
        import sounddevice as sd
        import numpy as np
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')

        print("\n[4.1] Initializing Gemini Live session...")
        client = genai.Client(api_key=api_key)

        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Puck"
                    )
                )
            )
        )
        
        # SINDH PATTERN: Store context manager and enter it
        session_context = client.aio.live.connect(
            model='models/gemini-2.0-flash-exp',
            config={
                "response_modalities": ["TEXT"],
                "input_audio_transcription": {}  # Enable transcription like SINDH
            }
        )
        session = await session_context.__aenter__()
        print("   Session connected")

        print("\n[4.2] Starting audio stream (speak for 10 seconds)...")
        print("   Listening... SPEAK NOW!")
        print("   Real-time transcripts will appear below:")
        print("  " + "-"*70)

        SAMPLE_RATE = 16000
        DURATION = 10

        audio_queue = asyncio.Queue(maxsize=50)
        stream_active = True
        transcript_received = False
        packet_count = 0

        # Audio callback
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"  ️  Audio status: {status}")

            # Convert to int16 PCM
            audio_data = (indata.copy() * 32767).astype(np.int16).tobytes()

            try:
                audio_queue.put_nowait(audio_data)
            except asyncio.QueueFull:
                pass  # Silently drop frames if queue is full

        # Stream audio to Gemini (SINDH-EXACT API)
        async def stream_audio():
            nonlocal stream_active, packet_count
            try:
                while stream_active:
                    try:
                        audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                        
                        # Use SINDH-exact API: send_realtime_input with types.Blob
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=audio_data,
                                mime_type=f"audio/pcm;rate={SAMPLE_RATE}"
                            )
                        )
                        packet_count += 1

                    except asyncio.TimeoutError:
                        continue
            except Exception as e:
                logger.error(f"Stream audio error: {e}")

        # Process transcripts (SINDH-EXACT API)
        async def process_transcripts():
            nonlocal stream_active, transcript_received
            try:
                async for response in session.receive():
                    # Check for input_transcription (SINDH-exact field)
                    if response.server_content and response.server_content.input_transcription:
                        text = response.server_content.input_transcription.text
                        if text and text.strip():
                            if not transcript_received:
                                print("\n  ️  Speech detected!")
                                transcript_received = True
                            print(f"   {text.strip()}")

            except Exception as e:
                logger.error(f"Process transcripts error: {e}")

        # Timeout manager
        async def manage_timeout():
            nonlocal stream_active
            await asyncio.sleep(DURATION)
            stream_active = False

        # Start all tasks
        with sd.InputStream(
            callback=audio_callback,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=int(SAMPLE_RATE * 0.1)  # 100ms chunks
        ):
            tasks = [
                asyncio.create_task(stream_audio()),
                asyncio.create_task(process_transcripts()),
                asyncio.create_task(manage_timeout())
            ]

            # Wait for all tasks
            await asyncio.gather(*tasks, return_exceptions=True)

        print()

        # Close session using SINDH pattern
        await session_context.__aexit__(None, None, None)

        if transcript_received:
            print("   SUCCESS: Transcripts received from Gemini!")
            return True
        else:
            print("   FAILURE: No transcripts received")
            print("     Possible issues:")
            print("     - No speech detected (speak louder)")
            print("     - API not transcribing (check quota)")
            print("     - Session configuration issue")
            return False

    except Exception as e:
        print(f"   Audio streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 5: Leibniz VAD Integration
# =============================================================================
async def test_leibniz_vad():
    """Test 5: Standalone Leibniz-style VAD using SINDH pattern (no module import)."""
    print("\n" + "="*80)
    print("TEST 5: Leibniz VAD Integration Test (SINDH Pattern)")
    print("="*80)

    try:
        from google import genai
        from google.genai import types
        import sounddevice as sd
        import numpy as np
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')

        print("\n[5.1] Creating standalone Leibniz-style VAD session...")
        print("  ℹ️  Note: Using SINDH pattern directly (no leibniz_vad module)")
        client = genai.Client(api_key=api_key)

        # SINDH PATTERN: Create persistent session with English config
        session_context = client.aio.live.connect(
            model='models/gemini-2.0-flash-exp',
            config={
                "response_modalities": ["TEXT"],
                "input_audio_transcription": {},  # Enable transcription
                "speech_config": {
                    "language_code": "en-US"  # English for Leibniz
                }
            }
        )
        session = await session_context.__aenter__()
        print("   Leibniz VAD session connected (English, SINDH pattern)")

        print("\n[5.2] Testing speech capture (speak in ENGLISH for 15 seconds)...")
        print("   Listening... SPEAK NOW IN ENGLISH!")
        print("   Real-time transcripts will appear below:")
        print("  " + "-"*70)

        SAMPLE_RATE = 16000
        DURATION = 15  # Leibniz uses longer timeout

        audio_queue = asyncio.Queue(maxsize=50)
        stream_active = True
        transcripts = []
        packet_count = 0

        # Audio callback (SINDH pattern)
        def audio_callback(indata, frames, time_info, status):
            audio_data = (indata.copy() * 32767).astype(np.int16).tobytes()
            try:
                audio_queue.put_nowait(audio_data)
            except asyncio.QueueFull:
                pass

        # Stream audio (SINDH pattern)
        async def stream_audio():
            nonlocal stream_active, packet_count
            while stream_active:
                try:
                    audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    await session.send_realtime_input(
                        audio=types.Blob(data=audio_data, mime_type=f"audio/pcm;rate={SAMPLE_RATE}")
                    )
                    packet_count += 1
                except asyncio.TimeoutError:
                    continue

        # Process transcripts (SINDH pattern)
        async def process_transcripts():
            async for response in session.receive():
                if response.server_content and response.server_content.input_transcription:
                    text = response.server_content.input_transcription.text
                    if text and text.strip():
                        if not transcripts:
                            print("\n  ️  Speech detected!")
                        transcripts.append(text.strip())
                        print(f"   {text.strip()}")

        # Timeout manager
        async def manage_timeout():
            nonlocal stream_active
            await asyncio.sleep(DURATION)
            stream_active = False

        # Run capture (SINDH pattern)
        with sd.InputStream(
            callback=audio_callback,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=int(SAMPLE_RATE * 0.1)
        ):
            await asyncio.gather(
                asyncio.create_task(stream_audio()),
                asyncio.create_task(process_transcripts()),
                asyncio.create_task(manage_timeout()),
                return_exceptions=True
            )

        print()

        # Close session (SINDH pattern)
        await session_context.__aexit__(None, None, None)

        if transcripts:
            final_transcript = ' '.join(transcripts)
            print(f"   SUCCESS: Leibniz VAD captured: {final_transcript}")
            print(f"   Total fragments: {len(transcripts)}")
            return True
        else:
            print("   FAILURE: No transcripts captured")
            print("     Speak in ENGLISH and louder")
            return False

        vad = get_leibniz_vad()
        print("   VAD instance created")

        print("\n[5.2] Enabling verbose logging...")
        vad.config.verbose = True
        vad.config.log_audio_callbacks = True
        vad.config.log_state_transitions = True
        print("   Verbose logging enabled")

        print("\n[5.3] Setting agent speaking state to False...")
        await vad.set_agent_speaking_state(False, 'test_init')
        print(f"   State: {vad.conversation_state}, Speaking: {vad.is_agent_speaking}")

        print("\n[5.4] Testing speech capture (speak for 10 seconds)...")
        print("   Listening... SPEAK NOW!")
        print("   Real-time fragments will appear below:")
        print("  " + "-"*70)

        fragments = []

        def streaming_callback(fragment: str, is_final: bool):
            fragments.append((fragment, is_final))
            status = "FINAL" if is_final else "partial"
            print(f"   [{status}] {fragment}")

        # Set timeout for test
        vad.set_dynamic_timeout(attempt_count=0, conversation_context="initial")

        transcript = await vad.capture_speech_bidirectional(
            streaming_callback=streaming_callback
        )

        print("\n")
        print(f"   Fragments received: {len(fragments)}")

        if transcript:
            print(f"   SUCCESS: Final transcript: {transcript}")
            
            # Get performance metrics
            metrics = vad.get_performance_metrics()
            print(f"  � Capture count: {metrics['capture_count']}")
            print(f"   Avg capture time: {metrics['avg_capture_time']:.2f}s")
            print(f"   Consecutive timeouts: {metrics['consecutive_timeouts']}")
            return True
        else:
            print("   FAILURE: No transcript captured")
            print(f"     should_accept_user_audio: {vad.should_accept_user_audio()}")
            print(f"     conversation_state: {vad.conversation_state}")
            print(f"     is_agent_speaking: {vad.is_agent_speaking}")
            return False

    except Exception as e:
        print(f"   Leibniz VAD test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main Test Runner
# =============================================================================
async def main():
    """Run all diagnostic tests."""
    print("\nStarting comprehensive diagnostic tests...")
    print("This will test microphone, Gemini API, and VAD integration.")
    print()

    results = {}

    # Run tests
    results['environment'] = await test_environment()

    if results['environment']:
        results['microphone'] = await test_microphone_hardware()
    else:
        print("\n️  Skipping microphone test due to environment issues")
        results['microphone'] = False

    if results['environment']:
        results['gemini'] = await test_gemini_connection()
    else:
        print("\n️  Skipping Gemini test due to environment issues")
        results['gemini'] = False

    if results['environment'] and results['microphone'] and results['gemini']:
        results['streaming'] = await test_audio_streaming()
    else:
        print("\n️  Skipping audio streaming test due to previous failures")
        results['streaming'] = False

    if results['streaming']:
        results['leibniz_vad'] = await test_leibniz_vad()
    else:
        print("\n️  Skipping Leibniz VAD test due to streaming failure")
        results['leibniz_vad'] = False

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC TEST SUMMARY")
    print("="*80)
    print()

    for test_name, passed in results.items():
        status = " PASS" if passed else " FAIL"
        print(f"  {status}  {test_name.replace('_', ' ').title()}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print()
    print(f"  Total: {total_passed}/{total_tests} tests passed")
    print()

    if all(results.values()):
        print("   ALL TESTS PASSED! Leibniz VAD should be working.")
    else:
        print("  ️  SOME TESTS FAILED. Review errors above.")
        print()
        print("  Common fixes:")
        if not results.get('microphone', True):
            print("  - Check microphone connection and permissions")
            print("  - Increase microphone volume in Windows settings")
        if not results.get('gemini', True):
            print("  - Verify GEMINI_API_KEY in .env file")
            print("  - Check internet connection")
        if not results.get('streaming', True):
            print("  - Speak louder and clearer")
            print("  - Check API quota limits")

    print()
    print("   Full log saved to: vad_diagnostic.log")
    print("="*80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n Fatal error: {e}")
        import traceback
        traceback.print_exc()
