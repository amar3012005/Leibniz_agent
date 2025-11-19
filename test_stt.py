#!/usr/bin/env python3
"""
Test Leibniz STT (Speech-to-Text) with Gemini Live Streaming

CORRECT USAGE:
  python -m leibniz_agent.test_stt  (from SINDH-Orchestra-Complete directory)
  OR
  python leibniz_agent/run_stt_test.py  (from SINDH-Orchestra-Complete directory)

WARNING: Do NOT run as:
  python test_stt.py  (from leibniz_agent/ directory)
  This causes ModuleNotFoundError for package imports

PREREQUISITES:
  - GEMINI_API_KEY set in .env.leibniz
  - AUDIO_INPUT_DEVICE (optional, defaults to system default)
  - Run device query to find correct microphone index
"""
import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from leibniz_agent.leibniz_stt import LeibnizSTT, LeibnizSTTConfig, get_leibniz_stt, leibniz_capture_audio
import sounddevice as sd


async def main():
    """Test STT with Gemini Live Streaming"""
    
    # Check if running correctly as module
    if __package__ is None:
        print("️  WARNING: This test may fail with import errors")
        print("   Recommended: Run as 'python -m leibniz_agent.test_stt' from parent directory")
        print("   Or use: python leibniz_agent/run_stt_test.py")
        print()
    
    print("="*70)
    print("Leibniz STT Test (Gemini Live Streaming)")
    print("="*70)
    
    # List available audio devices
    print("\n Available Audio Input Devices:")
    print("="*70)
    devices = sd.query_devices()
    
    # Comment 7: Guard against None in sd.default.device
    default_device = getattr(sd.default, 'device', (None, None))
    default_input_index = default_device[0] if default_device else None
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Only show input devices
            default_marker = " (DEFAULT)" if (default_input_index is not None and i == default_input_index) else ""
            print(f"  [{i}] {device['name']}{default_marker}")
            print(f"      Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']}")
    
    print("\n To use a specific device, set AUDIO_INPUT_DEVICE in .env.leibniz")
    print("   Example: AUDIO_INPUT_DEVICE=1")
    print("\n .env.leibniz file location:")
    env_path = Path(__file__).parent / ".env.leibniz"
    print(f"   {env_path}")
    
    device_config = os.getenv('AUDIO_INPUT_DEVICE', 'default')
    print(f"\nCurrent device setting: {device_config}")
    if device_config == 'default':
        print("️  WARNING: Using system default device")
        print("   If test fails with 'no speech detected', set AUDIO_INPUT_DEVICE explicitly")
        print("   Windows users: Run this command to list devices:")
        print('   python -c "import sounddevice as sd; print(sd.query_devices())"')
    print("="*70)
    
    # Initialize STT with config
    print("\n1. Initializing Leibniz STT...")
    config = LeibnizSTTConfig(
        model_name="gemini-live-2.5-flash-preview",  # Correct Gemini Live API model for real-time audio
        language_code="en-US",
        silence_timeout=2.0,
        start_timeout_s=12.0  # Allow more time for speech detection
    )
    
    stt = LeibnizSTT(config)  # Create instance with config
    
    print(f"   Model: {config.model_name}")
    print(f"   Language: {config.language_code}")
    print(f"   Silence timeout: {config.silence_timeout}s")
    print(f"   Start timeout: {config.start_timeout_s}s")
    
    # Test speech recognition
    print("\n2. Starting speech recognition test...")
    print("    Please speak now (will listen for speech)...")
    print("    Try saying: 'Hello, this is a test of the speech recognition system'")
    print("    The system will stop listening after {:.1f}s of silence".format(config.silence_timeout))
    print()
    
    transcript_received = False
    
    try:
        # Capture speech - returns Optional[str] (not tuple)
        transcript = await stt.capture_audio()
        
        # Handle Optional[str] return type
        if transcript is None:
            print(f"\n Speech capture failed (returned None)")
            print("   Possible causes:")
            print("   - Microphone error or permission denied")
            print("   - WebSocket connection failure")
            print("   - API rate limit or quota exceeded")
        elif transcript:
            transcript_received = True
            print(f"\n Transcript received!")
            print(f"   Text: '{transcript}'")
        else:
            print(f"\n️ No speech detected (empty string)")
            
    except Exception as e:
        print(f"\n Recognition error: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    print("\n3. Getting performance metrics...")
    try:
        metrics = stt.get_performance_metrics()
        # Use .get() with defaults to avoid KeyError
        print(f"   Total captures: {metrics.get('total_captures', 0)}")
        print(f"   Successful captures: {metrics.get('successful_captures', 0)}")
        print(f"   Failed captures: {metrics.get('failed_captures', 0)}")
        if metrics.get('avg_capture_time_s'):
            print(f"   Average capture time: {metrics['avg_capture_time_s']:.2f}s")
    except Exception as e:
        print(f"   ️ Metrics error: {e}")
    
    # Summary
    print("\n" + "="*70)
    if transcript_received:
        print(" STT TEST PASSED - Speech recognized successfully!")
    else:
        print("️ STT TEST INCOMPLETE - No speech recognized")
        print("\nTroubleshooting:")
        print("  1. Check microphone is connected and working")
        print("  2. Verify GEMINI_API_KEY is set in .env.leibniz")
        print("  3. Ensure no other app is using the microphone")
        print("  4. Try speaking louder or closer to the microphone")
        print("  5. Try running as module: python -m leibniz_agent.test_stt")
        print("  6. Check .env.leibniz file exists and has GEMINI_API_KEY set")
        print("  7. Verify you're in the parent directory (SINDH-Orchestra-Complete)")
        print("\n For detailed instructions, see TESTING_GUIDE.md")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
