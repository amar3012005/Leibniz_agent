#!/usr/bin/env python3
"""
Test Audio Playback Functionality
=================================

Tests audio synthesis and playback with sounddevice fallback guards.
Verifies that TTS synthesis works and audio can be played through pygame
with sounddevice fallback when available.

Usage:
    python test_audio_playback.py

Environment Variables:
- LEMONFOX_API_KEY: Required for synthesis
- SOUNDDEVICE_AVAILABLE: Auto-detected (optional dependency)
"""

import os
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for sounddevice availability (for fallback guards)
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    print("✅ sounddevice available for pygame fallback")
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️ sounddevice not available - pygame fallback disabled")

# Import TTS module (direct import since we're in leibniz_agent directory)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from leibniz_tts import get_leibniz_tts


async def test_audio_synthesis():
    """Test basic audio synthesis functionality"""
    print("\n🎤 Testing Audio Synthesis...")

    try:
        tts = get_leibniz_tts()

        # Test synthesis
        test_text = "Hello! This is a test of the Leibniz TTS system."
        result = await tts.synthesize_to_file(test_text, "test_output.wav", emotion="helpful")

        if result['success']:
            audio_path = result['file']
            abs_path = os.path.abspath(audio_path)
            print(f"✅ Synthesis successful: {abs_path}")
            print(f"   Duration: {result.get('duration', 'unknown'):.2f}s")
            print(f"   Cached: {result.get('cached', False)}")
            print(f"   Provider: {result.get('provider', 'unknown')}")

            # Verify file exists and has size
            if os.path.exists(abs_path) and os.path.getsize(abs_path) > 0:
                print(f"✅ File verification passed: {os.path.getsize(abs_path)} bytes")
                return abs_path
            else:
                print("❌ File verification failed")
                return None
        else:
            print(f"❌ Synthesis failed: {result.get('error', 'unknown error')}")
            return None

    except Exception as e:
        print(f"❌ Synthesis test failed: {e}")
        return None


async def test_pygame_playback(audio_path):
    """Test pygame audio playback with sounddevice fallback"""
    print(f"\n🔊 Testing Audio Playback: {audio_path}")

    try:
        # Import pygame for primary playback
        import pygame
        pygame.mixer.init()

        # Check file exists before loading
        abs_path = os.path.abspath(audio_path)
        if not os.path.exists(abs_path):
            print(f"❌ Audio file not found: {abs_path}")
            return False

        # Load and play with pygame
        pygame.mixer.music.load(abs_path)
        pygame.mixer.music.play()

        # Wait for playback to complete
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)

        pygame.mixer.quit()
        print("✅ Pygame playback successful")
        return True

    except Exception as e:
        print(f"❌ Pygame playback failed: {e}")

        # Try sounddevice fallback if available
        if SOUNDDEVICE_AVAILABLE:
            print("🔄 Attempting sounddevice fallback...")
            try:
                import soundfile as sf

                # Load audio with soundfile
                audio_data, sample_rate = sf.read(abs_path)

                # Play with sounddevice
                sd.play(audio_data, sample_rate)
                sd.wait()

                print("✅ Sounddevice fallback successful")
                return True

            except Exception as fallback_e:
                print(f"❌ Sounddevice fallback also failed: {fallback_e}")
                return False
        else:
            print("⚠️ No sounddevice fallback available")
            return False


async def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 Leibniz TTS Audio Playback Test")
    print("=" * 60)

    # Check environment
    api_key = os.getenv('LEMONFOX_API_KEY')
    if not api_key:
        print("❌ LEMONFOX_API_KEY environment variable not set")
        print("   Please set your LemonFox API key to run this test")
        return

    print(f"✅ LEMONFOX_API_KEY: {'*' * len(api_key)}")

    # Test synthesis
    audio_path = await test_audio_synthesis()
    if not audio_path:
        print("❌ Cannot proceed with playback test - synthesis failed")
        return

    # Test playback
    playback_success = await test_pygame_playback(audio_path)

    # Summary
    print("\n" + "=" * 60)
    if playback_success:
        print("🎉 All tests passed! TTS system is working correctly.")
    else:
        print("⚠️ Synthesis worked but playback failed. Check audio setup.")
    print("=" * 60)

    # Cleanup
    try:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
            print(f"🧹 Cleaned up test file: {audio_path}")
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())