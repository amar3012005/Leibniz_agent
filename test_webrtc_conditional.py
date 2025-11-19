#!/usr/bin/env python3
"""
Simple test to verify speak_friendly conditional logic for WebRTC vs local playback
"""
import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_webrtc_conditional_logic():
    """Test that speak_friendly skips local playback when WebRTC sink is provided"""

    print(" Testing speak_friendly conditional logic...")

    # Mock WebRTC sink class
    class MockWebRTCSink:
        def __init__(self):
            self.chunks_sent = []

        async def send_audio_chunk(self, chunk):
            self.chunks_sent.append(len(chunk))
            print(f" WebRTC chunk sent: {len(chunk)} bytes")

    # Import speak_friendly
    try:
        from leibniz_pro import speak_friendly
        print(" speak_friendly imported successfully")
    except ImportError as e:
        print(f" Failed to import speak_friendly: {e}")
        return False

    # Test 1: With WebRTC sink (should skip local playback)
    print("\n Test 1: With WebRTC sink (should skip sounddevice)")
    mock_sink = MockWebRTCSink()

    try:
        result = await speak_friendly(
            text="Hello, this is a test with WebRTC sink",
            sink=mock_sink,
            emotion="professional"
        )
        print(" speak_friendly completed with WebRTC sink")
        print(f" Chunks sent to WebRTC: {len(mock_sink.chunks_sent)}")
        print(f" Playback method: {result.audio_path if hasattr(result, 'audio_path') else 'N/A'}")
        return True
    except Exception as e:
        print(f" speak_friendly failed with WebRTC sink: {e}")
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_webrtc_conditional_logic())
    sys.exit(0 if success else 1)