#!/usr/bin/env python3
"""
Test script for unified WebRTC integration in Leibniz Pro.

This demonstrates how to use the LeibnizWebRTCIntegration class
for bidirectional WebRTC audio flow in the main pipeline.

Usage:
    # Run in unified WebRTC mode
    export LEIBNIZ_UNIFIED_WEBRTC=true
    python -m leibniz_agent.leibniz_pro

    # Or test the integration class directly
    python test_unified_webrtc.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from leibniz_agent.leibniz_pro import (
    LeibnizWebRTCIntegration,
    create_webrtc_integration,
    run_webrtc_conversation_loop
)


async def test_webrtc_integration():
    """Test the unified WebRTC integration class."""
    print("Testing Leibniz WebRTC Integration...")

    try:
        # Import WebRTC adapters
        from leibniz_agent.leibniz_webrtc_io import WebRTCSource, WebRTCSink

        # Create adapters
        source = WebRTCSource(webrtc_id="test_session", sample_rate=16000)
        sink = WebRTCSink(webrtc_id="test_session", sample_rate=16000)

        # Create integration
        integration = await create_webrtc_integration(
            source=source,
            sink=sink,
            session_id="test_session"
        )

        print(f"WebRTC integration ready: {integration.is_webrtc_ready()}")
        print(f"Session info: {integration.get_session_info()}")

        # Test speech capture (will timeout without real WebRTC connection)
        print("Testing speech capture (10s timeout)...")
        try:
            transcript = await asyncio.wait_for(
                integration.capture_speech_webrtc(),
                timeout=10.0
            )
            print(f"Captured: {transcript}")
        except asyncio.TimeoutError:
            print("Speech capture timed out (expected without WebRTC connection)")
        except Exception as e:
            print(f"Speech capture error: {e}")

        # Test TTS streaming (will work with mock audio)
        print("Testing TTS streaming...")
        result = await integration.speak_via_webrtc(
            "Hello from unified WebRTC integration!",
            emotion="helpful"
        )
        print(f"TTS result: {result}")

        print("WebRTC integration test completed successfully!")

    except ImportError as e:
        print(f"WebRTC adapters not available: {e}")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


async def demo_conversation_flow():
    """Demonstrate a mock conversation flow using the integration."""
    print("Demonstrating conversation flow...")

    try:
        # Create integration without real WebRTC (for demonstration)
        integration = LeibnizWebRTCIntegration(session_id="demo_session")

        # Mock the adapters for demonstration
        class MockSource:
            pass

        class MockSink:
            def __init__(self):
                self.audio_chunks = []

            async def put_frames(self, frames):
                self.audio_chunks.append(len(frames))
                print(f"Mock sink received {len(frames)} audio frames")

        mock_source = MockSource()
        mock_sink = MockSink()

        await integration.setup_webrtc_adapters(source=mock_source, sink=mock_sink)

        print(f"Demo integration ready: {integration.is_webrtc_ready()}")

        # Simulate conversation turn processing
        print("Simulating conversation turn...")

        # This would normally capture from WebRTC, but we'll skip that part
        # and just test the response processing
        test_transcript = "Hello, I need help with admissions"

        # Process transcript to response (this part works without WebRTC)
        response = await integration._process_transcript_to_response(test_transcript)
        print(f"Generated response: {response}")

        # Test TTS (will use mock sink)
        await integration.speak_via_webrtc("This is a test response", emotion="helpful")

        print(f"Mock sink received {len(mock_sink.audio_chunks)} audio chunks")

    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Leibniz Unified WebRTC Integration Test")
    print("=" * 50)

    # Run tests
    asyncio.run(test_webrtc_integration())
    print()
    asyncio.run(demo_conversation_flow())

    print("\nTo run the full WebRTC integration:")
    print("export LEIBNIZ_UNIFIED_WEBRTC=true")
    print("python -m leibniz_agent.leibniz_pro")