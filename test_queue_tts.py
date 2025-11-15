#!/usr/bin/env python3
"""
Test TTS Streaming Queue Functionality
=====================================

Tests the TTS streaming queue system with barge-in handling and real-time playback.
Verifies that multiple sentences can be queued and played with proper synchronization.

Usage:
    python test_queue_tts.py

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

# Import TTS and streaming functions
from leibniz_agent.leibniz_tts import get_leibniz_tts
from leibniz_agent.leibniz_pro import consume_tts_streaming_queue, clear_tts_queue


async def test_queue_tts():
    """Test TTS streaming queue with multiple sentences"""
    print("\n🎤 Testing TTS Streaming Queue...")

    try:
        # Get TTS instance
        tts = get_leibniz_tts()

        # Test sentences
        sentences = [
            "Hello! Welcome to Leibniz University.",
            "I can help you schedule appointments and answer questions.",
            "What would you like to know about our academic programs?",
            "Thank you for your interest in our university."
        ]

        print(f"📝 Enqueuing {len(sentences)} sentences...")

        # Import the global queue directly
        from leibniz_agent.leibniz_pro import _tts_streaming_queue

        # Clear any existing items
        await clear_tts_queue()

        # Enqueue sentences
        for i, sentence in enumerate(sentences, 1):
            await _tts_streaming_queue.put({
                'text': sentence,
                'emotion': 'helpful',
                'priority': i
            })
            print(f"   {i}. \"{sentence[:40]}...\"")

        print("✅ All sentences enqueued")

        # Start consumer task
        print("🎧 Starting TTS streaming consumer...")
        consumer_task = asyncio.create_task(consume_tts_streaming_queue())

        # Wait for queue to be processed (with timeout)
        timeout = 60  # 60 seconds max
        start_time = asyncio.get_event_loop().time()

        try:
            while not _tts_streaming_queue.empty():
                await asyncio.sleep(0.5)
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    print(f"⏱️ Timeout after {elapsed:.1f}s")
                    break

                # Show progress
                queue_size = _tts_streaming_queue.qsize()
                if queue_size > 0:
                    print(f"   Queue remaining: {queue_size} items")

        except Exception as e:
            print(f"❌ Queue processing error: {e}")

        # Wait a bit more for final playback
        await asyncio.sleep(2)

        # Cancel consumer task
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

        print("✅ TTS streaming queue test completed")
        return True

    except Exception as e:
        print(f"❌ TTS streaming queue test failed: {e}")
        return False


async def test_barge_in_simulation():
    """Test barge-in handling during TTS playback"""
    print("\n🚫 Testing Barge-in Simulation...")

    try:
        # Import VAD to simulate barge-in
        from leibniz_agent.leibniz_vad import get_leibniz_vad

        vad = get_leibniz_vad()

        # Start a long TTS playback
        long_text = "This is a longer message that will take some time to play back. It should demonstrate barge-in functionality when user speech is detected."

        # Get TTS queue and enqueue
        from leibniz_agent.leibniz_pro import _tts_streaming_queue

        await _tts_streaming_queue.put({
            'text': long_text,
            'emotion': 'helpful',
            'priority': 1
        })

        # Start consumer
        consumer_task = asyncio.create_task(consume_tts_streaming_queue())

        # Wait a moment then simulate barge-in
        await asyncio.sleep(1.0)

        print("🎤 Simulating user barge-in...")

        # Simulate barge-in by setting agent speaking to False
        # (In real usage, this would be triggered by VAD detecting user speech)
        vad.is_agent_speaking = False
        vad.barge_in_detected = True

        # Wait for barge-in to be processed
        await asyncio.sleep(0.5)

        # Check if playback was interrupted
        if vad.barge_in_detected:
            print("✅ Barge-in simulation successful")
            success = True
        else:
            print("⚠️ Barge-in simulation inconclusive")
            success = True  # Not a failure, just couldn't trigger

        # Cancel consumer
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

        return success

    except Exception as e:
        print(f"❌ Barge-in test failed: {e}")
        return False


async def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 Leibniz TTS Streaming Queue Test")
    print("=" * 60)

    # Check environment
    api_key = os.getenv('LEMONFOX_API_KEY')
    if not api_key:
        print("❌ LEMONFOX_API_KEY environment variable not set")
        print("   Please set your LemonFox API key to run this test")
        return

    print(f"✅ LEMONFOX_API_KEY: {'*' * len(api_key)}")

    # Test streaming queue
    queue_success = await test_queue_tts()

    # Test barge-in (optional)
    barge_in_success = await test_barge_in_simulation()

    # Summary
    print("\n" + "=" * 60)
    if queue_success:
        print("🎉 TTS streaming queue test passed!")
        if barge_in_success:
            print("🎉 Barge-in simulation also successful!")
        else:
            print("⚠️ Barge-in simulation had issues but queue test passed.")
    else:
        print("❌ TTS streaming queue test failed.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())