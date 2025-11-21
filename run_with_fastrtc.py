"""
Connection script for running Leibniz agent with FastRTC browser audio.

This script bridges the FastRTC server (leibniz_fastrtc_server.py) with the main
Leibniz conversation pipeline (leibniz_pro.py). It demonstrates the two-terminal
workflow for browser-based audio interaction.

Usage:
- Terminal 1: python leibniz_fastrtc_server.py
- Terminal 2: python run_with_fastrtc.py

Architecture Reference: docs/fastrtc-integration.md (lines 684-749)

This script waits for user speech via the FastRTC handler's event, then runs
conversation sessions using the handler's audio source and sink adapters.
"""

import asyncio
import logging
from leibniz_fastrtc_server import get_fastrtc_handler
from leibniz_pro import run_conversation_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """
    Main connection loop that bridges FastRTC server with Leibniz pipeline.

    Waits for the FastRTC server to initialize, then enters an infinite loop
    waiting for user speech events and running conversation sessions.
    """
    logger.info("Waiting for FastRTC server to start...")

    # Give the FastRTC server time to initialize
    await asyncio.sleep(2)

    # Get the handler from the FastRTC server
    handler = get_fastrtc_handler()

    if handler is None:
        logger.error("FastRTC handler not available. Start leibniz_fastrtc_server.py first.")
        return

    # Connection success banner
    logger.info("=" * 60)
    logger.info("✅ Connected to FastRTC server")
    logger.info("🌐 Open http://localhost:7860 in browser")
    logger.info("🎤 Click 'Record' and start speaking")
    logger.info("=" * 60)

    # Main conversation loop
    while True:
        try:
            # Wait for user to finish speaking (event set by FastRTC handler)
            await handler.user_finished_speaking.wait()

            # Run conversation session with FastRTC audio adapters
            await run_conversation_session(
                audio_source=handler.source,
                audio_sink=handler.sink
            )

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in conversation loop: {e}")
            # Brief pause before continuing
            await asyncio.sleep(1)


if __name__ == "__main__":
    # Run the connection script
    asyncio.run(main())