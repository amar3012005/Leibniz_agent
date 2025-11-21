"""
FastRTC server for Leibniz agent browser-based audio integration.

This module provides a standalone FastAPI application that serves as the FastRTC bridge
for browser-based audio interaction with the Leibniz agent. It runs the complete
conversation pipeline in the same process, providing seamless browser-based audio
interaction.

Architecture Reference: docs/fastrtc-integration.md

Single-process workflow:
- Terminal 1: python leibniz_fastrtc_server.py (starts server + conversation loop)
- Browser: http://localhost:7860 (direct audio interaction)

The server initializes a LeibnizFastRTCHandler and runs the conversation pipeline
in the same process, ensuring proper audio routing between browser and pipeline.
"""

import asyncio
import logging
from fastrtc import Stream, ReplyOnPause
from fastapi import FastAPI
import uvicorn
import gradio as gr
from leibniz_fastrtc_handler import LeibnizFastRTCHandler
from leibniz_pro import run_conversation_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app at module level for uvicorn compatibility
app = None
fastrtc_handler = None


def create_monochrome_theme():
    """
    Create a custom monochromatic Gradio theme with grayscale colors.

    Returns:
        Configured Gradio theme object with monochrome color scheme, or None if creation fails
    """
    try:
        # Use Soft theme as base for clean, minimal foundation
        theme = gr.themes.Soft()

        # Override all color tokens with grayscale values
        theme = theme.set(
            # Hues - all set to slate for consistent gray palette
            primary_hue="slate",
            secondary_hue="slate",
            neutral_hue="slate",

            # Backgrounds
            background_fill_primary="#ffffff",      # white (per user spec)
            background_fill_secondary="#f5f5f5",    # light gray (per user spec)
            background_fill_muted="#fafafa",        # very light gray for subtle contrast

            # Text colors
            body_text_color="#1a1a1a",              # near black (per user spec)
            body_text_color_subdued="#666666",      # medium gray for secondary text

            # Borders
            border_color_primary="#cccccc",         # medium gray (per user spec)
            border_color_secondary="#e0e0e0",       # lighter gray for subtle borders

            # Buttons (flat design, no gradients)
            button_primary_background_fill="#2a2a2a",       # dark gray (per user spec)
            button_primary_text_color="#ffffff",            # white text on dark button
            button_primary_background_fill_hover="#3a3a3a",  # slightly lighter on hover
            button_secondary_background_fill="#f5f5f5",      # light gray
            button_secondary_text_color="#1a1a1a",          # dark text on light button

            # Inputs
            input_background_fill="#ffffff",         # white
            input_border_color="#cccccc",            # medium gray
            input_border_color_focus="#2a2a2a",      # dark gray on focus
            input_background_fill_hover="#fafafa",   # very light gray on hover
            input_border_color_hover="#666666",      # medium gray on hover

            # Remove visual effects (flat design)
            shadow_drop="none",                      # no shadows
            shadow_spread="none",                    # no shadow spread

            # State indicators (monochrome)
            error_text_color="#1a1a1a",              # use dark gray instead of red
            error_background_fill="#e5e5e5",         # light gray
            success_text_color="#1a1a1a",            # use dark gray instead of green
            success_background_fill="#f5f5f5",       # light gray
            warning_text_color="#1a1a1a",            # use dark gray instead of yellow
            warning_background_fill="#f0f0f0",       # light gray

            # Links
            link_text_color="#2a2a2a",               # dark gray
            link_text_color_hover="#1a1a1a",         # darker on hover

            # Block elements (for strict monochrome)
            block_background_fill="#ffffff",         # white
            block_border_color="#e0e0e0",            # light gray
            block_border_width="1px",                # minimal border
            block_info_text_color="#666666",         # medium gray
            block_label_background_fill="#f5f5f5",   # light gray
            block_label_border_color="#cccccc",      # medium gray
            block_label_text_color="#1a1a1a",        # dark gray
            block_title_background_fill="#f5f5f5",   # light gray
            block_title_border_color="#cccccc",      # medium gray
            block_title_text_color="#1a1a1a",        # dark gray

            # Checkbox elements
            checkbox_background_color="#ffffff",     # white
            checkbox_border_color="#cccccc",         # medium gray
            checkbox_check="#2a2a2a",                # dark gray checkmark
            checkbox_label_background_fill="#ffffff", # white
            checkbox_label_background_fill_hover="#fafafa", # very light gray
            checkbox_label_text_color="#1a1a1a",     # dark gray
            checkbox_label_text_color_hover="#1a1a1a", # dark gray

            # Slider elements
            slider_color="#2a2a2a",                  # dark gray

            # Table elements
            table_border_color="#e0e0e0",            # light gray
            table_even_background_fill="#ffffff",    # white
            table_odd_background_fill="#fafafa",     # very light gray
            table_row_focus="#f5f5f5",               # light gray

            # Accent colors (override any remaining color accents)
            color_accent_soft="#f5f5f5",             # light gray
            color_accent="#2a2a2a",                  # dark gray

            # Layout
            radius_size="4px",                       # subtle rounded corners
            spacing_size="8px"                       # consistent, clean spacing
        )

        logger.info("Created monochromatic Gradio theme")
        return theme

    except Exception as e:
        logger.warning(f"Custom theme could not be applied due to incompatible Gradio version or unknown tokens: {e}")
        try:
            # Return safe fallback theme
            return gr.themes.Soft()
        except Exception as fallback_error:
            logger.warning(f"Fallback theme also failed: {fallback_error}")
            return None


def create_fastrtc_app() -> FastAPI:
    """
    Create and configure the FastRTC FastAPI application.

    Returns:
        Configured FastAPI application with FastRTC streaming and Gradio UI
    """
    global fastrtc_handler

    # Create FastAPI app
    app = FastAPI(
        title="Leibniz WebRTC Bridge",
        version="1.0.0"
    )

    # Initialize the global handler
    fastrtc_handler = LeibnizFastRTCHandler()

    # Create custom monochromatic theme
    monochrome_theme = create_monochrome_theme()

    # Build ui_args with theme if available
    ui_args = {
        "title": "Leibniz University Assistant",
        "description": "Speak naturally. I'll respond when you pause."
    }

    # Only add theme if creation succeeded
    if monochrome_theme is not None:
        ui_args["theme"] = monochrome_theme

    # Create FastRTC stream with ReplyOnPause for pause detection
    stream = Stream(
        handler=ReplyOnPause(fastrtc_handler),  # Detects user pauses and calls handler
        modality="audio",                       # Audio-only mode
        mode="send-receive",                    # Bidirectional audio streaming
        ui_args=ui_args
    )

    # Mount the Gradio UI to the FastAPI app at root path
    # For FastRTC integration, we need to run the stream directly
    # The stream.ui is a Gradio Blocks app that can be launched directly
    return stream.ui

    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "leibniz-fastrtc-bridge"}

    logger.info("FastRTC app created")
    return app


# Create the app instance at module level
app = create_fastrtc_app()


async def run_conversation_loop():
    """
    Run the main conversation loop that waits for user speech and processes conversations.

    This function runs in the same process as the FastRTC server, ensuring proper
    access to the handler instance and seamless audio routing.
    """
    logger.info("Starting conversation loop...")

    try:
        while True:
            # Wait for user to finish speaking (event set by FastRTC handler)
            await fastrtc_handler.user_finished_speaking.wait()

            # Run conversation session with FastRTC audio adapters
            await run_conversation_session(
                audio_source=fastrtc_handler.source,
                audio_sink=fastrtc_handler.sink
            )

    except KeyboardInterrupt:
        logger.info("Conversation loop interrupted")
    except Exception as e:
        logger.error(f"Error in conversation loop: {e}")


if __name__ == "__main__":
    # Startup banner
    logger.info("=" * 60)
    logger.info("🚀 Starting Leibniz FastRTC Server + Conversation Pipeline")
    logger.info("=" * 60)
    logger.info("📱 Browser UI: http://localhost:7860")
    logger.info("🎤 Single-process: Server + Pipeline integrated")
    logger.info("=" * 60)

    # Start both the Gradio server and conversation loop
    async def main():
        # Start the Gradio server in a separate thread (non-blocking)
        import threading
        gradio_thread = threading.Thread(
            target=lambda: app.launch(server_name="0.0.0.0", server_port=7860, show_error=True),
            daemon=True
        )
        gradio_thread.start()

        # Give Gradio a moment to start
        await asyncio.sleep(2)

        # Now run the conversation loop in the main async context
        await run_conversation_loop()

    # Run the integrated server and conversation loop
    asyncio.run(main())