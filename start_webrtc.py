"""
Quick Start Script for Leibniz WebRTC Server
Validates environment and starts the server
"""

import os
import sys
import logging
from pathlib import Path

# Load environment variables from .env.leibniz (same as leibniz_pro.py)
from dotenv import load_dotenv

# Get the leibniz_agent directory (where this file is located)
LEIBNIZ_DIR = Path(__file__).parent
ENV_FILE = LEIBNIZ_DIR / ".env.leibniz"

# Load Leibniz-specific environment variables
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
    logging.info(f" Loaded environment from: {ENV_FILE}")
else:
    logging.warning(f"️  .env.leibniz not found at: {ENV_FILE}")
    # Try loading default .env as fallback
    load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def check_environment():
    """Check that required environment variables are set"""
    logger.info(" Checking environment variables...")
    
    required_vars = {
        'GEMINI_API_KEY': 'Gemini API for STT and intent classification',
        'LEMONFOX_API_KEY': 'LemonFox API for TTS synthesis'
    }
    
    missing = []
    for var, purpose in required_vars.items():
        if not os.getenv(var):
            logger.error(f"    {var} not set ({purpose})")
            missing.append(var)
        else:
            logger.info(f"    {var} is set")
    
    if missing:
        logger.error("\n Missing required environment variables!")
        logger.error("Please set these variables in your .env.leibniz file or system environment:")
        for var in missing:
            logger.error(f"   - {var}")
        return False
    
    logger.info(" All required environment variables are set\n")
    return True


def check_dependencies():
    """Check that required packages are installed"""
    logger.info(" Checking dependencies...")
    
    required_packages = {
        'fastrtc': 'FastRTC for WebRTC streaming',
        'scipy': 'SciPy for audio resampling',
        'soundfile': 'SoundFile for audio I/O',
        'numpy': 'NumPy for array operations',
        'fastapi': 'FastAPI for web server',
        'uvicorn': 'Uvicorn ASGI server'
    }
    
    missing = []
    for package, purpose in required_packages.items():
        try:
            __import__(package)
            logger.info(f"    {package}")
        except ImportError:
            logger.error(f"    {package} ({purpose})")
            missing.append(package)
    
    if missing:
        logger.error("\n Missing required packages!")
        logger.error("Install them with:")
        logger.error(f"   pip install {' '.join(missing)}")
        return False
    
    logger.info(" All required packages are installed\n")
    return True


def start_server():
    """Start the WebRTC server"""
    logger.info(" Starting Leibniz WebRTC server...\n")
    logger.info("=" * 70)
    logger.info("Leibniz Agent - WebRTC Voice Interface")
    logger.info("=" * 70)
    logger.info("")
    logger.info(" Open your browser to: http://localhost:8000/")
    logger.info(" Click 'Connect' and allow microphone access")
    logger.info(" Start speaking - the agent will respond automatically!")
    logger.info("")
    logger.info(" Try these queries:")
    logger.info("   - What are your office hours?")
    logger.info("   - I need to schedule an appointment")
    logger.info("   - Tell me about the university programs")
    logger.info("")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 70)
    logger.info("")
    
    # Import and start the app
    try:
        import uvicorn
        from leibniz_webrtc_app import app
        
        port = int(os.getenv("LEIBNIZ_WEBRTC_PORT", "8000"))
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("\n\n Shutting down server...")
        logger.info("Thank you for using Leibniz Agent!")
    except Exception as e:
        logger.error(f"\n Server error: {e}")
        logger.error("Check the logs above for details")
        return 1
    
    return 0


def main():
    """Main entry point"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Leibniz Agent - WebRTC Quick Start")
    logger.info("=" * 70)
    logger.info("")
    
    # Step 1: Check environment
    if not check_environment():
        logger.error("\n Environment check failed")
        logger.error("Set the required environment variables and try again")
        return 1
    
    # Step 2: Check dependencies
    if not check_dependencies():
        logger.error("\n Dependency check failed")
        logger.error("Install the required packages and try again")
        return 1
    
    # Step 3: Start server
    return start_server()


if __name__ == "__main__":
    sys.exit(main())
