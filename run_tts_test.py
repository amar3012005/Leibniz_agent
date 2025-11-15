#!/usr/bin/env python3
"""
Test Runner for Leibniz TTS Module

This script properly executes the TTS test as a Python module and provides
diagnostic information about TTS provider configuration.

Usage:
    python leibniz_agent/run_tts_test.py
    (Run from SINDH-Orchestra-Complete directory)
"""
import subprocess
import sys
import os
from pathlib import Path


def main():
    """Execute TTS test as a Python module with diagnostics."""
    print("=" * 60)
    print("Leibniz TTS Test Runner")
    print("=" * 60)
    print()
    
    # Check TTS provider configuration
    print("TTS Provider Configuration:")
    print("-" * 60)
    
    # Check environment variables
    gemini_key = os.getenv("GEMINI_API_KEY")
    google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    tts_provider = os.getenv("LEIBNIZ_TTS_PROVIDER", "elevenlabs")  # Default: ElevenLabs (working)
    
    print(f"LEIBNIZ_TTS_PROVIDER: {tts_provider}")
    print(f"ELEVENLABS_API_KEY: {'Set ✅' if elevenlabs_key else 'Not set'}")
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {'Set' if google_creds else 'Not set'}")
    print(f"GEMINI_API_KEY: {'Set' if gemini_key else 'Not set'}")
    print()
    
    # Recommend ElevenLabs (proven working)
    if tts_provider == "elevenlabs":
        print("✅ ElevenLabs TTS - Recommended (proven working)")
        if not elevenlabs_key:
            print("   ⚠️  ELEVENLABS_API_KEY not set - get from https://elevenlabs.io")
        
        # Check voice_id format
        elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")
        if elevenlabs_voice_id:
            print(f"   Voice ID: {elevenlabs_voice_id}")
            # Voice IDs are long hashes (20+ chars), not names
            if len(elevenlabs_voice_id) < 20 or ' ' in elevenlabs_voice_id:
                print("   ⚠️  WARNING: Voice ID looks like a name, not a voice_id hash!")
                print("   Voice IDs are long hashes like: EXAVITQu4vr4xnSDxMaL")
                print("   Voice names like 'Rachel' or 'Sarah' will cause 404 errors")
                print("   Get voice IDs from: https://elevenlabs.io/app/voice-lab")
        print()
    
    # Warn about Gemini TTS instability
    if tts_provider == "gemini":
        print("⚠️  WARNING: Gemini TTS Preview Models - Known Issues")
        print("   Gemini TTS models are experiencing frequent 500 Internal Server errors")
        print("   This is a server-side issue confirmed as of October 2025")
        print("   Recommendation: Set LEIBNIZ_TTS_PROVIDER=elevenlabs in .env.leibniz")
        print()
    
    # Check if any provider is configured
    if not (gemini_key or google_creds or elevenlabs_key):
        print("❌ ERROR: No TTS provider API keys configured!")
        print("   Set at least one of:")
        print("   - ELEVENLABS_API_KEY (recommended - proven working)")
        print("   - GOOGLE_APPLICATION_CREDENTIALS (enterprise stable)")
        print("   - GEMINI_API_KEY (unstable preview - not recommended)")
        print()
        return 1
    
    print("Executing test as Python module...")
    print("Command: python -m leibniz_agent.test_leibniz_tts")
    print()
    
    # Get the parent directory (SINDH-Orchestra-Complete)
    parent_dir = Path(__file__).parent.parent
    
    # Construct the command to run test as module
    command = [sys.executable, "-m", "leibniz_agent.test_leibniz_tts"]
    
    print(f"Working directory: {parent_dir}")
    print(f"Python executable: {sys.executable}")
    print("=" * 60)
    print()
    
    # Execute the test
    result = subprocess.run(
        command,
        cwd=str(parent_dir),
        capture_output=False  # Show output in real-time
    )
    
    return result.returncode


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
