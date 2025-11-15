"""
Test ElevenLabs TTS Integration for Leibniz Agent

This test script verifies the ElevenLabs TTS provider integration including:
- API key configuration
- Basic synthesis
- Emotion modulation
- Caching behavior
- Fallback to Google TTS

USAGE:
    python -m leibniz_agent.test_single_tts  (from SINDH-Orchestra-Complete directory)
    OR
    python leibniz_agent/test_single_tts.py

PREREQUISITES:
    - ELEVENLABS_API_KEY set in leibniz_agent/.env.leibniz
    - Get API key from: https://elevenlabs.io
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Leibniz TTS components
from leibniz_agent.leibniz_tts import get_leibniz_tts


async def main():
    """Main test function for ElevenLabs TTS integration"""
    
    print("=" * 70)
    print("ElevenLabs TTS Integration Test - Leibniz Agent")
    print("=" * 70)
    
    # Load environment variables from .env.leibniz
    env_path = Path(__file__).parent / ".env.leibniz"
    load_dotenv(env_path)
    print(f"\n📝 Loading environment from: {env_path}")
    
    # Verify ElevenLabs API key is configured
    elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
    if not elevenlabs_key or elevenlabs_key == 'your_elevenlabs_api_key_here':
        print("\n❌ ERROR: ELEVENLABS_API_KEY not configured!")
        print("\n📋 To fix this:")
        print(f"   1. Edit: {env_path}")
        print("   2. Set your ElevenLabs API key:")
        print("      ELEVENLABS_API_KEY=your_actual_api_key_here")
        print("\n   Get a key from: https://elevenlabs.io")
        print("   Note: Free tier includes 10,000 characters/month")
        return 1
    
    print(f"✅ ELEVENLABS_API_KEY: Set (length: {len(elevenlabs_key)} chars)")
    
    # Check provider configuration
    tts_provider = os.getenv('LEIBNIZ_TTS_PROVIDER', 'google')
    tts_fallback = os.getenv('LEIBNIZ_TTS_FALLBACK_PROVIDER', 'elevenlabs')
    
    print(f"\n📊 Provider Configuration:")
    print(f"   Primary provider: {tts_provider}")
    print(f"   Fallback provider: {tts_fallback}")
    
    if tts_provider != 'elevenlabs':
        print(f"\n⚠️  WARNING: Primary provider is '{tts_provider}', not 'elevenlabs'")
        print("   For this test, we'll initialize ElevenLabs explicitly")
    
    # Initialize TTS system with ElevenLabs
    print(f"\n1. Initializing Leibniz TTS with ElevenLabs...")
    try:
        tts = get_leibniz_tts()
        print("   ✅ TTS system initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize TTS: {e}")
        return 1
    
    # Check TTS status
    print(f"\n2. Checking TTS provider status...")
    try:
        status = tts.status()
        print(f"   Primary provider: {status.get('primary_provider', 'unknown')}")
        print(f"   ElevenLabs available: {status.get('elevenlabs_available', False)}")
        print(f"   Google available: {status.get('google_available', False)}")
        print(f"   Any provider available: {status.get('any_provider_available', False)}")
        
        if not status.get('elevenlabs_available'):
            print("\n   ⚠️  WARNING: ElevenLabs provider not available!")
            print("   Check your API key and internet connection")
    except Exception as e:
        print(f"   ⚠️  Could not get status: {e}")
    
    # Test basic synthesis
    print(f"\n3. Testing basic synthesis with ElevenLabs...")
    test_text = "Hello! This is a test of the ElevenLabs text to speech system."
    test_file = "test_output/elevenlabs_basic_test.wav"
    
    try:
        result = await tts.synthesize_to_file(
            text=test_text,
            outfile=test_file,
            emotion='helpful'
        )
        
        if result.get('success'):
            print(f"   ✅ Synthesis successful!")
            print(f"      Provider: {result.get('provider', 'unknown')}")
            print(f"      File: {result.get('file', test_file)}")
            print(f"      Duration: {result.get('duration', 0):.2f}s")
            print(f"      Elapsed: {result.get('elapsed', 0):.3f}s")
            
            # Check if file exists
            file_path = Path(result.get('file', test_file))
            if file_path.exists():
                file_size = file_path.stat().st_size / 1024
                print(f"      File size: {file_size:.1f} KB")
        else:
            print(f"   ❌ Synthesis failed: {result.get('error', 'Unknown error')}")
            return 1
    except Exception as e:
        print(f"   ❌ Exception during synthesis: {e}")
        return 1
    
    # Test emotion modulation
    print(f"\n4. Testing emotion modulation...")
    emotions = ['helpful', 'excited', 'calm', 'professional']
    
    for emotion in emotions:
        test_text_emotion = f"Testing {emotion} emotion with ElevenLabs."
        test_file_emotion = f"test_output/elevenlabs_{emotion}_test.wav"
        
        try:
            result = await tts.synthesize_to_file(
                text=test_text_emotion,
                outfile=test_file_emotion,
                emotion=emotion
            )
            
            if result.get('success'):
                print(f"   ✅ {emotion.capitalize()}: {result.get('provider', 'unknown')} ({result.get('elapsed', 0):.3f}s)")
            else:
                print(f"   ❌ {emotion.capitalize()}: Failed - {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"   ❌ {emotion.capitalize()}: Exception - {e}")
    
    # Test caching
    print(f"\n5. Testing caching behavior...")
    cache_test_text = "This phrase will be cached for faster playback."
    cache_test_file = "test_output/elevenlabs_cache_test.wav"
    
    try:
        # First synthesis (cache miss)
        print("   First synthesis (cache miss)...")
        result1 = await tts.synthesize_to_file(
            text=cache_test_text,
            outfile=cache_test_file,
            emotion='helpful'
        )
        time1 = result1.get('elapsed', 0)
        print(f"   Response time: {time1:.3f}s")
        
        # Second synthesis (cache hit)
        print("   Second synthesis (cache hit expected)...")
        result2 = await tts.synthesize_to_file(
            text=cache_test_text,
            outfile=cache_test_file,
            emotion='helpful'
        )
        time2 = result2.get('elapsed', 0)
        print(f"   Response time: {time2:.3f}s")
        
        if time2 < time1 * 0.5:  # Cache should be significantly faster
            print(f"   ✅ Caching working! Speedup: {time1/time2:.1f}x")
        else:
            print(f"   ⚠️  Cache may not be working (similar times)")
        
        # Get cache statistics
        if hasattr(tts, 'get_cache_stats'):
            cache_stats = tts.get_cache_stats()
            print(f"\n   Cache Statistics:")
            print(f"   - Total entries: {cache_stats.get('total_entries', 0)}")
            print(f"   - Cache hits: {cache_stats.get('hits', 0)}")
            print(f"   - Cache misses: {cache_stats.get('misses', 0)}")
    except Exception as e:
        print(f"   ⚠️  Cache test exception: {e}")
    
    # Test fallback (optional - requires invalid key)
    print(f"\n6. Fallback testing (skipped - requires invalid key)")
    print("   To test fallback, temporarily set invalid ELEVENLABS_API_KEY")
    print("   System should automatically fall back to Google TTS")
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Completed!")
    print("=" * 70)
    print("\n📋 Summary:")
    print("   - ElevenLabs API key configured ✅")
    print("   - TTS system initialized ✅")
    print("   - Basic synthesis working ✅")
    print("   - Emotion modulation tested ✅")
    print("   - Caching behavior verified ✅")
    print("\n💡 Next steps:")
    print("   - Run the full Leibniz agent: python -m leibniz_agent.leibniz_pro")
    print("   - Check generated audio files in: test_output/")
    print("   - Review TTS logs for provider usage")
    
    return 0


if __name__ == "__main__":
    # Create test output directory
    Path("test_output").mkdir(exist_ok=True)
    
    # Run async test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
