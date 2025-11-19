#!/usr/bin/env python3
"""
Test script for Gemini Live TTS integration in Leibniz TTS module.

Comment 8: Quick self-test for Gemini TTS setup validation.

Usage:
    python -m leibniz_agent.test_gemini_live_tts

Prerequisites:
    1. Install google-genai: pip install google-genai>=1.33.0
    2. Set GEMINI_API_KEY in .env.leibniz
    3. Set LEIBNIZ_TTS_PROVIDER=gemini in .env.leibniz

This script tests:
1. GeminiLiveTTSProvider initialization
2. File-based synthesis
3. Streaming synthesis
4. Emotion-aware dialogue
5. Integration with LeibnizTTS triple-provider architecture

Model IDs (API identifiers, not file paths):
- gemini-2.5-flash-native-audio-preview-09-2025 (recommended)
- gemini-2.5-flash-preview-tts
- gemini-live-2.5-flash-preview
- gemini-2.0-flash-live-001
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env.leibniz")

async def test_gemini_provider_initialization():
    """Test 1: Initialize GeminiLiveTTSProvider"""
    print("\n" + "="*80)
    print("TEST 1: GeminiLiveTTSProvider Initialization")
    print("="*80)
    
    try:
        from leibniz_agent.leibniz_tts import GeminiLiveTTSProvider
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print(" GEMINI_API_KEY not found in environment")
            print("   Set GEMINI_API_KEY in .env.leibniz or environment")
            print("   Get key from: https://aistudio.google.com/apikey")
            return False
        
        # Test with preview TTS model
        # Comment 8: Model ID is an API identifier, not a file path
        provider = GeminiLiveTTSProvider(
            api_key=api_key,
            model="gemini-2.5-flash-preview-tts"
        )
        
        print(f" Provider initialized successfully")
        print(f"   Model: {provider.model}")
        print(f"   Sample rate: {provider.sample_rate}Hz")
        
        return True
    except Exception as e:
        print(f" Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        import traceback
        traceback.print_exc()
        return False


async def test_gemini_file_synthesis():
    """Test 2: File-based synthesis with Gemini Live"""
    print("\n" + "="*80)
    print("TEST 2: File-Based Synthesis")
    print("="*80)
    
    try:
        from leibniz_agent.leibniz_tts import GeminiLiveTTSProvider
        import wave
        
        api_key = os.getenv("GEMINI_API_KEY")
        provider = GeminiLiveTTSProvider(api_key=api_key)
        
        # Test text
        test_text = "Hello! Welcome to Leibniz University. How can I help you today?"
        print(f" Synthesizing: '{test_text}'")
        
        # Synthesize
        audio_bytes = await provider.synthesize(
            text=test_text,
            language="en-US",
            emotion="helpful"
        )
        
        print(f" Synthesis successful: {len(audio_bytes)} bytes")
        
        # Save to WAV file
        output_file = Path(__file__).parent / "test_gemini_output.wav"
        with wave.open(str(output_file), 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(24000)  # 24kHz
            wf.writeframes(audio_bytes)
        
        print(f" Saved to: {output_file}")
        print(f"   Size: {output_file.stat().st_size} bytes")
        
        return True
    except Exception as e:
        print(f" Synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_gemini_streaming():
    """Test 3: Streaming synthesis"""
    print("\n" + "="*80)
    print("TEST 3: Streaming Synthesis")
    print("="*80)
    
    try:
        from leibniz_agent.leibniz_tts import GeminiLiveTTSProvider
        
        api_key = os.getenv("GEMINI_API_KEY")
        provider = GeminiLiveTTSProvider(api_key=api_key)
        
        test_text = "This is a streaming test for Gemini Live TTS integration."
        print(f" Streaming: '{test_text}'")
        
        chunks = []
        chunk_count = 0
        
        async for chunk in provider.stream_synthesize(
            text=test_text,
            language="en-US",
            emotion="professional"
        ):
            chunks.append(chunk)
            chunk_count += 1
            print(f"   Chunk {chunk_count}: {len(chunk)} bytes")
        
        total_bytes = sum(len(c) for c in chunks)
        print(f" Streaming complete: {chunk_count} chunks, {total_bytes} total bytes")
        
        return True
    except Exception as e:
        print(f" Streaming failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_emotion_variations():
    """Test 4: Emotion-aware synthesis"""
    print("\n" + "="*80)
    print("TEST 4: Emotion-Aware Synthesis")
    print("="*80)
    
    try:
        from leibniz_agent.leibniz_tts import GeminiLiveTTSProvider
        
        api_key = os.getenv("GEMINI_API_KEY")
        provider = GeminiLiveTTSProvider(api_key=api_key)
        
        # Test different emotions
        emotions = {
            'helpful': "Let me help you with that!",
            'excited': "That's amazing news!",
            'calm': "Everything will be alright.",
            'professional': "Please review the following information.",
            'empathetic': "I understand how you feel."
        }
        
        for emotion, text in emotions.items():
            print(f"\n Testing emotion: {emotion}")
            print(f"   Text: '{text}'")
            
            audio_bytes = await provider.synthesize(
                text=text,
                language="en-US",
                emotion=emotion
            )
            
            print(f"    Synthesized: {len(audio_bytes)} bytes")
        
        print(f"\n All emotion tests passed")
        return True
        
    except Exception as e:
        print(f" Emotion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_leibniz_tts_integration():
    """Test 5: Integration with LeibnizTTS main class"""
    print("\n" + "="*80)
    print("TEST 5: LeibnizTTS Triple-Provider Integration")
    print("="*80)
    
    try:
        from leibniz_agent.leibniz_tts import LeibnizTTS, LeibnizTTSConfig
        
        # Configure with Gemini as primary provider
        config = LeibnizTTSConfig(
            provider='gemini',  # Use Gemini as primary
            fallback_provider='google',  # Fallback to Google if Gemini fails
            gemini_model='gemini-2.5-flash-preview-tts',
            gemini_emotion_support=True,
            enable_cache=True
        )
        
        print(" Initializing LeibnizTTS with Gemini as primary provider...")
        tts = LeibnizTTS(config=config)
        
        # Check which providers were initialized
        providers_initialized = []
        if tts.google_provider:
            providers_initialized.append("Google Cloud TTS")
        if tts.elevenlabs_provider:
            providers_initialized.append("ElevenLabs")
        if tts.gemini_provider:
            providers_initialized.append("Gemini Live")
        
        print(f" LeibnizTTS initialized")
        print(f"   Providers: {', '.join(providers_initialized)}")
        
        # Test synthesis
        test_text = "Welcome to Leibniz University! I'm Lexi, your virtual assistant."
        output_file = Path(__file__).parent / "test_leibniz_gemini_output.wav"
        
        print(f"\n Synthesizing with emotion: helpful")
        print(f"   Text: '{test_text}'")
        
        result = await tts.synthesize_to_file(
            text=test_text,
            outfile=str(output_file),
            emotion="helpful"
        )
        
        if result['success']:
            print(f" Synthesis successful")
            print(f"   Provider used: {result['provider']}")
            print(f"   Cached: {result['cached']}")
            print(f"   Duration: {result['duration']:.2f}s")
            print(f"   Elapsed: {result['elapsed']:.3f}s")
            print(f"   File: {output_file}")
        else:
            print(f" Synthesis failed: {result.get('error', 'Unknown error')}")
            return False
        
        # Test again to verify caching
        print(f"\n Testing cache (same text)...")
        result2 = await tts.synthesize_to_file(
            text=test_text,
            outfile=str(output_file),
            emotion="helpful"
        )
        
        if result2['cached']:
            print(f" Cache hit confirmed")
            print(f"   Elapsed: {result2['elapsed']:.3f}s (should be very fast)")
        else:
            print(f"️ Expected cache hit but got cache miss")
        
        return True
        
    except Exception as e:
        print(f" Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "Gemini Live TTS Integration Tests" + " "*24 + "║")
    print("╚" + "="*78 + "╝")
    
    # Check prerequisites
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n GEMINI_API_KEY not found in environment")
        print("   Please set GEMINI_API_KEY in .env.leibniz or environment")
        return
    
    print(f"\n GEMINI_API_KEY found: {api_key[:10]}...")
    
    # Run tests
    tests = [
        ("Provider Initialization", test_gemini_provider_initialization),
        ("File-Based Synthesis", test_gemini_file_synthesis),
        ("Streaming Synthesis", test_gemini_streaming),
        ("Emotion-Aware Synthesis", test_emotion_variations),
        ("LeibnizTTS Integration", test_leibniz_tts_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"\n Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = " PASS" if result else " FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*80}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*80}")
    
    if passed == total:
        print("\n All tests passed! Gemini Live TTS integration is working correctly.")
    else:
        print(f"\n️ {total - passed} test(s) failed. Please review errors above.")


if __name__ == "__main__":
    asyncio.run(main())
