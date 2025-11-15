"""
🎤 Voice Cloning Test Script for Leibniz/TARA Agents
Tests XTTS v2 voice cloning with a 1-minute speaker sample

Requirements:
- pip install TTS
- pip install "transformers<4.50"  # Critical for XTTS v2
- Clean 1-minute WAV file (mono, 16-22kHz, clear speech, no background noise)

Usage:
    # Basic voice cloning test
    python leibniz_agent/test_voice_cloning.py
    
    # With integration test
    TEST_INTEGRATION=true python leibniz_agent/test_voice_cloning.py
"""

import torch
from TTS.api import TTS
import os
import time
import soundfile as sf
import numpy as np
from pathlib import Path

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "leibniz_agent/voice_cloning_output"
SPEAKER_SAMPLE_PATH = "leibniz_agent/audio/test-k.wav"  # PUT YOUR 1-MINUTE SAMPLE HERE

# Test texts for different scenarios
TEST_TEXTS = {
    "greeting": {
        "en": "Hello! Welcome to Leibniz University International Office. How can I help you today?",
        "hi": "नमस्ते! लेइबनिज़ विश्वविद्यालय में आपका स्वागत है। मैं आपकी कैसे मदद कर सकता हूं?",
        "es": "¡Hola! Bienvenido a la Oficina Internacional de la Universidad Leibniz. ¿Cómo puedo ayudarte hoy?"
    },
    "short": {
        "en": "Your appointment has been confirmed for tomorrow at 2 PM.",
        "hi": "आपकी नियुक्ति कल दोपहर 2 बजे के लिए पुष्टि की गई है।",
        "es": "Su cita ha sido confirmada para mañana a las 2 PM."
    },
    "long": {
        "en": """To apply for the Master's program in Computer Science, you'll need a bachelor's degree 
with a GPA of at least 2.5, proof of English proficiency such as TOEFL or IELTS, 
and two letters of recommendation. The application deadline is March 15th. 
You can submit your documents through our online portal at apply dot leibniz dot edu.""",
        "hi": """कंप्यूटर साइंस में मास्टर प्रोग्राम के लिए आवेदन करने के लिए, आपको कम से कम 2.5 जीपीए 
के साथ स्नातक की डिग्री की आवश्यकता होगी, अंग्रेजी दक्षता का प्रमाण जैसे टीओईएफएल या आईईएलटीएस, 
और दो सिफारिश पत्र। आवेदन की अंतिम तिथि 15 मार्च है।""",
        "es": """Para aplicar al programa de Maestría en Ciencias de la Computación, necesitará un título 
de licenciatura con un GPA de al menos 2.5, prueba de dominio del inglés como TOEFL o IELTS, 
y dos cartas de recomendación. La fecha límite de solicitud es el 15 de marzo."""
    },
    "conversational": {
        "en": """Great question! The registrar's office is open Monday through Friday, 9 AM to 5 PM. 
You can reach them by email at registrar at leibniz dot edu, or call them at 555-0123. 
Is there anything else I can help you with?""",
        "hi": """बहुत अच्छा सवाल! रजिस्ट्रार का कार्यालय सोमवार से शुक्रवार, सुबह 9 बजे से शाम 5 बजे तक खुला रहता है। 
आप उन्हें ईमेल द्वारा registrar@leibniz.edu पर संपर्क कर सकते हैं। क्या मैं आपकी और कोई मदद कर सकता हूं?""",
        "es": """¡Excelente pregunta! La oficina del registrador está abierta de lunes a viernes, 
de 9 AM a 5 PM. Puede contactarlos por correo electrónico en registrar@leibniz.edu. 
¿Hay algo más en lo que pueda ayudarte?"""
    }
}


def check_speaker_sample(sample_path: str) -> dict:
    """Validate speaker sample audio file"""
    print("\n" + "="*70)
    print("🔍 Validating Speaker Sample")
    print("="*70)
    
    if not os.path.exists(sample_path):
        print(f"❌ Speaker sample not found: {sample_path}")
        print("\n📝 To use voice cloning:")
        print("   1. Record 1 minute of clear speech (no background noise)")
        print("   2. Save as WAV file (mono, 16-22kHz recommended)")
        print(f"   3. Place file at: {sample_path}")
        print("   4. Re-run this script")
        return {"valid": False, "reason": "File not found"}
    
    try:
        # Load and analyze audio
        audio_data, sample_rate = sf.read(sample_path)
        
        # Get audio properties
        duration = len(audio_data) / sample_rate
        channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
        bit_depth = audio_data.dtype
        
        # Check if stereo (need to convert to mono)
        is_stereo = channels == 2
        
        print(f"✅ Speaker sample found: {os.path.basename(sample_path)}")
        print(f"   Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Channels: {channels} ({'stereo' if is_stereo else 'mono'})")
        print(f"   Bit depth: {bit_depth}")
        print(f"   File size: {os.path.getsize(sample_path) / 1024 / 1024:.2f} MB")
        
        # Quality checks
        warnings = []
        
        if duration < 6:
            warnings.append(f"⚠️  Duration too short ({duration:.1f}s) - recommend 6-60 seconds")
        elif duration > 120:
            warnings.append(f"⚠️  Duration very long ({duration:.1f}s) - recommend 6-60 seconds")
        
        if sample_rate < 16000:
            warnings.append(f"⚠️  Low sample rate ({sample_rate}Hz) - recommend 16000-22050 Hz")
        
        if is_stereo:
            warnings.append("⚠️  Stereo audio detected - will convert to mono")
            # Convert to mono
            audio_data = np.mean(audio_data, axis=1)
            mono_path = sample_path.replace(".wav", "_mono.wav")
            sf.write(mono_path, audio_data, sample_rate)
            print(f"   ✅ Converted to mono: {mono_path}")
            sample_path = mono_path
        
        # Check signal quality (noise level)
        rms = np.sqrt(np.mean(audio_data**2))
        if rms < 0.01:
            warnings.append(f"⚠️  Very quiet audio (RMS: {rms:.4f}) - may affect quality")
        
        if warnings:
            print("\n⚠️  Quality Warnings:")
            for warning in warnings:
                print(f"   {warning}")
        else:
            print("\n✅ Audio quality looks good!")
        
        return {
            "valid": True,
            "path": sample_path,
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": 1 if is_stereo else channels,
            "warnings": warnings
        }
    
    except Exception as e:
        print(f"❌ Error reading audio file: {e}")
        return {"valid": False, "reason": str(e)}


def test_voice_cloning(speaker_sample_path: str, language: str = "en"):
    """Test XTTS v2 voice cloning with speaker sample"""
    
    print("\n" + "="*70)
    print(f"🎤 Testing Voice Cloning - Language: {language.upper()}")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Load XTTS v2 model
        print("\n📦 Loading XTTS v2 model...")
        print(f"   Device: {DEVICE}")
        
        start_load = time.time()
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
        load_time = time.time() - start_load
        
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Test each text type
        results = []
        
        for test_type, texts in TEST_TEXTS.items():
            if language not in texts:
                continue
            
            text = texts[language]
            
            print(f"\n{'─'*70}")
            print(f"🎙️  Test: {test_type.upper()}")
            print(f"📝 Text: {text[:80]}{'...' if len(text) > 80 else ''}")
            print(f"{'─'*70}")
            
            output_file = os.path.join(
                OUTPUT_DIR,
                f"cloned_{language}_{test_type}.wav"
            )
            
            try:
                # Synthesize with voice cloning
                print("⏳ Synthesizing...")
                start_synth = time.time()
                
                tts.tts_to_file(
                    text=text,
                    speaker_wav=speaker_sample_path,
                    language=language,
                    file_path=output_file
                )
                
                synth_time = time.time() - start_synth
                
                # Get output file info
                file_size = os.path.getsize(output_file)
                audio_data, sample_rate = sf.read(output_file)
                duration = len(audio_data) / sample_rate
                
                print(f"✅ Success!")
                print(f"   Output: {os.path.basename(output_file)}")
                print(f"   Duration: {duration:.1f}s")
                print(f"   Synthesis time: {synth_time:.2f}s")
                print(f"   Real-time factor: {synth_time/duration:.2f}x")
                print(f"   File size: {file_size/1024:.1f} KB")
                
                results.append({
                    "test_type": test_type,
                    "language": language,
                    "success": True,
                    "synth_time": synth_time,
                    "duration": duration,
                    "rtf": synth_time / duration,
                    "output_file": output_file
                })
                
            except Exception as e:
                print(f"❌ Synthesis failed: {e}")
                results.append({
                    "test_type": test_type,
                    "language": language,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def print_summary(all_results: dict):
    """Print comprehensive summary of all tests"""
    
    print("\n\n" + "="*70)
    print("📊 VOICE CLONING TEST SUMMARY")
    print("="*70)
    
    total_tests = 0
    successful_tests = 0
    
    for lang, results in all_results.items():
        lang_successful = sum(1 for r in results if r.get("success", False))
        total_tests += len(results)
        successful_tests += lang_successful
        
        print(f"\n🌐 Language: {lang.upper()}")
        print(f"   Tests: {lang_successful}/{len(results)} successful")
        
        successful_results = [r for r in results if r.get("success", False)]
        if successful_results:
            avg_rtf = np.mean([r["rtf"] for r in successful_results])
            avg_synth_time = np.mean([r["synth_time"] for r in successful_results])
            
            print(f"   Avg synthesis time: {avg_synth_time:.2f}s")
            print(f"   Avg real-time factor: {avg_rtf:.2f}x")
            
            if avg_rtf < 1.0:
                print(f"   ✅ Faster than real-time! (can stream)")
            else:
                print(f"   ⚠️  Slower than real-time (batch processing only)")
    
    print(f"\n{'─'*70}")
    print(f"Overall: {successful_tests}/{total_tests} tests successful")
    print(f"{'─'*70}")
    
    print(f"\n📂 All outputs saved to: {OUTPUT_DIR}")
    print("\n💡 Next Steps:")
    print("   1. Listen to generated samples in output directory")
    print("   2. Verify voice quality matches your speaker sample")
    print("   3. If satisfied, integrate into Leibniz/TARA TTS pipeline")
    print("   4. See integration example below")


def print_integration_guide(speaker_sample_path: str):
    """Print integration guide for production use"""
    
    print("\n\n" + "="*70)
    print("🚀 PRODUCTION INTEGRATION GUIDE")
    print("="*70)
    
    print("""
📝 Step 1: Update leibniz_tts.py (or tara_tts.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import torch
from TTS.api import TTS
from pathlib import Path

class VoiceClonedTTS:
    def __init__(self, speaker_sample_path: str, language: str = "en"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.speaker_sample = speaker_sample_path
        self.language = language
        
        # Load XTTS v2 model (one-time at startup)
        print(f"Loading XTTS v2 voice cloning model...")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        print(f"✅ Model loaded on {self.device}")
    
    def synthesize_to_file(self, text: str, output_path: str) -> str:
        \"\"\"Synthesize text using cloned voice\"\"\"
        self.tts.tts_to_file(
            text=text,
            speaker_wav=self.speaker_sample,
            language=self.language,
            file_path=output_path
        )
        return output_path
    
    def synthesize_streaming(self, text: str):
        \"\"\"Stream audio chunks (for low-latency playback)\"\"\"
        # XTTS v2 supports streaming with <200ms latency
        return self.tts.tts_stream(
            text=text,
            speaker_wav=self.speaker_sample,
            language=self.language
        )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 Step 2: Update leibniz_pro.py (main orchestrator)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Replace ElevenLabs/Google TTS with voice-cloned TTS

from leibniz_tts import VoiceClonedTTS

# Initialize TTS with your speaker sample
tts_engine = VoiceClonedTTS(
    speaker_sample_path="leibniz_agent/speaker_sample.wav",
    language="en"  # or "hi" for TARA, "es" for Spanish, etc.
)

# Use in conversation loop
response_text = "Welcome to Leibniz University!"
output_path = "temp_audio.wav"
tts_engine.synthesize_to_file(response_text, output_path)

# Play audio (existing playback logic)
play_audio(output_path)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 Step 3: For TARA (Multilingual Support)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# XTTS v2 supports 16 languages with same speaker voice!

# Detect language from user input
detected_language = intent_parser.extract_language(user_input)

# Initialize TTS with language switching
tts_engine = VoiceClonedTTS(
    speaker_sample_path="tara_agent/hindi_female_sample.wav",
    language="hi"  # Default Hindi
)

# Synthesize in detected language
if detected_language == "en":
    tts_engine.language = "en"
elif detected_language == "hi":
    tts_engine.language = "hi"
elif detected_language == "te":
    tts_engine.language = "te"  # Telugu

tts_engine.synthesize_to_file(response_text, output_path)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    
    print(f"\n🎤 Your speaker sample: {speaker_sample_path}")
    print(f"📂 Test outputs: {OUTPUT_DIR}")
    print("\n✅ Ready for production deployment!")


def main():
    print("="*70)
    print("🎤 VOICE CLONING TEST - LEIBNIZ/TARA AGENTS")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Model: XTTS v2 (Multilingual Voice Cloning)")
    print("="*70)
    
    # Step 1: Check speaker sample
    sample_info = check_speaker_sample(SPEAKER_SAMPLE_PATH)
    
    if not sample_info["valid"]:
        print("\n❌ Cannot proceed without valid speaker sample.")
        print("\n💡 Auto-generating 10-second test sample...")
        
        # Auto-generate test sample (no user input needed)
        print("\n📝 Generating test sample...")
        # Generate simple voice-like audio
        sample_rate = 22050
        duration = 10  # 10 seconds for better quality
        t = np.linspace(0, duration, sample_rate * duration)
        
        # Generate multiple harmonics for voice-like sound (female-ish pitch ~220 Hz)
        audio = np.zeros_like(t)
        fundamental = 220  # A3 note (female voice range)
        for i, harmonic in enumerate([1, 2, 3, 4, 5]):
            freq = fundamental * harmonic
            # Add vibrato
            vibrato = np.sin(2 * np.pi * 5 * t) * 0.02  # 5 Hz vibrato
            audio += np.sin(2 * np.pi * freq * t * (1 + vibrato)) / (i + 1)
        
        # Add some pink noise for realism
        audio += np.random.randn(len(audio)) * 0.02
        
        # Simple envelope to avoid clicks
        fade_samples = int(0.1 * sample_rate)
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        audio = audio / np.max(np.abs(audio)) * 0.5  # Normalize
        
        os.makedirs(os.path.dirname(SPEAKER_SAMPLE_PATH), exist_ok=True)
        sf.write(SPEAKER_SAMPLE_PATH, audio, sample_rate)
        print(f"✅ Test sample created: {SPEAKER_SAMPLE_PATH}")
        print("⚠️  This is synthetic - replace with real voice for production!")
        
        sample_info = check_speaker_sample(SPEAKER_SAMPLE_PATH)
    
    if not sample_info["valid"]:
        return
    
    speaker_path = sample_info["path"]
    
    # Step 2: Choose languages to test (auto-select English for quick test)
    print("\n📋 Auto-selecting: English only (quick test)")
    print("   (Edit script to test other languages: hi, es, etc.)")
    
    languages = ["en"]  # Auto-select English
    print(f"   Testing: {', '.join(languages).upper()}")
    
    # Step 3: Run tests
    all_results = {}
    
    for lang in languages:
        results = test_voice_cloning(speaker_path, lang)
        all_results[lang] = results
        
        # Small delay between languages
        if len(languages) > 1 and lang != languages[-1]:
            print("\n⏳ Waiting 2s before next language test...")
            time.sleep(2)
    
    # Step 4: Print summary
    print_summary(all_results)
    
    # Step 5: Integration guide
    print_integration_guide(speaker_path)
    
    # Step 6: Test integration (optional)
    run_integration = os.getenv('TEST_INTEGRATION', 'false').lower() == 'true'
    if run_integration:
        print("\n" + "="*70)
        print("🔗 TESTING LEIBNIZ TTS INTEGRATION")
        print("="*70)
        import asyncio
        asyncio.run(test_leibniz_integration())
    
    print("\n" + "="*70)
    print("✅ Voice cloning test complete!")
    print("="*70)


async def test_leibniz_integration():
    """Test XTTS integration with Leibniz TTS system"""
    try:
        # Add parent directory to path for imports
        import sys
        script_dir = Path(__file__).parent.parent.resolve()
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        # Import Leibniz TTS
        from leibniz_agent.leibniz_tts import LeibnizTTS, LeibnizTTSConfig
        
        # Initialize config with XTTS
        config = LeibnizTTSConfig(
            provider="xtts_local",
            xtts_speaker_sample=SPEAKER_SAMPLE_PATH,
            xtts_language="en",
            xtts_device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Get TTS instance
        print(f"\n🔄 Initializing Leibniz TTS with XTTS provider...")
        tts = LeibnizTTS(config=config)
        
        # Test synthesis
        test_text = "Hello! This is a test of the XTTS local provider integration."
        print(f"📝 Synthesizing: {test_text}")
        
        # Create output directory
        output_file = os.path.join(OUTPUT_DIR, "integration_test.wav")
        
        result = await tts.synthesize_to_file(text=test_text, outfile=output_file, emotion="helpful")
        
        if result['success']:
            print(f"\n✅ Integration test successful!")
            print(f"   Audio file: {result['file']}")
            print(f"   Duration: {result['duration']:.2f}s")
            print(f"   Provider: {result['provider']}")
            print(f"   Cached: {result.get('cached', False)}")
        else:
            print(f"\n❌ Integration test failed: {result.get('error')}")
    except ImportError as e:
        print(f"\n⚠️ Integration test skipped - Leibniz TTS not available: {e}")
        print("   This is normal if running from leibniz_agent/ directory.")
        print("   Run from repository root for integration test.")
    except Exception as e:
        print(f"\n❌ Integration test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
