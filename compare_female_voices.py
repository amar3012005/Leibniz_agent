"""
Compare best female humanized voice models from Coqui TTS
Tests quality, naturalness, and suitability for conversational AI
"""
import torch
from TTS.api import TTS
import os
import time

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {device}\n")

# Output directory
output_dir = "leibniz_agent/voice_comparison"
os.makedirs(output_dir, exist_ok=True)

# Test text - conversational university response
test_text = """Hello! Welcome to Leibniz University International Office. 
I can help you with information about exchange programs, admission requirements, 
and general university services. How can I assist you today?"""

# Alternative shorter test for faster comparison
short_test = "Hello! How can I help you with your university inquiries today?"

# Best models for female humanized conversational voices
models_to_test = [
    # === RECOMMENDED FOR LEIBNIZ/TARA ===
    {
        "name": "XTTS v2 (Voice Cloning)",
        "model_id": "tts_models/multilingual/multi-dataset/xtts_v2",
        "description": "🌟 BEST - Multilingual, voice cloning, most natural & human-like",
        "language": "en",
        "requires_speaker": True,
        "quality": "⭐⭐⭐⭐⭐",
        "speed": "Medium (2-3s)",
        "use_case": "Production - Best for natural conversations"
    },
    {
        "name": "YourTTS (Voice Cloning)",
        "model_id": "tts_models/multilingual/multi-dataset/your_tts",
        "description": "Multilingual voice cloning, good quality",
        "language": "en",
        "requires_speaker": True,
        "quality": "⭐⭐⭐⭐",
        "speed": "Medium (2-3s)",
        "use_case": "Alternative to XTTS v2"
    },
    
    # === ENGLISH MODELS (No voice cloning needed) ===
    {
        "name": "Jenny (English Female)",
        "model_id": "tts_models/en/jenny/jenny",
        "description": "High-quality English female voice, very natural",
        "language": "en",
        "requires_speaker": False,
        "quality": "⭐⭐⭐⭐",
        "speed": "Fast (1-2s)",
        "use_case": "English-only, fast inference"
    },
    {
        "name": "VCTK Multi-Speaker",
        "model_id": "tts_models/en/vctk/vits",
        "description": "109 English speakers (many female), good variety",
        "language": "en",
        "requires_speaker": False,
        "quality": "⭐⭐⭐⭐",
        "speed": "Fast (1-2s)",
        "use_case": "Multiple voices, accents available"
    },
    {
        "name": "LJSpeech VITS",
        "model_id": "tts_models/en/ljspeech/vits",
        "description": "Single female speaker, clear & professional",
        "language": "en",
        "requires_speaker": False,
        "quality": "⭐⭐⭐⭐",
        "speed": "Very Fast (<1s)",
        "use_case": "Fast, consistent female voice"
    },
    
    # === OTHER LANGUAGES (for TARA multilingual) ===
    {
        "name": "Italian Female (Mai)",
        "model_id": "tts_models/it/mai_female/vits",
        "description": "Italian female voice",
        "language": "it",
        "requires_speaker": False,
        "quality": "⭐⭐⭐",
        "speed": "Fast",
        "use_case": "Italian language support"
    },
    {
        "name": "Bengali Female",
        "model_id": "tts_models/bn/custom/vits-female",
        "description": "Bengali female voice (South Asian)",
        "language": "bn",
        "requires_speaker": False,
        "quality": "⭐⭐⭐",
        "speed": "Fast",
        "use_case": "South Asian language support"
    }
]

def test_model(model_info, test_text, use_short=False):
    """Test a single TTS model"""
    model_name = model_info["name"]
    model_id = model_info["model_id"]
    
    print(f"\n{'='*70}")
    print(f"🎤 Testing: {model_name}")
    print(f"   Model ID: {model_id}")
    print(f"   Description: {model_info['description']}")
    print(f"   Quality: {model_info['quality']} | Speed: {model_info['speed']}")
    print(f"   Use Case: {model_info['use_case']}")
    print(f"{'='*70}")
    
    try:
        # Load model
        print("📦 Loading model...")
        start_load = time.time()
        tts = TTS(model_name=model_id, progress_bar=False).to(device)
        load_time = time.time() - start_load
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Generate speech
        output_file = os.path.join(
            output_dir, 
            f"{model_name.replace(' ', '_').replace('/', '-')}.wav"
        )
        
        text_to_use = short_test if use_short else test_text
        print(f"🎙️ Synthesizing: '{text_to_use[:50]}...'")
        
        start_synth = time.time()
        
        if model_info["requires_speaker"]:
            # For voice cloning models, create a temp female voice sample
            import soundfile as sf
            import numpy as np
            
            # Create a simple voice sample (placeholder)
            sample_path = os.path.join(output_dir, "temp_female_voice.wav")
            if not os.path.exists(sample_path):
                # Generate 3s of pink noise (simulates voice better than white noise)
                sample_rate = 22050
                duration = 3
                samples = np.random.randn(sample_rate * duration) * 0.02
                # Simple lowpass filter to make it sound more voice-like
                for i in range(1, len(samples)):
                    samples[i] = 0.7 * samples[i] + 0.3 * samples[i-1]
                sf.write(sample_path, samples, sample_rate)
            
            tts.tts_to_file(
                text=text_to_use,
                speaker_wav=sample_path,
                language=model_info["language"],
                file_path=output_file
            )
        else:
            # For single-speaker or multi-speaker models without cloning
            if "vctk" in model_id.lower():
                # VCTK has multiple speakers - choose a female one
                # Speaker p225 is a female British English speaker
                tts.tts_to_file(
                    text=text_to_use,
                    speaker="p225",  # Female speaker
                    file_path=output_file
                )
            else:
                tts.tts_to_file(
                    text=text_to_use,
                    file_path=output_file
                )
        
        synth_time = time.time() - start_synth
        file_size = os.path.getsize(output_file)
        
        print(f"✅ Synthesis complete in {synth_time:.2f}s")
        print(f"📂 Output: {output_file}")
        print(f"💾 File size: {file_size / 1024:.1f} KB")
        print(f"⚡ Performance: Load={load_time:.2f}s, Synth={synth_time:.2f}s, Total={load_time + synth_time:.2f}s")
        
        return {
            "success": True,
            "load_time": load_time,
            "synth_time": synth_time,
            "file_size": file_size,
            "output_file": output_file
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

def main():
    print("🎯 FEMALE VOICE COMPARISON FOR LEIBNIZ/TARA AGENTS")
    print("="*70)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Test text length: {len(test_text)} chars")
    print("="*70)
    
    # Ask user preference
    print("\n📋 Choose test mode:")
    print("1. Quick test (short text, faster)")
    print("2. Full test (long text, realistic)")
    choice = input("Enter choice (1/2, default=1): ").strip() or "1"
    use_short = (choice == "1")
    
    results = []
    
    for i, model_info in enumerate(models_to_test, 1):
        print(f"\n\n{'#'*70}")
        print(f"# Test {i}/{len(models_to_test)}")
        print(f"{'#'*70}")
        
        result = test_model(model_info, test_text, use_short)
        results.append({
            "model": model_info["name"],
            "result": result
        })
        
        # Cleanup GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Summary
    print("\n\n" + "="*70)
    print("📊 SUMMARY RESULTS")
    print("="*70)
    
    successful = [r for r in results if r["result"]["success"]]
    failed = [r for r in results if not r["result"]["success"]]
    
    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        print("\n🏆 Best Models by Speed:")
        sorted_by_speed = sorted(
            successful, 
            key=lambda x: x["result"]["synth_time"]
        )
        for i, r in enumerate(sorted_by_speed[:3], 1):
            print(f"   {i}. {r['model']}: {r['result']['synth_time']:.2f}s")
        
        print("\n💎 RECOMMENDATION FOR PRODUCTION:")
        print("   1️⃣  XTTS v2 - Best quality, most natural, multilingual")
        print("       → Requires transformers<4.50 (pip install 'transformers<4.50')")
        print("   2️⃣  Jenny - Fast, high-quality English female voice")
        print("       → Works with current setup, no voice cloning")
        print("   3️⃣  LJSpeech VITS - Fastest, consistent female voice")
        print("       → Production-ready, very fast inference")
    
    print(f"\n📂 All audio files saved to: {output_dir}")
    print("🎧 Listen to the samples and choose the best for your use case!")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
