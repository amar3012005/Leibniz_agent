import torch
from TTS.api import TTS
import os
import soundfile as sf
import numpy as np

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using device: {device}")

# Init TTS with XTTS v2 (multilingual voice cloning model)
print(" Loading XTTS v2 model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print(" Model loaded successfully!")

# Test text (shortened for faster testing)
test_text = "Hello! Welcome to Leibniz University. How can I help you today?"

# Output directory
output_dir = "leibniz_agent/test_tts_output"
os.makedirs(output_dir, exist_ok=True)

# Create a simple voice sample WAV file for testing (1 second of silence at 22050 Hz)
# XTTS requires a voice sample, so we'll create a minimal one
sample_voice_path = os.path.join(output_dir, "temp_voice_sample.wav")

if not os.path.exists(sample_voice_path):
    print("\n Creating temporary voice sample...")
    # Generate 3 seconds of low-amplitude noise (simulates voice characteristics)
    sample_rate = 22050
    duration = 3
    samples = np.random.randn(sample_rate * duration) * 0.01  # Very quiet noise
    sf.write(sample_voice_path, samples, sample_rate)
    print(f" Created temp sample: {sample_voice_path}")

# Test with voice cloning using the temp sample
print("\n Testing TTS synthesis...")
output_file = os.path.join(output_dir, "test_output.wav")

try:
    tts.tts_to_file(
        text=test_text,
        speaker_wav=sample_voice_path,
        language="en",
        file_path=output_file
    )
    print(f" Generated audio: {output_file}")
    print(f"   File size: {os.path.getsize(output_file)} bytes")
    
    # Clean up temp sample
    if os.path.exists(sample_voice_path):
        os.remove(sample_voice_path)
        print(" Cleaned up temp voice sample")
    
    print("\n TTS test complete!")
    print(f" Output file: {output_file}")
    print("\n To use a real voice:")
    print("   1. Record 6-24 seconds of clear speech (WAV format)")
    print("   2. Replace speaker_wav parameter with your file path")
    
except Exception as e:
    print(f"\n TTS synthesis failed: {e}")
    import traceback
    traceback.print_exc()