"""
Simple TTS test using Tacotron2 + WaveGlow (no voice cloning, works with all transformers versions)
This is a fallback test if XTTS v2 has compatibility issues.
"""
import torch
from TTS.api import TTS
import os

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using device: {device}")

# Use a simpler model (Tacotron2) that doesn't require transformers
print(" Loading Tacotron2 model (English, single speaker)...")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True).to(device)
print(" Model loaded successfully!")

# Test text
test_text = "Hello! Welcome to Leibniz University. How can I help you today?"

# Output directory
output_dir = "leibniz_agent/test_tts_output"
os.makedirs(output_dir, exist_ok=True)

# Generate speech
print("\n Synthesizing speech...")
output_file = os.path.join(output_dir, "test_tacotron2.wav")

try:
    tts.tts_to_file(
        text=test_text,
        file_path=output_file
    )
    print(f" Generated audio: {output_file}")
    print(f"   File size: {os.path.getsize(output_file)} bytes")
    
    print("\n TTS test complete!")
    print(f" Output file: {output_file}")
    print("\n This is a simpler model without voice cloning.")
    print("   For voice cloning with XTTS v2, you need:")
    print("   pip install 'transformers<4.50'")
    
except Exception as e:
    print(f"\n TTS synthesis failed: {e}")
    import traceback
    traceback.print_exc()
