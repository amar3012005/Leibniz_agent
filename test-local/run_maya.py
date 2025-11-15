"""
Maya1 Voice AI Test Script
===========================
Maya1: Best open-source voice AI model with emotional speech synthesis.

Features:
- Natural language voice design
- Emotional expressions (<laugh>, <sigh>, etc.)
- Multiple accents and voice characteristics
- 24kHz high-quality output

Requirements:
- CUDA GPU (recommended)
- ~8GB VRAM for model
- transformers, torch, snac, soundfile

Usage:
    python run_maya.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import soundfile as sf
import time

def main():
    print("=" * 70)
    print("Maya1 Voice AI - Emotional Speech Generation")
    print("=" * 70)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  CUDA not available. Running on CPU (will be slow)")
        print("   Recommend: Install PyTorch with CUDA support")
    
    print("\n📥 Loading Maya1 model (this may take a few minutes)...")
    start_time = time.time()
    
    # Load the Maya1 voice AI model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "maya-research/maya1", 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True  # Maya1 may require custom code
        )
        tokenizer = AutoTokenizer.from_pretrained("maya-research/maya1", trust_remote_code=True)
        print(f"✅ Maya1 model loaded ({time.time() - start_time:.1f}s)")
    except Exception as e:
        print(f"❌ Failed to load Maya1 model: {e}")
        print("\nTroubleshooting:")
        print("  1. Install dependencies: pip install transformers accelerate")
        print("  2. Ensure HuggingFace login: huggingface-cli login")
        print("  3. Check model access: https://huggingface.co/maya-research/maya1")
        return
    
    print("\n📥 Loading SNAC audio decoder (24kHz)...")
    start_time = time.time()
    
    # Load SNAC audio decoder
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
        print(f"✅ SNAC decoder loaded ({time.time() - start_time:.1f}s)")
    except Exception as e:
        print(f"❌ Failed to load SNAC decoder: {e}")
        print("\nInstall SNAC: pip install snac")
        return
    
    # Design your voice with natural language
    print("\n" + "=" * 70)
    print("🎤 Voice Design Configuration")
    print("=" * 70)
    
    description = "Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing."
    text = "Hello! This is Maya1 <laugh> the best open source voice AI model with emotions."
    
    print(f"Voice Description: {description}")
    print(f"Text to Synthesize: {text}")
    print(f"\nSupported Emotions: <laugh>, <sigh>, <gasp>, <breath>, <pause>, <cry>, <whisper>, <angry>")
    
    # Create prompt with voice design (CORRECT FORMAT from HuggingFace docs)
    prompt = f'<description="{description}"> {text}'
    
    print(f"\n📝 Prompt: {prompt[:150]}...")
    
    print(f"\n🎯 Generating emotional speech...")
    start_time = time.time()
    
    # Generate emotional speech with enhanced parameters
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # CRITICAL FIX: Don't use eos_token_id in generation - it causes immediate stopping!
    # SNAC tokens (128266-156937) come BEFORE the final EOS
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1000,         # Enough tokens for ~7 seconds of audio
            temperature=0.7,             # Balanced randomness
            top_p=0.95,                  # Broad sampling
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            # NOTE: NOT setting eos_token_id to allow full generation
            repetition_penalty=1.1       # Prevent token loops
        )
    
    generation_time = time.time() - start_time
    print(f"✅ Speech tokens generated ({generation_time:.1f}s)")
    
    # Extract SNAC audio tokens
    print("\n📦 Decoding audio tokens...")
    generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
    
    # Debug: Print token range
    if len(generated_ids) > 0:
        token_min, token_max = generated_ids.min().item(), generated_ids.max().item()
        print(f"   Token range: {token_min} - {token_max}")
        print(f"   Expected SNAC range: 128266 - 156937")
    
    print(f"   Total generated tokens: {len(generated_ids)}")
    
    # Multiple extraction strategies
    snac_tokens = [t.item() for t in generated_ids if 128266 <= t <= 156937]
    print(f"   Strategy 1 (SNAC range): {len(snac_tokens)} tokens")
    
    if len(snac_tokens) == 0:
        print("\n⚠️  WARNING: No tokens in standard SNAC range. Trying alternatives...")
        
        # Strategy 2: Broader audio codec range
        snac_tokens = [t.item() for t in generated_ids if t > 128000]
        print(f"   Strategy 2 (>128000): {len(snac_tokens)} tokens")
        
        if len(snac_tokens) == 0:
            # Strategy 3: Look for any high-value tokens
            snac_tokens = [t.item() for t in generated_ids if t > 100000]
            print(f"   Strategy 3 (>100000): {len(snac_tokens)} tokens")
    
    print(f"   Total SNAC tokens extracted: {len(snac_tokens)}")
    
    if len(snac_tokens) == 0:
        print("\n❌ CRITICAL: No audio tokens found!")
        print("   Possible issues:")
        print("   1. Maya1 model may use different token range")
        print("   2. Prompt format may be incorrect")
        print("   3. Model not properly trained for SNAC codec")
        print("\n   Generated token IDs (first 50):")
        print(f"   {generated_ids[:50].tolist()}")
        
        # Check if tokens are actually text
        decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=False)
        print(f"\n   Decoded output: {decoded_output[:200]}")
        
        print("\n💡 Troubleshooting:")
        print("   - Check HuggingFace model card for updated usage")
        print("   - Verify model is maya-research/maya1 (not a fine-tune)")
        print("   - Try different prompt format")
        return
    
    # Decode SNAC tokens to audio frames
    frames = len(snac_tokens) // 7
    codes = [[], [], []]
    
    for i in range(frames):
        s = snac_tokens[i*7:(i+1)*7]
        codes[0].append((s[0]-128266) % 4096)
        codes[1].extend([(s[1]-128266) % 4096, (s[4]-128266) % 4096])
        codes[2].extend([
            (s[2]-128266) % 4096, 
            (s[3]-128266) % 4096, 
            (s[5]-128266) % 4096, 
            (s[6]-128266) % 4096
        ])
    
    print(f"   Audio frames: {frames}")
    print(f"   Code layers: {[len(c) for c in codes]}")
    
    # Generate final audio with SNAC decoder
    print("\n🎵 Generating audio waveform...")
    start_time = time.time()
    
    codes_tensor = [
        torch.tensor(c, dtype=torch.long, device=device).unsqueeze(0) 
        for c in codes
    ]
    
    with torch.inference_mode():
        audio = snac_model.decoder(
            snac_model.quantizer.from_codes(codes_tensor)
        )[0, 0].cpu().numpy()
    
    decode_time = time.time() - start_time
    print(f"✅ Audio decoded ({decode_time:.1f}s)")
    
    # Save your emotional voice output
    output_file = "maya1_output.wav"
    sf.write(output_file, audio, 24000)
    
    print("\n" + "=" * 70)
    print("✅ SUCCESS! Voice generated successfully!")
    print("=" * 70)
    print(f"📁 Output file: {output_file}")
    print(f"📊 Audio duration: {len(audio) / 24000:.2f} seconds")
    print(f"📈 Sample rate: 24000 Hz")
    print(f"⏱️  Total time: {generation_time + decode_time:.1f}s")
    print(f"\n🎧 Play the audio:")
    print(f"   PowerShell: Start-Process {output_file}")
    print(f"   Command: {output_file}")
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Tip: Ensure all dependencies are installed:")
        print("   pip install torch transformers accelerate snac soundfile")
