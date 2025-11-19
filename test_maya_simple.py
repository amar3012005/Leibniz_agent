"""
Simple Maya1 Test - Minimal Dependencies
Tests Maya1 voice AI without full XTTS stack
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("=" * 70)
    print("Maya1 Simple Test - Voice AI Diagnostic")
    print("=" * 70)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load model
    print("\n Loading Maya1 model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "maya-research/maya1",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained("maya-research/maya1", trust_remote_code=True)
        print(" Model loaded successfully!")
    except Exception as e:
        print(f" Model load failed: {e}")
        return
    
    # Test generation
    print("\n Testing token generation...")
    description = "Male voice, 30s, warm, conversational"
    text = "Hello world"
    prompt = f'<description="{description}"> {text}'
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"   Input tokens: {len(inputs['input_ids'][0])}")
    
    # Generate WITHOUT eos_token_id constraint
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
    print(f"   Generated tokens: {len(generated_ids)}")
    print(f"   Token range: {generated_ids.min().item()} - {generated_ids.max().item()}")
    
    # Check for SNAC tokens
    snac_count = sum(1 for t in generated_ids if 128266 <= t <= 156937)
    eos_count = sum(1 for t in generated_ids if t == 128009)
    
    print(f"\n Token Analysis:")
    print(f"   SNAC audio tokens (128266-156937): {snac_count}")
    print(f"   EOS tokens (128009): {eos_count}")
    print(f"   Other tokens: {len(generated_ids) - snac_count - eos_count}")
    
    if snac_count > 0:
        print("\n SUCCESS! Maya1 is generating audio tokens!")
        print(f"   Audio frames: {snac_count // 7}")
        print(f"   Approx duration: {(snac_count // 7) / 47:.2f} seconds")
    else:
        print("\n FAIL: No audio tokens generated")
        print("\n Debug info:")
        print(f"   First 20 tokens: {generated_ids[:20].tolist()}")
        print(f"   Decoded output: {tokenizer.decode(generated_ids, skip_special_tokens=False)[:200]}")
        
        print("\n Possible issues:")
        print("   1. Model may need specific prompt format")
        print("   2. Generation parameters may need tuning")
        print("   3. Model checkpoint may have issues")
        print("\n   Try official vLLM script from:")
        print("   https://huggingface.co/maya-research/maya1/blob/main/vllm_streaming_inference.py")

if __name__ == "__main__":
    main()
