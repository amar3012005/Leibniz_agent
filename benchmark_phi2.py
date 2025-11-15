"""
Direct Performance Comparison: HuggingFace Phi-2 vs GGUF Phi-2
===============================================================

Tests both approaches on the same queries to determine which is faster.
"""

import time
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Test query
TEST_QUERY = "What are the admission requirements for the Master's program?"
TEST_CONTEXT = """
Leibniz University Master's Program Admission Requirements:
- Bachelor's degree from accredited institution
- Minimum GPA of 3.0 on 4.0 scale
- Relevant field of study matching program
- English proficiency test (TOEFL/IELTS)
- Letters of recommendation (2-3)
- Statement of purpose
"""

def test_huggingface():
    """Test HuggingFace Phi-2 (optimized settings)"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        logging.info("\n" + "="*60)
        logging.info("🤗 TESTING HUGGINGFACE PHI-2 (Optimized)")
        logging.info("="*60)
        
        # Load model
        logging.info("📥 Loading HuggingFace model...")
        start = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        
        load_time = time.time() - start
        logging.info(f"✅ Loaded in {load_time:.2f}s")
        
        # Warmup
        logging.info("🔥 Warming up...")
        start = time.time()
        prompt = "Test warmup"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        logging.info(f"✅ Warmed up in {time.time()-start:.2f}s")
        
        # Test 3 generations
        logging.info("\n🧪 Running 3 test generations...")
        times = []
        
        for i in range(3):
            prompt = f"""Context: {TEST_CONTEXT}

Question: {TEST_QUERY}

Answer (friendly, 2-3 sentences):"""
            
            start = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                min_new_tokens=40,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
            answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            gen_time = (time.time() - start) * 1000
            times.append(gen_time)
            
            logging.info(f"   Run {i+1}: {gen_time:.0f}ms ({len(answer)} chars)")
        
        avg_time = sum(times) / len(times)
        logging.info(f"\n📊 HuggingFace Average: {avg_time:.0f}ms")
        
        return {"name": "HuggingFace Phi-2", "avg_ms": avg_time, "times": times}
    
    except Exception as e:
        logging.error(f"❌ HuggingFace test failed: {e}")
        return None

def test_gguf():
    """Test GGUF Phi-2"""
    try:
        from llama_cpp import Llama
        
        logging.info("\n" + "="*60)
        logging.info("⚡ TESTING GGUF PHI-2 (Q4_K_M)")
        logging.info("="*60)
        
        model_path = "models/phi-2.Q4_K_M.gguf"
        
        # Load model
        logging.info("📥 Loading GGUF model...")
        start = time.time()
        
        llm = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_threads=8,
            n_gpu_layers=-1,     # All layers to GPU
            n_batch=512,
            verbose=False,
            use_mmap=True,
            f16_kv=True
        )
        
        load_time = time.time() - start
        logging.info(f"✅ Loaded in {load_time:.2f}s")
        
        # Warmup
        logging.info("🔥 Warming up...")
        start = time.time()
        _ = llm("Test warmup", max_tokens=5, temperature=0.7)
        logging.info(f"✅ Warmed up in {time.time()-start:.2f}s")
        
        # Test 3 generations
        logging.info("\n🧪 Running 3 test generations...")
        times = []
        
        for i in range(3):
            prompt = f"""Context: {TEST_CONTEXT}

Question: {TEST_QUERY}

Answer (friendly, 2-3 sentences):"""
            
            start = time.time()
            output = llm(
                prompt,
                max_tokens=60,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.15,
                stop=["Question:", "\n\n"],
                echo=False,
                threads=8
            )
            answer = output['choices'][0]['text'].strip()
            gen_time = (time.time() - start) * 1000
            times.append(gen_time)
            
            logging.info(f"   Run {i+1}: {gen_time:.0f}ms ({len(answer)} chars)")
        
        avg_time = sum(times) / len(times)
        logging.info(f"\n📊 GGUF Average: {avg_time:.0f}ms")
        
        return {"name": "GGUF Phi-2 Q4", "avg_ms": avg_time, "times": times}
    
    except Exception as e:
        logging.error(f"❌ GGUF test failed: {e}")
        return None

def main():
    logging.info("\n" + "="*60)
    logging.info("🏁 PHI-2 PERFORMANCE BENCHMARK")
    logging.info("="*60)
    
    results = []
    
    # Test HuggingFace
    hf_result = test_huggingface()
    if hf_result:
        results.append(hf_result)
    
    # Test GGUF
    gguf_result = test_gguf()
    if gguf_result:
        results.append(gguf_result)
    
    # Compare
    if len(results) == 2:
        logging.info("\n" + "="*60)
        logging.info("🏆 FINAL COMPARISON")
        logging.info("="*60)
        
        for r in results:
            logging.info(f"\n{r['name']}:")
            logging.info(f"  Average: {r['avg_ms']:.0f}ms")
            logging.info(f"  Runs: {[f'{t:.0f}ms' for t in r['times']]}")
        
        if results[0]['avg_ms'] < results[1]['avg_ms']:
            faster = results[0]
            slower = results[1]
        else:
            faster = results[1]
            slower = results[0]
        
        speedup = slower['avg_ms'] / faster['avg_ms']
        
        logging.info(f"\n🏆 WINNER: {faster['name']}")
        logging.info(f"   {speedup:.2f}x faster than {slower['name']}")
        logging.info(f"   ({faster['avg_ms']:.0f}ms vs {slower['avg_ms']:.0f}ms)")
        logging.info("="*60)

if __name__ == "__main__":
    main()
