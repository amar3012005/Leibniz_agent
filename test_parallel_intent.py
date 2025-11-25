import asyncio
import time
import os
import torch
import logging
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HybridIntentParser")

# Load environment variables
load_dotenv()

# Override/Set Gemini API Key from user request
os.environ["GEMINI_API_KEY"] = "AIzaSyAhRwxvXku92nQqMsNM9jXijL3m6idH8OQ"

# Import existing Gemini parser
try:
    from leibniz_intent_parser import classify_leibniz_intent
except ImportError:
    logger.error("Could not import classify_leibniz_intent from leibniz_intent_parser.py")
    # Mock for testing if file missing
    async def classify_leibniz_intent(text, context=None):
        await asyncio.sleep(1.5) # Simulate latency
        return {"intent": "RAG_QUERY", "confidence": 0.95, "reasoning": "Mock LLM result"}

# Import fast SLM classifier (TinyLlama)
try:
    from fast_slm_classifier import HybridFastIntentParser, FastSLMIntentClassifier
    FAST_SLM_AVAILABLE = True
except ImportError:
    logger.warning("fast_slm_classifier.py not found, falling back to Phi-3")
    FAST_SLM_AVAILABLE = False

# ==================================================================================
# SLM SETUP (Phi-3 Mini - Fallback)
# ==================================================================================

class Phi3IntentClassifier:
    def __init__(self, model_id="microsoft/Phi-3-mini-4k-instruct"):
        self.model_id = model_id
        self.pipe = None
        self.tokenizer = None
        
    def load_model(self):
        logger.info(f"Loading SLM: {self.model_id}...")
        start_time = time.time()
        
        try:
            # Check for CUDA
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device_map}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=False)
            
            # Load model with 4-bit quantization for speed and memory efficiency if on CUDA
            # otherwise normal load
            model_kwargs = {
                "trust_remote_code": False,
                "device_map": device_map, 
            }
            
            if device_map == "cuda":
                # fast quantization
                try:
                    import bitsandbytes
                    logger.info("BitsAndBytes found, attempting 4-bit load...")
                    model_kwargs["load_in_4bit"] = True
                    model_kwargs["torch_dtype"] = torch.float16
                except ImportError:
                    logger.warning("BitsAndBytes not found, falling back to float16")
                    model_kwargs["torch_dtype"] = torch.float16
            else:
                model_kwargs["torch_dtype"] = torch.float32

            # Fix for 'DynamicCache' object has no attribute 'seen_tokens'
            model_kwargs["attn_implementation"] = "eager"

            logger.info(f"Loading model with kwargs: {model_kwargs}")
            
            # Use snapshot_download first to ensure files are present and show progress
            from huggingface_hub import snapshot_download
            logger.info("Ensuring model files are downloaded (this may take a while)...")
            snapshot_download(repo_id=self.model_id, allow_patterns=["*.json", "*.safetensors", "*.model"])

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                **model_kwargs
            )
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
            
            logger.info(f"SLM loaded in {time.time() - start_time:.2f}s")
            
            # Warmup inference
            logger.info("Warming up SLM inference...")
            self.pipe("Warmup", max_new_tokens=1)
            logger.info("SLM Warmup complete.")
            
        except Exception as e:
            logger.error(f"Failed to load SLM: {e}")
            raise

    async def classify(self, text: str) -> Dict[str, Any]:
        if not self.pipe:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Prompt engineering for Phi-3 (Optimized for Leibniz University context)
        # COMPACT PROMPT for lower latency (Prefill Optimization)
        prompt = f"""<|user|>
Classify intent:
1. APPOINTMENT_SCHEDULING ("book", "schedule")
2. RAG_QUERY (info, facts)
3. GREETING (hello, hi)
4. EXIT (bye, quit)
5. UNCLEAR

Input: "{text}"

JSON only: {{"intent": "...", "confidence": 0.9}}
<|end|>
<|assistant|>"""

        # Log token count for profiling
        if self.tokenizer:
            input_tokens = self.tokenizer(prompt, return_tensors="pt")
            token_count = input_tokens.input_ids.shape[1]
            logger.info(f"Prompt tokens: {token_count}")

        # Run blocking inference in a separate thread to avoid blocking the event loop
        # Using asyncio.to_thread (Python 3.9+)
        start_gen = time.time()
        result = await asyncio.to_thread(
            self.pipe,
            prompt, 
            max_new_tokens=15, 
            return_full_text=False,
            do_sample=False, # Greedy for determinism and speed
            temperature=0.0
        )
        gen_time = time.time() - start_gen
        
        # Estimate TTFT (Time To First Token) if possible or just log total generation time
        # With `pipeline`, we get the full result at once, so we track total time.
        # Assuming linear generation, TTFT ~= (Total Time / Tokens) * Prefill Overhead
        # But for accurate profiling, we just log the total time here.
        logger.info(f"SLM Generation Time: {gen_time*1000:.2f}ms for input '{text[:20]}...'")
        
        generated_text = result[0]['generated_text'].strip()
        
        # Parse JSON
        try:
            # Find JSON substring
            import re
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                return {
                    "intent": data.get("intent", "UNCLEAR"),
                    "confidence": float(data.get("confidence", 0.5)),
                    "source": "SLM_Phi3"
                }
            else:
                logger.warning(f"SLM output not JSON: {generated_text}")
                return {"intent": "UNCLEAR", "confidence": 0.0, "source": "SLM_Phi3_Error"}
        except Exception as e:
            logger.error(f"SLM parsing error: {e}")
            return {"intent": "UNCLEAR", "confidence": 0.0, "source": "SLM_Phi3_Error"}

# ==================================================================================
# HYBRID PARSER
# ==================================================================================

class HybridIntentParser:
    def __init__(self, use_fast_slm: bool = True):
        """
        Initialize hybrid parser.
        
        Args:
            use_fast_slm: If True and available, use TinyLlama (fast). Otherwise use Phi-3.
        """
        self.use_fast_slm = use_fast_slm and FAST_SLM_AVAILABLE
        
        if self.use_fast_slm:
            logger.info("Using FAST SLM (TinyLlama) for <100ms latency")
            self.fast_slm_parser = HybridFastIntentParser()
            self.slm = None  # Will use fast_slm_parser instead
        else:
            logger.info("Using Phi-3 Mini SLM (fallback)")
            self.slm = Phi3IntentClassifier()
            self.fast_slm_parser = None
        
        self.slm_threshold = 0.75 if self.use_fast_slm else 0.85
        # Timeout for SLM fast path
        self.slm_timeout = 0.1 if self.use_fast_slm else 0.5  # 100ms for TinyLlama, 500ms for Phi-3

    def load_models_sync(self):
        """Load models synchronously before async loop starts"""
        if self.use_fast_slm:
            self.fast_slm_parser.load_models()
        else:
            self.slm.load_model()

    async def classify_intent_slm_only(self, transcript: str):
        """Test SLM in isolation"""
        if self.use_fast_slm:
            result = await self.fast_slm_parser.classify(transcript)
            # Normalize format to match Phi-3 output
            return {
                "intent": result.get("intent", "UNCLEAR"),
                "confidence": result.get("confidence", 0.5),
                "source": "SLM_TinyLlama",
                "latency_ms": result.get("latency_ms", 0)
            }
        else:
            return await self.slm.classify(transcript)

    async def classify_intent_llm_only(self, transcript: str):
        """Test LLM in isolation"""
        return await classify_leibniz_intent(transcript)

    async def classify_intent_hybrid(self, transcript: str):
        """Parallel execution with SLM fast-path"""
        start_time = time.time()
        
        # Create tasks - use fast SLM if available, otherwise Phi-3
        if self.use_fast_slm:
            slm_task = asyncio.create_task(self.fast_slm_parser.classify(transcript))
        else:
            slm_task = asyncio.create_task(self.slm.classify(transcript))
        
        llm_task = asyncio.create_task(classify_leibniz_intent(transcript))
        
        final_result = None
        path_taken = "unknown"

        try:
            # Wait for SLM with timeout
            # If SLM finishes within timeout, we check confidence
            slm_result = await asyncio.wait_for(slm_task, timeout=self.slm_timeout)
            
            # Normalize fast SLM result format
            if self.use_fast_slm:
                slm_result = {
                    "intent": slm_result.get("intent", "UNCLEAR"),
                    "confidence": slm_result.get("confidence", 0.5),
                    "source": "SLM_TinyLlama"
                }
            
            if slm_result["confidence"] > self.slm_threshold:
                # SLM is confident! Cancel LLM
                path_taken = "SLM_FAST"
                final_result = slm_result
                
                # Ensure basic context structure exists
                if "context" not in final_result:
                     final_result["context"] = {
                        "user_goal": f"Intent: {final_result['intent']}",
                        "key_entities": {},
                        "extracted_meaning": transcript
                    }
                
                # Cancel LLM task
                llm_task.cancel()
                try:
                    await llm_task
                except asyncio.CancelledError:
                    logger.info(f"LLM cancelled after confident SLM result ({slm_result['confidence']})")
                
            else:
                # SLM not confident, wait for LLM
                logger.info(f"SLM low confidence ({slm_result['confidence']}), waiting for LLM...")
                path_taken = "LLM_FALLBACK_CONFIDENCE"
                final_result = await llm_task

        except asyncio.TimeoutError:
            logger.info("SLM timed out, waiting for LLM...")
            path_taken = "LLM_FALLBACK_TIMEOUT"
            try:
                final_result = await llm_task
            except asyncio.CancelledError:
                logger.error("LLM task was cancelled unexpectedly")
                raise
            except Exception as e:
                logger.error(f"LLM failed: {e}")
                # Try to get SLM result if it eventually finishes?
                try:
                    # Wait a bit longer for SLM if LLM failed? Or just fail.
                    # Here we just fallback to safe default
                    final_result = {"intent": "UNCLEAR", "confidence": 0.0}
                except:
                    final_result = {"intent": "UNCLEAR", "confidence": 0.0}

        except Exception as e:
            logger.error(f"Error in hybrid classification: {e}")
            # Fallback to LLM if SLM errors out immediately
            try:
                final_result = await llm_task
                path_taken = "LLM_FALLBACK_ERROR"
            except:
                final_result = {"intent": "UNCLEAR", "confidence": 0.0}

        duration = time.time() - start_time
        final_result["hybrid_path"] = path_taken
        final_result["total_latency"] = duration
        return final_result

# ==================================================================================
# TEST RUNNER
# ==================================================================================

async def run_tests_logic(parser: HybridIntentParser):
    test_cases = [
        ("Hello there!", "GREETING"),
        ("I want to schedule an appointment with admissions", "APPOINTMENT_SCHEDULING"),
        ("What are the requirements for Computer Science?", "RAG_QUERY"),
        ("Bye bye", "EXIT"),
        ("Tell me about the history of the university", "RAG_QUERY"),
        ("schedule meeting with advisor tomorrow", "APPOINTMENT_SCHEDULING"),
    ]
    
    # --- PHASE 1: SLM ONLY ---
    slm_name = "TinyLlama (Fast)" if parser.use_fast_slm else "Phi-3 Mini"
    print("\n" + "="*80)
    print(f"PHASE 1: TESTING SLM ONLY ({slm_name})")
    print("="*80)
    
    slm_latencies = []
    for text, expected in test_cases:
        print(f"\nInput: '{text}'")
        start = time.time()
        result = await parser.classify_intent_slm_only(text)
        latency = time.time() - start
        # Use latency_ms from result if available (more accurate for fast SLM)
        if "latency_ms" in result:
            latency = result["latency_ms"] / 1000.0
        slm_latencies.append(latency)
        
        intent = result.get("intent")
        conf = result.get("confidence")
        source = result.get("source", "SLM")
        is_correct = (intent == expected)
        status = "[PASS]" if is_correct else f"[FAIL] (Got {intent})"
        print(f"{status} | Latency: {latency*1000:.1f}ms | Conf: {conf:.2f} | Source: {source}")

    avg_slm = sum(slm_latencies) / len(slm_latencies)
    print(f"\nAvg SLM Latency: {avg_slm*1000:.1f}ms")
    if avg_slm < 0.1:
        print("[SUCCESS] SLM is FAST (<100ms) - Fast path viable!")
    elif avg_slm < 0.5:
        print("[WARNING] SLM is moderate (100-500ms) - May win race sometimes")
    else:
        print("[SLOW] SLM is SLOW (>500ms) - Will timeout, LLM fallback always used")

    # --- PHASE 2: LLM ONLY ---
    print("\n" + "="*80)
    print("PHASE 2: TESTING LLM ONLY (Gemini/Regex)")
    print("="*80)
    
    llm_latencies = []
    for text, expected in test_cases:
        print(f"\nInput: '{text}'")
        start = time.time()
        result = await parser.classify_intent_llm_only(text)
        latency = time.time() - start
        llm_latencies.append(latency)
        
        intent = result.get("intent")
        conf = result.get("confidence")
        is_correct = (intent == expected)
        status = "[PASS]" if is_correct else f"[FAIL] (Got {intent})"
        print(f"{status} | Latency: {latency*1000:.1f}ms | Conf: {conf:.2f}")

    avg_llm = sum(llm_latencies) / len(llm_latencies)
    print(f"\nAvg LLM Latency: {avg_llm*1000:.1f}ms")
    
    # --- PHASE 3: HYBRID PARALLEL ---
    print("\n" + "="*80)
    print("PHASE 3: TESTING HYBRID PARALLEL (SLM + LLM)")
    print("="*80)
    
    hybrid_latencies = []
    correct_count = 0
    
    for text, expected in test_cases:
        print(f"\nInput: '{text}'")
        
        result = await parser.classify_intent_hybrid(text)
        
        intent = result.get("intent")
        latency = result.get("total_latency", 0)
        path = result.get("hybrid_path")
        conf = result.get("confidence")
        
        hybrid_latencies.append(latency)
        is_correct = (intent == expected)
        if is_correct: correct_count += 1
        
        status = "[PASS]" if is_correct else f"[FAIL] (Got {intent})"
        print(f"{status} | Path: {path} | Latency: {latency*1000:.1f}ms | Conf: {conf:.2f}")
        
    avg_hybrid = sum(hybrid_latencies) / len(hybrid_latencies)
    print("\n" + "="*80)
    print(f"FINAL RESULTS: {correct_count}/{len(test_cases)} Correct")
    print(f"Avg Hybrid Latency: {avg_hybrid*1000:.1f}ms")
    print("="*80)

def main():
    # 1. Initialize and load models synchronously (BLOCKING)
    # This prevents blocking the asyncio event loop later
    parser = HybridIntentParser()
    
    print("Initializing models (Sync)...", flush=True)
    parser.load_models_sync()
    print("Models loaded.", flush=True)
    
    # 2. Run async tests
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(run_tests_logic(parser))

if __name__ == "__main__":
    main()
