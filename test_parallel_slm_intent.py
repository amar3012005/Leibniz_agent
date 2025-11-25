import asyncio
import logging
import time
import json
import os
import sys
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HybridIntentTest")

# Try to import llama_cpp for SLM
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. SLM will run in MOCK mode.")

# Import existing Gemini parser
try:
    from leibniz_intent_parser import LeibnizIntentParser, get_leibniz_parser
    LEIBNIZ_PARSER_AVAILABLE = True
except ImportError:
    LEIBNIZ_PARSER_AVAILABLE = False
    logger.warning("leibniz_intent_parser not found. LLM will run in MOCK mode.")

@dataclass
class IntentResult:
    intent: str
    confidence: float
    source: str  # "SLM" or "LLM"
    latency: float
    context: Dict[str, Any] = None

class SLMIntentClassifier:
    def __init__(self, model_path: str = "models/phi-2.Q3_K_M.gguf"):
        self.model_path = model_path
        self.llm = None
        
        if LLAMA_CPP_AVAILABLE and os.path.exists(model_path):
            logger.info(f"Loading SLM from {model_path}...")
            try:
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=4,
                    n_gpu_layers=-1, # Use all GPU layers if available
                    verbose=False
                )
                logger.info("SLM loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load SLM: {e}")
        else:
            if not os.path.exists(model_path):
                logger.warning(f"SLM model not found at {model_path}")
            logger.info("SLM initialized in MOCK mode (or fallback).")

    async def classify(self, text: str) -> IntentResult:
        start_time = time.time()
        
        if self.llm:
            # Real inference using SLM
            # We need to prompt it to act as an intent classifier
            prompt = f"""Classify the user intent into: APPOINTMENT_SCHEDULING, RAG_QUERY, GREETING, EXIT, UNCLEAR.
            
User: "{text}"
Intent:"""
            
            # Run in thread executor to not block async loop
            output = await asyncio.to_thread(
                self.llm,
                prompt,
                max_tokens=10,
                stop=["\n", "User:"],
                temperature=0.1
            )
            
            response_text = output['choices'][0]['text'].strip()
            
            # Simple parsing logic (can be improved)
            intent = "UNCLEAR"
            confidence = 0.5
            
            if "APPOINTMENT" in response_text.upper():
                intent = "APPOINTMENT_SCHEDULING"
                confidence = 0.95
            elif "RAG" in response_text.upper() or "QUERY" in response_text.upper():
                intent = "RAG_QUERY"
                confidence = 0.9
            elif "GREETING" in response_text.upper():
                intent = "GREETING"
                confidence = 0.95
            elif "EXIT" in response_text.upper():
                intent = "EXIT"
                confidence = 0.95
            
            # Simulate some confidence variation based on clarity
            # (In real scenario, logprobs would be used for confidence)
            
        else:
            # Mock inference
            await asyncio.sleep(0.05) # Simulate fast SLM (50ms)
            
            # Simple mock logic
            lower_text = text.lower()
            if "schedule" in lower_text or "appointment" in lower_text:
                intent = "APPOINTMENT_SCHEDULING"
                confidence = 0.92
            elif "hello" in lower_text:
                intent = "GREETING"
                confidence = 0.98
            elif "bye" in lower_text:
                intent = "EXIT"
                confidence = 0.98
            else:
                intent = "RAG_QUERY" # Default fallback
                confidence = 0.85
            
            response_text = intent

        latency = time.time() - start_time
        return IntentResult(
            intent=intent,
            confidence=confidence,
            source="SLM",
            latency=latency,
            context={"raw_output": response_text}
        )

class LLMIntentClassifier:
    def __init__(self):
        self.parser = None
        if LEIBNIZ_PARSER_AVAILABLE:
            # Initialize without full service manager for lighter test
            try:
                self.parser = LeibnizIntentParser()
                logger.info("Gemini LLM Parser initialized.")
            except Exception as e:
                logger.error(f"Failed to init Gemini parser: {e}")

    async def classify(self, text: str) -> IntentResult:
        start_time = time.time()
        
        if self.parser and self.parser.model:
            # Use real Gemini parser
            result = await self.parser._gemini_classification(text)
            intent = result.get("intent", "UNCLEAR")
            confidence = result.get("confidence", 0.0)
            context = result.get("context", {})
        else:
            # Mock LLM (slower)
            await asyncio.sleep(1.5) # Simulate LLM latency (1.5s)
            
            lower_text = text.lower()
            if "schedule" in lower_text:
                intent = "APPOINTMENT_SCHEDULING"
                confidence = 0.99
            else:
                intent = "RAG_QUERY"
                confidence = 0.95
            context = {"mock": True}

        latency = time.time() - start_time
        return IntentResult(
            intent=intent,
            confidence=confidence,
            source="LLM",
            latency=latency,
            context=context
        )

class HybridIntentClassifier:
    def __init__(self, slm_model_path: str = "models/phi-2.Q3_K_M.gguf"):
        self.slm = SLMIntentClassifier(model_path=slm_model_path)
        self.llm = LLMIntentClassifier()
        self.slm_timeout = 0.2 # 200ms timeout for SLM to decide before we commit to waiting for LLM? 
                               # Actually user said: "use low timeout (100 ms) on the SLM task: if it responds quickly... cancel LLM"
                               # But we launch BOTH.
        self.slm_threshold = 0.9
        
    async def classify(self, transcript: str) -> IntentResult:
        # Create tasks for both
        slm_task = asyncio.create_task(self.slm.classify(transcript))
        llm_task = asyncio.create_task(self.llm.classify(transcript))
        
        start_total = time.time()
        
        try:
            # Wait for SLM with short timeout
            # User suggested ~100ms. 
            slm_result = await asyncio.wait_for(slm_task, timeout=0.15)
            
            if slm_result.confidence > self.slm_threshold:
                # SLM is confident! Cancel LLM
                llm_task.cancel()
                try:
                    await llm_task
                except asyncio.CancelledError:
                    pass # Expected
                    
                logger.info(f"🚀 SLM Win: {slm_result.intent} ({slm_result.confidence:.2f}) in {slm_result.latency*1000:.1f}ms")
                return slm_result
            else:
                logger.info(f"⚠️ SLM Low Confidence: {slm_result.confidence:.2f}. Waiting for LLM...")
                
        except asyncio.TimeoutError:
            logger.info("⏱️ SLM Timeout. Waiting for LLM...")
        except Exception as e:
            logger.error(f"SLM Error: {e}")
        
        # Fallback to LLM
        try:
            llm_result = await llm_task
            logger.info(f"🐢 LLM Result: {llm_result.intent} ({llm_result.confidence:.2f}) in {llm_result.latency*1000:.1f}ms")
            return llm_result
        except asyncio.CancelledError:
            logger.warning("LLM Task Cancelled unexpectedly")
            raise
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return IntentResult("UNCLEAR", 0.0, "ERROR", 0.0)

async def run_accuracy_test():
    print("\n" + "="*80)
    print("HYBRID SLM + LLM INTENT CLASSIFICATION TEST")
    print("="*80 + "\n")
    
    # Initialize hybrid classifier
    # Note: Ensure you have a model at models/phi-2.Q3_K_M.gguf or similar if you want real SLM testing
    classifier = HybridIntentClassifier()
    
    test_cases = [
        "Hello there",
        "I want to schedule an appointment with Dr. Smith",
        "What are the admission requirements for CS?",
        "Bye now",
        "Can you tell me about the campus library?",
        "I need to book a meeting",
        "What is the tuition fee?"
    ]
    
    print(f"Running {len(test_cases)} test cases...\n")
    
    results = []
    
    for text in test_cases:
        print(f"Query: '{text}'")
        result = await classifier.classify(text)
        print(f"  -> Predicted: {result.intent} (Conf: {result.confidence:.2f}) via {result.source}")
        print("-" * 40)
        results.append(result)
        
    # Stats
    slm_wins = sum(1 for r in results if r.source == "SLM")
    llm_wins = sum(1 for r in results if r.source == "LLM")
    avg_latency = sum(r.latency for r in results) / len(results)
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Queries: {len(results)}")
    print(f"SLM Wins (Fast): {slm_wins}")
    print(f"LLM Wins (Slow): {llm_wins}")
    print(f"Average Latency: {avg_latency*1000:.1f}ms")
    
    if not LLAMA_CPP_AVAILABLE:
        print("\nNOTE: llama-cpp-python was not found. SLM ran in MOCK mode.")
        print("To enable real SLM testing:")
        print("1. pip install llama-cpp-python")
        print("2. Download a GGUF model (e.g. Phi-3-mini) to 'models/' directory")

if __name__ == "__main__":
    asyncio.run(run_accuracy_test())

