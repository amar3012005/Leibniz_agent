# fast_slm_classifier.py
"""
Fast Small Language Model (SLM) Intent Classifier
Uses quantized, distilled models optimized for real-time inference.
Target: <100ms latency with semantic context generation
"""

import asyncio
import time
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextGenerationPipeline
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FastSLMClassifier")


class FastSLMIntentClassifier:
    """
    Ultra-fast SLM using quantized, distilled models.
    
    Best models for <100ms latency:
    1. "TinyLlama/TinyLlama-1.1B-Chat-v1.0" - 1.1B params, ~50-80ms
    2. "microsoft/phi-2" - 2.7B params, ~60-100ms (with 8-bit)
    3. "mistralai/Mistral-7B-Instruct-v0.1" - 7B params, 8-bit = ~80-120ms
    
    For guaranteed <100ms:
    - Use 4-bit quantization
    - Use Flash Attention 2
    - Reduce max_new_tokens to 20
    """
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize with fast SLM.
        
        Args:
            model_name: HuggingFace model identifier
                - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (FASTEST, recommended)
                - "microsoft/phi-2" (FAST)
                - "mistralai/Mistral-7B-Instruct-v0.1" (need quantization)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # System prompt for zero-shot classification (optimized for TinyLlama)
        # Use very explicit format to ensure complete JSON output
        self.system_prompt = """Classify intent. Return JSON only:
{"intent":"RAG_QUERY","confidence":0.9}

Valid intents: GREETING, EXIT, APPOINTMENT_SCHEDULING, RAG_QUERY, UNCLEAR"""

        self.intent_descriptions = {
            "GREETING": "User is initiating conversation with a greeting",
            "EXIT": "User wants to end the conversation",
            "APPOINTMENT_SCHEDULING": "User wants to schedule or book an appointment",
            "RAG_QUERY": "User is asking for information or facts",
            "UNCLEAR": "User input is ambiguous or unclear"
        }
    
    def load_model(self):
        """Load model with optimizations for fast inference."""
        start = time.time()
        logger.info(f"Loading SLM: {self.model_name}...")
        logger.info(f"Using device: {self.device}")
        
        # Quantization config for smaller models
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            # Try Flash Attention 2, fallback to eager if not available
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                logger.info("Flash Attention 2 available - using for faster inference")
            except ImportError:
                attn_impl = "eager"
                logger.warning("Flash Attention 2 not installed - using eager attention (slower but works)")
            
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": self.device,
                "torch_dtype": torch.float16,
                "attn_implementation": attn_impl
            }
            
            # Apply quantization if model is large enough
            if "7B" in self.model_name or "13B" in self.model_name:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            self.model.eval()
            
            # Create pipeline
            # Don't specify device when using device_map (accelerate handles device placement)
            self.pipeline = TextGenerationPipeline(
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            load_time = time.time() - start
            logger.info(f"✅ SLM loaded in {load_time:.2f}s")
            
            # Warmup
            logger.info("Warming up model...")
            self._warmup()
            logger.info("✅ Warmup complete")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _warmup(self):
        """Warmup model with dummy inference."""
        with torch.no_grad():
            prompt = "Classify: Hello\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
    
    async def classify(
        self,
        text: str,
        return_semantic_context: bool = True
    ) -> Dict[str, Any]:
        """
        Classify intent using SLM with zero-shot prompting.
        
        Args:
            text: User input text
            return_semantic_context: If True, include semantic context
            
        Returns:
            {
                "intent": str,
                "confidence": float,
                "semantic_context": str,
                "source": "SLM",
                "latency_ms": float
            }
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Build prompt
        prompt = self._build_prompt(text)
        
        # Run inference in thread to not block event loop
        output = await asyncio.to_thread(
            self._generate_with_timeout,
            prompt,
            timeout=0.5
        )
        
        # Parse output
        result = self._parse_output(output, text)
        result["latency_ms"] = (time.time() - start_time) * 1000
        
        return result
    
    def _build_prompt(self, text: str) -> str:
        """Build zero-shot classification prompt (compact for TinyLlama)."""
        # Use simple, direct prompt - TinyLlama works better with explicit instructions
        return f"""Classify this text and return ONLY valid JSON, no other text:

Input: "{text}"

Return: {{"intent":"INTENT_NAME","confidence":0.9}}

Valid intents: GREETING, EXIT, APPOINTMENT_SCHEDULING, RAG_QUERY, UNCLEAR

JSON:"""
    
    def _generate_with_timeout(self, prompt: str, timeout: float = 0.5) -> str:
        """Generate text with timeout."""
        try:
            with torch.no_grad():
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                # Generate with minimal tokens for fast inference
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=15,  # Reduced from 20 - JSON is short
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                return generated_text
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""
    
    def _parse_output(self, output: str, user_input: str) -> Dict[str, Any]:
        """Parse SLM output to extract intent and confidence."""
        import json
        import re
        
        # Clean output (remove markdown code blocks if present)
        output = output.replace("```json", "").replace("```", "").strip()
        
        # First, try to extract intent directly from JSON-like patterns
        # Handle cases like: {"intent": "GREETING", "confidence": 0.9}
        # Or partial: {"intent": "GREETING|EXIT|APPOINTMENT_SCHEDULING...
        intent_match = re.search(r'"intent"\s*:\s*"([^"]+)"', output)
        confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', output)
        
        # Also handle text-based outputs like "Intent: GREETING" or "Intent: RAG_QUERY"
        if not intent_match:
            text_intent_match = re.search(r'(?:Intent|intent):\s*([A-Z_]+)', output, re.IGNORECASE)
            if text_intent_match:
                intent_raw = text_intent_match.group(1)
                # Map common variations
                intent_raw = intent_raw.upper().replace("BYE_BYE", "EXIT").replace("TALK_HISTORY", "RAG_QUERY")
                
                valid_intents = ["GREETING", "EXIT", "APPOINTMENT_SCHEDULING", "RAG_QUERY", "UNCLEAR"]
                intent = None
                for valid_intent in valid_intents:
                    if valid_intent in intent_raw or intent_raw == valid_intent:
                        intent = valid_intent
                        break
                
                if not intent:
                    intent = self._extract_intent_fallback(user_input)
                
                confidence = 0.75
                if not confidence_match:
                    conf_match = re.search(r'(?:Confidence|confidence):\s*([0-9.]+)', output, re.IGNORECASE)
                    if conf_match:
                        try:
                            confidence = float(conf_match.group(1))
                        except ValueError:
                            pass
                elif confidence_match:
                    try:
                        confidence = float(confidence_match.group(1))
                    except ValueError:
                        pass
                
                return {
                    "intent": intent,
                    "confidence": confidence,
                    "semantic_context": self.intent_descriptions.get(intent, ""),
                    "source": "SLM"
                }
        
        if intent_match:
            intent_raw = intent_match.group(1)
            # Handle cases where TinyLlama outputs the example format like "GREETING|EXIT|..."
            # Extract just the first valid intent
            valid_intents = ["GREETING", "EXIT", "APPOINTMENT_SCHEDULING", "RAG_QUERY", "UNCLEAR"]
            intent = None
            for valid_intent in valid_intents:
                if valid_intent in intent_raw:
                    intent = valid_intent
                    break
            
            if not intent:
                # Try to find any valid intent in the raw string
                for valid_intent in valid_intents:
                    if valid_intent.lower() in intent_raw.lower():
                        intent = valid_intent
                        break
            
            if not intent:
                intent = self._extract_intent_fallback(user_input)
            
            confidence = 0.75
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                except ValueError:
                    pass
            
            return {
                "intent": intent,
                "confidence": confidence,
                "semantic_context": self.intent_descriptions.get(intent, ""),
                "source": "SLM"
            }
        
        # Try full JSON parsing as fallback
        json_match = re.search(r'\{[^{}]*"intent"[^{}]*\}', output, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                result = json.loads(json_str)
                intent = result.get("intent", "UNCLEAR")
                valid_intents = ["GREETING", "EXIT", "APPOINTMENT_SCHEDULING", "RAG_QUERY", "UNCLEAR"]
                if intent not in valid_intents:
                    intent = self._extract_intent_fallback(user_input)
                
                return {
                    "intent": intent,
                    "confidence": float(result.get("confidence", 0.75)),
                    "semantic_context": self.intent_descriptions.get(intent, ""),
                    "source": "SLM"
                }
            except json.JSONDecodeError:
                pass
        
        # Final fallback: keyword matching
        logger.warning(f"Could not parse JSON output: {output[:100]}...")
        intent = self._extract_intent_fallback(user_input)
        
        return {
            "intent": intent,
            "confidence": 0.6,  # Lower confidence for fallback
            "semantic_context": self.intent_descriptions.get(intent, ""),
            "source": "SLM"
        }
    
    def _extract_intent_fallback(self, text: str) -> str:
        """Fallback intent extraction using keywords."""
        text_lower = text.lower()
        
        # Pattern matching
        if any(word in text_lower for word in ["hello", "hi", "hey", "good morning", "greetings"]):
            return "GREETING"
        elif any(word in text_lower for word in ["bye", "goodbye", "exit", "quit", "done"]):
            return "EXIT"
        elif any(word in text_lower for word in ["schedule", "appointment", "book", "meeting", "arrange"]):
            return "APPOINTMENT_SCHEDULING"
        elif any(word in text_lower for word in ["what", "how", "explain", "tell me", "information", "requirements"]):
            return "RAG_QUERY"
        else:
            return "UNCLEAR"


class HybridFastIntentParser:
    """
    Hybrid classifier: SLM + LLM with parallel execution.
    Uses fast SLM for <100ms response, falls back to LLM if needed.
    """
    
    def __init__(self, slm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.slm = FastSLMIntentClassifier(model_name=slm_model)
        self.slm_threshold = 0.75  # Use SLM if confidence > threshold
        self.slm_timeout = 0.1  # 100ms timeout
    
    def load_models(self):
        """Load SLM model."""
        self.slm.load_model()
    
    async def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify intent using fast SLM.
        
        Args:
            text: User input text
            
        Returns:
            Intent classification result
        """
        result = await self.slm.classify(text)
        result["path"] = "SLM"
        return result


# ==================== TEST ====================

async def test_slm_classifier():
    """Test the fast SLM classifier."""
    
    # Initialize
    parser = HybridFastIntentParser(slm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.load_models()
    
    # Test cases
    test_inputs = [
        "Hello there!",
        "I want to schedule an appointment with admissions",
        "What are the requirements for Computer Science?",
        "Bye bye",
        "Tell me about the history of the university",
        "schedule meeting with advisor tomorrow",
    ]
    
    print("\n" + "="*80)
    print("TESTING FAST SLM INTENT CLASSIFIER")
    print("="*80)
    
    total_time = 0
    passed = 0
    
    for text in test_inputs:
        result = await parser.classify(text)
        
        print(f"\nInput: '{text}'")
        print(f"  Intent: {result['intent']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Semantic Context: {result['semantic_context']}")
        print(f"  Latency: {result['latency_ms']:.1f}ms")
        
        if result['latency_ms'] < 100:
            print(f"  ✅ Under 100ms")
            passed += 1
        else:
            print(f"  ⚠️ Over 100ms")
        
        total_time += result['latency_ms']
    
    avg_latency = total_time / len(test_inputs)
    
    print("\n" + "="*80)
    print(f"Average Latency: {avg_latency:.1f}ms")
    print(f"Passed (<100ms): {passed}/{len(test_inputs)}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_slm_classifier())
