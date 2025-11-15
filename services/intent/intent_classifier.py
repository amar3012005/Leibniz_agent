"""
Intent Classification Core Logic

Extracted from leibniz_intent_parser.py for microservice deployment.

Reference:
    leibniz_agent/leibniz_intent_parser.py (lines 400-920) - Original classification logic
"""

import re
import json
import time
import asyncio
import logging
from typing import Dict, Any, Optional
import google.generativeai as genai

from config import IntentConfig
from patterns import load_intent_patterns, compile_patterns, get_system_prompt

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Two-tier intent classifier for Leibniz University Agent.
    
    Classification Strategy:
        1. Fast pattern matching (target >80% of requests)
        2. Gemini LLM fallback for complex cases (target <20%)
    
    Intents:
        - APPOINTMENT_SCHEDULING: Book/schedule meetings
        - RAG_QUERY: Information requests (most common)
        - GREETING: Standalone greetings (strict - very rare)
        - EXIT: End conversation
        - UNCLEAR: Ambiguous/nonsensical input
    """
    
    def __init__(self, config: IntentConfig):
        """
        Initialize intent classifier.
        
        Args:
            config: Intent configuration with Gemini API key, thresholds, etc.
        """
        self.config = config
        
        # Load and compile patterns
        self.patterns = load_intent_patterns()
        self.compiled_patterns = compile_patterns(self.patterns)
        
        # Initialize Gemini client (optional - for fallback)
        self.model = None
        if config.gemini_api_key:
            try:
                genai.configure(api_key=config.gemini_api_key)
                self.model = genai.GenerativeModel(config.gemini_model)
                logger.info(f"✅ Gemini model initialized: {config.gemini_model}")
            except Exception as e:
                logger.warning(f"⚠️ Gemini initialization failed: {e}. Only fast pattern matching will be available.")
        else:
            logger.warning("⚠️ No Gemini API key. Only fast pattern matching available.")
        
        # Performance tracking
        self.fast_route_count = 0
        self.gemini_route_count = 0
        self.total_confidence = 0.0
    
    async def classify_intent(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classify user intent with two-tier approach.
        
        Args:
            text: User input text to classify
            context: Optional conversation context for better classification
            
        Returns:
            Dictionary with:
                - intent: str (APPOINTMENT_SCHEDULING, RAG_QUERY, GREETING, EXIT, UNCLEAR)
                - confidence: float (0.0-1.0)
                - context: dict with user_goal, key_entities, extracted_meaning
                - reasoning: str (classification explanation)
                - fast_route: bool (true if pattern match, false if LLM)
                - response_time: float (seconds)
        """
        # Validate input
        if not text or not text.strip():
            return {
                "intent": "UNCLEAR",
                "confidence": 0.0,
                "context": {
                    "user_goal": "empty input received",
                    "key_entities": {},
                    "extracted_meaning": ""
                },
                "reasoning": "Empty or whitespace-only input",
                "fast_route": True,
                "response_time": 0.0
            }
        
        # Start performance timer
        start_time = time.time()
        
        try:
            # Try fast pattern classification first
            fast_result = self._fast_pattern_classification(text, context)
            
            # If fast result has high confidence, return immediately
            if fast_result["confidence"] >= self.config.confidence_threshold:
                self.fast_route_count += 1
                self.total_confidence += fast_result["confidence"]
                fast_result["fast_route"] = True
                fast_result["response_time"] = time.time() - start_time
                
                if self.config.log_classifications:
                    logger.info(
                        f"✅ FAST: {fast_result['intent']} "
                        f"(conf={fast_result['confidence']:.2f}, "
                        f"time={fast_result['response_time']:.3f}s)"
                    )
                
                return fast_result
            
            # Fallback to Gemini classification for complex cases
            if self.model:
                try:
                    gemini_result = await self._gemini_classification(text, context)
                    self.gemini_route_count += 1
                    self.total_confidence += gemini_result["confidence"]
                    gemini_result["fast_route"] = False
                    gemini_result["response_time"] = time.time() - start_time
                    
                    if self.config.log_classifications:
                        logger.info(
                            f"🤖 GEMINI: {gemini_result['intent']} "
                            f"(conf={gemini_result['confidence']:.2f}, "
                            f"time={gemini_result['response_time']:.3f}s)"
                        )
                    
                    return gemini_result
                
                except Exception as e:
                    logger.warning(f"⚠️ Gemini classification failed: {e}")
                    # Return fast result as final fallback
                    self.fast_route_count += 1
                    self.total_confidence += fast_result["confidence"]
                    fast_result["fast_route"] = True
                    fast_result["response_time"] = time.time() - start_time
                    return fast_result
            else:
                # No Gemini available, use fast result
                self.fast_route_count += 1
                self.total_confidence += fast_result["confidence"]
                fast_result["fast_route"] = True
                fast_result["response_time"] = time.time() - start_time
                
                if self.config.log_classifications:
                    logger.info(
                        f"✅ FAST (no Gemini): {fast_result['intent']} "
                        f"(conf={fast_result['confidence']:.2f})"
                    )
                
                return fast_result
                
        except Exception as e:
            logger.error(f"❌ Classification error: {e}")
            return {
                "intent": "UNCLEAR",
                "confidence": 0.2,
                "context": {
                    "user_goal": "classification error occurred",
                    "key_entities": {},
                    "extracted_meaning": text.lower()
                },
                "reasoning": f"Error during classification: {str(e)}",
                "fast_route": True,
                "response_time": time.time() - start_time
            }
    
    def _fast_pattern_classification(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fast pattern matching for common university intents.
        
        Args:
            text: User input text
            context: Optional conversation context
            
        Returns:
            Classification result with intent, confidence, context dict, reasoning
        """
        # Light normalization before pattern checks
        text_normalized = text.lower().strip()
        # Remove common punctuation for pattern matching
        text_normalized = re.sub(r'[?!.,;]', '', text_normalized)
        text_lower = text_normalized
        
        # Priority 1: Appointment Detection (highest priority)
        appointment_patterns = self.patterns["appointment"]
        
        # Negative keywords to exclude false positives
        appointment_negative_keywords = ["class schedule", "course schedule", "exam schedule", "schedule of classes"]
        has_negative_keyword = any(neg_kw in text_lower for neg_kw in appointment_negative_keywords)
        
        # Check keywords
        appointment_keywords_found = [kw for kw in appointment_patterns["keywords"] if kw in text_lower]
        
        # Check regex patterns
        appointment_regex_patterns = self.compiled_patterns["appointment"]["regex_patterns"]
        appointment_regex_match = any(pattern.search(text_lower) for pattern in appointment_regex_patterns)
        
        # Refined appointment detection:
        # Accept if: (1) regex match, or (2) specific appointment/meeting keywords, 
        # or (3) scheduling verb + person/office/department
        is_appointment = False
        
        if not has_negative_keyword:
            # Strategy 1: Accept regex pattern match
            if appointment_regex_match:
                is_appointment = True
            # Strategy 2: Accept if "appointment" or "meeting"/"meet" keywords present
            elif any(kw in ["appointment", "meeting", "meet"] for kw in appointment_keywords_found):
                is_appointment = True
            # Strategy 3: Accept if scheduling verb + department/office/person reference
            elif any(kw in ["schedule", "book", "booking"] for kw in appointment_keywords_found):
                # Check for person/office/department reference
                has_person_office = re.search(r"\b(with|see|visit)\s+(an?|the)?\s*\w+\b", text_lower) or \
                                   re.search(r"\b(advisor|counselor|professor|dean|staff|office|department)\b", text_lower)
                if has_person_office:
                    is_appointment = True
        
        if is_appointment:
            # Extract entities
            key_entities = {}
            
            # Extract department
            entity_patterns = self.compiled_patterns["appointment"]["entity_patterns"]
            dept_match = entity_patterns["department"].search(text_lower)
            if dept_match:
                key_entities["department"] = dept_match.group(0)
            
            # Extract date/time
            datetime_match = entity_patterns["datetime"].search(text_lower)
            if datetime_match:
                key_entities["datetime"] = datetime_match.group(0)
            
            # Extract purpose
            purpose_match = entity_patterns["purpose"].search(text_lower)
            if purpose_match:
                key_entities["purpose"] = purpose_match.group(0)
            
            # Build context
            user_goal = "wants to schedule appointment"
            if "department" in key_entities:
                user_goal += f" with {key_entities['department']}"
            if "purpose" in key_entities:
                user_goal += f" for {key_entities['purpose']}"
            
            # Build extracted meaning
            meaning_parts = ["schedule", "appointment"]
            if "department" in key_entities:
                meaning_parts.append(key_entities["department"])
            if "purpose" in key_entities:
                meaning_parts.append(key_entities["purpose"])
            extracted_meaning = " ".join(meaning_parts)
            
            return {
                "intent": "APPOINTMENT_SCHEDULING",
                "confidence": 0.95,
                "context": {
                    "user_goal": user_goal,
                    "key_entities": key_entities,
                    "extracted_meaning": extracted_meaning
                },
                "reasoning": f"Appointment keywords detected: {appointment_keywords_found}"
            }
        
        # Priority 2: Exit Detection
        exit_patterns = self.patterns["exit"]
        exit_keywords_found = [kw for kw in exit_patterns["keywords"] if kw in text_lower]
        exit_regex_patterns = self.compiled_patterns["exit"]["regex_patterns"]
        exit_regex_match = any(pattern.search(text_lower) for pattern in exit_regex_patterns)
        
        if exit_keywords_found or exit_regex_match:
            return {
                "intent": "EXIT",
                "confidence": 0.95,
                "context": {
                    "user_goal": "ending conversation",
                    "key_entities": {},
                    "extracted_meaning": ""
                },
                "reasoning": "Exit phrase detected"
            }
        
        # Priority 3: Greeting Detection (STRICT - must be standalone greeting)
        greeting_patterns = self.patterns["greeting"]
        greeting_keywords_found = [kw for kw in greeting_patterns["keywords"] if kw in text_lower]
        greeting_regex_patterns = self.compiled_patterns["greeting"]["regex_patterns"]
        greeting_regex_match = any(pattern.search(text_lower) for pattern in greeting_regex_patterns)
        
        # CRITICAL FIX: Greetings must NOT contain question words or information request phrases
        has_question_words = re.search(r'\b(can you|could you|tell me|talk about|what|when|where|who|why|how|explain|describe)\b', text_lower)
        is_information_request = re.search(r'\b(about|information|help me with|assist|know more)\b', text_lower)
        
        # Only classify as GREETING if: (1) greeting pattern matches AND (2) no question/info request words
        if (greeting_keywords_found or greeting_regex_match) and not has_question_words and not is_information_request:
            # Additional check: greeting should be SHORT (< 10 words typically)
            word_count = len(text_normalized.split())
            if word_count <= 10:
                return {
                    "intent": "GREETING",
                    "confidence": 0.95,
                    "context": {
                        "user_goal": "greeting the assistant",
                        "key_entities": {},
                        "extracted_meaning": ""
                    },
                    "reasoning": "Greeting detected"
                }
        
        # Priority 4: General Query Detection (RAG routing)
        query_patterns = self.patterns["general_query"]
        
        # Check question words
        has_question_word = any(text_lower.startswith(qw) or f" {qw} " in f" {text_lower} " 
                                for qw in query_patterns["question_words"])
        
        # Check university topics
        topics_found = [topic for topic in query_patterns["university_topics"] if topic in text_lower]
        
        # Check information request patterns
        query_regex_patterns = self.compiled_patterns["general_query"]["regex_patterns"]
        info_request_match = any(pattern.search(text_lower) for pattern in query_regex_patterns)
        
        if has_question_word or topics_found or info_request_match:
            # Extract entities
            key_entities = {}
            entity_patterns = self.compiled_patterns["entities"]
            
            # Extract program
            program_match = entity_patterns["program"].search(text_lower)
            if program_match:
                key_entities["program"] = program_match.group(0)
            
            # Extract course code (case-sensitive)
            course_match = entity_patterns["course_code"].search(text)
            if course_match:
                key_entities["course"] = course_match.group(0)
            
            # Extract topics from university_topics list
            if topics_found:
                key_entities["topic"] = topics_found[0]  # Use first matched topic
            
            # Build user_goal
            user_goal = "asking about"
            if "program" in key_entities:
                user_goal += f" {key_entities['program']} program"
            if "topic" in key_entities:
                user_goal += f" {key_entities['topic']}"
            if not key_entities:
                user_goal += " university information"
            
            # Build extracted_meaning (normalize query)
            # Remove filler words and question words
            filler_words = ["um", "uh", "like", "you know", "i mean", "well", "so"]
            meaning_text = text_lower
            for filler in filler_words:
                meaning_text = meaning_text.replace(filler, "")
            
            # Keep key nouns and verbs, remove excessive question words at start
            meaning_text = re.sub(r"^(what|when|where|who|why|how)\s+(is|are|do|does|can|could)\s+", "", meaning_text)
            meaning_text = re.sub(r"\s+", " ", meaning_text).strip()
            
            return {
                "intent": "RAG_QUERY",
                "confidence": 0.85,
                "context": {
                    "user_goal": user_goal,
                    "key_entities": key_entities,
                    "extracted_meaning": meaning_text
                },
                "reasoning": "General information query detected"
            }
        
        # Default: UNCLEAR
        return {
            "intent": "UNCLEAR",
            "confidence": 0.3,
            "context": {
                "user_goal": "unclear intent",
                "key_entities": {},
                "extracted_meaning": text_lower
            },
            "reasoning": "No clear pattern matched"
        }
    
    async def _gemini_classification(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        LLM-based classification for complex cases using Gemini.
        
        Args:
            text: User input text
            context: Optional conversation context
            
        Returns:
            Classification result with intent, confidence, context dict, reasoning
        """
        try:
            # Get system prompt
            system_prompt = get_system_prompt()
            
            # Build context information if provided
            context_info = ""
            if context:
                context_info = f"Context: {json.dumps(context)}. "
            
            # Build full prompt
            full_prompt = f"{system_prompt}\n\n{context_info}User input: '{text}'. Classify this input:"
            
            # Call Gemini API with timeout
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.model.generate_content,
                        full_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.1,  # Low temperature for consistent classification
                            max_output_tokens=500
                        )
                    ),
                    timeout=self.config.gemini_timeout
                )
            except asyncio.TimeoutError:
                return {
                    "intent": "UNCLEAR",
                    "confidence": 0.3,
                    "context": {
                        "user_goal": "Gemini classification timeout",
                        "key_entities": {},
                        "extracted_meaning": text.lower()
                    },
                    "reasoning": f"Gemini classification timed out after {self.config.gemini_timeout}s"
                }
            
            # Extract response text
            response_text = response.text.strip()
            
            # Robust JSON extraction: Remove code fences and extra wrappers
            # Remove markdown code fences
            response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            response_text = response_text.strip()
            
            # Try multiple extraction strategies
            json_text = None
            
            # Strategy 1: Try parsing the entire response as JSON
            try:
                result = json.loads(response_text)
                json_text = response_text
            except json.JSONDecodeError:
                # Strategy 2: Find first balanced JSON object
                json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    # Strategy 3: Find last JSON-looking block
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(0)
            
            if not json_text:
                raise ValueError("No JSON found in Gemini response")
            
            result = json.loads(json_text)
            
            # Validate result structure
            if "intent" not in result:
                result["intent"] = "UNCLEAR"
            if "confidence" not in result:
                result["confidence"] = 0.5
            if "context" not in result:
                result["context"] = {
                    "user_goal": "unclear intent",
                    "key_entities": {},
                    "extracted_meaning": text.lower()
                }
            else:
                # Ensure context has required fields
                if "user_goal" not in result["context"]:
                    result["context"]["user_goal"] = "unclear intent"
                if "key_entities" not in result["context"]:
                    result["context"]["key_entities"] = {}
                if "extracted_meaning" not in result["context"]:
                    result["context"]["extracted_meaning"] = text.lower()
            
            if "reasoning" not in result:
                result["reasoning"] = "Gemini classification"
            
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parse error in Gemini response: {e}")
            return {
                "intent": "UNCLEAR",
                "confidence": 0.4,
                "context": {
                    "user_goal": "JSON parse error",
                    "key_entities": {},
                    "extracted_meaning": text.lower()
                },
                "reasoning": "Failed to parse Gemini JSON response"
            }
        except Exception as e:
            logger.warning(f"⚠️ Gemini API error: {e}")
            return {
                "intent": "UNCLEAR",
                "confidence": 0.2,
                "context": {
                    "user_goal": "Gemini API error",
                    "key_entities": {},
                    "extracted_meaning": text.lower()
                },
                "reasoning": f"Gemini API error: {str(e)}"
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get classification performance statistics.
        
        Returns:
            Dictionary with performance metrics:
                - total_requests: Total number of classifications
                - fast_route_count: Number of pattern matches
                - gemini_route_count: Number of LLM fallbacks
                - fast_route_percentage: Percentage of fast route hits
                - average_confidence: Average classification confidence
        """
        total_requests = self.fast_route_count + self.gemini_route_count
        fast_route_percentage = (self.fast_route_count / total_requests * 100) if total_requests > 0 else 0
        average_confidence = (self.total_confidence / total_requests) if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "fast_route_count": self.fast_route_count,
            "gemini_route_count": self.gemini_route_count,
            "fast_route_percentage": round(fast_route_percentage, 2),
            "average_confidence": round(average_confidence, 3)
        }
