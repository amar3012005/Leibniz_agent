"""
Leibniz University Institute - Intent Classification Module

⚠️ DEPRECATION NOTICE (2024-12):
This module is being phased out in favor of TARA's fast parser architecture.
New code should use:
- sindh_finetuned_parser.SINDHFineTunedParser
- fast_intent_router.FastIntentRouter

Migration is complete in leibniz_pro.py and leibniz_persistent_services.py.
This module remains for backward compatibility only.

----

This module provides intent classification for the Leibniz University customer service agent.
It classifies user inputs into 5 core intent categories and extracts structured context to
improve information retrieval and response generation.

Intent Categories:
- APPOINTMENT_SCHEDULING: User wants to schedule/book appointments (admissions, advising, counseling)
- RAG_QUERY: General questions about university (courses, admissions, campus, services, policies)
- GREETING: Basic greetings and conversation starters
- EXIT: User wants to end conversation (goodbye, that's all, thanks bye)
- UNCLEAR: Cannot determine intent or ambiguous input

Key Innovation - Context Extraction:
Instead of just returning intent and confidence, this parser extracts structured context:
- user_goal: High-level description of what user wants (1 sentence)
- key_entities: Extracted entities as key-value pairs (department, program, date, etc.)
- extracted_meaning: Paraphrased/normalized query for semantic search

This enriched context is passed to the RAG system for better document retrieval and more
accurate responses, rather than using the raw transcript alone.

Architecture:
Two-tier classification for optimal performance:
1. Fast pattern matching (80%+ queries) - instant classification using regex and keywords
2. Gemini LLM fallback (complex cases) - AI-powered classification with context extraction

Performance target: >80% fast route for optimal response times.

⚠️ DEPRECATED: Use TARA's fast parser instead (sindh_finetuned_parser + fast_intent_router)
"""

import warnings
import re
import json
import asyncio
import time
import hashlib
from typing import Dict, Any, Optional, List
from collections import OrderedDict
import google.generativeai as genai
import os
from dotenv import load_dotenv

# PHASE 3: Leibniz Intent Parser is ACTIVE for university customer service
# This parser is specifically designed for Leibniz University with university-specific intents:
# - APPOINTMENT_SCHEDULING: Schedule meetings with admissions, advisors, counselors
# - RAG_QUERY: Information requests about programs, courses, admissions, campus services
# - GREETING: Standalone greetings without information requests
# - EXIT: Conversation termination
# - UNCLEAR: Ambiguous inputs
#
# DO NOT use SINDH parser (job placement intents) for Leibniz agent!

# Load environment variables
load_dotenv()


class LeibnizIntentParser:
    """Intent classifier for Leibniz University agent with context extraction"""
    
    def __init__(self):
        """Initialize the Leibniz intent parser"""
        # Load Gemini API key
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Load Gemini model name from environment (Comment 4)
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
        
        # Load confidence threshold from environment (Comment 2)
        try:
            self.fast_threshold = float(os.getenv("LEIBNIZ_INTENT_PARSER_CONFIDENCE_THRESHOLD", "0.8"))
        except ValueError:
            self.fast_threshold = 0.8
        
        # Load Gemini timeout from environment (Comment 2)
        try:
            self.gemini_timeout = float(os.getenv("LEIBNIZ_INTENT_PARSER_GEMINI_TIMEOUT", "5.0"))
        except ValueError:
            self.gemini_timeout = 5.0
        
        # OPTIMIZATION: LRU cache for repeated queries (instant classification)
        self.classification_cache = OrderedDict()
        self.cache_max_size = 128  # Cache last 128 unique queries
        self.cache_ttl_seconds = 300  # 5 minute TTL
        self.cache_hits = 0
        
        # Configure Gemini client
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel(gemini_model)
            print(f"🎓 Leibniz Parser: Gemini AI ({gemini_model}) enabled for complex classification")
        else:
            self.model = None
            print("⚠️ Leibniz Parser: No Gemini API key found - fast patterns only")
        
        # Load Leibniz-specific patterns
        self.leibniz_patterns = self._load_leibniz_patterns()
        
        # Performance tracking
        self.fast_route_count = 0
        self.gemini_route_count = 0
        self.total_confidence = 0.0
        
        print(f"🎓 Leibniz Intent Parser initialized (threshold={self.fast_threshold}, timeout={self.gemini_timeout}s, cache={self.cache_max_size})")
    
    def _load_leibniz_patterns(self) -> Dict[str, Any]:
        """
        Load simplified English-only patterns for university context
        
        Returns:
            Dictionary with pattern categories for fast classification
        """
        patterns = {
            # Appointment patterns - highest priority
            "appointment": {
                "keywords": [
                    "appointment", "schedule", "book", "booking", "meeting", 
                    "meet", "visit", "consultation", "advising", "counseling"
                ],
                "regex_patterns": [
                    r"\b(schedule|book|make)\s+(an?\s+)?appointment\b",
                    r"\bappointment\s+(with|for)\b",
                    r"\b(want|need|like)\s+to\s+(schedule|book|meet)\b",
                    r"\bwhen\s+can\s+i\s+(meet|see|visit)\b",
                    r"\b(set up|arrange)\s+(a\s+)?(meeting|appointment)\b"
                ],
                "entity_patterns": {
                    "department": r"\b(admissions?|registrar|financial aid|counseling|advising|career services?|academic|student services?)\b",
                    "datetime": r"\b(today|tomorrow|next week|monday|tuesday|wednesday|thursday|friday|\d{1,2}\s*(?:am|pm)|\d{1,2}/\d{1,2})\b",
                    "purpose": r"\b(admission|enrollment|transcript|financial|academic|career|degree|program)\b"
                }
            },
            
            # Greeting patterns (Comment 10: expanded coverage)
            "greeting": {
                "keywords": [
                    "hello", "hi", "hey", "greetings", "good morning", 
                    "good afternoon", "good evening", "howdy", "hiya",
                    "what's up", "yo", "sup"
                ],
                "regex_patterns": [
                    r"^(hello|hi|hey|hiya|greetings|howdy)\b",
                    r"\bgood\s+(morning|afternoon|evening|day)\b",
                    r"^(what'?s?\s+up|yo|sup)\b"
                ]
            },
            
            # Exit patterns (Comment 10: expanded coverage)
            "exit": {
                "keywords": [
                    "bye", "goodbye", "exit", "quit", "stop", "end", 
                    "thanks bye", "that's all", "no more questions",
                    "see you", "i'm done", "that is all", "nothing else",
                    "talk later", "gotta go", "have a good day"
                ],
                "regex_patterns": [
                    r"\b(bye|goodbye|see you|talk later|gotta go)\b",
                    r"\b(that'?s?\s+all|no\s+more|i'?m\s+done|nothing\s+else)\b",
                    r"\b(thanks?|thank you).*(bye|goodbye|all)\b",
                    r"\bhave\s+a\s+good\s+(day|night|one)\b"
                ]
            },
            
            # General query patterns (for RAG routing)
            "general_query": {
                "question_words": [
                    "what", "when", "where", "who", "why", "how", 
                    "can", "could", "would", "is", "are", "do", "does"
                ],
                "university_topics": [
                    "course", "class", "program", "degree", "admission", 
                    "enrollment", "tuition", "fee", "scholarship", "campus", 
                    "library", "dorm", "housing", "faculty", "professor", 
                    "department", "major", "minor", "credit", "semester", 
                    "exam", "grade", "transcript", "schedule", "registration"
                ],
                "regex_patterns": [
                    r"^(what|when|where|who|why|how)\b",
                    r"\b(tell me|explain|describe|information about)\b",
                    r"\b(how do i|how can i|where can i)\b"
                ]
            },
            
            # Entity extraction patterns (used across all intents)
            "entities": {
                "course_code": r"\b[A-Z]{2,4}\s*\d{3,4}\b",
                "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                "date": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
                "program": r"\b(computer science|cs|engineering|business|biology|chemistry|physics|mathematics|english|history)\b"
            }
        }
        
        return patterns
    
    def _create_leibniz_system_prompt(self) -> str:
        """
        Create Gemini system prompt for university context with context extraction
        
        Returns:
            System prompt string for Gemini API
        """
        prompt = """You are an EXPERT intent classifier for Leibniz University Institute customer service with advanced natural language understanding.

🎯 YOUR MISSION: Accurately classify user intent by deeply analyzing the SEMANTIC MEANING and CONTEXT, not just surface keywords.

CORE PRINCIPLE: Prioritize ACCURACY over speed. Think through the user's true intention before classifying.

═══════════════════════════════════════════════════════════════════════════════
CLASSIFICATION CATEGORIES (5 INTENTS)
═══════════════════════════════════════════════════════════════════════════════

1. 🗓️ APPOINTMENT_SCHEDULING
   Definition: User explicitly wants to schedule, book, or arrange a meeting/appointment
   Key Indicators: "schedule", "book", "appointment", "meeting", "when can I meet"
   Confidence Threshold: Require explicit appointment-related action verbs
   
2. 📚 RAG_QUERY (Most Common - Default for Information Requests)
   Definition: User seeking information, asking questions, or requesting explanations
   Key Indicators: Question words (what, how, why, where, when, who), "tell me", "explain", "describe"
   Confidence Threshold: ANY information-seeking behavior → RAG_QUERY
   
3. 👋 GREETING (EXTREMELY STRICT - Rare)
   Definition: ONLY standalone social pleasantries with NO information request
   Key Indicators: "hi", "hello", "hey", "good morning" (ALONE, no follow-up)
   Confidence Threshold: Must be < 5 words AND contain zero question/request elements
   
4. 🚪 EXIT
   Definition: User wants to end the conversation
   Key Indicators: "bye", "goodbye", "thanks, that's all", "I'm done"
   Confidence Threshold: Clear termination intent
   
5. ❓ UNCLEAR
   Definition: Genuinely ambiguous, incomplete, or nonsensical input
   Confidence Threshold: Use sparingly - most inputs have classifiable intent

═══════════════════════════════════════════════════════════════════════════════
CRITICAL CLASSIFICATION RULES (READ CAREFULLY)
═══════════════════════════════════════════════════════════════════════════════

🚨 RULE 1: GREETING vs RAG_QUERY DISTINCTION (Most Common Error)

GREETING requires ALL of these conditions:
  ✅ Contains greeting word ("hi", "hello", "hey", "good morning")
  ✅ Standalone (no additional requests or questions)
  ✅ Word count ≤ 5 words
  ✅ NO question words (what, how, why, where, when, who, can, could, would)
  ✅ NO action requests ("tell me", "show me", "explain", "talk about")
  ✅ NO topic mentions (programs, courses, admission, etc.)

If ANY condition fails → Classify as RAG_QUERY, NOT GREETING

Examples of FALSE GREETINGS (actually RAG_QUERY):
  ❌ "can you specifically talk about talk about any" → RAG_QUERY (has "can you talk about")
  ❌ "can you tell me about programs" → RAG_QUERY (information request)
  ❌ "what can you help me with" → RAG_QUERY (question about services)
  ❌ "hello, how do I apply?" → RAG_QUERY (has follow-up question)
  ❌ "hey, what programs do you offer?" → RAG_QUERY (asking about programs)
  ❌ "hi there, I need information" → RAG_QUERY (information request)

Examples of TRUE GREETINGS:
  ✅ "hi" (standalone)
  ✅ "hello" (standalone)
  ✅ "good morning" (standalone)
  ✅ "hey there" (casual greeting only)
  ✅ "what's up" (colloquial greeting)

  ✅ "what's up" (colloquial greeting)

🚨 RULE 2: RAG_QUERY is the DEFAULT for Information Requests

Classify as RAG_QUERY if user:
  ✅ Asks a question (contains what, how, why, where, when, who)
  ✅ Requests information ("tell me", "explain", "describe", "talk about")
  ✅ Seeks clarification ("can you...", "could you...", "would you...")
  ✅ Mentions university topics (programs, courses, admission, tuition, etc.)
  ✅ Uses imperative verbs ("show", "list", "give me", "provide")

Even if input is poorly formed or contains typos, extract the underlying information-seeking intent.

🚨 RULE 3: APPOINTMENT_SCHEDULING Requires Explicit Scheduling Intent

Classify as APPOINTMENT_SCHEDULING ONLY if:
  ✅ Explicit scheduling verbs: "schedule", "book", "make", "arrange", "set up"
  ✅ Meeting/appointment nouns: "appointment", "meeting", "consultation", "visit"
  ✅ Time-related requests: "when can I meet", "available times", "book a slot"

DO NOT classify as appointment if:
  ❌ Asking ABOUT scheduling process (that's RAG_QUERY)
  ❌ General questions about appointments (that's RAG_QUERY)
  ❌ "How do I schedule..." without explicit action request (that's RAG_QUERY)

🚨 RULE 4: Context Extraction is MANDATORY (Every Classification)

For EVERY intent, extract rich structured context:

user_goal: One clear sentence describing user's objective
  - Focus on the WHAT (what they want to achieve)
  - Use natural language, not keywords
  - Examples:
    * "wants to learn about computer science program requirements"
    * "seeking information on campus housing options and costs"
    * "trying to understand the admission application process"

key_entities: Dictionary of extracted entities
  - Program/major names: {"program": "computer science"}
  - Departments: {"department": "admissions"}
  - Topics: {"topic": "tuition fees", "subtopic": "payment plans"}
  - Time references: {"time_frame": "fall 2024"}
  - Empty dict {} for greetings/exits

extracted_meaning: Normalized semantic query for RAG search
  - Remove filler words (um, uh, like, you know, basically)
  - Remove question words at start (what, how, when, etc.)
  - Keep core semantic content
  - Expand abbreviations (CS → computer science)
  - Examples:
    * Input: "um, so like, what are the requirements for getting into the CS program?"
    * Output: "computer science program admission requirements prerequisites"

═══════════════════════════════════════════════════════════════════════════════
ROBUST CLASSIFICATION ALGORITHM (Follow This Process)
═══════════════════════════════════════════════════════════════════════════════

Step 1: SEMANTIC ANALYSIS
  - What is the user REALLY trying to accomplish?
  - Ignore surface-level keyword matches
  - Focus on underlying intent

Step 2: ELIMINATE IMPOSSIBLE INTENTS
  - Has appointment-scheduling action verb? → Could be APPOINTMENT_SCHEDULING
  - Has question/information request? → Could be RAG_QUERY
  - Standalone greeting (< 5 words, no request)? → Could be GREETING
  - Clear goodbye/termination? → Could be EXIT
  - Genuinely unclear/nonsensical? → Could be UNCLEAR

Step 3: APPLY STRICT RULES
  - Use GREETING rules to eliminate false positives
  - Default to RAG_QUERY for information-seeking
  - Require explicit action for APPOINTMENT_SCHEDULING

Step 4: ASSIGN CONFIDENCE
  - High confidence (0.90-0.99): Clear, unambiguous intent
  - Medium confidence (0.70-0.89): Likely correct, minor ambiguity
  - Low confidence (0.50-0.69): Best guess, consider fallback

Step 5: EXTRACT CONTEXT
  - Build rich semantic context (user_goal, key_entities, extracted_meaning)
  - Ensure downstream RAG system has maximum information

═══════════════════════════════════════════════════════════════════════════════
RESPONSE FORMAT (JSON ONLY - No Markdown, No Code Fences)
═══════════════════════════════════════════════════════════════════════════════

{
  "intent": "RAG_QUERY",
  "confidence": 0.95,
  "context": {
    "user_goal": "asking about computer science program admission requirements and prerequisites",
    "key_entities": {
      "program": "computer science",
      "topic": "admission requirements",
      "subtopic": "prerequisites"
    },
    "extracted_meaning": "computer science program admission requirements prerequisites courses needed"
  },
  "reasoning": "User asking informational question about CS program - clear RAG query with question words and topic focus"
}

═══════════════════════════════════════════════════════════════════════════════
COMPREHENSIVE EXAMPLES (Study These Carefully)
═══════════════════════════════════════════════════════════════════════════════

EXAMPLE 1: Tricky Case (Looks Like Greeting, Is Actually RAG_QUERY)
Input: "can you specifically talk about talk about any"
Analysis:
  - Contains "can you" (request for action)
  - Contains "talk about" (information request)
  - User wants information (even if unclear what)
  - NOT a standalone greeting
Output: {
  "intent": "RAG_QUERY",
  "confidence": 0.80,
  "context": {
    "user_goal": "requesting general information about university topics or services available",
    "key_entities": {},
    "extracted_meaning": "university information topics services available general inquiry"
  },
  "reasoning": "Contains information request phrase 'can you talk about' - this is an open-ended RAG query, NOT a greeting despite casual phrasing"
}

EXAMPLE 2: Clear RAG Query
Input: "What are the requirements for the computer science program?"
Output: {
  "intent": "RAG_QUERY",
  "confidence": 0.98,
  "context": {
    "user_goal": "asking about specific admission or enrollment requirements for computer science program",
    "key_entities": {
      "program": "computer science",
      "topic": "requirements"
    },
    "extracted_meaning": "computer science program admission requirements prerequisites eligibility criteria"
  },
  "reasoning": "Direct question about program requirements - textbook RAG query with clear topic and question structure"
}

EXAMPLE 3: Appointment Scheduling
Input: "I'd like to schedule an appointment with the admissions office"
Output: {
  "intent": "APPOINTMENT_SCHEDULING",
  "confidence": 0.98,
  "context": {
    "user_goal": "wants to schedule appointment meeting with admissions office for consultation",
    "key_entities": {
      "department": "admissions",
      "appointment_type": "consultation"
    },
    "extracted_meaning": "schedule appointment admissions office consultation meeting"
  },
  "reasoning": "Explicit scheduling intent with 'schedule appointment' and target department - clear appointment request"
}

EXAMPLE 4: True Greeting
Input: "hello"
Output: {
  "intent": "GREETING",
  "confidence": 0.99,
  "context": {
    "user_goal": "greeting the assistant to initiate conversation",
    "key_entities": {},
    "extracted_meaning": ""
  },
  "reasoning": "Standalone greeting with no follow-up request - meets all strict greeting criteria (< 5 words, no questions, no requests)"
}

EXAMPLE 5: Exit Intent
Input: "Thanks for the help, that's all I needed"
Output: {
  "intent": "EXIT",
  "confidence": 0.97,
  "context": {
    "user_goal": "ending conversation with gratitude after receiving needed information",
    "key_entities": {},
    "extracted_meaning": ""
  },
  "reasoning": "Clear termination intent with 'that's all I needed' - user signaling conversation completion"
}

EXAMPLE 6: Ambiguous Input (Badly Formed but Still Classifiable)
Input: "um like how does the uh admission thing work or whatever"
Output: {
  "intent": "RAG_QUERY",
  "confidence": 0.85,
  "context": {
    "user_goal": "asking about admission process procedures and how it works",
    "key_entities": {
      "topic": "admission",
      "subtopic": "process"
    },
    "extracted_meaning": "admission process procedures how it works steps requirements"
  },
  "reasoning": "Despite filler words and casual phrasing, underlying intent is clear - asking how admission works, which is informational RAG query"
}

EXAMPLE 7: False Appointment (Actually RAG_QUERY)
Input: "How do I schedule an appointment?"
Output: {
  "intent": "RAG_QUERY",
  "confidence": 0.90,
  "context": {
    "user_goal": "asking about appointment scheduling process and procedures",
    "key_entities": {
      "topic": "appointments",
      "subtopic": "scheduling process"
    },
    "extracted_meaning": "appointment scheduling process how to book procedure"
  },
  "reasoning": "Asking ABOUT scheduling process, not requesting to schedule - this is informational query, not action request"
}

═══════════════════════════════════════════════════════════════════════════════
FINAL REMINDERS
═══════════════════════════════════════════════════════════════════════════════

✅ When in doubt between GREETING and RAG_QUERY → Choose RAG_QUERY
✅ Greetings are EXTREMELY rare in customer service context
✅ Extract maximum context - downstream RAG system depends on it
✅ Confidence reflects your certainty - be honest about ambiguity
✅ Focus on SEMANTIC intent, not surface keywords
✅ Return ONLY valid JSON (no markdown, no code fences, no extra text)

Now classify the user input with expert-level accuracy."""
        
        return prompt
    
    async def classify_intent(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Classify user intent with two-tier approach and context extraction
        
        OPTIMIZED with LRU cache for instant repeated query classification
        
        Args:
            text: User input text to classify
            context: Optional conversation context for better classification
            
        Returns:
            Dictionary with intent, confidence, context (user_goal, key_entities, extracted_meaning),
            reasoning, fast_route flag, and response_time
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
        
        # OPTIMIZATION: Check cache first (0.1-0.5ms for cache hits)
        cache_key = hashlib.md5(text.lower().strip().encode()).hexdigest()
        
        if cache_key in self.classification_cache:
            cached_result, cache_time = self.classification_cache[cache_key]
            
            # Check TTL
            if time.time() - cache_time < self.cache_ttl_seconds:
                # Cache hit - return immediately
                self.cache_hits += 1
                cached_result["response_time"] = time.time() - start_time
                cached_result["cache_hit"] = True
                
                # Move to end (LRU)
                self.classification_cache.move_to_end(cache_key)
                
                return cached_result
            else:
                # Expired - remove from cache
                del self.classification_cache[cache_key]
        
        try:
            # Try fast pattern classification first
            fast_result = self._fast_pattern_classification(text, context)
            
            # If fast result has high confidence, return immediately (Comment 2: use configurable threshold)
            if fast_result["confidence"] > self.fast_threshold:
                self.fast_route_count += 1
                self.total_confidence += fast_result["confidence"]
                fast_result["fast_route"] = True
                fast_result["response_time"] = time.time() - start_time
                fast_result["cache_hit"] = False
                
                # Cache the result
                self._cache_result(cache_key, fast_result)
                
                return fast_result
            
            # Fallback to Gemini classification for complex cases
            if self.model:
                try:
                    gemini_result = await self._gemini_classification(text, context)
                    self.gemini_route_count += 1
                    self.total_confidence += gemini_result["confidence"]
                    gemini_result["fast_route"] = False
                    gemini_result["response_time"] = time.time() - start_time
                    gemini_result["cache_hit"] = False
                    
                    # Cache the result
                    self._cache_result(cache_key, gemini_result)
                    
                    return gemini_result
                except Exception as e:
                    print(f"⚠️ Gemini classification failed: {e}")
                    # Return fast result as final fallback
                    self.fast_route_count += 1
                    self.total_confidence += fast_result["confidence"]
                    fast_result["fast_route"] = True
                    fast_result["response_time"] = time.time() - start_time
                    fast_result["cache_hit"] = False
                    
                    # Cache the fallback result
                    self._cache_result(cache_key, fast_result)
                    
                    return fast_result
            else:
                # No Gemini available, use fast result
                self.fast_route_count += 1
                self.total_confidence += fast_result["confidence"]
                fast_result["fast_route"] = True
                fast_result["response_time"] = time.time() - start_time
                fast_result["cache_hit"] = False
                
                # Cache the result
                self._cache_result(cache_key, fast_result)
                
                return fast_result
                
        except Exception as e:
            print(f"❌ Classification error: {e}")
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
                "response_time": time.time() - start_time,
                "cache_hit": False
            }
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """
        Cache classification result with LRU eviction
        
        Args:
            cache_key: MD5 hash of normalized query
            result: Classification result to cache
        """
        # Remove cache_hit and response_time before caching (session-specific)
        result_copy = result.copy()
        result_copy.pop("cache_hit", None)
        result_copy.pop("response_time", None)
        
        # Store with timestamp
        self.classification_cache[cache_key] = (result_copy, time.time())
        
        # LRU eviction if over size
        if len(self.classification_cache) > self.cache_max_size:
            # Remove oldest entry
            self.classification_cache.popitem(last=False)
    
    def _fast_pattern_classification(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Fast pattern matching for common university intents
        
        Args:
            text: User input text
            context: Optional conversation context
            
        Returns:
            Classification result with intent, confidence, context dict, reasoning
        """
        # Comment 10: Light normalization before pattern checks
        text_normalized = text.lower().strip()
        # Remove common punctuation for pattern matching
        text_normalized = re.sub(r'[?!.,;]', '', text_normalized)
        text_lower = text_normalized
        
        # Priority 1: Appointment Detection (highest priority)
        appointment_patterns = self.leibniz_patterns["appointment"]
        
        # Negative keywords to exclude false positives (Comment 1)
        appointment_negative_keywords = ["class schedule", "course schedule", "exam schedule", "schedule of classes"]
        has_negative_keyword = any(neg_kw in text_lower for neg_kw in appointment_negative_keywords)
        
        # Check keywords
        appointment_keywords_found = [kw for kw in appointment_patterns["keywords"] if kw in text_lower]
        
        # Check regex patterns
        appointment_regex_match = any(re.search(pattern, text_lower) for pattern in appointment_patterns["regex_patterns"])
        
        # Refined appointment detection (Comment 1):
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
            dept_match = re.search(appointment_patterns["entity_patterns"]["department"], text_lower)
            if dept_match:
                key_entities["department"] = dept_match.group(0)
            
            # Extract date/time
            datetime_match = re.search(appointment_patterns["entity_patterns"]["datetime"], text_lower)
            if datetime_match:
                key_entities["datetime"] = datetime_match.group(0)
            
            # Extract purpose
            purpose_match = re.search(appointment_patterns["entity_patterns"]["purpose"], text_lower)
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
        exit_patterns = self.leibniz_patterns["exit"]
        exit_keywords_found = [kw for kw in exit_patterns["keywords"] if kw in text_lower]
        exit_regex_match = any(re.search(pattern, text_lower) for pattern in exit_patterns["regex_patterns"])
        
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
        greeting_patterns = self.leibniz_patterns["greeting"]
        greeting_keywords_found = [kw for kw in greeting_patterns["keywords"] if kw in text_lower]
        greeting_regex_match = any(re.search(pattern, text_lower) for pattern in greeting_patterns["regex_patterns"])
        
        # CRITICAL FIX: Greetings must NOT contain question words or information request phrases
        has_question_words = re.search(r'\b(can you|could you|tell me|talk about|what|when|where|who|why|how|explain|describe)\b', text_lower)
        is_information_request = re.search(r'\b(about|information|help me with|assist|know more)\b', text_lower)
        
        # Only classify as GREETING if: (1) greeting pattern matches AND (2) no question/info request words
        if (greeting_keywords_found or greeting_regex_match) and not has_question_words and not is_information_request:
            # Additional check: greeting should be SHORT (< 8 words typically)
            word_count = len(text_normalized.split())
            if word_count <= 10:  # Allow up to 10 words for compound greetings like "good morning, how are you"
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
        
        # Priority 4: General Query Detection (RAG routing) - OPTIMIZED
        query_patterns = self.leibniz_patterns["general_query"]
        
        # OPTIMIZATION 1: Quick word-level checks (faster than regex)
        words = text_lower.split()
        first_word = words[0] if words else ""
        
        # Check question words (optimized with first word check)
        has_question_word = (first_word in query_patterns["question_words"]) or \
                           any(f" {qw} " in f" {text_lower} " for qw in query_patterns["question_words"])
        
        # Check university topics (optimized substring search)
        topics_found = [topic for topic in query_patterns["university_topics"] if topic in text_lower]
        
        # Check information request patterns (only if needed)
        info_request_match = any(re.search(pattern, text_lower) for pattern in query_patterns["regex_patterns"])
        
        # OPTIMIZATION 2: Calculate confidence score based on signal strength
        confidence_score = 0.70  # Base score
        
        if has_question_word:
            confidence_score += 0.15  # Strong signal
        if topics_found:
            confidence_score += 0.10 * min(len(topics_found), 2)  # More topics = higher confidence
        if info_request_match:
            confidence_score += 0.10
        
        # Boost for "tell me" / "explain" / "describe" patterns (clear info requests)
        if re.search(r'\b(tell me|explain|describe|talk about|information about)\b', text_lower):
            confidence_score += 0.15
        
        # Cap at 0.98 (leave room for appointment/greeting higher priority)
        confidence_score = min(confidence_score, 0.98)
        
        if has_question_word or topics_found or info_request_match:
            # Extract entities (optimized - only regex if needed)
            key_entities = {}
            entity_patterns = self.leibniz_patterns["entities"]
            
            # Extract program (only if program-related words present)
            if any(word in text_lower for word in ["program", "degree", "major", "course"]):
                program_match = re.search(entity_patterns["program"], text_lower)
                if program_match:
                    key_entities["program"] = program_match.group(0)
            
            # Extract course code (only if course pattern present)
            if re.search(r'\b[A-Z]{2,4}\s*\d{3,4}\b', text):
                course_match = re.search(entity_patterns["course_code"], text)
                if course_match:
                    key_entities["course"] = course_match.group(0)
            
            # Extract topics from university_topics list
            if topics_found:
                key_entities["topic"] = topics_found[0]  # Use first matched topic
            
            # Build user_goal (simplified)
            if "program" in key_entities:
                user_goal = f"seeking information about {key_entities['program']} program"
            elif "topic" in key_entities:
                user_goal = f"seeking information about {key_entities['topic']}"
            elif topics_found:
                user_goal = f"seeking information about {topics_found[0]}"
            else:
                user_goal = "seeking general university information"
            
            # Build extracted_meaning (optimized normalization)
            # Quick filler removal with single pass
            meaning_text = text_lower
            for filler in ["um ", "uh ", "like ", "you know ", "i mean ", "well ", "so "]:
                meaning_text = meaning_text.replace(filler, " ")
            
            # Remove question starters efficiently
            meaning_text = re.sub(r"^(what|when|where|who|why|how)\s+(is|are|do|does|can|could)\s+", "", meaning_text)
            meaning_text = re.sub(r"\s+", " ", meaning_text).strip()
            
            # Add context to extracted meaning for better RAG matching
            if key_entities:
                entity_str = " ".join(f"{k} {v}" for k, v in key_entities.items())
                meaning_text = f"{user_goal}: {entity_str} {meaning_text}".strip()
            else:
                meaning_text = f"{user_goal}: {meaning_text}".strip()
            
            return {
                "intent": "RAG_QUERY",
                "confidence": confidence_score,  # DYNAMIC confidence based on signals
                "context": {
                    "user_goal": user_goal,
                    "key_entities": key_entities,
                    "extracted_meaning": meaning_text
                },
                "reasoning": f"Information query (conf={confidence_score:.2f}, signals: question_word={has_question_word}, topics={len(topics_found)}, info_request={info_request_match})"
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
    
    async def _gemini_classification(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        LLM-based classification for complex cases using Gemini
        
        Args:
            text: User input text
            context: Optional conversation context
            
        Returns:
            Classification result with intent, confidence, context dict, reasoning
        """
        try:
            # Get system prompt
            system_prompt = self._create_leibniz_system_prompt()
            
            # Build context information if provided
            context_info = ""
            if context:
                context_info = f"Context: {json.dumps(context)}. "
            
            # Build full prompt
            full_prompt = f"{system_prompt}\n\n{context_info}User input: '{text}'. Classify this input:"
            
            # Call Gemini API with timeout (Comment 2)
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
                    timeout=self.gemini_timeout
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
                    "reasoning": f"Gemini classification timed out after {self.gemini_timeout}s"
                }
            
            # Extract response text
            response_text = response.text.strip()
            
            # Robust JSON extraction (Comment 5): Remove code fences and extra wrappers
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
            print(f"⚠️ JSON parse error in Gemini response: {e}")
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
            print(f"⚠️ Gemini API error: {e}")
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
        Get classification performance statistics
        
        Returns:
            Dictionary with performance metrics
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
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get LRU cache performance statistics
        
        Returns:
            Dictionary with cache size, hits, hit rate, oldest entry age
        """
        total_classifications = self.fast_route_count + self.gemini_route_count
        cache_hit_rate = (self.cache_hits / total_classifications * 100) if total_classifications > 0 else 0.0
        
        # Get age of oldest entry
        oldest_age = 0.0
        if self.classification_cache:
            oldest_entry = next(iter(self.classification_cache.values()))
            oldest_age = time.time() - oldest_entry[1]
        
        return {
            "cache_size": len(self.classification_cache),
            "cache_max_size": self.cache_max_size,
            "cache_hits": self.cache_hits,
            "total_classifications": total_classifications,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "oldest_entry_age_seconds": round(oldest_age, 1)
        }


# Global instance
leibniz_parser: Optional[LeibnizIntentParser] = None


def get_leibniz_parser() -> LeibnizIntentParser:
    """
    Get global Leibniz intent parser instance (singleton pattern)
    
    Returns:
        Shared LeibnizIntentParser instance
    """
    global leibniz_parser
    if leibniz_parser is None:
        leibniz_parser = LeibnizIntentParser()
    return leibniz_parser


async def classify_leibniz_intent(text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main async function for intent classification (external API)
    
    Args:
        text: User input text to classify
        context: Optional conversation context
        
    Returns:
        Normalized classification result with should_use_rag and language fields
        (optionally minimal schema if LEIBNIZ_INTENT_PARSER_MINIMAL_OUTPUT=true)
    """
    # Get parser instance
    parser = get_leibniz_parser()
    
    # Classify intent
    result = await parser.classify_intent(text, context)
    
    # Normalize output format
    result["should_use_rag"] = (result["intent"] == "RAG_QUERY")
    result["language"] = "english"
    
    # Check if minimal output is requested (Comment 6)
    minimal_output = os.getenv("LEIBNIZ_INTENT_PARSER_MINIMAL_OUTPUT", "false").lower() == "true"
    
    if minimal_output:
        # Strip non-essential fields, keep only intent, confidence, and context
        result = {
            "intent": result.get("intent"),
            "confidence": result.get("confidence"),
            "context": result.get("context", {})
        }
    
    return result


async def test_leibniz_parser():
    """Comprehensive test suite for Leibniz intent parser"""
    print("\n" + "="*80)
    print("LEIBNIZ INTENT PARSER TEST SUITE")
    print("="*80 + "\n")
    
    test_cases = [
        # Appointment scheduling
        ("I'd like to schedule an appointment with admissions", {}),
        ("Can I book a meeting with an advisor?", {}),
        ("When can I meet with the registrar?", {}),
        ("I need to schedule a visit to the financial aid office", {}),
        
        # RAG queries
        ("What are the requirements for the computer science program?", {}),
        ("How do I apply for financial aid?", {}),
        ("Tell me about campus housing options", {}),
        ("Where is the library located?", {}),
        ("What courses are required for a business major?", {}),
        ("Can you explain the enrollment process?", {}),
        ("How much is tuition for the fall semester?", {}),
        
        # Greetings
        ("Hello!", {}),
        ("Good morning", {}),
        ("Hi there", {}),
        
        # Exits
        ("Goodbye", {}),
        ("Thanks, that's all", {}),
        ("I'm done, bye", {}),
        
        # Unclear
        ("asdfghjkl", {}),
        ("", {}),
    ]
    
    print(f"Running {len(test_cases)} test cases...\n")
    
    for idx, (text, ctx) in enumerate(test_cases, 1):
        print(f"Test {idx}: '{text}'")
        print("-" * 80)
        
        result = await classify_leibniz_intent(text, ctx)
        
        print(f"  Intent: {result['intent']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Fast Route: {result['fast_route']}")
        print(f"  Response Time: {result['response_time']:.4f}s")
        print(f"  Context:")
        print(f"    - User Goal: {result['context']['user_goal']}")
        print(f"    - Key Entities: {result['context']['key_entities']}")
        print(f"    - Extracted Meaning: {result['context']['extracted_meaning']}")
        print(f"  Reasoning: {result['reasoning']}")
        print(f"  Should Use RAG: {result['should_use_rag']}")
        print()
    
    # Print performance statistics
    parser = get_leibniz_parser()
    stats = parser.get_performance_stats()
    
    print("="*80)
    print("PERFORMANCE STATISTICS")
    print("="*80)
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Fast Route Count: {stats['fast_route_count']}")
    print(f"Gemini Route Count: {stats['gemini_route_count']}")
    print(f"Fast Route Percentage: {stats['fast_route_percentage']:.2f}%")
    print(f"Average Confidence: {stats['average_confidence']:.3f}")
    print("\nTarget: >80% fast route for optimal performance")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(test_leibniz_parser())
