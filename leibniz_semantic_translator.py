"""
Leibniz Semantic Context Generator with Translation Support

This module provides semantic understanding and translation capabilities for the Leibniz
appointment booking FSM. It enables slot filling from multilingual inputs (Hindi, Telugu, etc.)
by generating English semantic context that can be used directly by the FSM.

Key Features:
- Language detection (Hindi, Telugu, English, mixed)
- Translation to English using Gemini API
- Semantic entity extraction for slot filling
- Field-specific context generation for FSM states

Integration with Appointment FSM:
- Called before each FSM process_input() to generate semantic context
- Returns normalized English text + extracted entities
- Enables natural multilingual conversation without modifying FSM logic

Author: SINDH Technologies
Date: November 2025
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class LeibnizSemanticTranslator:
    """
    Semantic context generator with translation for multilingual appointment booking
    """
    
    def __init__(self):
        """Initialize the semantic translator with Gemini API"""
        # Load Gemini API key
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        
        # Use lite model for fast translation
        model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash-lite")
        self.model = genai.GenerativeModel(model_name)
        
        # Translation timeout
        self.timeout = float(os.getenv("LEIBNIZ_SEMANTIC_TRANSLATOR_TIMEOUT", "5.0"))
        
        # Language detection patterns
        self.language_patterns = {
            'hindi': re.compile(r'[\u0900-\u097F]+'),  # Devanagari script
            'telugu': re.compile(r'[\u0C00-\u0C7F]+'),  # Telugu script
            'tamil': re.compile(r'[\u0B80-\u0BFF]+'),   # Tamil script
            'kannada': re.compile(r'[\u0C80-\u0CFF]+'), # Kannada script
            'malayalam': re.compile(r'[\u0D00-\u0D7F]+') # Malayalam script
        }
        
        logger.info(f"✅ Leibniz Semantic Translator initialized ({model_name}, timeout={self.timeout}s)")
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect primary language in user input
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (language, confidence)
            - language: 'english', 'hindi', 'telugu', 'tamil', 'mixed', etc.
            - confidence: 0.0-1.0
        """
        if not text or not text.strip():
            return ('english', 1.0)
        
        # Count characters by script
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return ('english', 1.0)
        
        script_counts = {}
        for lang, pattern in self.language_patterns.items():
            matches = pattern.findall(text)
            char_count = sum(len(match) for match in matches)
            if char_count > 0:
                script_counts[lang] = char_count
        
        # If no Indic scripts found, assume English
        if not script_counts:
            return ('english', 1.0)
        
        # Find dominant script
        dominant_lang = max(script_counts, key=script_counts.get)
        dominant_count = script_counts[dominant_lang]
        
        # Calculate confidence
        confidence = dominant_count / total_chars
        
        # Check if mixed (both English and Indic)
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        if english_chars > 0 and dominant_count > 0:
            # Mixed language
            if dominant_count > english_chars:
                return (dominant_lang, confidence)
            else:
                return ('mixed', 0.7)
        
        return (dominant_lang, confidence)
    
    async def translate_to_english(
        self,
        text: str,
        source_language: Optional[str] = None,
        fsm_state: Optional[str] = None,
        field_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate text to English and extract semantic entities for FSM
        
        Args:
            text: User input text (any language)
            source_language: Optional detected language ('hindi', 'telugu', etc.)
            fsm_state: Optional current FSM state (e.g., 'collect_name', 'collect_email')
            field_context: Optional field being collected (e.g., 'name', 'email', 'phone')
            
        Returns:
            Dictionary with:
                - english_text: Translated/normalized English text
                - original_text: Original user input
                - detected_language: Auto-detected or provided language
                - entities: Extracted entities relevant to current field
                - confidence: Translation confidence (0.0-1.0)
                - semantic_meaning: Paraphrased meaning for slot filling
        """
        try:
            # Detect language if not provided
            if not source_language:
                detected_lang, lang_confidence = self.detect_language(text)
            else:
                detected_lang = source_language
                lang_confidence = 1.0
            
            # If already English, just normalize
            if detected_lang == 'english':
                return {
                    'english_text': text.strip(),
                    'original_text': text,
                    'detected_language': 'english',
                    'entities': self._extract_entities_from_english(text, field_context),
                    'confidence': 1.0,
                    'semantic_meaning': text.lower().strip()
                }
            
            # Build translation prompt with FSM context
            prompt = self._build_translation_prompt(
                text=text,
                source_language=detected_lang,
                fsm_state=fsm_state,
                field_context=field_context
            )
            
            # Call Gemini API with timeout
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.model.generate_content,
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.1,  # Low temperature for accurate translation
                            max_output_tokens=300
                        )
                    ),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Translation timeout after {self.timeout}s, returning original text")
                return {
                    'english_text': text,
                    'original_text': text,
                    'detected_language': detected_lang,
                    'entities': {},
                    'confidence': 0.5,
                    'semantic_meaning': text.lower()
                }
            
            # Parse response
            response_text = response.text.strip()
            
            # Extract JSON from response
            result = self._parse_translation_response(response_text)
            
            # Add original metadata
            result['original_text'] = text
            result['detected_language'] = detected_lang
            
            # Ensure required fields
            if 'english_text' not in result:
                result['english_text'] = text
            if 'entities' not in result:
                result['entities'] = {}
            if 'confidence' not in result:
                result['confidence'] = 0.8
            if 'semantic_meaning' not in result:
                result['semantic_meaning'] = result.get('english_text', text).lower()
            
            logger.info(f"🌐 Translated {detected_lang} → English: '{text}' → '{result['english_text']}'")
            
            return result
        
        except Exception as e:
            logger.error(f"Translation error: {e}", exc_info=True)
            
            # Fallback: return original text
            return {
                'english_text': text,
                'original_text': text,
                'detected_language': detected_lang if 'detected_lang' in locals() else 'unknown',
                'entities': {},
                'confidence': 0.3,
                'semantic_meaning': text.lower(),
                'error': str(e)
            }
    
    def _build_translation_prompt(
        self,
        text: str,
        source_language: str,
        fsm_state: Optional[str],
        field_context: Optional[str]
    ) -> str:
        """
        Build Gemini prompt for translation with FSM context
        
        Args:
            text: Text to translate
            source_language: Detected source language
            fsm_state: Current FSM state
            field_context: Current field being collected
            
        Returns:
            Formatted prompt string
        """
        # Base translation instruction
        prompt = f"""You are a professional translator for a university appointment booking system.

Translate the following {source_language.upper()} text to natural English and extract relevant entities.

"""
        
        # Add FSM context if available
        if field_context:
            context_instructions = {
                'name': "The user is providing their NAME. Extract first name and last name if possible.",
                'email': "The user is providing their EMAIL ADDRESS. Extract the email if mentioned.",
                'phone': "The user is providing their PHONE NUMBER. Extract the phone number with country code if possible.",
                'department': "The user is selecting a DEPARTMENT or SERVICE. Extract the department name.",
                'appointment_type': "The user is selecting an APPOINTMENT TYPE. Extract the service they need.",
                'datetime': "The user is providing a PREFERRED DATE/TIME. Extract date and time information.",
                'purpose': "The user is describing the PURPOSE/REASON for their appointment. Summarize their needs."
            }
            
            if field_context in context_instructions:
                prompt += f"Context: {context_instructions[field_context]}\n\n"
        
        prompt += f"""Input text: "{text}"

Provide your response as JSON with these fields:
{{
    "english_text": "translated text in natural English",
    "entities": {{
        "extracted_value": "main value relevant to current field",
        "additional_info": "any additional context"
    }},
    "confidence": 0.9,
    "semantic_meaning": "paraphrased meaning for understanding"
}}

Respond ONLY with valid JSON, no markdown, no code fences."""
        
        return prompt
    
    def _parse_translation_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Gemini translation response
        
        Args:
            response_text: Raw Gemini response
            
        Returns:
            Parsed dictionary
        """
        # Remove markdown code fences
        response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
        response_text = re.sub(r'\s*```$', '', response_text)
        response_text = response_text.strip()
        
        # Try parsing as JSON
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            # Try extracting JSON object
            json_match = re.search(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                # Fallback: treat entire response as english_text
                return {
                    'english_text': response_text,
                    'entities': {},
                    'confidence': 0.7,
                    'semantic_meaning': response_text.lower()
                }
    
    def _extract_entities_from_english(
        self,
        text: str,
        field_context: Optional[str]
    ) -> Dict[str, Any]:
        """
        Extract entities from English text using simple patterns
        
        Args:
            text: English text
            field_context: Current field being collected
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        
        if not field_context:
            return entities
        
        # Simple entity extraction patterns
        if field_context == 'email':
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            if email_match:
                entities['extracted_value'] = email_match.group(0)
        
        elif field_context == 'phone':
            phone_match = re.search(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,9}', text)
            if phone_match:
                entities['extracted_value'] = phone_match.group(0)
        
        elif field_context == 'name':
            # Extract capitalized words (likely names)
            name_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            if name_words:
                entities['extracted_value'] = ' '.join(name_words)
        
        return entities


# Singleton instance
_semantic_translator_instance = None


def get_semantic_translator() -> LeibnizSemanticTranslator:
    """
    Get singleton semantic translator instance
    
    Returns:
        LeibnizSemanticTranslator instance
    """
    global _semantic_translator_instance
    
    if _semantic_translator_instance is None:
        _semantic_translator_instance = LeibnizSemanticTranslator()
    
    return _semantic_translator_instance


# ============================================================================
# Helper Functions
# ============================================================================

async def generate_semantic_context_for_fsm(
    user_input: str,
    fsm_state: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate semantic context for FSM slot filling with translation support
    
    This is the main entry point for FSM integration. Call this before
    each fsm.process_input() to get English semantic context.
    
    Args:
        user_input: Raw user input (any language)
        fsm_state: Current FSM state (e.g., 'collect_name', 'confirm_email')
        
    Returns:
        Dictionary with:
            - translated_text: English version of user input
            - original_text: Original user input
            - detected_language: Detected language
            - field_value: Extracted value for current field
            - confidence: Translation confidence
            
    Usage in FSM:
        context = await generate_semantic_context_for_fsm(user_input, fsm.state.value)
        result = await fsm.process_input(context['translated_text'])
    """
    translator = get_semantic_translator()
    
    # Extract field context from FSM state
    field_context = None
    if fsm_state:
        # Map FSM states to field contexts
        if 'name' in fsm_state:
            field_context = 'name'
        elif 'email' in fsm_state:
            field_context = 'email'
        elif 'phone' in fsm_state:
            field_context = 'phone'
        elif 'department' in fsm_state:
            field_context = 'department'
        elif 'appointment_type' in fsm_state or 'type' in fsm_state:
            field_context = 'appointment_type'
        elif 'datetime' in fsm_state or 'date' in fsm_state or 'time' in fsm_state:
            field_context = 'datetime'
        elif 'purpose' in fsm_state:
            field_context = 'purpose'
    
    # Translate and extract entities
    result = await translator.translate_to_english(
        text=user_input,
        fsm_state=fsm_state,
        field_context=field_context
    )
    
    # Format for FSM consumption
    return {
        'translated_text': result['english_text'],
        'original_text': result['original_text'],
        'detected_language': result['detected_language'],
        'field_value': result['entities'].get('extracted_value', ''),
        'confidence': result['confidence'],
        'semantic_meaning': result['semantic_meaning']
    }
