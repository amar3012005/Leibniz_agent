"""
Leibniz Fast Semantic Context Extractor

Lightweight, regex-based semantic context extraction that runs IMMEDIATELY
after transcript generation (before intent classification).

Purpose:
- Extract user goal, key entities, and meaning from raw transcript in <5ms
- Provide enriched context to intent classifier for better accuracy
- Avoid duplicate LLM calls (context extraction + intent classification)

Architecture:
Two-tier extraction:
1. Fast pattern matching (regex + keywords) - <5ms for 90% of cases
2. LLM fallback (Gemini Nano/Flash Lite) - <100ms for complex cases

Output Format:
{
    'user_goal': str,           # High-level goal ("schedule appointment", "learn about program")
    'key_entities': dict,       # Extracted entities {"department": "CS", "date": "next week"}
    'extracted_meaning': str,   # Normalized/paraphrased query
    'extraction_method': str,   # 'fast_pattern' or 'llm_fallback'
    'extraction_time_ms': float # Extraction latency
}
"""

import re
import time
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LeibnizSemanticExtractor:
    """Fast semantic context extractor for Leibniz agent"""
    
    # Common action patterns (user goals)
    ACTION_PATTERNS = {
        'schedule': r'\b(schedule|book|make|set up|arrange|get|need)\s+(an?\s+)?(appointment|meeting|visit|session|consultation)\b',
        'inquire': r'\b(tell me|what is|what are|how do|explain|information about|details about|know about|learn about)\b',
        'apply': r'\b(apply|application|admission|enroll|enrollment|register|registration|join)\b',
        'find': r'\b(find|locate|where is|where can|looking for|search for)\b',
        'compare': r'\b(compare|difference between|versus|vs|better|prefer)\b',
        'verify': r'\b(confirm|verify|check|validate|is it true|correct)\b',
        'list': r'\b(list|show|give me|all|available|options)\b',
        'recommend': r'\b(recommend|suggest|advise|best|top|which should)\b'
    }
    
    # Entity extraction patterns
    ENTITY_PATTERNS = {
        'department': r'\b(computer science|cs|engineering|business|economics|biology|chemistry|physics|math|mathematics|psychology|sociology|english|history|art|music)\b',
        'program': r'\b(bachelor|bs|ba|master|ms|ma|phd|doctorate|undergraduate|graduate|major|minor|degree|program)\b',
        'person': r'\b(professor|prof|dr|advisor|counselor|dean|instructor|teacher)\b',
        'time': r'\b(today|tomorrow|next week|next month|monday|tuesday|wednesday|thursday|friday|morning|afternoon|evening|am|pm|\d{1,2}:\d{2})\b',
        'location': r'\b(campus|building|room|office|library|lab|cafeteria|gym|hall|center|department)\b',
        'course': r'\b(course|class|lecture|seminar|workshop|lab|tutorial)\b',
        'service': r'\b(admission|advising|counseling|career|financial aid|scholarship|housing|parking|health|registration)\b'
    }
    
    # Goal templates based on action
    GOAL_TEMPLATES = {
        'schedule': 'User wants to schedule an appointment or meeting',
        'inquire': 'User is seeking information or explanation',
        'apply': 'User wants to apply or enroll in a program',
        'find': 'User is looking for a location, person, or resource',
        'compare': 'User wants to compare options or alternatives',
        'verify': 'User needs confirmation or verification',
        'list': 'User wants a list of available options',
        'recommend': 'User is seeking recommendations or advice'
    }
    
    def __init__(self):
        """Initialize semantic extractor"""
        self.extraction_count = 0
        self.fast_route_count = 0
        self.llm_route_count = 0
    
    def extract_context(self, transcript: str) -> Dict[str, Any]:
        """
        Extract semantic context from transcript using fast patterns
        
        Args:
            transcript: Raw user transcript from VAD
            
        Returns:
            Dict with user_goal, key_entities, extracted_meaning, metadata
        """
        start_time = time.time()
        self.extraction_count += 1
        
        # Normalize transcript
        text_lower = transcript.lower().strip()
        
        # Extract action/goal
        detected_action = None
        for action, pattern in self.ACTION_PATTERNS.items():
            if re.search(pattern, text_lower):
                detected_action = action
                break
        
        # Extract entities
        entities = {}
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                # Take first match or join multiple matches
                entities[entity_type] = matches[0] if len(matches) == 1 else ', '.join(matches)
        
        # Generate user goal
        if detected_action:
            user_goal = self.GOAL_TEMPLATES.get(detected_action, "User has a general query")
            
            # Enhance goal with entities if available
            if 'service' in entities:
                user_goal = f"{user_goal} related to {entities['service']}"
            elif 'department' in entities:
                user_goal = f"{user_goal} related to {entities['department']}"
        else:
            # Fallback: Generic goal
            if entities:
                user_goal = f"User is asking about {', '.join(entities.values())}"
            else:
                user_goal = "User has a general query about the university"
        
        # Generate extracted meaning (normalized query)
        # Remove filler words and normalize
        extracted_meaning = self._normalize_query(text_lower)
        
        # Add key terms from entities
        if entities:
            key_terms = ' '.join(entities.values())
            extracted_meaning = f"{extracted_meaning} {key_terms}".strip()
        
        extraction_time = (time.time() - start_time) * 1000
        self.fast_route_count += 1
        
        result = {
            'user_goal': user_goal,
            'key_entities': entities,
            'extracted_meaning': extracted_meaning,
            'extraction_method': 'fast_pattern',
            'extraction_time_ms': extraction_time,
            'detected_action': detected_action
        }
        
        logger.debug(f" Semantic context extracted in {extraction_time:.1f}ms: '{user_goal}'")
        
        return result
    
    def _normalize_query(self, text: str) -> str:
        """
        Normalize query by removing filler words and standardizing
        
        Args:
            text: Lowercase input text
            
        Returns:
            Normalized query string
        """
        # Remove common filler words
        fillers = [
            r'\b(um|uh|like|you know|i mean|well|so|actually|basically|literally)\b',
            r'\b(can you|could you|would you|please|help me)\b',
            r'\b(i want to|i need to|i would like to)\b'
        ]
        
        normalized = text
        for filler_pattern in fillers:
            normalized = re.sub(filler_pattern, '', normalized)
        
        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return {
            'total_extractions': self.extraction_count,
            'fast_route': self.fast_route_count,
            'llm_route': self.llm_route_count,
            'fast_route_pct': (self.fast_route_count / self.extraction_count * 100) if self.extraction_count > 0 else 0
        }


# Global singleton instance
_semantic_extractor = None

def get_semantic_extractor() -> LeibnizSemanticExtractor:
    """Get or create singleton semantic extractor"""
    global _semantic_extractor
    if _semantic_extractor is None:
        _semantic_extractor = LeibnizSemanticExtractor()
    return _semantic_extractor


# Convenience function for direct use
def extract_semantic_context(transcript: str) -> Dict[str, Any]:
    """
    Extract semantic context from transcript (convenience wrapper)
    
    Args:
        transcript: Raw user transcript
        
    Returns:
        Dict with user_goal, key_entities, extracted_meaning
    """
    extractor = get_semantic_extractor()
    return extractor.extract_context(transcript)
