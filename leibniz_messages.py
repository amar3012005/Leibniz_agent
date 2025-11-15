#!/usr/bin/env python3
"""
Leibniz University Agent Messages - Standardized Data Contracts
==============================================================

This module contains standard data contracts for inter-module communication in the Leibniz University agent system.
These dataclasses provide clear, typed message formats for component interaction and prepare the codebase 
for potential microservices architecture.

These dataclasses are language-agnostic and reusable across different agent implementations.

Message Flow:
    STT → TranscriptMessage → Intent Classification → IntentMessage → 
    RAG (if needed) → RAGMessage → TTS → TTSMessage

Future Compatibility:
    - These dataclasses are designed to be serializable for protocol communication
    - They can be easily converted to JSON for network transmission
    - They maintain clear contracts between system components
    - Subsequent phases will integrate these messages into the Leibniz orchestration system
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone


@dataclass
class TranscriptMessage:
    """
    Output from Speech-to-Text (STT) system.
    Represents transcribed audio with confidence and normalization.
    
    Used in the Leibniz agent's STT module (OpenAI Whisper).
    """
    transcript: str  # The raw transcribed text from audio
    confidence: float = 0.0  # Confidence score from STT system (0.0 to 1.0)
    normalized: Optional[str] = None  # Normalized/cleaned version of transcript
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # When transcription occurred
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional STT metadata (e.g., audio_path, duration)
    
    def __post_init__(self):
        """Validate and clamp probability fields to [0, 1]"""
        # Clamp confidence to [0, 1]
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class IntentMessage:
    """
    Output from Intent Classification system.
    Represents classified user intent with routing information.
    
    Used in the Leibniz agent's intent parser to route queries to appropriate handlers.
    """
    intent: str  # The classified intent (e.g., "RAG_QUERY", "APPOINTMENT_SCHEDULING", "UNCLEAR_REQUEST")
    confidence: float  # Classification confidence score (0.0 to 1.0)
    entities: Dict[str, Any] = field(default_factory=dict)  # Extracted entities from user input
    should_use_rag: bool = False  # Flag indicating if RAG system should be used (routing logic)
    user_context: Optional[str] = None  # Clarified/enriched version of user query (TARA pattern - same as semantic context)
    language: Optional[str] = None  # Detected language (e.g., "english" for Leibniz agent)
    response_style: Optional[str] = None  # Suggested response style (e.g., "helpful", "apologetic")
    reasoning: Optional[str] = None  # Explanation of classification decision
    response_time: float = 0.0  # Time taken for classification in seconds (deprecated - use processing_time)
    processing_time: float = 0.0  # Total processing time in seconds (including context generation)
    timing_breakdown: Dict[str, float] = field(default_factory=dict)  # Detailed timing (classification_ms, context_generation_ms)
    method: Optional[str] = None  # Classification method used (e.g., "persistent_parser", "fallback")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # When classification occurred

    def __post_init__(self):
        """Validate and clamp probability fields to [0, 1]"""
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class RAGMessage:
    """
    Output from RAG (Retrieval-Augmented Generation) system.
    Represents knowledge base query results.
    
    Used in the Leibniz agent's RAG module to retrieve and generate responses from the university knowledge base.
    """
    answer: str  # The generated answer from RAG system
    sources: List[Dict[str, Any]] = field(default_factory=list)  # List of source documents/chunks with metadata (supports richer source information)
    relevance_score: float = 0.0  # Relevance score of retrieved documents (0.0 to 1.0)
    confidence: float = 0.0  # Confidence in the generated answer (0.0 to 1.0)
    method: Optional[str] = None  # Retrieval method (e.g., "persistent_rag", "cache_hit", "speculative_hit", "fallback_timeout")
    cached: bool = False  # Whether result came from cache (performance tracking)
    timing_breakdown: Dict[str, Any] = field(default_factory=dict)  # Detailed timing metrics (embedding_ms, search_ms, response_gen_ms, relevance_score)
    processing_time_ms: float = 0.0  # Total processing time in milliseconds (performance tracking)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # When RAG query was processed

    def __post_init__(self):
        """Validate and clamp probability fields to [0, 1]"""
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class TTSMessage:
    """
    Output from Text-to-Speech (TTS) system.
    Represents synthesized audio output.
    
    Used in the Leibniz agent's TTS module (ElevenLabs or Google TTS).
    """
    audio_path: str  # File path to generated audio file
    text: str  # The text that was synthesized
    duration_ms: float = 0.0  # Duration of audio in milliseconds
    speaker: Optional[str] = None  # Voice speaker used (e.g., "Rachel" for ElevenLabs)
    pace: float = 1.0  # Speech pace/speed multiplier
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # When synthesis occurred

    def __post_init__(self):
        """Validate duration and pace fields"""
        self.duration_ms = max(0.0, self.duration_ms)
        self.pace = 1.0 if self.pace <= 0 else self.pace


# Helper methods for backward compatibility with existing dict-based code
# These facilitate gradual migration from dict-based to dataclass-based message passing

def transcript_to_dict(msg: TranscriptMessage) -> Dict[str, Any]:
    """Convert TranscriptMessage to dictionary for backward compatibility"""
    return {
        "transcript": msg.transcript,
        "confidence": msg.confidence,
        "normalized": msg.normalized,
        "timestamp": msg.timestamp.isoformat(),
        "metadata": msg.metadata
    }


def transcript_from_dict(data: Dict[str, Any]) -> TranscriptMessage:
    """Create TranscriptMessage from dictionary"""
    timestamp = data.get("timestamp")
    if isinstance(timestamp, str):
        # Handle 'Z' suffix and common ISO 8601 variants
        if timestamp.endswith('Z'):
            timestamp = timestamp[:-1] + '+00:00'
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except Exception:
            timestamp = datetime.now(timezone.utc)
    elif timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    # Treat naive datetimes as UTC and normalize to timezone-aware
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    return TranscriptMessage(
        transcript=data.get("transcript", ""),
        confidence=data.get("confidence", 0.0),
        normalized=data.get("normalized"),
        timestamp=timestamp,
        metadata=data.get("metadata", {})
    )


def intent_to_dict(msg: IntentMessage) -> Dict[str, Any]:
    """Convert IntentMessage to dictionary for backward compatibility"""
    return {
        "intent": msg.intent,
        "confidence": msg.confidence,
        "entities": msg.entities,
        "should_use_rag": msg.should_use_rag,
        "language": msg.language,
        "response_style": msg.response_style,
        "reasoning": msg.reasoning,
        "response_time": msg.response_time,
        "method": msg.method,
        "timestamp": msg.timestamp.isoformat()
    }


def intent_from_dict(data: Dict[str, Any]) -> IntentMessage:
    """Create IntentMessage from dictionary"""
    timestamp = data.get("timestamp")
    if isinstance(timestamp, str):
        # Handle 'Z' suffix and common ISO 8601 variants
        if timestamp.endswith('Z'):
            timestamp = timestamp[:-1] + '+00:00'
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except Exception:
            timestamp = datetime.now(timezone.utc)
    elif timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    # Treat naive datetimes as UTC and normalize to timezone-aware
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    return IntentMessage(
        intent=data.get("intent", "UNCLEAR"),
        confidence=data.get("confidence", 0.0),
        entities=data.get("entities", {}),
        should_use_rag=data.get("should_use_rag", False),
        language=data.get("language"),
        response_style=data.get("response_style"),
        reasoning=data.get("reasoning"),
        response_time=data.get("response_time", 0.0),
        method=data.get("method"),
        timestamp=timestamp
    )


def rag_to_dict(msg: RAGMessage) -> Dict[str, Any]:
    """Convert RAGMessage to dictionary for backward compatibility. Sources may contain richer metadata (List[Dict[str, Any]])."""
    return {
        "answer": msg.answer,
        "sources": msg.sources,
        "relevance_score": msg.relevance_score,
        "confidence": msg.confidence,
        "method": msg.method,
        "cached": msg.cached,
        "timing_breakdown": msg.timing_breakdown,
        "processing_time_ms": msg.processing_time_ms,
        "timestamp": msg.timestamp.isoformat()
    }


def rag_from_dict(data: Dict[str, Any]) -> RAGMessage:
    """Create RAGMessage from dictionary. Supports richer source metadata (List[Dict[str, Any]])."""
    timestamp = data.get("timestamp")
    if isinstance(timestamp, str):
        # Handle 'Z' suffix and common ISO 8601 variants
        if timestamp.endswith('Z'):
            timestamp = timestamp[:-1] + '+00:00'
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except Exception:
            timestamp = datetime.now(timezone.utc)
    elif timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    # Treat naive datetimes as UTC and normalize to timezone-aware
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    return RAGMessage(
        answer=data.get("answer", data.get("response", "")),
        sources=data.get("sources", []),
        relevance_score=data.get("relevance_score", 0.0),
        confidence=data.get("confidence", 0.0),
        method=data.get("method"),
        cached=data.get("cached", False),
        timing_breakdown=data.get("timing_breakdown", {}),
        processing_time_ms=data.get("processing_time_ms", 0.0),
        timestamp=timestamp
    )


def tts_to_dict(msg: TTSMessage) -> Dict[str, Any]:
    """Convert TTSMessage to dictionary for backward compatibility"""
    return {
        "audio_path": msg.audio_path,
        "text": msg.text,
        "duration_ms": msg.duration_ms,
        "speaker": msg.speaker,
        "pace": msg.pace,
        "timestamp": msg.timestamp.isoformat()
    }


def tts_from_dict(data: Dict[str, Any]) -> TTSMessage:
    """Create TTSMessage from dictionary"""
    timestamp = data.get("timestamp")
    if isinstance(timestamp, str):
        # Handle 'Z' suffix and common ISO 8601 variants
        if timestamp.endswith('Z'):
            timestamp = timestamp[:-1] + '+00:00'
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except Exception:
            timestamp = datetime.now(timezone.utc)
    elif timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    # Treat naive datetimes as UTC and normalize to timezone-aware
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    return TTSMessage(
        audio_path=data.get("audio_path", ""),
        text=data.get("text", ""),
        duration_ms=data.get("duration_ms", 0.0),
        speaker=data.get("speaker"),
        pace=data.get("pace", 1.0),
        timestamp=timestamp
    )
