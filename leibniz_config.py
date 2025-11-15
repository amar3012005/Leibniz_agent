#!/usr/bin/env python3
"""
Leibniz University Agent Configuration File
==========================================
Centralized configuration for the Leibniz University Institute customer service agent.
Provides friendly, casual English-only support for university-related inquiries.

Technical Architecture:
- STT: Gemini Live API with SINDH bidirectional VAD for robust English-only speech recognition
- LLM: Gemini for intent parsing and response generation
- RAG: FAISS vector store over university knowledge base
- TTS: Google Cloud TTS (primary) or ElevenLabs (fallback) for English speech synthesis

Enhanced Features (SINDH Integration):
- Bidirectional VAD with smart prompting and background warmup
- Dynamic timeout adjustment based on conversation context
- Barge-in detection for natural conversation flow
- Persistent session management for low-latency responses

Note: The system uses Gemini Live API for both STT and LLM operations,
providing unified infrastructure and better integration. The dual-provider TTS
architecture provides automatic fallback for reliability.
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Tuple
import os
import datetime

# Debug mode for semantic context flow tracking
SEMANTIC_CONTEXT_DEBUG = os.getenv('LEIBNIZ_SEMANTIC_CONTEXT_DEBUG', 'false').lower() == 'true'

@dataclass
class VoiceConfig:
    """Voice and speech configuration for Leibniz agent"""
    # TTS Settings
    # Changed default to elevenlabs (premium quality) - Google TTS as stable fallback
    tts_provider: str = "xtts_local"  # TTS provider: xtts_local, elevenlabs, google, gemini, auto (with fallback)
    tts_fallback_provider: str = "google"  # Fallback TTS provider
    voice_id: str = "EXAVITQu4vr4xnSDxMaL"  # ElevenLabs voice_id (Sarah - soft female). Get IDs from GET /v2/voices API
    google_voice: str = "en-US-Neural2-F"  # Google TTS voice (Neural2-F: female, Neural2-C: female, Neural2-A: male)
    elevenlabs_model: str = "eleven_multilingual_v2"  # ElevenLabs model
    # Gemini TTS model options (verified 2025-10-27 from official docs):
    # UNSTABLE - frequent 500 errors as of Oct 2025: gemini-2.5-flash-preview-tts
    # VERY UNSTABLE - more prone to 500 errors: gemini-2.5-pro-preview-tts
    # Recommended: Use Google Cloud TTS (google) or ElevenLabs (elevenlabs) as primary provider
    # See: https://ai.google.dev/gemini-api/docs/speech-generation
    gemini_model: str = "gemini-2.5-flash-preview-tts"  # Only used if tts_provider="gemini" (not recommended)
    
    # Gemini TTS Valid Voice Names (30 voices - Capital first letter required)
    # See full list: https://ai.google.dev/gemini-api/docs/speech-generation#voice-options
    gemini_voice: str = "Pulcherrima"  # Default voice (Forward)
    gemini_voice_helpful: str = "Pulcherrima"  # Forward
    gemini_voice_calm: str = "Vindemiatrix"  # Gentle
    gemini_voice_excited: str = "Fenrir"  # Excitable
    gemini_voice_professional: str = "Achernar"  # Soft
    
    # XTTS Local TTS Settings
    xtts_speaker_sample: str = "leibniz_agent/audio/speaker_sample.wav"  # Path to speaker sample for voice cloning (6-60s WAV, mono, 16-22kHz recommended)
    xtts_language: str = "en"  # XTTS language code (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi)
    xtts_device: str = "auto"  # Device for XTTS: cuda (GPU, recommended), cpu (slower), or auto (detect)
    xtts_stream_chunk_size: int = 20  # Streaming chunk size (lower = faster first audio, higher = better quality)
    xtts_overlap_wav_len: int = 1024  # Overlap length for smooth streaming
    
    language: str = "en-US"  # English - United States
    
    # Voice Characteristics
    pitch: float = 0.0  # Neutral pitch (varies by provider)
    speed: float = 1.0  # Natural speech speed
    volume: float = 1.0  # Full volume (0.0-1.0)
    tone: str = "friendly"  # friendly, professional, casual, warm
    
    # Speech Patterns
    pause_short: float = 0.3  # Short pause between sentences
    pause_medium: float = 0.6  # Medium pause for emphasis
    pause_long: float = 1.0  # Long pause for dramatic effect
    breathing_sounds: bool = False  # Cleaner English speech without breathing
    
    # Emotional Voice Variations
    emotion_pitch_map: Dict[str, float] = None
    emotion_speed_map: Dict[str, float] = None
    
    def __post_init__(self):
        if self.emotion_pitch_map is None:
            self.emotion_pitch_map = {
                'happy': 0.1, 'excited': 0.15, 'calm': 0.0, 'sad': -0.1,
                'helpful': 0.05, 'professional': 0.0, 'enthusiastic': 0.2
            }
        if self.emotion_speed_map is None:
            self.emotion_speed_map = {
                'happy': 1.1, 'excited': 1.2, 'calm': 0.95, 'sad': 0.9,
                'helpful': 1.0, 'professional': 1.0, 'enthusiastic': 1.15
            }

@dataclass
class PersonalityConfig:
    """Leibniz agent's personality and behavior configuration"""
    # Core Identity
    name: str = "Lexi"  # Friendly, approachable name
    name_english: str = "Lexi"
    full_name: str = "Leibniz University Institute Assistant"
    age_persona: str = "mid-20s university support specialist"
    personality_type: str = "ENFJ"  # The Protagonist - helpful, empathetic
    background: str = "I'm a friendly and helpful assistant for Leibniz University Institute, here to help with your questions about courses, admissions, appointments, and campus services."
    
    # Behavioral Traits (0.0-1.0 scale)
    friendliness_level: float = 0.85  # Friendly but professional
    formality_level: float = 0.3  # Casual but respectful for university context
    empathy_level: float = 0.8  # High empathy
    patience_level: float = 0.9  # Very patient
    humor_level: float = 0.6  # Moderate, appropriate for university setting
    enthusiasm_level: float = 0.75  # Moderately enthusiastic
    
    # Communication Style
    speaking_style: str = "conversational"  # conversational, professional, casual
    local_touch: bool = False  # No regional expressions needed for English
    emotional_intelligence: bool = True  # Respond to user emotions
    encouragement_frequency: str = "medium"  # low, medium, high
    use_honorifics: bool = False  # English doesn't require formal titles
    
    # Natural Speech Patterns
    filler_words: List[str] = None
    greeting_phrases: List[str] = None
    expressions: List[str] = None
    avoid_phrases: List[str] = None
    
    def __post_init__(self):
        if self.filler_words is None:
            self.filler_words = ["um", "well", "you know", "let me see", "okay", "right", "so"]
        if self.greeting_phrases is None:
            self.greeting_phrases = ["Hi there!", "Hello!", "Hey!", "Good to hear from you!", "How can I help you today?", "Welcome!"]
        if self.expressions is None:
            self.expressions = ["That's great!", "Awesome!", "Perfect!", "Sounds good!", "Excellent!", "Nice!", "Wonderful!", "Fantastic!"]
        if self.avoid_phrases is None:
            self.avoid_phrases = ["I'm Lexi", "Leibniz customer service", "As an AI assistant"]

@dataclass
class ConversationConfig:
    """Conversation flow and dialogue configuration"""
    # Response Timing
    response_delay_min: float = 0.5  # Minimum response delay (seconds)
    response_delay_max: float = 1.5  # Maximum response delay (seconds)
    thinking_time: float = 0.3  # Time before "processing" responses
    
    # Conversation Flow
    max_retry_attempts: int = 3  # Max retries for data collection
    patience_retry_messages: bool = True  # Different messages each retry
    escalation_threshold: int = 2  # When to escalate to human
    
    # Context Management
    memory_span: int = 10  # Remember last 10 exchanges
    context_awareness: bool = True  # Maintain conversation context
    topic_switching_smoothness: float = 0.8  # How smoothly to switch topics
    
    # Interruption Handling
    allow_interruptions: bool = True  # User can interrupt agent
    interruption_grace_period: float = 0.5  # Seconds to wait
    interruption_response: str = "Yes, go ahead?"  # Response to interruption
    
    # Silence Handling
    silence_timeout: int = 5  # Seconds before prompting user
    silence_prompts: List[str] = None
    
    def __post_init__(self):
        if self.silence_prompts is None:
            self.silence_prompts = [
                "Is there anything else you'd like to know?",
                "I'm here to help, what would you like to ask?",
                "Do you have any other questions?",
                "Feel free to ask me anything about the university."
            ]

@dataclass
class EmotionalConfig:
    """Emotional intelligence and response configuration"""
    # Emotion Detection
    detect_user_emotions: bool = True
    respond_to_emotions: bool = True
    emotion_confidence_threshold: float = 0.7
    
    # Emotional Responses
    excitement_expressions: List[str] = None
    sympathy_expressions: List[str] = None
    encouragement_expressions: List[str] = None
    celebration_expressions: List[str] = None
    
    # Mood Adaptation
    adapt_to_user_mood: bool = True
    mirror_energy_level: bool = True  # Match user's energy
    comfort_distressed_users: bool = True
    
    # Energy Levels by Time
    energy_schedule: Dict[str, str] = None
    
    def __post_init__(self):
        if self.excitement_expressions is None:
            self.excitement_expressions = [
                "Wow!", "That's fantastic!", "Excellent!", "Great news!", "Wonderful!",
                "Amazing!", "Brilliant!", "Outstanding!", "Superb!", "Terrific!"
            ]
        if self.sympathy_expressions is None:
            self.sympathy_expressions = [
                "I understand", "I see", "That makes sense", "No worries", "I'm here to help",
                "I get it", "Don't worry about it", "That's okay", "I can help with that"
            ]
        if self.encouragement_expressions is None:
            self.encouragement_expressions = [
                "You've got this!", "Don't worry", "I'm here to help you", "Let's try again",
                "Great job!", "You're doing well", "Keep going!", "Almost there!"
            ]
        if self.celebration_expressions is None:
            self.celebration_expressions = [
                "Congratulations!", "That's wonderful!", "Well done!", "Fantastic!", "Excellent work!",
                "You did it!", "Awesome achievement!", "Great success!", "Brilliant!"
            ]
        if self.energy_schedule is None:
            self.energy_schedule = {
                'morning': 'high',     # 6-12
                'afternoon': 'medium', # 12-17
                'evening': 'calm',     # 17-21
                'night': 'gentle'      # 21-6
            }

@dataclass
class TechnicalConfig:
    """Technical and integration configuration"""
    # API Settings
    stt_provider: str = "gemini_live"  # Speech-to-text provider (Gemini Live API)
    llm_provider: str = "gemini"  # Primary LLM provider
    fallback_llm: str = "openai"  # Fallback LLM
    
    # STT-specific settings (SINDH-enhanced)
    stt_language: str = "en-US"  # English language code for STT
    stt_strict_english: bool = True  # Enforce English-only validation
    stt_timeout: float = 30.0  # File transcription timeout in seconds
    stt_streaming_timeout: float = 10.0  # Streaming capture start timeout
    stt_silence_timeout: float = 2.5  # Silence detection timeout
    stt_enable_language_detection: bool = True  # Validate language
    
    # SINDH VAD features (aligned with bidirectional VAD capabilities)
    stt_smart_prompt_enabled: bool = True  # Enable smart prompting during silence
    stt_smart_prompt_threshold: float = 6.0  # Seconds before triggering smart prompts
    stt_barge_in_threshold: float = 0.5  # Threshold for barge-in detection
    stt_warmup_trigger_delay: float = 2.0  # Delay for background warmup
    stt_max_timeout_s: float = 15.0  # Maximum timeout for complex responses
    
    # TTS-specific settings
    tts_timeout: float = 30.0  # TTS synthesis timeout in seconds
    tts_retry_attempts: int = 3  # Number of retry attempts for TTS
    tts_retry_delay: float = 1.0  # Delay between retries in seconds
    tts_enable_fallback: bool = True  # Enable automatic provider fallback
    tts_cache_enabled: bool = True  # Enable TTS caching
    tts_cache_max_size: int = 500  # Maximum cache entries
    tts_cache_ttl_days: int = 30  # Cache time-to-live in days
    tts_sample_rate: int = 24000  # Audio sample rate for TTS
    
    # Performance Settings
    response_timeout: float = 10.0  # Maximum response time
    response_time_target: float = 2.0  # Target response time
    api_retry_attempts: int = 3
    cache_responses: bool = True
    cache_duration: int = 3600  # 1 hour in seconds
    
    # Audio Processing
    audio_quality: str = "high"  # low, medium, high
    noise_reduction: bool = True
    echo_cancellation: bool = True
    automatic_gain_control: bool = True
    
    # RAG Settings
    rag_document_limit: int = 8  # Max documents to retrieve (suitable for university KB)
    rag_confidence_threshold: float = 0.7
    
    # RAG Feature Toggles (TARA pattern adoption)
    rag_enable_cache: bool = True  # Enable query response caching
    rag_enable_speculative: bool = True  # Enable speculative execution
    rag_adaptive_timeout: bool = True  # Enable adaptive timeouts based on intent
    rag_enable_streaming: bool = True  # Enable streaming TTS for long responses
    rag_enable_streaming_generation: bool = True  # Stream RAG responses progressively
    rag_enable_fallbacks: bool = True  # Enable intent-specific fallback responses
    rag_enable_metrics: bool = True  # Enable performance tracking
    tts_enable_parallel_synthesis: bool = True  # Prefetch next sentence during playback
    
    # RAG Cache Configuration
    rag_cache_ttl: int = 3600  # Cache time-to-live in seconds (1 hour)
    rag_speculative_max_age: int = 60  # Max age for speculative results in seconds
    
    # RAG Quality Configuration
    rag_min_confidence: float = 0.5  # Minimum confidence threshold for responses
    rag_response_min_length: int = 20  # Minimum response length (chars)
    
    # RAG Fallback Configuration
    rag_fallback_timeout: float = 30.0  # Timeout before using fallback response
    
    # VAD Verbose Logging Configuration (SINDH pattern)
    vad_verbose_logging: bool = True  # Enable detailed VAD session logs
    vad_log_audio_callbacks: bool = False  # Log audio callback details (debug only)
    vad_log_state_transitions: bool = True  # Log state changes (listening, speaking)
    vad_log_timeout_checks: bool = False  # Log timeout management (debug only)
    rag_retry_attempts: int = 1  # Number of retry attempts for RAG queries
    
    # RAG Metrics Configuration
    rag_metrics_verbose: bool = False  # Enable verbose metrics logging
    rag_metrics_log_interval: int = 10  # Log metrics every N queries
    
    # Security
    data_encryption: bool = True
    log_conversations: bool = True
    anonymize_logs: bool = True
    retention_period: int = 30  # Days to keep logs

@dataclass
class LanguageConfig:
    """Language and localization configuration"""
    # Primary Language
    primary_language: str = "english"
    secondary_language: Optional[str] = None  # English-only system
    code_switching: bool = False  # No code-switching in English-only
    
    # English Settings
    english_dialect: str = "american"  # american, british, australian
    
    # Response Language Rules
    respond_in_user_language: bool = True
    force_english_responses: bool = True  # Ensure English-only responses
    
    # Pronunciation Guide for university-specific terms
    pronunciation_guide: Dict[str, str] = None
    difficult_words_slowdown: bool = False  # English TTS handles well
    spell_out_numbers: bool = False  # English TTS handles numbers well
    
    def __post_init__(self):
        if self.pronunciation_guide is None:
            self.pronunciation_guide = {
                "Leibniz": "LIBE-nits",
                "Institut": "IN-sti-toot",
                "Hannover": "hah-NOH-ver",
                "Universität": "oo-nee-ver-zee-TAYT",
                "Mensa": "MEN-sah",
                "Studierendenwerk": "shtoo-dee-REN-den-verk"
            }

@dataclass
class AudioConfig:
    """Audio and sound configuration"""
    # Background Music
    background_music_enabled: bool = False  # Professional setting, no background music
    background_volume: float = 0.6  # Pre-duck volume (reduced to 50% during speech)
    background_ducking_factor: float = 0.5  # Reduce to 50% during speech
    music_fade_in_time: float = 2.0  # Seconds
    music_fade_out_time: float = 1.0  # Seconds
    
    # Sound Effects
    notification_sounds: bool = False  # Minimalist professional approach
    success_sound: bool = False
    error_sound: bool = False
    
    # Audio Files (if enabled)
    background_music_file: str = ""
    success_sound_file: str = ""
    notification_sound_file: str = ""

class LeibnizConfig:
    """Main Leibniz agent configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize Leibniz agent configuration"""
        # Core configurations
        self.voice = VoiceConfig()
        self.personality = PersonalityConfig()
        self.conversation = ConversationConfig()
        self.emotional = EmotionalConfig()
        self.technical = TechnicalConfig()
        self.language = LanguageConfig()
        self.audio = AudioConfig()
        
        # Store config file path for reload
        self._config_path = config_file
        
        # Load custom config if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update configurations
            for section_name, section_data in config_data.items():
                if hasattr(self, section_name):
                    section = getattr(self, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
            
            print(f"✅ Leibniz configuration loaded from {config_file}")
        except Exception as e:
            print(f"⚠️ Error loading config file: {e}")
    
    def save_to_file(self, config_file: str = None):
        """Save current configuration to JSON file"""
        if config_file is None:
            config_file = "leibniz_config.json"
        
        try:
            config_data = {
                'voice': asdict(self.voice),
                'personality': asdict(self.personality),
                'conversation': asdict(self.conversation),
                'emotional': asdict(self.emotional),
                'technical': asdict(self.technical),
                'language': asdict(self.language),
                'audio': asdict(self.audio)
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Configuration saved to {config_file}")
        except Exception as e:
            print(f"❌ Error saving config file: {e}")
    
    def get_tts_settings(self, emotion: str = 'helpful') -> Dict:
        """Get TTS settings for audio generation"""
        pitch_value = self.voice.emotion_pitch_map.get(emotion, self.voice.pitch)
        
        # Clamp pitch to safe range for most TTS providers
        # Note: Provider-specific handling to be completed in TTS implementation phase
        # - ElevenLabs: typically expects stability/similarity_boost (0.0-1.0)
        # - Google TTS: expects pitch in semitones (SSML <prosody pitch="+2st">)
        # - OpenAI TTS: may not support pitch adjustment directly
        # For now, clamp to a generic [-1.0, 1.0] range for float-based providers
        clamped_pitch = max(-1.0, min(1.0, pitch_value))
        
        return {
            'provider': self.voice.tts_provider,
            'fallback_provider': self.voice.tts_fallback_provider,
            'voice_id': self.voice.voice_id,
            'google_voice': self.voice.google_voice,
            'elevenlabs_model': self.voice.elevenlabs_model,
            'xtts_speaker_sample': self.voice.xtts_speaker_sample,
            'xtts_language': self.voice.xtts_language,
            'xtts_device': self.voice.xtts_device,
            'language': self.voice.language,
            'pitch': clamped_pitch,
            'speed': self.voice.emotion_speed_map.get(emotion, self.voice.speed),
            'volume': self.voice.volume,
            'tone': self.voice.tone,
            'sample_rate': self.technical.tts_sample_rate
        }
    
    def get_response_style(self) -> Dict:
        """Get response style settings"""
        return {
            'friendliness': self.personality.friendliness_level,
            'formality': self.personality.formality_level,
            'empathy': self.personality.empathy_level,
            'humor': self.personality.humor_level,
            'speaking_style': self.personality.speaking_style,
            'enthusiasm': self.personality.enthusiasm_level
        }
    
    def get_conversation_rules(self) -> Dict:
        """Get conversation flow rules"""
        return {
            'max_retries': self.conversation.max_retry_attempts,
            'response_delay': (self.conversation.response_delay_min, self.conversation.response_delay_max),
            'memory_span': self.conversation.memory_span,
            'allow_interruptions': self.conversation.allow_interruptions,
            'context_awareness': self.conversation.context_awareness,
            'silence_timeout': self.conversation.silence_timeout
        }
    
    def should_force_english(self) -> bool:
        """Check if responses should be forced to English"""
        return self.language.force_english_responses
    
    def get_current_energy_level(self) -> str:
        """Get current energy level based on time of day"""
        current_hour = datetime.datetime.now().hour
        
        if 6 <= current_hour < 12:
            return self.emotional.energy_schedule['morning']
        elif 12 <= current_hour < 17:
            return self.emotional.energy_schedule['afternoon']
        elif 17 <= current_hour < 21:
            return self.emotional.energy_schedule['evening']
        else:
            return self.emotional.energy_schedule['night']
    
    def get_personality_prompt(self) -> str:
        """Generate personality prompt for LLMs"""
        energy = self.get_current_energy_level()
        
        return f"""You are {self.personality.name}, a {self.personality.age_persona}. {self.personality.background}

Personality Traits:
- Friendliness level: {self.personality.friendliness_level:.1f}/1.0 (warm and welcoming)
- Formality: {self.personality.formality_level:.1f}/1.0 (casual but respectful)
- Empathy: {self.personality.empathy_level:.1f}/1.0 (understanding and caring)
- Patience: {self.personality.patience_level:.1f}/1.0 (very patient)
- Enthusiasm: {self.personality.enthusiasm_level:.1f}/1.0 (positive energy)
- Current energy level: {energy}

Communication Style:
- Speak in a {self.personality.speaking_style} manner
- Show emotional intelligence: {self.personality.emotional_intelligence}
- Provide encouragement: {self.personality.encouragement_frequency} frequency
- Use natural expressions like: {', '.join(self.personality.expressions[:5])}

Avoid phrases like: {', '.join(self.personality.avoid_phrases)}

Always remember you're a helpful friend, not a formal chatbot. Keep responses concise, friendly, and focused on solving the user's university-related questions."""
    
    def apply_pronunciation_guide(self, text: str) -> str:
        """Apply pronunciation guide to university-specific terms"""
        result = text
        for term, pronunciation in self.language.pronunciation_guide.items():
            # For TTS, we might want to keep original or use SSML tags
            # This is a placeholder for future TTS-specific formatting
            pass
        return result
    
    def get_emotion_expression(self, emotion: str) -> str:
        """Get a random expression for an emotion"""
        import random
        
        expressions_map = {
            'excitement': self.emotional.excitement_expressions,
            'sympathy': self.emotional.sympathy_expressions,
            'encouragement': self.emotional.encouragement_expressions,
            'celebration': self.emotional.celebration_expressions
        }
        
        expressions = expressions_map.get(emotion, ["Okay"])
        return random.choice(expressions)

# Global configuration instance and path tracking
_leibniz_config = None
_leibniz_config_path = None

def get_leibniz_config() -> LeibnizConfig:
    """Get global Leibniz configuration instance"""
    global _leibniz_config
    if _leibniz_config is None:
        _leibniz_config = LeibnizConfig()
    return _leibniz_config

def initialize_leibniz_config(config_file: Optional[str] = None):
    """Initialize Leibniz configuration with optional config file"""
    global _leibniz_config, _leibniz_config_path
    _leibniz_config = LeibnizConfig(config_file)
    _leibniz_config_path = config_file
    return _leibniz_config

def reload_leibniz_config():
    """Reload global Leibniz configuration"""
    global _leibniz_config, _leibniz_config_path
    
    # Resolve config path: instance path > module-level path > env var > default
    config_path = None
    if _leibniz_config and hasattr(_leibniz_config, '_config_path') and _leibniz_config._config_path:
        config_path = _leibniz_config._config_path
    elif _leibniz_config_path:
        config_path = _leibniz_config_path
    else:
        config_path = os.getenv('LEIBNIZ_CONFIG_FILE', 'leibniz_config.json')
    
    if _leibniz_config and config_path and os.path.exists(config_path):
        _leibniz_config.load_from_file(config_path)
    else:
        _leibniz_config = LeibnizConfig()
    print("🔄 Global Leibniz configuration reloaded")

# Convenience functions for common settings
def get_voice_settings(emotion: str = 'helpful'):
    """Quick access to voice settings"""
    return get_leibniz_config().get_tts_settings(emotion)

def get_personality_settings():
    """Quick access to personality settings"""
    return get_leibniz_config().get_response_style()

def get_english_mode():
    """Check if English mode is enforced"""
    return get_leibniz_config().should_force_english()

def resolve_language_mode() -> Dict[str, Union[str, bool]]:
    """
    Resolve language mode based on config and environment variables.
    
    Returns a dictionary with resolved language settings:
    - language: The active language code (from config or LEIBNIZ_LANGUAGE env var)
    - force_english: Whether English-only responses are enforced
    - dialect: The English dialect to use
    
    This function reads environment variables without side effects and returns
    a resolved language mode dict for consumers to use.
    """
    config = get_leibniz_config()
    
    # Read from environment variable if set, otherwise use config
    env_language = os.getenv('LEIBNIZ_LANGUAGE', config.language.primary_language)
    
    # Normalize language code
    is_english = 'en' in env_language.lower() or env_language.lower() == 'english'
    
    return {
        'language': env_language if is_english else 'en-US',
        'force_english': config.language.force_english_responses,
        'dialect': config.language.english_dialect,
        'primary_language': config.language.primary_language
    }

def get_max_retries():
    """Get maximum retry attempts"""
    return get_leibniz_config().conversation.max_retry_attempts

def get_personality_prompt():
    """Get personality prompt for LLMs"""
    return get_leibniz_config().get_personality_prompt()

def get_agent_name():
    """Get agent's name"""
    return get_leibniz_config().personality.name

def get_emotion_expression(emotion: str):
    """Get emotional expression"""
    return get_leibniz_config().get_emotion_expression(emotion)

if __name__ == "__main__":
    # Test the configuration
    print("🎓 Leibniz University Agent Configuration Test")
    print("=" * 60)
    
    # Initialize config
    config = LeibnizConfig()
    
    # Test voice settings
    print("🎤 Voice Settings:")
    voice_settings = config.get_tts_settings('helpful')
    for key, value in voice_settings.items():
        print(f"   {key}: {value}")
    
    # Test personality settings
    print("\n😊 Personality Settings:")
    personality = config.get_response_style()
    for key, value in personality.items():
        print(f"   {key}: {value}")
    
    # Test conversation rules
    print("\n💬 Conversation Rules:")
    rules = config.get_conversation_rules()
    for key, value in rules.items():
        print(f"   {key}: {value}")
    
    # Test current energy
    print(f"\n⚡ Current Energy Level: {config.get_current_energy_level()}")
    
    # Test emotions
    print(f"\n😊 Excitement Expression: {config.get_emotion_expression('excitement')}")
    print(f"🤝 Sympathy Expression: {config.get_emotion_expression('sympathy')}")
    
    # Test personality prompt
    print("\n🎯 Personality Prompt:")
    print(config.get_personality_prompt()[:300] + "...")
    
    # Save example config
    config.save_to_file("leibniz_config_example.json")
    print(f"\n✅ Example configuration saved!")
