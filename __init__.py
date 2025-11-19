"""
Leibniz University Agent Package
=================================

A friendly, casual English-only customer service agent for Leibniz University Institute.
Provides voice-first conversational support for university-related inquiries including:
- Admissions and enrollment
- Course information
- Faculty and departments
- Student services
- Appointment scheduling
- Campus facilities and resources

Architecture:
    - STT: Gemini Live API for English speech recognition
    - Intent Parser: Two-tier classification (fast patterns + Gemini fallback) with context extraction
      5 intents: APPOINTMENT_SCHEDULING, RAG_QUERY, GREETING, EXIT, UNCLEAR
      Extracts: user_goal, key_entities, extracted_meaning for improved RAG retrieval
    - RAG: Context-aware retrieval with FAISS vector store over 63 university knowledge base documents
      Accepts structured context (user_goal, key_entities, extracted_meaning) for enhanced semantic matching
      Uses intelligent chunking (FAQ Q&A, section headers, semantic paragraphs) with entity-based filtering
      Generates friendly casual English responses via Gemini 2.0 Flash
    - TTS: Google Cloud TTS (primary) or ElevenLabs (fallback) for natural English speech synthesis with dual-provider architecture
    - Memory: MongoDB-backed conversation history and user context
    - VAD: Bidirectional voice activity detection with barge-in support and persistent session optimization
    - Main Orchestration: Conversation loop with greeting → intent classification → RAG/appointment FSM → response

Usage Example:
    ```python
    from leibniz_agent import LeibnizConfig, get_leibniz_config
    
    # Initialize with default configuration
    config = get_leibniz_config()
    
    # Or load custom configuration
    from leibniz_agent import initialize_leibniz_config
    config = initialize_leibniz_config("custom_config.json")
    
    # Get voice settings
    voice_settings = config.get_tts_settings(emotion='helpful')
    
    # Get personality prompt for LLM
    personality = config.get_personality_prompt()
    
    # Intent Parser - classify with context extraction
    from leibniz_agent import classify_leibniz_intent
    result = await classify_leibniz_intent("What are CS program requirements?")
    # Returns: intent, confidence, context (user_goal, key_entities, extracted_meaning)
    
    # RAG - context-aware knowledge retrieval
    from leibniz_agent import process_leibniz_query
    
    # Context-aware query (recommended - uses intent parser output)
    context = {
        'user_goal': 'asking about CS program requirements',
        'key_entities': {'program': 'computer science', 'topic': 'requirements'},
        'extracted_meaning': 'computer science program admission requirements'
    }
    response = process_leibniz_query(context=context)
    
    # Raw query (fallback mode)
    response = process_leibniz_query(query="What are the CS program requirements?")
    
    # TTS - synthesize and speak
    from leibniz_agent import leibniz_speak, leibniz_stream_speak
    await leibniz_speak("Hello! How can I help you?", emotion="helpful")
    
    # Appointment FSM - book appointments
    from leibniz_agent import create_appointment_fsm
    
    # Create appointment booking FSM
    fsm = create_appointment_fsm()
    
    # Process user input
    result = await fsm.process_input("John Smith")
    print(result['response'])  # System response
    print(result['state'])     # Current FSM state
    print(result['complete'])  # Whether booking is complete
    
    # When complete, get booking data
    if result['complete']:
        booking_data = format_appointment_for_submission(fsm.data)
        # Submit to university booking system
    
    # Main Orchestration - run full conversation session
    from leibniz_agent import initialize_leibniz_services, run_conversation_session
    
    # Initialize all services
    await initialize_leibniz_services()
    
    # Run conversation session
    await run_conversation_session()
    
    # Or run the full multi-session orchestrator
    from leibniz_agent.leibniz_pro import main
    asyncio.run(main())
```Message Flow:
    Audio Input → STT (TranscriptMessage) → Intent Classification (IntentMessage with context) 
    → RAG/FSM (RAGMessage with enriched context) → Response Generation → TTS (TTSMessage) → Audio Output

For detailed documentation, see leibniz_agent/README.md
"""

from leibniz_agent.leibniz_config import (
    LeibnizConfig,
    VoiceConfig,
    PersonalityConfig,
    ConversationConfig,
    EmotionalConfig,
    TechnicalConfig,
    LanguageConfig,
    AudioConfig,
    get_leibniz_config,
    initialize_leibniz_config,
    reload_leibniz_config,
    get_voice_settings,
    get_personality_settings,
    get_english_mode,
    resolve_language_mode,
    get_max_retries,
    get_personality_prompt,
    get_agent_name,
    get_emotion_expression,
)

from leibniz_agent.leibniz_messages import (
    TranscriptMessage,
    IntentMessage,
    RAGMessage,
    TTSMessage,
    transcript_to_dict,
    transcript_from_dict,
    intent_to_dict,
    intent_from_dict,
    rag_to_dict,
    rag_from_dict,
    tts_to_dict,
    tts_from_dict,
)

from leibniz_agent.leibniz_stt import (
    LeibnizSTT,
    get_leibniz_stt,
    leibniz_transcribe_file,
    leibniz_capture_audio,
    warmup_leibniz_stt,
    cleanup_leibniz_stt,
)

from leibniz_agent.leibniz_tts import (
    LeibnizTTS,
    get_leibniz_tts,
    leibniz_synthesize,
    leibniz_speak,
    leibniz_stream_speak,
    warmup_leibniz_tts,
    cleanup_leibniz_tts,
    get_available_voices,
)

from leibniz_agent.leibniz_intent_parser import (
    LeibnizIntentParser,
    get_leibniz_parser,
    classify_leibniz_intent,
)

from leibniz_agent.leibniz_rag import (
    LeibnizRAG,
    get_leibniz_rag,
    process_leibniz_query,
)

from leibniz_agent.leibniz_persistent_services import (
    PersistentServicesManager,
    get_leibniz_services_manager,
    fast_classify_leibniz_intent,
    process_leibniz_rag_query,
    get_leibniz_service_status,
    prewarm_leibniz_during_tts,
)

from leibniz_agent.leibniz_appointment_fsm import (
    LeibnizAppointmentFSM,
    create_appointment_fsm,
    format_appointment_for_submission,
    AppointmentData,
    AppointmentState,
)

from leibniz_agent.leibniz_vad import (
    get_leibniz_vad,
    capture_leibniz_speech,
    set_leibniz_agent_speaking,
    reset_leibniz_conversation,
    cleanup_leibniz_vad,
)

from leibniz_agent.leibniz_pro import (
    run_conversation_session,
    initialize_leibniz_services,
    speak_friendly,
    handle_rag_query,
    handle_appointment_booking,
    capture_and_transcribe,
    transcribe_and_classify,
)

__version__ = "1.0.0"
__author__ = "SINDH Development Team"
__description__ = "Leibniz University Institute Customer Service Agent"

__all__ = [
    # Configuration classes
    "LeibnizConfig",
    "VoiceConfig",
    "PersonalityConfig",
    "ConversationConfig",
    "EmotionalConfig",
    "TechnicalConfig",
    "LanguageConfig",
    "AudioConfig",
    
    # Configuration functions
    "get_leibniz_config",
    "initialize_leibniz_config",
    "reload_leibniz_config",
    "get_voice_settings",
    "get_personality_settings",
    "get_english_mode",
    "resolve_language_mode",
    "get_max_retries",
    "get_personality_prompt",
    "get_agent_name",
    "get_emotion_expression",
    
    # Message dataclasses
    "TranscriptMessage",
    "IntentMessage",
    "RAGMessage",
    "TTSMessage",
    
    # Message conversion helpers
    "transcript_to_dict",
    "transcript_from_dict",
    "intent_to_dict",
    "intent_from_dict",
    "rag_to_dict",
    "rag_from_dict",
    "tts_to_dict",
    "tts_from_dict",
    
    # STT Module
    "LeibnizSTT",
    "get_leibniz_stt",
    "leibniz_transcribe_file",
    "leibniz_capture_audio",
    "warmup_leibniz_stt",
    "cleanup_leibniz_stt",
    
    # TTS Module
    "LeibnizTTS",
    "get_leibniz_tts",
    "leibniz_synthesize",
    "leibniz_speak",
    "leibniz_stream_speak",
    "warmup_leibniz_tts",
    "cleanup_leibniz_tts",
    "get_available_voices",
    
    # Intent Parser Module
    "LeibnizIntentParser",
    "get_leibniz_parser",
    "classify_leibniz_intent",
    
    # RAG Module
    "LeibnizRAG",
    "get_leibniz_rag",
    "process_leibniz_query",
    
    # Persistent Services (Async Queue-Based)
    "PersistentServicesManager",
    "get_leibniz_services_manager",
    "fast_classify_leibniz_intent",
    "process_leibniz_rag_query",
    "get_leibniz_service_status",
    "prewarm_leibniz_during_tts",
    
    # Appointment Booking FSM
    "LeibnizAppointmentFSM",
    "create_appointment_fsm",
    "format_appointment_for_submission",
    "AppointmentData",
    "AppointmentState",
    
    # VAD Module (Bidirectional Voice Activity Detection)
    "get_leibniz_vad",
    "capture_leibniz_speech",
    "set_leibniz_agent_speaking",
    "reset_leibniz_conversation",
    "cleanup_leibniz_vad",
    
    # Main Orchestration Module
    "run_conversation_session",
    "initialize_leibniz_services",
    "speak_friendly",
    "handle_rag_query",
    "handle_appointment_booking",
    "capture_and_transcribe",
    "transcribe_and_classify",
    
    # Package metadata
    "__version__",
    "__author__",
    "__description__",
]
