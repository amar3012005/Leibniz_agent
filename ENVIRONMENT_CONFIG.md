# Leibniz University Agent Environment Configuration

## Required Configuration

Create a `.env` file in the project root with the following configuration:

```bash
# Required API Keys
# -----------------

# Gemini API Key (REQUIRED)
# Used for: STT (Gemini Live), Intent Classification, RAG Response Generation
# Get your key at: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Knowledge Base Configuration (REQUIRED)
# ----------------------------------------

# Knowledge Base Path - The knowledge base lives at the repository root
# Use absolute paths for reliability as relative paths are resolved from working directory
LEIBNIZ_RAG_KNOWLEDGE_BASE_PATH=C:/Users/AMAR/SINDHv2/SINDH-Orchestra-Complete/leibniz_knowledge_base

# Optional TTS Provider API Keys
# ------------------------------

# Google Cloud Text-to-Speech (OPTIONAL)
# Used for: High-quality TTS synthesis
# Setup: https://cloud.google.com/text-to-speech/docs/quickstart
# GOOGLE_APPLICATION_CREDENTIALS=path/to/google/credentials.json

# ElevenLabs API Key (OPTIONAL)
# Used for: Premium voice synthesis with emotion support
# Get your key at: https://elevenlabs.io/
# ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

## Model Configuration

```bash
# Gemini Models
GEMINI_MODEL=gemini-2.0-flash-lite
LEIBNIZ_TTS_GEMINI_MODEL=gemini-2.5-flash-preview-tts
LEIBNIZ_STT_MODEL=gemini-2.5-flash-preview-tts
```

## Performance Tuning

```bash
# Intent Parser Settings
LEIBNIZ_INTENT_PARSER_CONFIDENCE_THRESHOLD=0.8
LEIBNIZ_INTENT_PARSER_GEMINI_TIMEOUT=5.0

# RAG System Settings
LEIBNIZ_RAG_TOP_K=8
LEIBNIZ_RAG_TOP_N=5
LEIBNIZ_RAG_SIMILARITY_THRESHOLD=0.3
LEIBNIZ_RAG_MAX_RESPONSE_LENGTH=500

# Appointment FSM Settings
LEIBNIZ_APPOINTMENT_MAX_RETRIES=3
LEIBNIZ_APPOINTMENT_MAX_BOOKING_MONTHS=3
LEIBNIZ_APPOINTMENT_BUSINESS_HOURS_START=08:00
LEIBNIZ_APPOINTMENT_BUSINESS_HOURS_END=18:00
```

## Important Notes

1. **Knowledge Base Path**: The knowledge base directory must contain 12 category subdirectories with markdown files
2. **Absolute Paths**: Recommended for reliability, especially for the knowledge base path
3. **TTS Providers**: Configure at least one TTS provider (Gemini, Google Cloud, or ElevenLabs) for audio output
4. **API Quotas**: Use personal API keys to avoid quota limits during testing and production use

## Provider Requirements

| Provider | Required For | API Key Variable | Setup Instructions |
|----------|--------------|------------------|-------------------|
| **Gemini** | STT, Intent, RAG | `GEMINI_API_KEY` | https://makersuite.google.com/app/apikey |
| **Google Cloud** | TTS (optional) | `GOOGLE_APPLICATION_CREDENTIALS` | https://cloud.google.com/text-to-speech/docs/quickstart |
| **ElevenLabs** | TTS (optional) | `ELEVENLABS_API_KEY` | https://elevenlabs.io/ |

## Testing Configuration

For testing purposes, you can enable mock modes:

```bash
# Mock modes for testing without hardware
MOCK_STT=true
MOCK_TTS=true
ALLOW_NO_TTS=true

# Debug logging
LOG_LEVEL=DEBUG
LEIBNIZ_INTENT_PARSER_LOG_CLASSIFICATIONS=true
```
