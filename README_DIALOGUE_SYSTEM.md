# Leibniz Agent Dialogue System

## Overview

The Leibniz Agent implements a comprehensive category-based dialogue system that loads dialogues from JSON databases, organizes audio files by category, and enables audio reuse to improve performance and consistency.

## Architecture

### Category-Based Organization

Dialogues are organized into five categories:

- **intro**: Introduction and greeting dialogues
- **fsm**: Appointment booking FSM dialogues
- **rag**: RAG query response dialogues
- **errors**: Error handling and apology dialogues
- **prompts**: User prompts and continuation dialogues

### File Structure

```
leibniz_agent/
├── dialogue_database/          # JSON dialogue files
│   ├── greetings.json
│   ├── appointments.json
│   ├── responses.json
│   └── errors.json
├── audio/
│   └── dialogues/              # Pregenerated audio files
│       ├── intro/
│       ├── fsm/
│       ├── rag/
│       ├── errors/
│       └── prompts/
└── leibniz_dialogue_manager.py # Dialogue management system
```

## Configuration

### Environment Variables (.env.leibniz)

```bash
# Dialogue database configuration
LEIBNIZ_DIALOGUE_DATABASE_PATH=dialogue_database
LEIBNIZ_DIALOGUE_ENABLE_JSON_LOADING=true

# Category-specific environment file references
LEIBNIZ_DIALOGUE_INTRO_FILE=.env.intro
LEIBNIZ_DIALOGUE_FSM_FILE=.env.fsm
LEIBNIZ_DIALOGUE_RAG_FILE=.env.rag
LEIBNIZ_DIALOGUE_ERRORS_FILE=.env.errors
LEIBNIZ_DIALOGUE_PROMPTS_FILE=.env.prompts

# Audio reuse configuration
LEIBNIZ_DIALOGUE_ENABLE_AUDIO_REUSE=true
LEIBNIZ_DIALOGUE_AUDIO_BASE_PATH=leibniz_agent/audio/dialogues

# Dialogue archiving
LEIBNIZ_ENABLE_DIALOGUE_ARCHIVE=true
LEIBNIZ_DIALOGUE_ARCHIVE_PATH=leibniz_agent/audio/dialogues/archive
```

### Category-Specific .env Files

Each category has its own .env file (e.g., `.env.intro`, `.env.fsm`) containing dialogue key-value pairs:

```bash
# .env.intro
LEIBNIZ_DIALOGUE_GREETING="Hello! I'm the university receptionist. How may I assist you today?"
LEIBNIZ_DIALOGUE_WELCOME="Welcome to Leibniz University. I'm here to help with admissions, appointments, and information."
```

## Usage

### Basic Dialogue Access

```python
from leibniz_agent.leibniz_dialogue_manager import get_leibniz_dialogue_manager

# Get dialogue manager instance
dialogue_mgr = get_leibniz_dialogue_manager()

# Get dialogue text
greeting = dialogue_mgr.get_dialogue('greeting')
error_msg = dialogue_mgr.get_dialogue('error')

# Get dialogue category
category = dialogue_mgr.get_dialogue_category('greeting')  # Returns 'intro'

# Check if audio exists
audio_exists = dialogue_mgr.check_audio_exists('intro', 'greeting')

# Get audio file path
audio_path = dialogue_mgr.get_audio_path('intro', 'greeting')
```

### Audio Reuse in speak_friendly()

The `speak_friendly()` function automatically checks for existing audio files:

```python
# This will use existing audio if available, otherwise synthesize new
await speak_friendly(dialogue_key='greeting', emotion='helpful')
```

### TTS Integration

Dialogues support all TTS providers (ElevenLabs, Google, Gemini) with automatic audio reuse:

```python
# Configure TTS provider
LEIBNIZ_TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your_key_here
ELEVENLABS_VOICE_ID=voice_id_here
```

## Setup and Maintenance

### 1. Generate Category .env Files

Run the generation script to create category-specific .env files from JSON databases:

```bash
python scripts/generate_dialogue_env_files.py
```

This creates:
- `.env.intro` - Introduction dialogues
- `.env.fsm` - Appointment booking dialogues
- `.env.rag` - RAG response dialogues
- `.env.errors` - Error handling dialogues
- `.env.prompts` - User prompt dialogues

### 2. Pregenerate Audio Files

Generate audio files for all dialogues to enable reuse:

```bash
python scripts/pregenerate_dialogue_audio.py
```

This creates audio files in category subdirectories with MD5-based filenames for uniqueness.

### 3. Update JSON Databases

Add new dialogues to the appropriate JSON file in `dialogue_database/`:

```json
{
  "new_greeting": "Hello! Welcome to our university information system.",
  "appointment_success": "Your appointment has been successfully booked!"
}
```

Then regenerate the .env files and audio files.

## JSON Database Format

Dialogue databases are JSON files with simple key-value pairs:

```json
{
  "greeting": "Hello! How may I assist you today?",
  "error": "I'm sorry, I encountered an error. Please try again.",
  "appointment_confirm": "Your appointment is confirmed for [date] at [time].",
  "help_offer": "How can I help you today?"
}
```

## Audio File Organization

Audio files are organized by category with MD5-based unique names:

```
leibniz_agent/audio/dialogues/
├── intro/
│   ├── greeting_a1b2c3d4.wav
│   └── welcome_e5f6g7h8.wav
├── fsm/
│   ├── appointment_confirm_i9j0k1l2.wav
│   └── booking_success_m3n4o5p6.wav
├── errors/
│   └── error_q7r8s9t0.wav
└── prompts/
    └── help_offer_u1v2w3x4.wav
```

## Performance Benefits

### Audio Reuse
- **Faster Response Times**: Skip TTS synthesis for frequently used dialogues
- **Consistent Audio Quality**: Same audio file used for identical dialogues
- **Reduced API Costs**: Fewer TTS API calls for common phrases
- **Lower Latency**: Instant playback of pregenerated audio

### Category-Based Organization
- **Easier Maintenance**: Dialogues grouped by function
- **Selective Updates**: Update only relevant categories
- **Clear Separation**: Intro, FSM, RAG, errors, and prompts separated

### JSON Database
- **Version Control Friendly**: Text-based dialogue storage
- **Easy Editing**: JSON format supports comments and structure
- **Multi-language Support**: Easy to add language variants
- **Backup/Restore**: Simple file-based persistence

## Troubleshooting

### Audio Files Not Found
1. Check if `scripts/pregenerate_dialogue_audio.py` has been run
2. Verify TTS provider configuration
3. Check file permissions on audio directories

### Dialogue Keys Not Found
1. Run `scripts/generate_dialogue_env_files.py` after adding new dialogues
2. Check JSON syntax in dialogue_database files
3. Verify dialogue keys match between JSON and code

### TTS Provider Issues
1. Ensure API keys are configured in `.env.leibniz`
2. Check TTS provider status and quotas
3. Verify voice IDs are valid for the selected provider

## Migration from Legacy System

The new system maintains backward compatibility:

- **Legacy Environment Variables**: Still supported via fallback
- **Old Dialogue Keys**: Continue to work through category mapping
- **Existing Audio Files**: Can be migrated to category structure

To migrate:
1. Export existing dialogues to JSON format
2. Run generation and pregeneration scripts
3. Update code to use new dialogue keys where possible
4. Gradually phase out legacy environment variables

## Best Practices

### Dialogue Key Naming
- Use descriptive, lowercase names with underscores
- Include category hints: `greeting_main`, `error_timeout`, `prompt_continue`
- Keep keys consistent across similar dialogues

### Audio Quality
- Use consistent voice settings across categories
- Test audio files for clarity and naturalness
- Regenerate audio when voice settings change

### Category Assignment
- **intro**: First-time user interactions
- **fsm**: Multi-step processes (appointments, forms)
- **rag**: Information responses and answers
- **errors**: System errors, timeouts, failures
- **prompts**: User guidance, questions, continuations

### Maintenance
- Run pregeneration script after dialogue updates
- Archive old audio files before major voice changes
- Test all dialogue categories after updates
- Monitor audio file disk usage

## API Reference

### DialogueManager Class

#### Methods
- `get_dialogue(key: str) -> str`: Get dialogue text by key
- `get_dialogue_category(key: str) -> str`: Get category for dialogue key
- `check_audio_exists(category: str, key: str) -> bool`: Check if audio file exists
- `get_audio_path(category: str, key: str) -> str`: Get full audio file path
- `load_from_json_database() -> None`: Load dialogues from JSON files
- `archive_audio(audio_file: str, session_id: str, turn_number: int, text: str) -> None`: Archive dialogue audio

#### Properties
- `dialogues`: Dictionary of all loaded dialogues
- `categories`: Dictionary of dialogues by category

## Future Enhancements

- **Multi-language Support**: Language-specific JSON databases
- **Dynamic Audio Generation**: On-demand audio generation with caching
- **Audio Compression**: Optimize file sizes for storage
- **Voice Variants**: Different voices for different dialogue types
- **Performance Monitoring**: Track audio reuse statistics</content>
<parameter name="filePath">c:\Users\AMAR\SINDHv2\SINDH-Orchestra-Complete\leibniz_agent\README_DIALOGUE_SYSTEM.md