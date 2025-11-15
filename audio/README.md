# Leibniz Agent Audio Files

This directory stores optional audio files for the Leibniz University customer service agent.

## Files

### intro.wav
- **Purpose**: Professional intro greeting audio
- **Format**: WAV, mono, 16kHz or higher
- **Duration**: 3-5 seconds recommended
- **Content**: "Hi there! I'm Lexi, your friendly assistant for Leibniz University."
- **Status**: Optional (if missing, TTS will speak greeting instead)

### background.mp3
- **Purpose**: Ambient background audio for natural feel
- **Format**: MP3, stereo or mono
- **Duration**: Loopable (3-5 minutes recommended)
- **Content**: Soft instrumental music or ambient sounds
- **Status**: Optional (disabled by default)

## Dialogue Caching

The `leibniz_agent/voices/` directory stores frequently used dialogue audio with content verification:

**Cache Structure**:
- `intro_greeting.wav` + `intro_greeting.txt` - Welcome message
- `outro_farewell.wav` + `outro_farewell.txt` - Goodbye message
- `help_prompt.wav` + `help_prompt.txt` - "How can I help you?"
- `error_message.wav` + `error_message.txt` - Error response
- `timeout_message.wav` + `timeout_message.txt` - Timeout handling

**Benefits**:
- ⚡ Sub-100ms response time for cached dialogues (vs 1-3s for synthesis)
- 💰 Reduced API token usage (ElevenLabs/Google Cloud TTS)
- 🔊 Consistent audio quality for repeated phrases
- 📝 Content verification via `.txt` files prevents stale cache

**Usage in Code**:
```python
from leibniz_agent.leibniz_pro import speak_friendly, DIALOGUE_CACHE_NAMES

# Automatic caching using predefined names
await speak_friendly("Hello! How can I help you?", cache_name=DIALOGUE_CACHE_NAMES['greeting'])

# Manual cache name
await speak_friendly("Custom message", cache_name="custom_phrase")
```

## Dialogue Archiving

The `leibniz_agent/audio/dialogues/` directory stores conversation audio for replay/debugging:

**Archive Structure**:
```
dialogues/
  session_20240315_143022/
    turn_001.wav    # First agent response
    turn_001.txt    # Metadata (text, timestamp)
    turn_002.wav    # Second agent response
    turn_002.txt
    ...
```

**Usage**:
```python
from leibniz_agent.leibniz_pro import archive_dialogue_audio

# Archive after TTS synthesis
session_id = "20240315_143022"
turn = 1
archived_path = await archive_dialogue_audio(
    audio_file="temp_audio.wav",
    session_id=session_id,
    turn_number=turn,
    text="Hello! How can I help you?"
)
```

**Benefits**:
- 🎧 Replay entire conversations for quality assurance
- 🐛 Debug TTS issues by inspecting actual audio output
- 📊 Analyze dialogue pacing and timing
- 🗂️ Session-based organization for easy retrieval

## Configuration

Configure audio files in `.env.leibniz`:

```bash
# Enable intro audio
LEIBNIZ_ENABLE_INTRO_AUDIO=true
LEIBNIZ_INTRO_AUDIO_PATH=./leibniz_agent/audio/intro.wav

# Enable background audio (optional)
LEIBNIZ_ENABLE_BACKGROUND_AUDIO=false
LEIBNIZ_BACKGROUND_AUDIO_PATH=./leibniz_agent/audio/background.mp3
LEIBNIZ_BACKGROUND_AUDIO_VOLUME=0.3

# Dialogue caching (enabled by default)
LEIBNIZ_DIALOGUE_CACHE_DIR=./leibniz_agent/voices
LEIBNIZ_ENABLE_DIALOGUE_CACHE=true

# Dialogue archiving (optional)
LEIBNIZ_DIALOGUE_ARCHIVE_DIR=./leibniz_agent/audio/dialogues
LEIBNIZ_ENABLE_DIALOGUE_ARCHIVE=false
```

## Creating Audio Files

### Intro Audio (intro.wav)

**Option 1: Use Leibniz TTS** (recommended)
```python
from leibniz_agent import get_leibniz_tts

tts = get_leibniz_tts()
result = await tts.synthesize_to_file(
    text="Hi there! I'm Lexi, your friendly assistant for Leibniz University. How can I help you today?",
    emotion="excited",
    output_path="leibniz_agent/audio/intro.wav"
)
```

**Option 2: Record manually**
1. Record greeting in professional studio quality
2. Convert to WAV format: `ffmpeg -i input.mp3 -ar 16000 -ac 1 intro.wav`
3. Normalize audio: `ffmpeg -i intro.wav -filter:a loudnorm intro_normalized.wav`

### Background Audio (background.mp3)

**Option 1: Use free music libraries**
- [YouTube Audio Library](https://www.youtube.com/audiolibrary) (royalty-free)
- [Free Music Archive](https://freemusicarchive.org/)
- [Incompetech](https://incompetech.com/music/) (Creative Commons)

**Option 2: Create ambient sounds**
- Use tools like Audacity to create loopable ambient sounds
- Mix soft instrumental music with subtle nature sounds
- Keep volume low (background only)

**Conversion to MP3**:
```bash
ffmpeg -i input.wav -b:a 128k -ac 2 background.mp3
```

## Usage

The main orchestration module (`leibniz_pro.py`) automatically plays these files if enabled:

```python
# Intro audio plays at session start
await play_natural_intro()

# Background audio plays throughout session (optional)
start_background_audio()
```

## Notes

- Audio files are **optional** - system works without them
- If intro audio is missing, TTS will speak greeting instead
- Background audio is **disabled by default** (set `LEIBNIZ_ENABLE_BACKGROUND_AUDIO=true` to enable)
- All audio playback requires `pygame` library
- System gracefully degrades if `pygame` is not installed

## Troubleshooting

**Issue**: Intro audio not playing
- **Solution**: Check file exists at `LEIBNIZ_INTRO_AUDIO_PATH`
- **Solution**: Verify `LEIBNIZ_ENABLE_INTRO_AUDIO=true` in `.env.leibniz`
- **Solution**: Ensure `pygame` is installed: `pip install pygame`
- **Solution**: Check audio format (WAV, 16kHz, mono recommended)

**Issue**: Background audio too loud
- **Solution**: Adjust `LEIBNIZ_BACKGROUND_AUDIO_VOLUME` (0.0-1.0)
- **Solution**: Re-encode audio with lower volume: `ffmpeg -i input.mp3 -filter:a "volume=0.3" output.mp3`

**Issue**: Audio playback errors
- **Solution**: Check pygame initialization: `pygame.mixer.init()`
- **Solution**: Verify audio file format is supported (WAV, MP3)
- **Solution**: Check file permissions (read access required)

**Issue**: "[WinError 5] Access is denied" when moving temp files (Windows)
- **Root Cause**: File locking by audio playback (pygame.mixer)
- **Solution**: Unique temp files now generated automatically (no more fixed "out.wav")
- **Solution**: Retry logic with fallback to copy (implemented in `_convert_to_wav()`)
- **Solution**: Ensure no other processes are accessing TTS output files
- **Technical Details**: System now uses `tempfile.NamedTemporaryFile` with timestamps to avoid conflicts

**Issue**: Dialogue cache not hitting
- **Solution**: Check cache name matches `DIALOGUE_CACHE_NAMES` mapping
- **Solution**: Verify `.txt` file content matches synthesized text
- **Solution**: Clear cache if stale: delete `.wav` + `.txt` pairs in `voices/` directory
- **Solution**: Enable cache debugging: check logs for "💾 Dialogue cache hit" messages

**Issue**: Dialogue archiving fails
- **Solution**: Verify `DIALOGUE_ARCHIVE_DIR` exists and is writable
- **Solution**: Check disk space (archives can accumulate over time)
- **Solution**: Clean old sessions: `rm -rf leibniz_agent/audio/dialogues/session_*`

## License

Audio files should be licensed for commercial use or created in-house. Always verify licensing before using third-party audio in production.
