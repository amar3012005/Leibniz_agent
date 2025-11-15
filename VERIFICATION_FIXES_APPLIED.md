# Verification Fixes Applied - Leibniz Agent

## Summary

All 5 verification comments have been successfully implemented with precise adherence to the instructions.

---

## Comment 1: Fixed inline comments in .env values ✅

**File**: `leibniz_agent/.env.leibniz`

**Problem**: Inline comments after variable assignments (using `#`) can be parsed as part of the value by many dotenv loaders.

**Solution**: Moved all trailing inline comments to separate preceding comment lines.

**Changes Made**:

1. `OPENAI_ORG_ID=your_org_id_here  # Optional`
   - Changed to: `# Optional` on preceding line + `OPENAI_ORG_ID=your_org_id_here`

2. `WHISPER_MODEL=whisper-1  # or specify local model path...`
   - Changed to: `# or specify local model path...` on preceding line + `WHISPER_MODEL=whisper-1`

3. `ELEVENLABS_VOICE_ID=Rachel  # or other voice ID...`
   - Changed to: `# or other voice ID...` on preceding line + `ELEVENLABS_VOICE_ID=Rachel`

4. `GEMINI_MODEL=gemini-2.0-flash-exp  # or latest model available`
   - Changed to: `# or latest model available` on preceding line + `GEMINI_MODEL=gemini-2.0-flash-exp`

5. `RAG_TOP_K=8  # Number of documents to retrieve`
   - Changed to: `# Number of documents to retrieve` on preceding line + `RAG_TOP_K=8`

6. `LEIBNIZ_CONFIG_FILE=./leibniz_agent/leibniz_config.json  # Optional custom config`
   - Changed to: `# Optional custom config` on preceding line + `LEIBNIZ_CONFIG_FILE=...`

7. `LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR`
   - Changed to: `# DEBUG, INFO, WARNING, ERROR` on preceding line + `LOG_LEVEL=INFO`

8. `ENVIRONMENT=development  # development, staging, production`
   - Changed to: `# development, staging, production` on preceding line + `ENVIRONMENT=development`

9. `MOCK_STT=false  # For testing without microphone`
   - Changed to: `# For testing without microphone` on preceding line + `MOCK_STT=false`

10. `MOCK_TTS=false  # For testing without audio output`
    - Changed to: `# For testing without audio output` on preceding line + `MOCK_TTS=false`

**Impact**: Environment variable parsing will now work correctly across all dotenv loaders (python-dotenv, docker-compose, etc.).

---

## Comment 2: Fixed reload to use stored config path ✅

**File**: `leibniz_agent/leibniz_config.py`

**Problem**: `reload_leibniz_config()` hard-coded `"leibniz_config.json"` and ignored custom file paths or environment-configured paths.

**Solution**: 
1. Added `_config_path` attribute to `LeibnizConfig` class to persist the initialization path
2. Added module-level `_leibniz_config_path` variable to track path at global level
3. Updated `reload_leibniz_config()` to resolve path using priority order:
   - Instance `_config_path` attribute (highest priority)
   - Module-level `_leibniz_config_path` variable
   - `LEIBNIZ_CONFIG_FILE` environment variable
   - Default `"leibniz_config.json"` (fallback)

**Changes Made**:

1. **LeibnizConfig.__init__()**: Added `self._config_path = config_file` to store path
2. **Module-level variables**: Changed from `_leibniz_config = None` to:
   ```python
   _leibniz_config = None
   _leibniz_config_path = None
   ```
3. **initialize_leibniz_config()**: Added `_leibniz_config_path = config_file` assignment
4. **reload_leibniz_config()**: Complete rewrite with path resolution logic:
   ```python
   config_path = None
   if _leibniz_config and hasattr(_leibniz_config, '_config_path') and _leibniz_config._config_path:
       config_path = _leibniz_config._config_path
   elif _leibniz_config_path:
       config_path = _leibniz_config_path
   else:
       config_path = os.getenv('LEIBNIZ_CONFIG_FILE', 'leibniz_config.json')
   ```

**Impact**: Config reload now respects the original initialization path, making it work correctly in production environments with custom config files.

---

## Comment 3: Updated README file names to match planned phases ✅

**File**: `leibniz_agent/README.md`

**Problem**: File names and function references diverged from the planned subsequent phases.

**Solution**: Replaced all occurrences with correct naming:
- `leibniz_orchestrator.py` → `leibniz_pro.py`
- `leibniz_fsm.py` → `leibniz_appointment_fsm.py`
- `transcribe_audio()` → `transcribe_file()`
- `synthesize_speech()` → `synthesize_to_file()`
- `LeibnizOrchestrator` class → `start_conversation_loop()` function

**Changes Made** (9 replacements):

1. **Component Overview** (line 88-94):
   - Changed: `leibniz_fsm.py` → `leibniz_appointment_fsm.py`
   - Changed: `leibniz_orchestrator.py` → `leibniz_pro.py`

2. **Usage Examples - Basic Conversation** (line 281-287):
   - Changed: `from leibniz_agent import LeibnizOrchestrator` → `from leibniz_agent.leibniz_pro import start_conversation_loop`
   - Changed: `agent = LeibnizOrchestrator()` + `agent.start_conversation()` → `start_conversation_loop()`

3. **Testing Individual Components - STT** (line 302-304):
   - Changed: `transcribe_audio("test_audio.wav")` → `transcribe_file("test_audio.wav")`

4. **Testing Individual Components - TTS** (line 331-333):
   - Changed: `synthesize_speech(...)` → `synthesize_to_file(...)`

5. **Project Structure** (line 349-355):
   - Changed: `leibniz_fsm.py` → `leibniz_appointment_fsm.py`
   - Changed: `leibniz_orchestrator.py` → `leibniz_pro.py`

6. **Adding New Intents** (line 376):
   - Changed: `leibniz_orchestrator.py` → `leibniz_pro.py`

7. **Modifying Appointment Form** (line 404):
   - Changed: `leibniz_fsm.py` → `leibniz_appointment_fsm.py`

8. **Manual Testing** (line 446):
   - Changed: `python -m leibniz_agent.leibniz_orchestrator` → `python -m leibniz_agent.leibniz_pro`

9. **Debug Mode** (line 534):
   - Changed: `python -m leibniz_agent.leibniz_orchestrator` → `python -m leibniz_agent.leibniz_pro`

**Impact**: Documentation now aligns with the actual implementation plan for Phase 2, preventing confusion during development.

---

## Comment 4: Added pitch clamping with provider-specific notes ✅

**File**: `leibniz_agent/leibniz_config.py`

**Problem**: No validation or clamping of pitch values, which could cause issues with different TTS providers.

**Solution**: Added pitch clamping to [-1.0, 1.0] range with comprehensive comments noting provider-specific handling for future TTS implementation.

**Changes Made** in `get_tts_settings()`:

```python
# Clamp pitch to safe range for most TTS providers
# Note: Provider-specific handling to be completed in TTS implementation phase
# - ElevenLabs: typically expects stability/similarity_boost (0.0-1.0)
# - Google TTS: expects pitch in semitones (SSML <prosody pitch="+2st">)
# - OpenAI TTS: may not support pitch adjustment directly
# For now, clamp to a generic [-1.0, 1.0] range for float-based providers
clamped_pitch = max(-1.0, min(1.0, pitch_value))

return {
    'provider': self.voice.tts_provider,
    'voice_id': self.voice.voice_id,
    'language': self.voice.language,
    'pitch': clamped_pitch,  # Now using clamped value
    'speed': self.voice.emotion_speed_map.get(emotion, self.voice.speed),
    'volume': self.voice.volume,
    'tone': self.voice.tone
}
```

**Impact**: 
- Prevents invalid pitch values from being passed to TTS providers
- Documents provider-specific requirements for TTS implementation phase
- Provides safe default behavior while allowing future specialization

---

## Comment 5: Added resolve_language_mode() helper function ✅

**Files**: 
- `leibniz_agent/leibniz_config.py` (implementation)
- `leibniz_agent/__init__.py` (export)

**Problem**: No unified way to resolve language mode based on both config and environment variables.

**Solution**: Added `resolve_language_mode()` function that reads `LEIBNIZ_LANGUAGE` env var and config without side effects, returning a resolved language mode dict.

**Implementation** in `leibniz_config.py`:

```python
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
```

**Exported in `__init__.py`**:
- Added `resolve_language_mode` to imports from `leibniz_config`
- Added `"resolve_language_mode"` to `__all__` list

**Backward Compatibility**:
- Kept `get_english_mode()` intact for backward compatibility
- New function provides richer information without replacing existing API

**Impact**: 
- Consumers can now get complete language configuration in one call
- Environment variable overrides work correctly
- No side effects - pure data resolution
- Backward compatible with existing code using `get_english_mode()`

---

## Verification Summary

| Comment | Status | Files Modified | Lines Changed |
|---------|--------|----------------|---------------|
| 1 - .env inline comments | ✅ Complete | 1 | ~10 |
| 2 - Reload config path | ✅ Complete | 1 | ~15 |
| 3 - README file names | ✅ Complete | 1 | ~9 replacements |
| 4 - Pitch clamping | ✅ Complete | 1 | ~10 |
| 5 - Language mode helper | ✅ Complete | 2 | ~25 |

**Total Files Modified**: 3
- `leibniz_agent/.env.leibniz`
- `leibniz_agent/leibniz_config.py`
- `leibniz_agent/README.md`
- `leibniz_agent/__init__.py`

**Total Lines Changed**: ~70

---

## Testing Recommendations

1. **Comment 1 (.env parsing)**:
   ```python
   from dotenv import load_dotenv
   import os
   load_dotenv('.env.leibniz')
   assert os.getenv('OPENAI_ORG_ID') == 'your_org_id_here'  # No trailing comment
   ```

2. **Comment 2 (config reload)**:
   ```python
   from leibniz_agent import initialize_leibniz_config, reload_leibniz_config
   config = initialize_leibniz_config('custom_path.json')
   reload_leibniz_config()  # Should reload from 'custom_path.json', not default
   ```

3. **Comment 3 (README alignment)**:
   - Verify all imports work: `from leibniz_agent.leibniz_pro import start_conversation_loop`
   - Verify function names match: `transcribe_file()`, `synthesize_to_file()`

4. **Comment 4 (pitch clamping)**:
   ```python
   from leibniz_agent import get_voice_settings
   settings = get_voice_settings('excited')
   assert -1.0 <= settings['pitch'] <= 1.0
   ```

5. **Comment 5 (language mode)**:
   ```python
   from leibniz_agent import resolve_language_mode
   import os
   os.environ['LEIBNIZ_LANGUAGE'] = 'en-GB'
   mode = resolve_language_mode()
   assert mode['language'] == 'en-GB'
   assert 'force_english' in mode
   assert 'dialect' in mode
   ```

---

## Conclusion

All verification comments have been implemented exactly as specified. The codebase is now more robust with:

1. ✅ Proper .env file formatting for universal parser compatibility
2. ✅ Intelligent config reload respecting initialization paths
3. ✅ Documentation aligned with planned Phase 2 implementation
4. ✅ Safe TTS pitch handling with provider-specific documentation
5. ✅ Flexible language mode resolution with environment variable support

The Leibniz agent is ready for Phase 2 implementation with these foundational improvements in place.
