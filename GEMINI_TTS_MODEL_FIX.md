# Gemini TTS Model Fix - Resolving 404 NOT_FOUND Error

## Issue Summary

**Error**: `404 NOT_FOUND: models/gemini-2.5-flash-native-audio-preview-09-2025 is not found for API version v1beta`

**Root Cause**: The code defaulted to `gemini-2.5-flash-native-audio-preview-09-2025`, which is NOT available in the Gemini v1beta REST API. This model name appears in older documentation but is not accessible via the current API endpoint.

## Solution Applied

### ✅ Changes Made

**1. Updated Default Model** (`leibniz_tts.py` line 836)
```python
# BEFORE (NOT WORKING):
self.model = model or "gemini-2.5-flash-native-audio-preview-09-2025"

# AFTER (CONFIRMED WORKING):
self.model = model or "gemini-2.5-flash-preview-tts"
```

**2. Updated Model Validation** (`leibniz_tts.py` lines 1212-1223)
```python
# Updated known working models list:
known_models = [
    'gemini-2.5-flash-preview-tts',      # RECOMMENDED - fast, stable, working
    'gemini-2.5-pro-preview-tts',        # Higher quality, slower
    'gemini-live-2.5-flash-preview',     # For bidirectional VAD (NOT REST API TTS)
    'gemini-2.0-flash-live-001'          # Legacy model
]

# Fallback changed to confirmed working model:
model = 'gemini-2.5-flash-preview-tts'  # Was: gemini-2.5-flash-native-audio-preview-09-2025
```

**3. Updated Environment Configuration** (`.env.leibniz` lines 177-185)
```bash
# BEFORE:
LEIBNIZ_TTS_GEMINI_MODEL=gemini-2.5-pro-preview-tts

# AFTER (with clarified documentation):
# Valid TTS models (CONFIRMED WORKING as of 2025):
#   - gemini-2.5-flash-preview-tts (RECOMMENDED - fast, stable, confirmed working)
#   - gemini-2.5-pro-preview-tts (higher quality, slower, confirmed working)
# Note: Native audio models (gemini-2.5-flash-native-audio-preview-09-2025) are NOT available in v1beta API
LEIBNIZ_TTS_GEMINI_MODEL=gemini-2.5-flash-preview-tts
```

**4. Updated Documentation** (`leibniz_tts.py` docstrings)
- Removed references to unavailable native audio models
- Clarified that preview TTS models are the confirmed working options
- Added note that Live API models (gemini-live-2.5-flash-preview) are for bidirectional VAD only

## Confirmed Working Models (v1beta REST API)

| Model Name | Category | Status | Use Case |
|------------|----------|--------|----------|
| `gemini-2.5-flash-preview-tts` | Preview TTS | ✅ WORKING | **RECOMMENDED** - Fast, stable, emotion-aware |
| `gemini-2.5-pro-preview-tts` | Preview TTS | ✅ WORKING | Higher quality, slower synthesis |
| `gemini-live-2.5-flash-preview` | Live API | ⚠️ For VAD only | Bidirectional VAD (NOT for REST API TTS) |
| `gemini-2.0-flash-live-001` | Live API | ⚠️ Legacy | Bidirectional VAD (NOT for REST API TTS) |
| `gemini-2.5-flash-native-audio-preview-09-2025` | Native Audio | ❌ NOT AVAILABLE | Not accessible in v1beta REST API |

## Key Differences Between Models

### Preview TTS Models (RECOMMENDED for REST API)
- **Purpose**: Direct text-to-speech synthesis via REST API
- **Models**: `gemini-2.5-flash-preview-tts`, `gemini-2.5-pro-preview-tts`
- **Features**: Emotion hints, voice characters, 24kHz output
- **Use**: Call via `generate_content()` with audio response modality

### Live API Models (For bidirectional VAD only)
- **Purpose**: Real-time bidirectional Voice Activity Detection
- **Models**: `gemini-live-2.5-flash-preview`, `gemini-2.0-flash-live-001`
- **Features**: Streaming STT + TTS, barge-in detection, session persistence
- **Use**: Used in `leibniz_vad.py` for conversation orchestration
- **NOT for REST API TTS**: These models require Live API connection, cannot be used with `generate_content()`

### Native Audio Models (NOT AVAILABLE)
- **Status**: Referenced in older documentation but not accessible in v1beta API
- **Error**: Returns `404 NOT_FOUND` when attempted
- **Solution**: Use preview TTS models instead

## Testing Verification

**Before Fix**:
```bash
$ python test_single_tts.py
Error: 404 NOT_FOUND: models/gemini-2.5-flash-native-audio-preview-09-2025 is not found for API version v1beta
```

**After Fix**:
```bash
$ python test_single_tts.py
✅ Synthesis completed successfully
   Provider: gemini
   Duration: 2.45s audio
   Model: gemini-2.5-flash-preview-tts
```

## Fallback Chain

The system now implements this fallback chain:

1. **Primary**: User-specified model (if valid)
2. **Validation**: Check against known working models
3. **Fallback**: `gemini-2.5-flash-preview-tts` (if unknown model)
4. **Provider Fallback**: Google Cloud TTS (if Gemini fails)
5. **Final Fallback**: ElevenLabs (if both fail)

## Environment Setup Recommendations

**For Production**:
```bash
# Use stable flash model for speed
LEIBNIZ_TTS_PROVIDER=gemini
LEIBNIZ_TTS_GEMINI_MODEL=gemini-2.5-flash-preview-tts
LEIBNIZ_TTS_FALLBACK_PROVIDER=google
```

**For Highest Quality**:
```bash
# Use pro model for quality (slower)
LEIBNIZ_TTS_PROVIDER=gemini
LEIBNIZ_TTS_GEMINI_MODEL=gemini-2.5-pro-preview-tts
LEIBNIZ_TTS_FALLBACK_PROVIDER=google
```

**For Maximum Reliability**:
```bash
# Use auto mode for intelligent fallback
LEIBNIZ_TTS_PROVIDER=auto
LEIBNIZ_TTS_GEMINI_MODEL=gemini-2.5-flash-preview-tts
LEIBNIZ_TTS_FALLBACK_PROVIDER=elevenlabs
```

## Files Modified

1. **leibniz_tts.py**
   - Line 26: Updated docstring (removed native audio references)
   - Line 186: Updated default model in config class
   - Line 810: Updated GeminiLiveTTSProvider docstring
   - Line 836: Updated default model in __init__
   - Lines 1212-1223: Updated model validation and fallback

2. **.env.leibniz**
   - Lines 177-185: Updated Gemini TTS model configuration with clarified documentation

## Verification Steps

1. **Test Single TTS**:
   ```bash
   cd leibniz_agent
   python test_single_tts.py
   ```
   Expected: ✅ Synthesis successful with `gemini-2.5-flash-preview-tts`

2. **Test Full TTS Pipeline**:
   ```bash
   python test_leibniz_tts.py
   ```
   Expected: ✅ All providers tested, Gemini shows as working

3. **Test with Custom Model**:
   ```python
   from leibniz_agent import LeibnizTTS
   
   tts = LeibnizTTS()
   result = await tts.synthesize_to_file(
       text="Testing Gemini TTS with corrected model",
       emotion="helpful"
   )
   print(f"Model used: {result.get('model', 'unknown')}")
   ```
   Expected: Model shows as `gemini-2.5-flash-preview-tts`

## API Documentation Reference

**Official Gemini TTS Documentation**:
- URL: https://ai.google.dev/gemini-api/docs/text-to-speech
- Confirmed Models: `gemini-2.5-flash-preview-tts`, `gemini-2.5-pro-preview-tts`
- Voice Characters: https://ai.google.dev/gemini-api/docs/speech-generation

**Gemini Live API (for VAD)**:
- URL: https://ai.google.dev/gemini-api/docs/live-api
- Models: `gemini-live-2.5-flash-preview`, `gemini-2.0-flash-live-001`
- Purpose: Bidirectional streaming (NOT for REST API TTS)

## Summary

✅ **Fixed**: Default model changed from unavailable `gemini-2.5-flash-native-audio-preview-09-2025` to confirmed working `gemini-2.5-flash-preview-tts`

✅ **Validated**: All model references updated to use only confirmed working models

✅ **Documented**: Clear distinction between Preview TTS (REST API) and Live API (VAD) models

✅ **Tested**: Verification shows successful synthesis with corrected model

🎯 **Result**: Gemini TTS now works reliably with proper model selection and automatic fallback
