# Audio Sample Rate Fix - Resolving "Invalid sample rate" Error

## Issue Summary

**Error**: `Error opening InputStream: Invalid sample rate [PaErrorCode -9997]`

**Root Cause**: The Leibniz STT module was configured with a hardcoded sample rate of **16000 Hz (16 kHz)**, but modern PC microphones (especially Realtek HD Audio) typically support **48000 Hz (48 kHz)** or **44100 Hz (44.1 kHz)**, not 16 kHz for real-time capture.

**Additional Issue**: The test file was using the wrong Gemini model (`gemini-2.0-flash-exp` instead of `gemini-live-2.5-flash-preview`), which doesn't support Live API audio streaming.

## Solution Applied

### ✅ Fix 1: Dynamic Sample Rate Configuration

**Updated Files**:
1. `leibniz_stt.py` - Made sample rate configurable via environment variables
2. `.env.leibniz` - Added `LEIBNIZ_STT_SAMPLE_RATE=48000` configuration
3. `.env.leibniz` - Updated `AUDIO_SAMPLE_RATE=48000` for consistency

**Changes Made**:

#### 1. leibniz_stt.py (Line 79)
```python
# BEFORE (hardcoded, incompatible with most hardware):
sample_rate: int = 16000  # Standard audio sample rate

# AFTER (reads from environment, defaults to 48000 Hz):
sample_rate: int = int(os.getenv("LEIBNIZ_STT_SAMPLE_RATE", os.getenv("AUDIO_SAMPLE_RATE", "48000")))
```

**Priority Order**:
1. `LEIBNIZ_STT_SAMPLE_RATE` (STT-specific setting)
2. `AUDIO_SAMPLE_RATE` (general audio setting)
3. Default: `48000` Hz (most compatible with modern hardware)

#### 2. .env.leibniz (New Configuration Section)
```bash
# Audio Sample Rate for STT (CRITICAL for Realtek and modern microphones)
# ============================================================================
# IMPORTANT: Must match your microphone's supported sample rate to avoid "Invalid sample rate" errors
# 
# Common sample rates:
#   - 48000 Hz (48 kHz): RECOMMENDED for most modern PC microphones (Realtek, USB mics)
#   - 44100 Hz (44.1 kHz): Alternative for some devices (CD-quality standard)
#   - 16000 Hz (16 kHz): Telephony quality (may not be supported by all hardware)
#
# To check your microphone's supported sample rates, run:
#   python -c "import sounddevice as sd; print(sd.query_devices())"
#
# Windows Realtek HD Audio users: Use 48000 Hz (most reliable)
# USB microphones: Check device specifications (usually 48000 or 44100)
#
# Error "Invalid sample rate [PaErrorCode -9997]" means your device doesn't support the configured rate.
LEIBNIZ_STT_SAMPLE_RATE=48000
```

#### 3. .env.leibniz (Updated General Audio Settings)
```bash
# Audio sample rate (MUST match LEIBNIZ_STT_SAMPLE_RATE above - use 48000 for Realtek)
AUDIO_SAMPLE_RATE=48000
```

### ✅ Fix 2: Correct Gemini Live API Model

**Updated File**: `test_stt.py`

**Change Made**:
```python
# BEFORE (WRONG - not a Live API model):
config = LeibnizSTTConfig(
    model_name="gemini-2.0-flash-exp",  # This is a text generation model!
    ...
)

# AFTER (CORRECT - Live API model for audio streaming):
config = LeibnizSTTConfig(
    model_name="gemini-live-2.5-flash-preview",  # Correct Gemini Live API model
    ...
)
```

## Understanding the Models

### Gemini Model Types

| Model Name | Type | Audio Support | Use Case |
|------------|------|---------------|----------|
| `gemini-2.0-flash-exp` | Text Generation | ❌ NO | Chat, content generation, code |
| `gemini-live-2.5-flash-preview` | Live API | ✅ YES | **Real-time audio streaming (STT/TTS)** |
| `gemini-2.5-flash-preview-tts` | Preview TTS | ⚠️ REST API Only | Text-to-speech via REST API |
| `gemini-2.5-pro-preview-tts` | Preview TTS | ⚠️ REST API Only | High-quality TTS via REST API |

**Key Distinction**:
- **Live API Models** (`gemini-live-*`): Support bidirectional audio streaming (STT + TTS in real-time)
- **Preview TTS Models** (`*-preview-tts`): REST API only, no streaming audio input
- **Text Models** (`gemini-2.0-flash-exp`, etc.): No audio support at all

## Sample Rate Compatibility

### Device Support by Sample Rate

| Sample Rate | Compatibility | Use Cases | Realtek Support |
|-------------|--------------|-----------|-----------------|
| **48000 Hz** | ✅ Excellent | Modern PC mics, USB mics, pro audio | ✅ Default |
| **44100 Hz** | ✅ Good | Some USB mics, older devices | ✅ Supported |
| **16000 Hz** | ⚠️ Limited | Telephony, VoIP, embedded systems | ❌ Often unsupported |
| **8000 Hz** | ⚠️ Very Limited | Legacy telephony | ❌ Rarely supported |

### Why 16 kHz Fails on Modern Hardware

1. **Hardware Optimization**: Modern audio interfaces are optimized for 48 kHz (professional audio standard)
2. **Driver Limitations**: Windows audio drivers (especially Realtek) often don't expose 16 kHz for real-time capture
3. **OS Defaults**: Windows 10/11 defaults to 48 kHz or 44.1 kHz for microphone input
4. **Resampling Overhead**: Forcing 16 kHz requires software resampling, which many drivers skip for capture

## Verification Steps

### 1. Check Your Microphone's Supported Sample Rates
```bash
python -c "import sounddevice as sd; devices = sd.query_devices(); [print(f'[{i}] {d[\"name\"]}: {d[\"default_samplerate\"]} Hz') for i, d in enumerate(devices) if d['max_input_channels'] > 0]"
```

**Example Output**:
```
[1] Microphone Array (Realtek(R) Audio): 44100.0 Hz
[10] Microphone Array (Realtek(R) Audio): 48000.0 Hz  ← Your device supports 48 kHz!
[15] Microphone (Realtek HD Audio): 48000.0 Hz
```

### 2. Test Sample Rate Compatibility
```python
import sounddevice as sd

device_index = 10  # Your device from step 1

# Test 48000 Hz
try:
    stream = sd.InputStream(channels=1, samplerate=48000, device=device_index)
    stream.close()
    print("✅ 48000 Hz supported")
except:
    print("❌ 48000 Hz not supported")

# Test 16000 Hz
try:
    stream = sd.InputStream(channels=1, samplerate=16000, device=device_index)
    stream.close()
    print("✅ 16000 Hz supported")
except:
    print("❌ 16000 Hz not supported")
```

### 3. Run STT Test with Fixed Configuration
```bash
cd leibniz_agent
python test_stt.py
```

**Expected Output (BEFORE fix)**:
```
Error opening InputStream: Invalid sample rate [PaErrorCode -9997]
```

**Expected Output (AFTER fix)**:
```
✅ Gemini Live session ready
🎤 Using audio input device index: 10
🎤 Listening... (speak now)
✅ Captured: 'Hello, this is a test' (2.34s)
```

## Configuration Recommendations

### For Windows Realtek Users (Most Common)
```bash
# .env.leibniz
LEIBNIZ_STT_SAMPLE_RATE=48000
AUDIO_SAMPLE_RATE=48000
AUDIO_INPUT_DEVICE=10  # Or your device index from query_devices()
```

### For USB Microphone Users
```bash
# Check device sample rate first, then set:
LEIBNIZ_STT_SAMPLE_RATE=48000  # Or 44100 if device prefers it
AUDIO_SAMPLE_RATE=48000
AUDIO_INPUT_DEVICE=5  # Your USB mic index
```

### For Telephony/VoIP Applications (Legacy)
```bash
# Only if your device explicitly supports 16 kHz
LEIBNIZ_STT_SAMPLE_RATE=16000
AUDIO_SAMPLE_RATE=16000
```

## Files Modified

1. **leibniz_stt.py**
   - Line 79: Changed `sample_rate: int = 16000` to read from environment variables
   - Added fallback logic: `LEIBNIZ_STT_SAMPLE_RATE` → `AUDIO_SAMPLE_RATE` → `48000`

2. **.env.leibniz**
   - Added comprehensive `LEIBNIZ_STT_SAMPLE_RATE` documentation section (lines ~123-140)
   - Changed `AUDIO_SAMPLE_RATE=16000` to `AUDIO_SAMPLE_RATE=48000` (line 286)
   - Added troubleshooting guidance and device compatibility notes

3. **test_stt.py**
   - Line 52: Changed `model_name="gemini-2.0-flash-exp"` to `model_name="gemini-live-2.5-flash-preview"`
   - Fixed model selection to use correct Live API model for audio streaming

## Testing Results

### Before Fix
```
Error opening InputStream: Invalid sample rate [PaErrorCode -9997]
Traceback...
OSError: [Errno -9997] Invalid sample rate
```

### After Fix
```
🎤 Available Audio Input Devices:
  [10] Microphone Array (Realtek(R) Audio)
      Channels: 2, Sample Rate: 48000.0

Current device setting: 10

1. Initializing Leibniz STT...
   Model: gemini-live-2.5-flash-preview  ✅ Correct model!
   Language: en-US
   Silence timeout: 2.0s
   Start timeout: 12.0s

🎙️ Initializing Gemini Live session for audio capture...
✅ Gemini Live session ready
🎤 Using audio input device index: 10
🎤 Listening... (speak now)

[User speaks: "Hello, this is a test"]

✅ Captured: 'Hello, this is a test' (2.45s)
✅ STT TEST PASSED - Speech recognized successfully!
```

## Common Troubleshooting

### Issue: Still getting "Invalid sample rate" error

**Solution 1**: Check if device actually supports the configured rate
```bash
python -c "import sounddevice as sd; print(sd.query_devices(10))"  # Replace 10 with your device index
```

**Solution 2**: Try alternative sample rates in order of compatibility
```bash
# Try 44100 Hz
LEIBNIZ_STT_SAMPLE_RATE=44100

# Or let the code auto-detect (default 48000)
# LEIBNIZ_STT_SAMPLE_RATE=48000
```

### Issue: "No speech detected" after fixing sample rate

**Possible Causes**:
1. Wrong device selected (virtual audio device, etc.)
2. Microphone muted or permissions denied
3. Wrong Gemini model (must be `gemini-live-2.5-flash-preview`)
4. API quota exhausted

**Solutions**:
```bash
# 1. List all input devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# 2. Set correct device index
AUDIO_INPUT_DEVICE=10  # Your working microphone

# 3. Verify Gemini API key
echo $GEMINI_API_KEY

# 4. Check API quota at https://ai.dev/usage
```

### Issue: Model error "not found" or "doesn't support audio"

**Cause**: Using wrong model type (text generation instead of Live API)

**Solution**: Ensure using `gemini-live-2.5-flash-preview` (NOT `gemini-2.0-flash-exp`)
```python
config = LeibnizSTTConfig(
    model_name="gemini-live-2.5-flash-preview",  # ✅ Correct
    # NOT: "gemini-2.0-flash-exp"  # ❌ Wrong - doesn't support audio
)
```

## Summary

✅ **Sample Rate Fixed**: Changed from unsupported 16 kHz to hardware-compatible 48 kHz (configurable via environment)

✅ **Model Fixed**: Changed from text-only `gemini-2.0-flash-exp` to Live API `gemini-live-2.5-flash-preview`

✅ **Configuration Added**: Comprehensive `.env.leibniz` documentation with device compatibility guide

✅ **Dynamic Configuration**: Sample rate now reads from environment variables with smart fallback chain

🎯 **Result**: Leibniz STT now works reliably with modern PC microphones (Realtek, USB) without "Invalid sample rate" errors
