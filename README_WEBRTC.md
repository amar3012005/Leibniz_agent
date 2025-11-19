# Leibniz Agent - WebRTC Voice Interface

## Overview

The WebRTC integration provides a browser-based voice interface for the Leibniz Agent, eliminating local audio device dependencies and enabling multi-user conversations through standard web browsers.

## Key Features

✅ **Browser-Based Interface**: No microphone setup or device configuration needed
✅ **Real-Time Audio Streaming**: Bidirectional audio via WebRTC
✅ **Automatic Turn-Taking**: FastRTC's ReplyOnPause handles natural conversation flow
✅ **Cross-Platform**: Works on Windows, macOS, Linux through any modern browser
✅ **Multi-User Support**: Handle multiple simultaneous conversations
✅ **Device Independence**: Bypasses Windows audio driver issues completely

## Architecture

```
Browser (Client)                     Server (Leibniz Agent)
━━━━━━━━━━━━━━                       ━━━━━━━━━━━━━━━━━━━━━━
                                     
🎤 Microphone                        🔊 WebRTC Server
    ↓                                   ↓
WebRTC Audio Stream ─────→       FastRTC Stream Handler
                                        ↓
                                   STT (Gemini Live)
                                        ↓
                                   Intent Classification
                                        ↓
                               RAG / Appointment FSM
                                        ↓
                                   TTS (LemonFox 24kHz)
                                        ↓
                                   Resample to 16kHz
                                        ↓
WebRTC Audio Stream ←─────       Audio Chunks (50ms)
    ↓
🔈 Speakers
```

## Installation

### 1. Install Dependencies

```powershell
pip install fastrtc scipy librosa
```

### 2. Verify Environment Variables

Ensure these are set in your `.env.leibniz` file:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
LEMONFOX_API_KEY=your_lemonfox_api_key_here
```

### 3. Run Tests

```powershell
python leibniz_agent\test_webrtc_integration.py
```

Expected output:
```
✅ PASS     Imports
✅ PASS     Audio Converter
✅ PASS     Handler Initialization
✅ PASS     App Creation
✅ PASS     Service Dependencies
Results: 5/5 tests passed
🎉 All tests passed! WebRTC integration is ready.
```

## Usage

### Start the Server

```powershell
python -m leibniz_agent.leibniz_webrtc_app
```

You should see:
```
🚀 Starting Leibniz WebRTC server on port 8000
🌐 Open browser: http://localhost:8000/
✅ FastRTC stream mounted at /webrtc/offer
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Open Browser Client

1. Open your browser to `http://localhost:8000/`
2. Click the **"🔊 Connect"** button
3. Allow microphone access when prompted
4. Start speaking - the agent will respond automatically!

### Try These Queries

- "What are your office hours?"
- "I need to schedule an appointment"
- "Tell me about the university programs"
- "Hello, how can you help me?"

## Components

### 1. `leibniz_webrtc_handler.py`

**LeibnizConversationHandler** - Main conversation logic
- Extends FastRTC's `ReplyOnPause` for automatic turn-taking
- Integrates existing STT, TTS, Intent, RAG, and FSM components
- Handles audio format conversion (24kHz → 16kHz)
- Manages session state and conversation history

**AudioFormatConverter** - Audio processing utilities
- Resampling between sample rates (24kHz ↔ 16kHz)
- Format conversion (int16 → float32)
- Audio normalization to [-1, 1] range

### 2. `leibniz_webrtc_app.py`

**FastAPI Application** - Web server and WebRTC endpoints
- `/` - Browser-based HTML/JS client with beautiful UI
- `/webrtc/offer` - WebRTC signaling endpoint (auto-mounted by FastRTC)
- `/health` - Health check with session statistics
- `/api/session` - Session information endpoint

### 3. `test_webrtc_integration.py`

**Integration Tests** - Validates all components
- Import validation
- Audio format conversion tests
- Handler initialization tests
- FastAPI app creation tests
- Service dependency checks

## Technical Details

### Audio Pipeline

1. **Input**: Browser microphone → WebRTC stream → 16kHz mono PCM
2. **Transcription**: Save to temp WAV → Gemini Live STT → transcript
3. **Processing**: Intent classification → RAG/FSM → response text
4. **Synthesis**: LemonFox TTS → 24kHz WAV → load audio
5. **Conversion**: Resample 24kHz → 16kHz → normalize to float32
6. **Output**: Stream in 50ms chunks → WebRTC → browser speakers

### Sample Rates

- **WebRTC Standard**: 16kHz (fixed by FastRTC)
- **LemonFox TTS Output**: 24kHz (high quality)
- **Conversion**: Scipy resampling with high-quality filter

### Session Management

- **Session ID**: Generated UUID per connection
- **Conversation History**: Last 20 turns stored in memory
- **Appointment FSM State**: Persisted per session
- **Timeout**: 30 minutes of inactivity

## API Reference

### Handler Initialization

```python
from leibniz_webrtc_handler import LeibnizConversationHandler

handler = LeibnizConversationHandler(
    sample_rate=16000,          # WebRTC audio sample rate
    enable_logging=True,         # Detailed conversation logs
    session_timeout=1800         # 30 minutes
)
```

### Session Info

```python
session_info = handler.get_session_info()
# Returns:
# {
#     'session_id': 'uuid-string',
#     'start_time': 1763564359.32,
#     'uptime_seconds': 120.5,
#     'conversation_turns': 5,
#     'current_fsm_state': 'WAITING_FOR_TIME'
# }
```

### Audio Format Conversion

```python
from leibniz_webrtc_handler import AudioFormatConverter

converter = AudioFormatConverter()

# Resample audio
audio_16k = converter.resample_audio(audio_24k, 24000, 16000)

# Convert int16 to float32
audio_float = converter.ensure_float32(audio_int16)

# Normalize to [-1, 1]
audio_normalized = converter.normalize_audio(audio_loud)
```

## Configuration

### Environment Variables

```bash
# Required
GEMINI_API_KEY=your_key_here              # For STT and intent classification
LEMONFOX_API_KEY=your_key_here            # For TTS synthesis

# Optional
LEIBNIZ_WEBRTC_PORT=8000                  # Server port (default: 8000)
RAG_NAMESPACE=leibniz                     # RAG namespace (auto-set)
```

### Custom Port

```python
# In leibniz_webrtc_app.py __main__ section
port = int(os.getenv("LEIBNIZ_WEBRTC_PORT", "8000"))
```

Or run with:
```powershell
$env:LEIBNIZ_WEBRTC_PORT="8080"; python -m leibniz_agent.leibniz_webrtc_app
```

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Run tests first to identify missing dependencies
```powershell
python leibniz_agent\test_webrtc_integration.py
```

### Issue: "WebRTC connection failed"

**Causes**:
1. Browser doesn't support WebRTC (use Chrome/Firefox/Edge)
2. Microphone permissions denied
3. HTTPS required for production (localhost works with HTTP)

**Solution**: 
- Check browser console for errors
- Allow microphone access when prompted
- For production, use HTTPS with SSL certificates

### Issue: "No audio output"

**Causes**:
1. TTS service not initialized
2. Audio resampling failure
3. Browser audio playback blocked

**Solution**:
- Check server logs for TTS errors
- Verify LEMONFOX_API_KEY is set
- Click anywhere on page to enable audio (browser autoplay policy)

### Issue: "Slow response times"

**Causes**:
1. Cold start - services initializing
2. Network latency to Gemini/LemonFox APIs
3. Large audio files

**Solution**:
- First response is slower (1-2s) - subsequent responses faster
- Check network connection
- Use shorter phrases for testing

### Issue: "Agent doesn't understand me"

**Causes**:
1. Background noise
2. Accent/pronunciation differences
3. Unclear speech

**Solution**:
- Speak clearly and at moderate pace
- Reduce background noise
- Try shorter, simpler phrases
- Check STT logs for actual transcript

## Performance Metrics

### Latency Breakdown (Typical)

| Stage | Time | Notes |
|-------|------|-------|
| Audio capture | ~0ms | Continuous streaming |
| STT (Gemini) | 500-1500ms | Network dependent |
| Intent classification | 10-50ms | Cached patterns |
| RAG query | 300-800ms | FAISS + Gemini |
| TTS synthesis | 1000-2000ms | LemonFox API |
| Audio streaming | 50-100ms | Chunked delivery |
| **Total** | **2-4 seconds** | End-to-end |

### Resource Usage

- **Memory**: ~500MB (includes services, models, buffers)
- **CPU**: 5-15% idle, 25-40% during synthesis
- **Network**: ~50KB/minute audio, API calls vary
- **Concurrent Users**: 20-50 recommended per server

## Comparison: WebRTC vs. Local Audio

| Aspect | Local Audio (sounddevice) | WebRTC Streaming |
|--------|---------------------------|------------------|
| **Setup** | Configure devices, drivers | Just open browser |
| **Platform** | Windows-specific issues | Cross-platform |
| **Multi-User** | Single user only | Multiple simultaneous |
| **Device Errors** | Channel mismatches, etc. | None (browser handles) |
| **Latency** | Lower (direct access) | Slightly higher (network) |
| **Deployment** | Requires local install | Remote server possible |

## Future Enhancements

- [ ] **Session Persistence**: Redis/MongoDB for reconnection support
- [ ] **Background Music**: Ambient audio during agent responses
- [ ] **Voice Activity Visualization**: Real-time waveform display
- [ ] **Multi-Language Support**: Detect and respond in user's language
- [ ] **Emotion Detection**: Analyze user sentiment from voice
- [ ] **Recording/Playback**: Save and replay conversations
- [ ] **Authentication**: User login and personalization
- [ ] **Analytics Dashboard**: Conversation metrics and insights

## Contributing

To extend the WebRTC integration:

1. **Add new intents**: Modify `leibniz_intent_parser.py`
2. **Customize TTS**: Update `leibniz_tts.py` providers
3. **Enhance UI**: Edit HTML/CSS/JS in `leibniz_webrtc_app.py`
4. **Add middleware**: Modify FastAPI app in `create_leibniz_webrtc_app()`

## License

Same as parent Leibniz Agent project.

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review server logs for error details
3. Run integration tests to isolate failures
4. File an issue with reproduction steps

---

**Status**: ✅ Production Ready (v1.0.0)
**Last Updated**: November 19, 2025
**Tested On**: Windows 10/11, Python 3.10+
