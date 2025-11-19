# FastRTC WebRTC Streaming Integration Plan for Leibniz Agent

**Research Agent Analysis Report**  
**Generated:** 2025-01-19  
**Target System:** `leibniz_pro.py` (4431 lines) + supporting modules

---

## Executive Summary

This document provides a comprehensive plan for integrating **FastRTC WebRTC streaming** into the Leibniz Agent, transforming it from a local desktop application into a real-time web-based conversation system. The integration leverages FastRTC's `ReplyOnPause` handler for automatic turn-taking while preserving the existing VAD/STT/TTS/RAG/FSM architecture.

**Key Insight:** The current architecture is remarkably well-suited for WebRTC integration due to its modular design and streaming-first approach in TTS. The primary work involves wrapping existing logic in FastRTC handlers and adapting audio I/O from local sounddevice to WebRTC streams.

---

## 1. Current Architecture Analysis

### 1.1 Audio Flow Pipeline

```
┌─────────────┐
│  Microphone │ (sounddevice @ 16kHz PCM)
└──────┬──────┘
       │
       v
┌─────────────────────────────┐
│  Leibniz VAD                │ (leibniz_vad.py)
│  - Gemini Live API          │
│  - Bidirectional state mgmt │
│  - Barge-in detection       │
└──────────┬──────────────────┘
           │ transcript
           v
┌─────────────────────────────┐
│  Intent Classifier          │ (leibniz_intent_parser.py)
│  - Gemini 2.0 based         │
│  - Semantic context extract │
└──────────┬──────────────────┘
           │ intent + entities
           v
┌─────────────────────────────┐
│  Router (leibniz_pro.py)    │
│  - APPOINTMENT_SCHEDULING   │──► FSM (leibniz_appointment_fsm.py)
│  - RAG_QUERY               │──► RAG (leibniz_rag.py)
│  - GREETING/EXIT           │──► Static responses
└──────────┬──────────────────┘
           │ response text
           v
┌─────────────────────────────┐
│  TTS Synthesis              │ (leibniz_tts.py)
│  - LemonFox API (24kHz WAV) │
│  - Streaming support        │
│  - Emotion modulation       │
└──────────┬──────────────────┘
           │ audio bytes
           v
┌─────────────────────────────┐
│  Audio Playback             │
│  - sounddevice (device 12)  │
│  - pygame fallback          │
└─────────────────────────────┘
```

### 1.2 Key Functions to Modify/Replace

#### **1.2.1 Audio Input (leibniz_pro.py lines 2116-2150)**

**Current:** `capture_and_transcribe()`
- Captures from local microphone via sounddevice
- Returns transcript string (no audio file saved)
- Uses Gemini Live VAD for speech detection
- Supports streaming callbacks for real-time display

**WebRTC Adaptation:**
- **Source:** FastRTC's `AudioStream` (client microphone)
- **Sink:** Pass audio chunks directly to Gemini Live API
- **Key Change:** Replace sounddevice capture with FastRTC stream reader
- **Preserve:** Existing VAD logic, transcript normalization, streaming callbacks

#### **1.2.2 Audio Output (leibniz_pro.py lines 1922-2115)**

**Current:** `speak_friendly()`
- Synthesizes text via LemonFox API (24kHz WAV)
- Plays locally via sounddevice or pygame
- Supports streaming TTS for long responses
- Manages agent speaking state for barge-in

**WebRTC Adaptation:**
- **Source:** LemonFox TTS synthesis (keep existing)
- **Sink:** FastRTC's `AudioStream` (client speakers)
- **Key Change:** Replace local playback with WebRTC stream writer
- **Preserve:** TTS synthesis logic, emotion modulation, streaming queue

#### **1.2.3 Session Management (leibniz_pro.py lines 3828-4050)**

**Current:** `run_conversation_session()`
- Enter-to-start loop for desktop sessions
- Max 5 attempts per session with retry logic
- Tracks conversation state (greeting, rag_query, appointment)
- Handles barge-in, timeouts, and errors

**WebRTC Adaptation:**
- **Trigger:** WebSocket connection (not Enter keypress)
- **Lifecycle:** Session per WebRTC connection
- **Key Change:** Wrap session logic in FastRTC `ReplyOnPause` handler
- **Preserve:** Retry logic, state tracking, error handling

### 1.3 Supporting Modules

#### **1.3.1 VAD (leibniz_vad.py - 1200+ lines)**
- **Gemini Live API integration:** Direct WebSocket to Google
- **Bidirectional state:** Agent speaking vs listening
- **Barge-in detection:** User interrupt during TTS
- **WebRTC Impact:** Medium - must adapt audio input source
- **Key APIs:** `capture_leibniz_speech()`, `set_leibniz_agent_speaking()`

#### **1.3.2 TTS (leibniz_tts.py - 2061 lines)**
- **LemonFox provider:** REST API for synthesis (24kHz WAV)
- **Emotion modulation:** Speed adjustment (1.0=normal, 1.2=excited)
- **Caching:** MD5-based with LRU cleanup
- **WebRTC Impact:** Low - synthesis stays same, only playback changes
- **Key APIs:** `synthesize_to_file()`, `stream_tts()`

#### **1.3.3 RAG (leibniz_rag.py - referenced)**
- **FAISS vector store:** University knowledge base
- **Ensemble retrieval:** Hybrid search (dense + sparse)
- **Streaming generation:** Progressive response assembly
- **WebRTC Impact:** None - purely text processing
- **Key APIs:** `process_rag_query()`

#### **1.3.4 Appointment FSM (leibniz_appointment_fsm.py - referenced)**
- **State machine:** Multi-step slot filling (date, time, email, name)
- **Semantic translation:** Multilingual support (Hindi/Telugu → English)
- **Confirmation loops:** User validation at each step
- **WebRTC Impact:** None - text-based state machine
- **Key APIs:** `create_appointment_fsm()`, `process_input()`

---

## 2. FastRTC Integration Design

### 2.1 WebRTC Handler Architecture

```python
# leibniz_agent/leibniz_fastrtc_handler.py (NEW FILE)

from fastrtc import ReplyOnPause, AudioStream
from fastapi import FastAPI, WebSocket
import asyncio
import numpy as np

class LeibnizConversationHandler(ReplyOnPause):
    """
    FastRTC handler wrapping Leibniz conversation logic.
    
    Lifecycle:
    1. __init__: Load services (VAD, TTS, RAG, FSM)
    2. setup: Initialize session state
    3. process_audio_frame: Convert WebRTC → Gemini VAD format
    4. reply: Generate responses (intent → RAG/FSM → TTS)
    5. on_client_speaking: Handle barge-in
    6. cleanup: Release resources
    """
    
    def __init__(self):
        super().__init__()
        # Import existing services
        from leibniz_agent.leibniz_vad import get_leibniz_vad
        from leibniz_agent.leibniz_tts import get_leibniz_tts
        from leibniz_agent.leibniz_rag import get_leibniz_rag
        from leibniz_agent.leibniz_intent_parser import get_leibniz_parser
        
        self.vad = get_leibniz_vad()
        self.tts = get_leibniz_tts()
        self.rag = get_leibniz_rag()
        self.intent_parser = get_leibniz_parser()
        
        # Session state (per WebRTC connection)
        self.session_id = None
        self.conversation_state = "greeting"  # greeting, rag_query, appointment
        self.consecutive_no_input = 0
        self.current_fsm = None  # Appointment FSM if active
    
    async def setup(self, session_id: str):
        """Called when WebRTC connection established"""
        self.session_id = session_id
        await self.vad.set_agent_speaking_state(False, "Session start")
        
        # Play intro greeting
        greeting_text = get_dialogue_text('greeting', "Hello! Welcome to Leibniz University.")
        audio_bytes = await self.synthesize_audio(greeting_text, emotion="helpful")
        await self.send_audio_to_client(audio_bytes)
    
    async def process_audio_frame(self, audio_frame: np.ndarray):
        """
        Convert WebRTC audio to Gemini Live API format.
        
        FastRTC provides: 16kHz PCM mono (float32)
        Gemini expects: 16kHz PCM mono (int16 bytes)
        """
        # Convert float32 [-1, 1] to int16 PCM
        pcm_data = (audio_frame * 32767).astype(np.int16).tobytes()
        
        # Forward to Gemini Live session (existing VAD logic)
        # Note: VAD already handles this in capture_speech_bidirectional()
        # We just need to adapt the input source
        return pcm_data
    
    async def reply(self, transcript: str) -> str:
        """
        Core conversation logic - same as run_conversation_session().
        
        Args:
            transcript: User's speech (from VAD)
            
        Returns:
            Response text to speak
        """
        if not transcript:
            self.consecutive_no_input += 1
            if self.consecutive_no_input >= 5:
                return get_dialogue_text('errors.general', "I'm having trouble hearing you.")
            return get_dialogue_text('timeout', "I didn't catch that, could you repeat?")
        
        # Reset no-input counter
        self.consecutive_no_input = 0
        
        # Check for exit
        if is_exit_phrase(transcript):
            return get_dialogue_text('farewells.exit', "Goodbye! Have a great day.")
        
        # Classify intent (existing logic)
        intent_result = await self.intent_parser.classify_intent(transcript)
        intent = intent_result.get('intent', 'UNCLEAR')
        
        # Route to appropriate handler
        if intent == "APPOINTMENT_SCHEDULING":
            # Appointment FSM logic (keep existing)
            if not self.current_fsm:
                from leibniz_agent.leibniz_appointment_fsm import create_appointment_fsm
                self.current_fsm = create_appointment_fsm()
            
            result = await self.current_fsm.process_input(transcript)
            return result['response']
        
        elif intent == "RAG_QUERY":
            # RAG query logic (keep existing)
            from leibniz_agent.leibniz_pro import handle_rag_query
            rag_msg = await handle_rag_query(
                text=transcript,
                context={'last_intent': intent_result},
                enable_streaming=False
            )
            return rag_msg.answer
        
        elif intent == "GREETING":
            return get_dialogue_text('greetings.intro', "Hello! How can I help you?")
        
        elif intent == "EXIT":
            return get_dialogue_text('farewells.exit', "Thanks for chatting!")
        
        else:  # UNCLEAR
            return get_dialogue_text('prompts.clarify', "I didn't quite understand. Could you rephrase?")
    
    async def synthesize_audio(self, text: str, emotion: str = "helpful") -> bytes:
        """Synthesize text to audio bytes (keep existing TTS logic)"""
        result = await self.tts.synthesize_to_file(
            text=text,
            emotion=emotion,
            cache_name=None  # No caching for WebRTC (each user unique)
        )
        
        # Read audio file as bytes
        if result and result.get('success'):
            audio_path = result.get('audio_file')
            with open(audio_path, 'rb') as f:
                return f.read()
        return b''
    
    async def send_audio_to_client(self, audio_bytes: bytes):
        """Send audio to WebRTC client (FastRTC handles streaming)"""
        # FastRTC's AudioStream will handle this
        # We just need to yield audio chunks in the correct format
        # LemonFox outputs 24kHz WAV, need to convert to 16kHz for WebRTC
        
        # Option 1: Resample 24kHz → 16kHz
        # Option 2: Configure LemonFox for 16kHz output (if supported)
        # Option 3: Use FastRTC's built-in resampling
        
        pass  # Implementation depends on FastRTC's API
    
    async def on_client_speaking(self):
        """Handle user barge-in during agent speech"""
        await self.vad.set_agent_speaking_state(False, "Barge-in interrupt")
        # FastRTC will stop audio playback automatically
    
    async def cleanup(self):
        """Release session resources"""
        if self.current_fsm:
            self.current_fsm = None
        await self.vad.set_agent_speaking_state(False, "Session end")
```

### 2.2 FastAPI Application

```python
# leibniz_agent/leibniz_fastrtc_app.py (NEW FILE)

from fastapi import FastAPI, WebSocket
from fastrtc import FastRTC, AudioStream
from leibniz_agent.leibniz_fastrtc_handler import LeibnizConversationHandler
import uvicorn

app = FastAPI(title="Leibniz WebRTC Agent")
rtc = FastRTC(app)

@rtc.route("/conversation")
async def conversation_endpoint(audio_stream: AudioStream):
    """
    WebRTC endpoint for Leibniz conversations.
    
    Client connects via WebSocket + WebRTC handshake.
    Handler manages bidirectional audio streams.
    """
    handler = LeibnizConversationHandler()
    return handler

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "leibniz-webrtc"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

## 3. Step-by-Step Implementation Plan

### Phase 1: Environment Setup (Day 1)

#### 1.1 Install FastRTC
```bash
pip install fastrtc
# or
pip install git+https://github.com/cartesia-ai/fastrtc.git
```

#### 1.2 Create New Module Structure
```
leibniz_agent/
├── leibniz_fastrtc_handler.py  (NEW - 300 lines)
├── leibniz_fastrtc_app.py      (NEW - 50 lines)
├── leibniz_fastrtc_utils.py    (NEW - 100 lines - audio format conversions)
└── tests/
    ├── test_fastrtc_handler.py (NEW - unit tests)
    └── test_fastrtc_integration.py (NEW - E2E tests)
```

#### 1.3 Update Dependencies
```toml
# pyproject.toml or requirements.txt
fastrtc>=0.1.0
aiohttp>=3.9.0
numpy>=1.24.0
soundfile>=0.12.0  # For audio resampling
```

### Phase 2: Audio Format Adaptation (Day 2-3)

#### 2.1 Create Audio Conversion Utilities
```python
# leibniz_agent/leibniz_fastrtc_utils.py

import numpy as np
import soundfile as sf
from scipy import signal

def resample_audio(audio_data: np.ndarray, 
                   source_rate: int, 
                   target_rate: int) -> np.ndarray:
    """
    Resample audio from source_rate to target_rate.
    
    Example: LemonFox outputs 24kHz, WebRTC expects 16kHz
    """
    if source_rate == target_rate:
        return audio_data
    
    # Calculate resampling ratio
    num_samples = int(len(audio_data) * target_rate / source_rate)
    return signal.resample(audio_data, num_samples)

def pcm_float_to_int16(audio_float: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] to int16 PCM bytes"""
    audio_int16 = (audio_float * 32767).astype(np.int16)
    return audio_int16.tobytes()

def pcm_int16_to_float(audio_bytes: bytes) -> np.ndarray:
    """Convert int16 PCM bytes to float32 [-1, 1]"""
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32767.0

def wav_bytes_to_pcm(wav_bytes: bytes, target_rate: int = 16000) -> bytes:
    """
    Convert WAV file bytes to raw PCM at target sample rate.
    
    LemonFox outputs: 24kHz WAV
    WebRTC expects: 16kHz PCM
    """
    import io
    
    # Read WAV from bytes
    audio_data, source_rate = sf.read(io.BytesIO(wav_bytes))
    
    # Resample if needed
    if source_rate != target_rate:
        audio_data = resample_audio(audio_data, source_rate, target_rate)
    
    # Convert to int16 PCM
    return pcm_float_to_int16(audio_data)
```

#### 2.2 Test Audio Pipeline
```python
# tests/test_fastrtc_utils.py

import numpy as np
from leibniz_agent.leibniz_fastrtc_utils import *

def test_resample_24khz_to_16khz():
    # Generate 1 second of 24kHz audio
    audio_24k = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000))
    
    # Resample to 16kHz
    audio_16k = resample_audio(audio_24k, 24000, 16000)
    
    assert len(audio_16k) == 16000, "Output should be 16000 samples"
    assert audio_16k.dtype == np.float32, "Output should be float32"

def test_pcm_conversion_roundtrip():
    audio_float = np.random.rand(1000) * 2 - 1  # [-1, 1]
    audio_bytes = pcm_float_to_int16(audio_float)
    audio_recovered = pcm_int16_to_float(audio_bytes)
    
    # Check approximate equality (quantization loss expected)
    np.testing.assert_allclose(audio_float, audio_recovered, atol=1/32767)
```

### Phase 3: VAD Integration (Day 4-5)

#### 3.1 Adapt VAD for WebRTC Audio Source

**Current (leibniz_vad.py lines 700-900):**
```python
# Uses sounddevice for local microphone
stream = sd.InputStream(
    samplerate=16000,
    channels=1,
    dtype='float32',
    callback=audio_callback
)
```

**WebRTC Adaptation:**
```python
# leibniz_agent/leibniz_fastrtc_handler.py

async def capture_speech_from_webrtc(
    self, 
    audio_stream: AudioStream,
    streaming_callback: Optional[Callable]
) -> Optional[str]:
    """
    Capture speech from WebRTC audio stream.
    
    Replaces sounddevice capture with FastRTC stream.
    Forwards audio to Gemini Live API (existing VAD logic).
    """
    # Get Gemini Live session (existing persistent session)
    session = await LeibnizPersistentSession.get_session(
        self.vad.client,
        self.vad.config.model_name,
        self.vad.config
    )
    
    # Audio queue for async processing
    audio_queue = asyncio.Queue()
    
    # Read from WebRTC stream instead of sounddevice
    async def read_webrtc_audio():
        async for audio_chunk in audio_stream.read():
            # audio_chunk is np.ndarray (float32, 16kHz mono)
            # Convert to int16 PCM for Gemini
            pcm_data = pcm_float_to_int16(audio_chunk)
            await audio_queue.put(pcm_data)
    
    # Send to Gemini (keep existing logic)
    async def send_to_gemini():
        while True:
            try:
                pcm_data = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=pcm_data,
                        mime_type="audio/pcm;rate=16000"
                    )
                )
            except asyncio.TimeoutError:
                continue
    
    # Process transcripts (keep existing logic from leibniz_vad.py)
    transcript_buffer = TranscriptBuffer()
    async def receive_transcripts():
        async for response in session.receive():
            if response.server_content and response.server_content.input_transcription:
                text = response.server_content.input_transcription.text
                complete_text = transcript_buffer.add_fragment(text)
                
                if streaming_callback and complete_text:
                    streaming_callback(complete_text, is_final=False)
    
    # Run all tasks concurrently
    tasks = [
        asyncio.create_task(read_webrtc_audio()),
        asyncio.create_task(send_to_gemini()),
        asyncio.create_task(receive_transcripts())
    ]
    
    await asyncio.gather(*tasks)
    
    # Return final transcript
    return transcript_buffer.get_final_transcript()
```

#### 3.2 Handle Barge-In Detection

**Current:** `check_leibniz_barge_in()` checks global flag  
**WebRTC:** FastRTC's `ReplyOnPause` handles this automatically

```python
async def on_client_speaking(self):
    """
    Called by FastRTC when client starts speaking during agent speech.
    Stop TTS playback and cancel current audio stream.
    """
    # Set barge-in flag (existing infrastructure)
    self.vad.barge_in_detected = True
    await self.vad.set_agent_speaking_state(False, "Barge-in")
    
    # FastRTC stops audio output automatically
    logger.info("⚡ Barge-in detected - user interrupted agent")
```

### Phase 4: TTS Integration (Day 6-7)

#### 4.1 Adapt TTS Synthesis (Keep Existing)

**Current:** LemonFox API → 24kHz WAV file  
**WebRTC:** Same synthesis, convert format before streaming

```python
async def synthesize_and_stream_audio(
    self, 
    text: str, 
    emotion: str = "helpful"
) -> AsyncGenerator[bytes, None]:
    """
    Synthesize text and yield audio chunks for WebRTC streaming.
    
    Flow:
    1. LemonFox API → 24kHz WAV bytes
    2. Convert WAV → 16kHz PCM
    3. Chunk PCM into 20ms frames (320 samples @ 16kHz)
    4. Yield chunks to FastRTC AudioStream
    """
    # Synthesize (existing TTS logic)
    result = await self.tts.synthesize_to_file(
        text=text,
        emotion=emotion,
        cache_name=None
    )
    
    if not result or not result.get('success'):
        logger.error("TTS synthesis failed")
        return
    
    # Read WAV file
    audio_path = result.get('audio_file')
    with open(audio_path, 'rb') as f:
        wav_bytes = f.read()
    
    # Convert to 16kHz PCM
    pcm_bytes = wav_bytes_to_pcm(wav_bytes, target_rate=16000)
    
    # Chunk into 20ms frames (320 samples = 640 bytes @ 16kHz int16)
    chunk_size = 640  # 20ms frame
    for i in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[i:i+chunk_size]
        yield chunk
        await asyncio.sleep(0.02)  # 20ms delay for real-time feel
```

#### 4.2 Streaming TTS Queue Adaptation

**Current:** In-memory queue + pygame playback  
**WebRTC:** Same queue logic, send to AudioStream

```python
async def consume_tts_queue_webrtc(self, audio_stream: AudioStream):
    """
    Consumer for streaming TTS queue (adapted for WebRTC).
    
    Same sentence-by-sentence logic as leibniz_pro.py,
    but outputs to WebRTC instead of sounddevice.
    """
    while True:
        try:
            item = await asyncio.wait_for(_tts_streaming_queue.get(), timeout=0.5)
            
            if item is None:  # Sentinel
                break
            
            sentence, pace = item
            
            # Synthesize sentence
            async for audio_chunk in self.synthesize_and_stream_audio(sentence):
                await audio_stream.write(audio_chunk)
        
        except asyncio.TimeoutError:
            continue
```

### Phase 5: Session Management (Day 8-9)

#### 5.1 WebSocket Lifecycle

**Current:** `main()` with Enter-to-start loop  
**WebRTC:** Session per WebSocket connection

```python
# leibniz_agent/leibniz_fastrtc_app.py

@rtc.route("/conversation")
async def conversation_endpoint(audio_stream: AudioStream):
    """
    Each WebRTC connection gets a new session.
    
    Lifecycle:
    1. Connection established → setup()
    2. Audio frames arrive → process_audio_frame()
    3. User pauses → reply()
    4. Agent speaks → on_client_speaking() (barge-in)
    5. Connection closed → cleanup()
    """
    handler = LeibnizConversationHandler()
    
    # Generate session ID
    session_id = f"webrtc_{int(time.time())}_{id(audio_stream)}"
    
    # Initialize session
    await handler.setup(session_id)
    
    # Return handler to FastRTC
    # FastRTC will manage the conversation loop
    return handler
```

#### 5.2 Error Handling and Reconnection

```python
class LeibnizConversationHandler(ReplyOnPause):
    """Add error handling for production use"""
    
    async def on_error(self, error: Exception):
        """Called when handler encounters an error"""
        logger.error(f"Session {self.session_id} error: {error}")
        
        # Send error message to client
        error_msg = get_dialogue_text('errors.general', "Sorry, I encountered an error.")
        audio_bytes = await self.synthesize_audio(error_msg, emotion="apologetic")
        await self.send_audio_to_client(audio_bytes)
        
        # Reset state
        self.consecutive_no_input = 0
        self.conversation_state = "error_recovery"
    
    async def on_timeout(self):
        """Called if no speech detected for extended period"""
        self.consecutive_no_input += 1
        
        if self.consecutive_no_input >= 3:
            # Close session after 3 timeouts
            farewell = get_dialogue_text('farewells.timeout', "Goodbye!")
            audio_bytes = await self.synthesize_audio(farewell)
            await self.send_audio_to_client(audio_bytes)
            await self.cleanup()
        else:
            # Prompt user
            prompt = get_dialogue_text('timeout', "Are you still there?")
            audio_bytes = await self.synthesize_audio(prompt)
            await self.send_audio_to_client(audio_bytes)
```

### Phase 6: Testing Strategy (Day 10-12)

#### 6.1 Unit Tests

```python
# tests/test_fastrtc_handler.py

import pytest
from leibniz_agent.leibniz_fastrtc_handler import LeibnizConversationHandler

@pytest.mark.asyncio
async def test_handler_initialization():
    """Test handler initializes all services"""
    handler = LeibnizConversationHandler()
    assert handler.vad is not None
    assert handler.tts is not None
    assert handler.rag is not None
    assert handler.intent_parser is not None

@pytest.mark.asyncio
async def test_greeting_flow():
    """Test initial greeting synthesis"""
    handler = LeibnizConversationHandler()
    await handler.setup("test_session")
    
    # Verify greeting was synthesized
    assert handler.session_id == "test_session"
    assert handler.conversation_state == "greeting"

@pytest.mark.asyncio
async def test_rag_query_flow():
    """Test RAG query handling"""
    handler = LeibnizConversationHandler()
    await handler.setup("test_session")
    
    # Simulate user question
    response = await handler.reply("What are the admission requirements?")
    
    assert response is not None
    assert len(response) > 0
    assert "admission" in response.lower()

@pytest.mark.asyncio
async def test_appointment_flow():
    """Test appointment booking FSM"""
    handler = LeibnizConversationHandler()
    await handler.setup("test_session")
    
    # Start appointment
    response1 = await handler.reply("I want to schedule an appointment")
    assert "date" in response1.lower()
    
    # Provide date
    response2 = await handler.reply("Tomorrow at 2pm")
    assert "email" in response2.lower() or "confirm" in response2.lower()

@pytest.mark.asyncio
async def test_barge_in_handling():
    """Test user interrupt during agent speech"""
    handler = LeibnizConversationHandler()
    await handler.setup("test_session")
    
    # Simulate barge-in
    await handler.on_client_speaking()
    
    # Verify agent speaking state cleared
    assert not handler.vad.is_agent_speaking
```

#### 6.2 Integration Tests

```python
# tests/test_fastrtc_integration.py

import pytest
from fastrtc import AudioStream
from leibniz_agent.leibniz_fastrtc_app import app

@pytest.mark.asyncio
async def test_webrtc_connection():
    """Test WebRTC connection establishment"""
    # Use FastRTC test client
    async with app.test_client() as client:
        # Connect to /conversation endpoint
        rtc_client = await client.webrtc("/conversation")
        
        # Verify connection
        assert rtc_client.connected
        
        # Send test audio
        test_audio = np.random.rand(320).astype(np.float32)
        await rtc_client.send_audio(test_audio)
        
        # Receive response
        response_audio = await rtc_client.receive_audio()
        assert response_audio is not None

@pytest.mark.asyncio
async def test_end_to_end_conversation():
    """Test complete conversation flow"""
    async with app.test_client() as client:
        rtc_client = await client.webrtc("/conversation")
        
        # Receive greeting
        greeting_audio = await rtc_client.receive_audio()
        assert len(greeting_audio) > 0
        
        # Send question (pre-recorded audio file)
        question_audio = load_audio_file("tests/data/question_admission.wav")
        await rtc_client.send_audio(question_audio)
        
        # Receive answer
        answer_audio = await rtc_client.receive_audio()
        assert len(answer_audio) > 0
        
        # Transcribe answer (for verification)
        answer_text = await transcribe_audio(answer_audio)
        assert "admission" in answer_text.lower()
```

#### 6.3 Browser Testing

**HTML/JavaScript Client:**
```html
<!-- tests/webrtc_client.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Leibniz WebRTC Client</title>
</head>
<body>
    <h1>Leibniz University Agent - WebRTC Demo</h1>
    <button id="connect">Connect</button>
    <button id="disconnect" disabled>Disconnect</button>
    <div id="status">Not connected</div>
    <div id="transcript"></div>
    
    <script>
        let pc = null;
        let audioStream = null;
        
        document.getElementById('connect').onclick = async () => {
            try {
                // Get microphone access
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioStream = stream;
                
                // Create RTCPeerConnection
                pc = new RTCPeerConnection({
                    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
                });
                
                // Add audio track
                stream.getTracks().forEach(track => pc.addTrack(track, stream));
                
                // Handle incoming audio
                pc.ontrack = (event) => {
                    const audio = new Audio();
                    audio.srcObject = event.streams[0];
                    audio.play();
                };
                
                // Create offer
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                
                // Send offer to server
                const response = await fetch('http://localhost:8080/conversation', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sdp: offer.sdp, type: offer.type })
                });
                
                const answer = await response.json();
                await pc.setRemoteDescription(new RTCSessionDescription(answer));
                
                document.getElementById('status').innerText = 'Connected';
                document.getElementById('connect').disabled = true;
                document.getElementById('disconnect').disabled = false;
                
            } catch (error) {
                console.error('Connection error:', error);
                document.getElementById('status').innerText = 'Error: ' + error.message;
            }
        };
        
        document.getElementById('disconnect').onclick = () => {
            if (pc) {
                pc.close();
                pc = null;
            }
            if (audioStream) {
                audioStream.getTracks().forEach(track => track.stop());
                audioStream = null;
            }
            document.getElementById('status').innerText = 'Disconnected';
            document.getElementById('connect').disabled = false;
            document.getElementById('disconnect').disabled = true;
        };
    </script>
</body>
</html>
```

---

## 4. Code Patterns from FastRTC Examples

### 4.1 Basic Handler Pattern

```python
from fastrtc import ReplyOnPause, AudioStream

class SimpleEchoHandler(ReplyOnPause):
    """Simple echo bot for testing"""
    
    async def reply(self, transcript: str) -> str:
        # Return text to speak
        return f"You said: {transcript}"

# Mount to FastAPI
app = FastAPI()
rtc = FastRTC(app)

@rtc.route("/echo")
async def echo_endpoint(audio_stream: AudioStream):
    return SimpleEchoHandler()
```

### 4.2 Audio Streaming Pattern

```python
class StreamingTTSHandler(ReplyOnPause):
    """Stream TTS audio chunk-by-chunk"""
    
    async def reply(self, transcript: str) -> AsyncGenerator[bytes, None]:
        # Generate response text
        response_text = self.generate_response(transcript)
        
        # Synthesize and stream audio
        for audio_chunk in self.synthesize_streaming(response_text):
            yield audio_chunk
            await asyncio.sleep(0.02)  # 20ms chunks
```

### 4.3 State Management Pattern

```python
class StatefulHandler(ReplyOnPause):
    """Handler with conversation state"""
    
    def __init__(self):
        super().__init__()
        self.state = {}  # Session state
        self.turn_count = 0
    
    async def setup(self, session_id: str):
        """Called when connection established"""
        self.state['session_id'] = session_id
        self.state['start_time'] = time.time()
    
    async def reply(self, transcript: str) -> str:
        self.turn_count += 1
        self.state['last_transcript'] = transcript
        return f"Turn {self.turn_count}: {transcript}"
    
    async def cleanup(self):
        """Called when connection closed"""
        duration = time.time() - self.state['start_time']
        logger.info(f"Session ended after {duration:.1f}s, {self.turn_count} turns")
```

---

## 5. Identified Issues and Solutions

### Issue 1: Audio Sample Rate Mismatch

**Problem:** LemonFox outputs 24kHz, WebRTC expects 16kHz  
**Solution:** Use `scipy.signal.resample()` for high-quality resampling

```python
from scipy import signal

def resample_24k_to_16k(audio_24k: np.ndarray) -> np.ndarray:
    """Downsample 24kHz to 16kHz (1.5x compression)"""
    target_length = int(len(audio_24k) * 16000 / 24000)
    return signal.resample(audio_24k, target_length)
```

**Alternative:** Configure LemonFox API for 16kHz output (if supported)

### Issue 2: Session Persistence Across Reconnections

**Problem:** WebRTC connections can drop and reconnect (network issues)  
**Solution:** Store session state in Redis/database with session_id

```python
# Add Redis backend for state persistence
from redis.asyncio import Redis

class LeibnizConversationHandler(ReplyOnPause):
    def __init__(self, redis: Redis):
        super().__init__()
        self.redis = redis
    
    async def setup(self, session_id: str):
        # Restore state if reconnection
        state = await self.redis.get(f"session:{session_id}")
        if state:
            self.conversation_state = json.loads(state)
            logger.info(f"Restored session {session_id}")
        else:
            self.conversation_state = {"turn": 0}
    
    async def cleanup(self):
        # Save state for potential reconnection
        await self.redis.setex(
            f"session:{self.session_id}",
            600,  # 10 minute TTL
            json.dumps(self.conversation_state)
        )
```

### Issue 3: Appointment FSM State Management

**Problem:** FSM state is local to handler instance, lost on reconnection  
**Solution:** Serialize FSM state to Redis between turns

```python
async def save_fsm_state(self):
    """Serialize FSM state for persistence"""
    if self.current_fsm:
        state_dict = {
            'current_state': self.current_fsm.state.value,
            'data': self.current_fsm.data,
            'attempt_counts': self.current_fsm.attempt_counts
        }
        await self.redis.set(
            f"fsm:{self.session_id}",
            json.dumps(state_dict)
        )

async def restore_fsm_state(self):
    """Restore FSM from saved state"""
    state_json = await self.redis.get(f"fsm:{self.session_id}")
    if state_json:
        state_dict = json.loads(state_json)
        self.current_fsm = create_appointment_fsm()
        self.current_fsm.state = AppointmentState(state_dict['current_state'])
        self.current_fsm.data = state_dict['data']
        self.current_fsm.attempt_counts = state_dict['attempt_counts']
```

### Issue 4: Dialogue Archiving in WebRTC Context

**Problem:** Current archiving saves to local filesystem  
**Solution:** Stream archives to S3/cloud storage with session_id prefix

```python
async def archive_dialogue_audio_webrtc(
    self,
    audio_bytes: bytes,
    turn_number: int,
    text: str
):
    """Archive audio to cloud storage (S3/Azure Blob)"""
    import boto3
    
    s3 = boto3.client('s3')
    key = f"leibniz/sessions/{self.session_id}/turn_{turn_number:03d}.wav"
    
    s3.put_object(
        Bucket='leibniz-audio-archive',
        Key=key,
        Body=audio_bytes,
        Metadata={
            'text': text[:1000],
            'turn': str(turn_number),
            'timestamp': str(time.time())
        }
    )
```

### Issue 5: Background Audio Playback Compatibility

**Problem:** Background music (pygame) not applicable to WebRTC  
**Solution:** Remove background audio for WebRTC sessions, or mix client-side

```python
# Option 1: Disable background audio
ENABLE_BACKGROUND_AUDIO = False  # Always false for WebRTC

# Option 2: Send background audio as separate track
@rtc.route("/conversation")
async def conversation_endpoint(audio_stream: AudioStream):
    handler = LeibnizConversationHandler()
    
    # Add background audio track (if needed)
    if ENABLE_BACKGROUND_AUDIO:
        background_track = await audio_stream.add_track("background")
        asyncio.create_task(stream_background_audio(background_track))
    
    return handler
```

### Issue 6: Gemini Live API Session Sharing

**Problem:** Current VAD uses singleton session for local desktop  
**Solution:** Session per WebRTC connection (no sharing)

```python
# Instead of global singleton:
# _leibniz_vad_instance = None

# Use per-session instances:
class LeibnizConversationHandler(ReplyOnPause):
    def __init__(self):
        super().__init__()
        # Create dedicated VAD instance per session
        self.vad_config = LeibnizVADConfig()
        self.vad = LeibnizBidirectionalVAD(config=self.vad_config)
        # No shared state between sessions
```

### Issue 7: Real-Time Latency Optimization

**Problem:** Target <200ms end-to-end latency  
**Solution:** Profile and optimize each stage

**Current Latencies (Desktop):**
- VAD capture: ~100ms (sounddevice startup)
- Gemini transcription: ~50-150ms (streaming)
- Intent classification: ~200ms (Gemini 2.0)
- RAG retrieval: ~300-800ms (FAISS + generation)
- TTS synthesis: ~500-1500ms (LemonFox API)
- Audio playback: ~50ms (local sounddevice)

**Total:** ~1200-2800ms (excluding network)

**WebRTC Targets:**
- VAD capture: ~50ms (WebRTC stream already open)
- Transcription: ~50-150ms (same Gemini Live)
- Intent: ~150ms (with pre-warming)
- RAG: ~200-500ms (with caching + speculative execution)
- TTS: ~300-800ms (with caching)
- Audio streaming: ~50ms (WebRTC)

**Target Total:** ~800-1650ms

**Optimizations:**
1. **Pre-warm models:** Start intent + RAG during VAD capture
2. **Speculative execution:** Predict likely queries, pre-compute responses
3. **Aggressive caching:** Cache frequent questions/responses
4. **Streaming TTS:** Send first audio chunks before full synthesis
5. **Intent fast-path:** Pattern matching for common intents (skip LLM)

```python
async def optimized_reply(self, transcript: str) -> str:
    """Optimized reply with parallel execution"""
    
    # Start intent + RAG in parallel (don't await)
    intent_task = asyncio.create_task(self.classify_intent(transcript))
    rag_task = asyncio.create_task(self.precompute_rag(transcript))
    
    # Wait for intent (fast)
    intent = await intent_task
    
    # Check cache (fastest)
    cached_response = self.cache.get(transcript)
    if cached_response:
        return cached_response
    
    # Use RAG result if ready
    try:
        rag_result = await asyncio.wait_for(rag_task, timeout=0.5)
        return rag_result
    except asyncio.TimeoutError:
        # Fallback to synchronous RAG
        return await self.handle_rag_sync(transcript, intent)
```

---

## 6. Testing Checklist

### 6.1 Functional Tests
- [ ] WebRTC connection establishment
- [ ] Audio streaming (client → server)
- [ ] Audio playback (server → client)
- [ ] Speech transcription accuracy
- [ ] Intent classification correctness
- [ ] RAG query responses
- [ ] Appointment booking flow (full FSM)
- [ ] Barge-in detection and handling
- [ ] Session timeout and cleanup
- [ ] Error handling (network, API failures)

### 6.2 Performance Tests
- [ ] End-to-end latency < 2 seconds
- [ ] First audio chunk < 500ms
- [ ] Concurrent sessions (10+ simultaneous users)
- [ ] Memory usage per session < 100MB
- [ ] Audio quality (no clipping, distortion)
- [ ] Sample rate conversion accuracy

### 6.3 Edge Cases
- [ ] Very long questions (>2 minutes)
- [ ] Rapid barge-ins (user interrupts every word)
- [ ] Network disconnection and reconnection
- [ ] Microphone permission denied
- [ ] Speaker/headphone unavailable
- [ ] Multi-turn appointment booking with corrections
- [ ] Back-to-back questions (no pause)

### 6.4 Browser Compatibility
- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)
- [ ] Mobile Chrome (Android)
- [ ] Mobile Safari (iOS)

---

## 7. Deployment Considerations

### 7.1 Infrastructure Requirements

**Minimum:**
- CPU: 4 cores (2 for FastRTC, 2 for services)
- RAM: 8GB (2GB per concurrent session)
- GPU: None required (TTS via API, no local models)
- Network: 1 Gbps (WebRTC bandwidth: ~128 kbps per session)

**Recommended:**
- CPU: 8 cores
- RAM: 16GB
- GPU: Optional (for future local TTS/STT models)
- Network: 10 Gbps

### 7.2 Scaling Strategy

**Vertical Scaling (Single Server):**
- Handle ~20-50 concurrent sessions
- Simple deployment, no orchestration needed

**Horizontal Scaling (Multiple Servers):**
- Load balancer (NGINX/HAProxy) with WebRTC routing
- Sticky sessions (same user → same server)
- Shared Redis for session state
- Health checks on `/health` endpoint

**Example NGINX Config:**
```nginx
upstream leibniz_webrtc {
    least_conn;  # Route to least busy server
    server 10.0.0.1:8080 max_fails=3 fail_timeout=30s;
    server 10.0.0.2:8080 max_fails=3 fail_timeout=30s;
    server 10.0.0.3:8080 max_fails=3 fail_timeout=30s;
}

server {
    listen 443 ssl http2;
    server_name leibniz.university.edu;
    
    ssl_certificate /etc/ssl/certs/leibniz.crt;
    ssl_certificate_key /etc/ssl/private/leibniz.key;
    
    location /conversation {
        proxy_pass http://leibniz_webrtc;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /health {
        proxy_pass http://leibniz_webrtc;
    }
}
```

### 7.3 Monitoring and Observability

```python
# Add Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

# Metrics
sessions_total = Counter('leibniz_sessions_total', 'Total sessions')
session_duration = Histogram('leibniz_session_duration_seconds', 'Session duration')
active_sessions = Gauge('leibniz_active_sessions', 'Active sessions')
rag_latency = Histogram('leibniz_rag_latency_seconds', 'RAG query latency')
tts_latency = Histogram('leibniz_tts_latency_seconds', 'TTS synthesis latency')

@rtc.route("/conversation")
async def conversation_endpoint(audio_stream: AudioStream):
    sessions_total.inc()
    active_sessions.inc()
    start_time = time.time()
    
    try:
        handler = LeibnizConversationHandler()
        return handler
    finally:
        active_sessions.dec()
        session_duration.observe(time.time() - start_time)
```

**Grafana Dashboard:**
- Active sessions over time
- Average session duration
- P50/P90/P99 latencies (RAG, TTS, E2E)
- Error rates
- Audio quality metrics (sample rate, bit rate)

### 7.4 Security Considerations

**HTTPS/WSS Required:**
```python
# Enforce HTTPS for WebRTC
if not request.url.scheme == "https":
    raise HTTPException(status_code=403, detail="HTTPS required")
```

**Rate Limiting:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/conversation")
@limiter.limit("10/minute")  # 10 sessions per minute per IP
async def conversation_endpoint(request: Request, audio_stream: AudioStream):
    ...
```

**Authentication (Optional):**
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.get("/conversation")
async def conversation_endpoint(
    audio_stream: AudioStream,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify JWT token
    user = await verify_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    handler = LeibnizConversationHandler()
    handler.user_id = user.id
    return handler
```

---

## 8. Migration Path (Desktop → WebRTC)

### Phase 1: Parallel Deployment (Week 1)
- Keep existing desktop app (`leibniz_pro.py`)
- Deploy WebRTC version on separate URL (`/webrtc`)
- Beta test with small user group

### Phase 2: Feature Parity (Week 2-3)
- Ensure all desktop features work in WebRTC
- Match or exceed desktop latency
- Fix issues found in beta testing

### Phase 3: Gradual Rollout (Week 4)
- Redirect 10% of users to WebRTC
- Monitor metrics (latency, errors, user feedback)
- Increase to 50%, then 100%

### Phase 4: Sunset Desktop App (Week 5+)
- Deprecate desktop app
- Redirect all users to WebRTC
- Remove desktop-specific code

---

## 9. Summary and Next Steps

### What We're Building

A **WebRTC-based voice conversation agent** that:
- Accepts audio from web browsers via FastRTC
- Processes speech using existing Gemini Live VAD
- Classifies intent and generates responses (RAG/FSM)
- Synthesizes speech via LemonFox TTS
- Streams audio back to client in real-time
- Handles barge-in, timeouts, and errors gracefully

### Key Advantages Over Desktop

✅ **Accessibility:** No installation, works in any browser  
✅ **Scalability:** Multiple concurrent users on same server  
✅ **Flexibility:** Easy to add features (chat, video, screen sharing)  
✅ **Analytics:** Track user interactions, improve responses  
✅ **Security:** HTTPS/WSS encryption, authentication options  

### Estimated Effort

- **Core Integration:** 5-7 days (1 developer)
- **Testing + Bug Fixes:** 3-5 days
- **Deployment + Monitoring:** 2-3 days
- **Total:** 10-15 days (~2-3 weeks)

### Immediate Next Steps

1. **Install FastRTC:** `pip install fastrtc`
2. **Create handler stub:** `leibniz_fastrtc_handler.py` with basic structure
3. **Test audio conversion:** Verify 24kHz → 16kHz resampling works
4. **Adapt VAD capture:** Replace sounddevice with FastRTC stream
5. **Test E2E locally:** Browser → FastRTC → Gemini → LemonFox → Browser

### Critical Path Items

🔴 **Must Have (MVP):**
- Audio streaming (bidirectional)
- VAD transcription
- Intent classification
- RAG query handling
- TTS synthesis and playback

🟡 **Should Have (Beta):**
- Barge-in detection
- Appointment FSM
- Error handling
- Session persistence

🟢 **Nice to Have (V2):**
- Multi-user support
- Analytics dashboard
- Mobile optimization
- Chat interface (text fallback)

---

## 10. References

### Documentation
- **FastRTC:** https://github.com/cartesia-ai/fastrtc
- **Gemini Live API:** https://ai.google.dev/gemini-api/docs/live
- **LemonFox TTS:** https://www.lemonfox.ai/apis/text-to-speech
- **WebRTC Spec:** https://webrtc.org/

### Existing Codebase
- `leibniz_pro.py` (4431 lines) - Main orchestration
- `leibniz_vad.py` (1200+ lines) - Gemini Live VAD
- `leibniz_tts.py` (2061 lines) - LemonFox TTS
- `leibniz_rag.py` (referenced) - FAISS RAG system
- `leibniz_appointment_fsm.py` (referenced) - Appointment booking

### Similar Projects
- **CartesiaAI's examples:** FastRTC reference implementations
- **SINDH VAD:** Bidirectional VAD inspiration (leibniz_vad.py based on this)
- **TARA pattern:** Conversation loop structure (leibniz_pro.py follows this)

---

**END OF RESEARCH REPORT**

*This plan provides a complete roadmap for FastRTC integration. Execute the phases sequentially, testing thoroughly at each stage. The existing architecture is well-structured for this transition - most work is adapting audio I/O, not rewriting core logic.*
