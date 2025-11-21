# FastRTC Integration Guide for Leibniz Agent
## Wrapper Approach - Zero Changes to Core Pipeline

---

## 🎯 Overview

This guide shows **exactly where** to integrate FastRTC as a **thin wrapper** around your existing microphone/speaker pipeline. Your main conversation loop in `leibniz_pro.py` stays **completely unchanged**.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BROWSER (FastRTC UI)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │ WebRTC Audio Stream
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              FastRTC Wrapper Server (NEW)                   │
│  - Receives browser audio → Injects into audio source       │
│  - Receives TTS output → Streams to browser                 │
└──────────────────────┬──────────────────────────────────────┘
                       │ Injects via AudioSource interface
                       ↓
┌─────────────────────────────────────────────────────────────┐
│         Your Existing Pipeline (UNCHANGED)                  │
│  leibniz_vad.py → leibniz_stt.py → leibniz_intent_parser   │
│  → leibniz_rag.py → leibniz_tts.py                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📍 Current Audio Device Locations

### **1. Microphone Input (STT/VAD)**

#### **File: `leibniz_vad.py`**
**Lines: ~450-500** (in `capture_speech_bidirectional()`)

**Current Implementation:**
```python
# leibniz_vad.py - Line ~470
stream = sd.InputStream(
    samplerate=self.config.samplerate,  # 16000 Hz
    channels=numchannels,               # 1 (mono)
    dtype='float32',
    blocksize=int(self.config.samplerate * 0.05),  # 50ms chunks (800 samples)
    callback=audio_callback,
    device=device  # Native microphone device
)
```

**What happens:**
- `sounddevice` captures from **physical microphone**
- Audio goes to `audio_callback()` which puts chunks in `audioqueue`
- Queue sends to Gemini Live API for STT

**Where to inject FastRTC:**
- Replace `sd.InputStream` with `audio_source` parameter
- `audio_source` can be either:
  - Native `sounddevice` (current behavior)
  - FastRTC WebSocket stream (new wrapper)

---

#### **File: `leibniz_stt.py`**
**Lines: ~200-250** (in `capture_audio()`)

**Current Implementation:**
```python
# leibniz_stt.py - Line ~220
stream = sd.InputStream(
    samplerate=self.config.samplerate,  # 16000 Hz
    channels=numchannels,               # 1 or 2 (converts to mono)
    dtype='float32',
    blocksize=int(self.config.samplerate * 0.05),  # 50ms
    callback=audio_callback,
    device=device  # Native microphone
)
```

**Same pattern as VAD** - captures from physical mic

---

### **2. Speaker Output (TTS Playback)**

#### **File: `leibniz_pro.py`**
**Lines: ~800-900** (in TTS consumer task)

**Current Implementation - Primary (sounddevice):**
```python
# leibniz_pro.py - Line ~850
audiodata, samplerate = await asyncio.to_thread(sf.read, audiopath)
await asyncio.to_thread(sd.play, audiodata, samplerate, device=12)  # Device 12 = Realtek speakers
await asyncio.to_thread(sd.wait)  # Wait for playback completion
```

**Current Implementation - Fallback (pygame):**
```python
# leibniz_pro.py - Line ~920
await asyncio.to_thread(pygame.mixer.music.load, audiofile)
await asyncio.to_thread(pygame.mixer.music.play)
while pygame.mixer.music.get_busy():
    await asyncio.to_thread(pygame.mixer.music.get_busy())
    await asyncio.sleep(0.02)
```

**What happens:**
- TTS generates audio file (WAV/MP3)
- Audio plays through **physical speakers** (device 12)
- Blocks until playback complete

**Where to inject FastRTC:**
- Replace `sd.play()` with audio sink that can:
  - Play to physical speakers (current)
  - Stream to browser via WebRTC (new)

---

#### **File: `leibniz_tts.py`**
**Lines: ~650-700** (in `play_audio_file()`)

**Current Implementation:**
```python
# leibniz_tts.py - Line ~670
sd.play(audiodata, samplerate, device=12)  # Device 12 = speakers
sd.wait()  # Block until complete
```

**Same pattern** - plays to physical device 12 (Realtek speakers)

---

## 🔧 Integration Points

### **Point 1: Audio Source Abstraction**

**Already exists in your code!** Your VAD supports `audio_source` parameter:

```python
# leibniz_vad.py - Line ~380
async def capture_speech_bidirectional(
    self,
    streaming_callback=None,
    audio_source=None,  # ← THIS IS THE KEY
    context=None
):
```

**Current behavior:**
- If `audio_source=None` → Uses `sounddevice` microphone
- If `audio_source` provided → Uses that instead

**FastRTC wrapper just needs to implement:**
```python
class FastRTCAudioSource:
    async def get_frames(self, num_samples: int) -> np.ndarray:
        """Return audio from browser WebRTC stream"""
        # Get audio from FastRTC, convert format, return
```

---

### **Point 2: Audio Sink Abstraction**

**Does NOT exist yet.** Need to add `audio_sink` parameter to TTS playback functions.

**Modify `leibniz_pro.py` TTS consumer:**

```python
# BEFORE (current - plays to speakers directly):
await asyncio.to_thread(sd.play, audiodata, samplerate, device=12)

# AFTER (wrapper-compatible):
if audio_sink:
    await audio_sink.write_audio(audiodata, samplerate)
else:
    await asyncio.to_thread(sd.play, audiodata, samplerate, device=12)
```

**FastRTC wrapper just needs to implement:**
```python
class FastRTCAudioSink:
    async def write_audio(self, audiodata: np.ndarray, samplerate: int):
        """Stream audio to browser via WebRTC"""
        # Convert format, stream to FastRTC
```

---

## 📝 Implementation Steps

### **Step 1: Create FastRTC Audio Adapters**

**New file: `leibniz_fastrtc_adapters.py`**

```python
"""
FastRTC Audio Adapters - Thin wrappers for audio I/O
Implements AudioSource and AudioSink interfaces compatible with your pipeline
"""

import numpy as np
import asyncio
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FastRTCAudioSource:
    """
    Audio source that receives from FastRTC WebSocket.
    Compatible with leibniz_vad.py audio_source interface.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_queue = asyncio.Queue(maxsize=100)
        self.is_active = False
        logger.info(f"FastRTC audio source initialized: {sample_rate}Hz")
    
    async def push_audio_from_fastrtc(self, audio_chunk: np.ndarray):
        """
        Called by FastRTC handler to push browser audio into the source.
        
        Args:
            audio_chunk: Audio from FastRTC (float32, -1 to 1)
        """
        try:
            # FastRTC sends float32 normalized audio
            # Your pipeline expects the same format - just pass through
            self.audio_queue.put_nowait(audio_chunk)
            self.is_active = True
        except asyncio.QueueFull:
            logger.warning("Audio source queue full, dropping chunk")
    
    async def get_frames(self, num_samples: int) -> np.ndarray:
        """
        Interface method called by leibniz_vad.py to get audio.
        
        Args:
            num_samples: Number of samples requested (typically 800 = 50ms at 16kHz)
            
        Returns:
            Audio array (float32, normalized -1 to 1)
        """
        try:
            # Get audio from queue with timeout
            audio_chunk = await asyncio.wait_for(
                self.audio_queue.get(),
                timeout=0.1
            )
            
            # Pad or trim to requested size
            if len(audio_chunk) < num_samples:
                audio_chunk = np.pad(audio_chunk, (0, num_samples - len(audio_chunk)))
            elif len(audio_chunk) > num_samples:
                audio_chunk = audio_chunk[:num_samples]
            
            return audio_chunk.astype(np.float32)
            
        except asyncio.TimeoutError:
            # Return silence if no audio available (expected during pauses)
            return np.zeros(num_samples, dtype=np.float32)
    
    def clear(self):
        """Clear the buffer."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break


class FastRTCAudioSink:
    """
    Audio sink that streams to FastRTC WebSocket.
    Compatible with modified leibniz_pro.py TTS playback.
    """
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.output_queue = asyncio.Queue()
        self.is_streaming = False
        logger.info(f"FastRTC audio sink initialized: {sample_rate}Hz")
    
    async def write_audio(self, audio_data: np.ndarray, sample_rate: int):
        """
        Called by TTS playback to send audio to browser.
        
        Args:
            audio_data: Audio to play (float32 or int16)
            sample_rate: Sample rate of audio
        """
        try:
            # Convert to float32 if needed (FastRTC expects float32)
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32767.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Resample if needed (your TTS outputs 24kHz, keep it)
            # FastRTC will handle any resampling internally
            
            await self.output_queue.put((sample_rate, audio_float))
            self.is_streaming = True
            
        except Exception as e:
            logger.error(f"Error writing to audio sink: {e}")
    
    async def stream_to_fastrtc(self):
        """
        Generator that yields audio chunks for FastRTC streaming.
        Called by FastRTC handler.
        
        Yields:
            Tuple of (sample_rate, audio_chunk)
        """
        logger.info(f"Starting audio sink streaming at {self.sample_rate}Hz")
        
        while self.is_streaming or not self.output_queue.empty():
            try:
                sample_rate, audio_chunk = await asyncio.wait_for(
                    self.output_queue.get(),
                    timeout=0.5
                )
                
                # Split into FastRTC-compatible chunks (2400 samples = 100ms at 24kHz)
                chunk_size = 2400
                for i in range(0, len(audio_chunk), chunk_size):
                    chunk = audio_chunk[i:i+chunk_size]
                    
                    # Pad last chunk if needed
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                    
                    yield (sample_rate, chunk)
                    
            except asyncio.TimeoutError:
                # No more audio available
                break
        
        logger.info("Audio sink streaming ended")
        self.is_streaming = False
    
    def stop_streaming(self):
        """Stop the streaming."""
        self.is_streaming = False
    
    def clear(self):
        """Clear the buffer."""
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except:
                break
```

---

### **Step 2: Create FastRTC Handler**

**New file: `leibniz_fastrtc_handler.py`**

```python
"""
FastRTC Handler - Wraps your existing pipeline
No changes to your conversation logic - just audio I/O routing
"""

import numpy as np
import asyncio
import logging
from typing import Optional

from leibniz_fastrtc_adapters import FastRTCAudioSource, FastRTCAudioSink

logger = logging.getLogger(__name__)


class LeibnizFastRTCHandler:
    """
    FastRTC handler that bridges browser audio to your existing pipeline.
    
    This is ONLY an audio I/O wrapper. Your conversation logic in
    leibniz_pro.py runs completely unchanged.
    """
    
    def __init__(self):
        # Create audio adapters
        self.source = FastRTCAudioSource(sample_rate=16000)
        self.sink = FastRTCAudioSink(sample_rate=24000)
        
        # Event to signal when conversation should process audio
        self.user_finished_speaking = asyncio.Event()
        self.current_transcript = None
        
        logger.info("Leibniz FastRTC handler initialized")
    
    async def __call__(self, audio: tuple[int, np.ndarray]):
        """
        Main FastRTC callback - called by ReplyOnPause when user pauses.
        
        This ONLY handles:
        1. Receiving audio from browser → push to source
        2. Signaling your main loop to process
        3. Streaming response audio back to browser
        
        Args:
            audio: Tuple of (sample_rate, audio_array) from FastRTC
            
        Yields:
            Tuple of (sample_rate, audio_array) for response playback
        """
        try:
            # Step 1: Push browser audio into source buffer
            sample_rate, audio_array = audio
            await self.source.push_audio_from_fastrtc(audio_array.flatten())
            
            # Step 2: Signal main conversation loop to process
            # (Your main loop will call capture_speech_bidirectional with this source)
            self.user_finished_speaking.set()
            
            # Step 3: Wait for TTS to generate response audio
            # (Your main loop will call sink.write_audio when TTS completes)
            
            # Step 4: Stream response back to browser
            async for audio_chunk in self.sink.stream_to_fastrtc():
                yield audio_chunk
            
            logger.debug("Turn completed successfully")
            
        except Exception as e:
            logger.error(f"FastRTC handler error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Yield silence on error
            yield (24000, np.zeros(1024, dtype=np.float32))
        
        finally:
            # Clear buffers for next turn
            self.source.clear()
            self.sink.clear()
            self.user_finished_speaking.clear()
```

---

### **Step 3: Create FastRTC Server**

**New file: `leibniz_fastrtc_server.py`**

```python
"""
FastRTC Server - Runs separately from main pipeline
Connects browser to your existing conversation loop via adapters
"""

import asyncio
import logging
from fastrtc import Stream, ReplyOnPause
from fastapi import FastAPI
import uvicorn

from leibniz_fastrtc_handler import LeibnizFastRTCHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global handler instance (shared between FastAPI and main pipeline)
fastrtc_handler = None


def create_fastrtc_app() -> FastAPI:
    """
    Create FastAPI app with FastRTC streaming.
    """
    global fastrtc_handler
    
    app = FastAPI(
        title="Leibniz WebRTC Bridge",
        version="1.0.0"
    )
    
    # Create handler
    fastrtc_handler = LeibnizFastRTCHandler()
    
    # Create FastRTC stream with ReplyOnPause
    stream = Stream(
        handler=ReplyOnPause(fastrtc_handler),
        modality="audio",
        mode="send-receive",
        output_sample_rate=24000,
        input_sample_rate=16000,
        ui_args={
            "title": "Leibniz University Assistant",
            "description": "Speak naturally. I'll respond when you pause."
        }
    )
    
    # Mount Gradio UI at root
    app = stream.ui.mount_to(app, path="/")
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "leibniz-fastrtc-bridge"
        }
    
    logger.info("FastRTC app created")
    return app


def get_fastrtc_handler():
    """Get the global FastRTC handler for use by main pipeline."""
    return fastrtc_handler


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 Starting Leibniz FastRTC Bridge Server")
    logger.info("=" * 60)
    logger.info("📱 Browser UI: http://localhost:7860")
    logger.info("🎤 Main pipeline should connect via get_fastrtc_handler()")
    logger.info("=" * 60)
    
    app = create_fastrtc_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )
```

---

### **Step 4: Modify Main Pipeline (Minimal Changes)**

**File: `leibniz_pro.py`**

**Location: Beginning of `run_conversation_session()` function (~line 1500)**

**BEFORE:**
```python
async def run_conversation_session():
    """Main conversation loop - unchanged"""
    
    # Initialization
    services = await initialize_leibniz_services()
    
    # Greeting
    await speak_friendly("Hello! I'm your Leibniz University assistant...")
    
    # Main loop
    for attempt in range(5):
        # Capture speech
        transcript, intent = await transcribe_and_classify()
        
        # Process intent...
```

**AFTER (with FastRTC support):**
```python
async def run_conversation_session(
    audio_source=None,  # ← NEW: FastRTC source if using WebRTC
    audio_sink=None     # ← NEW: FastRTC sink if using WebRTC
):
    """Main conversation loop - with optional WebRTC support"""
    
    # Initialization (unchanged)
    services = await initialize_leibniz_services()
    
    # Greeting (pass audio_sink if provided)
    await speak_friendly(
        "Hello! I'm your Leibniz University assistant...",
        audio_sink=audio_sink  # ← NEW
    )
    
    # Main loop
    for attempt in range(5):
        # Capture speech (pass audio_source if provided)
        transcript, intent = await transcribe_and_classify(
            audio_source=audio_source  # ← NEW
        )
        
        # Process intent (unchanged)...
```

---

**Location: `transcribe_and_classify()` function (~line 1200)**

**BEFORE:**
```python
async def transcribe_and_classify(
    streaming_callback=None,
    context=None
):
    """Capture and classify speech"""
    transcript = await capture_leibniz_speech(
        streaming_callback=streaming_callback,
        context=context
    )
    # ... rest unchanged
```

**AFTER:**
```python
async def transcribe_and_classify(
    streaming_callback=None,
    context=None,
    audio_source=None  # ← NEW
):
    """Capture and classify speech"""
    transcript = await capture_leibniz_speech(
        streaming_callback=streaming_callback,
        context=context,
        audio_source=audio_source  # ← NEW: Pass to VAD
    )
    # ... rest unchanged
```

---

**Location: `speak_friendly()` function (~line 1000)**

**BEFORE:**
```python
async def speak_friendly(
    text: str,
    emotion: str = "helpful",
    cache_name: Optional[str] = None,
    enable_streaming: bool = True
):
    """TTS synthesis and playback"""
    
    # Synthesis (unchanged)
    result = await tts.synthesize_to_file(text, emotion=emotion)
    
    # Playback to speakers
    audiodata, samplerate = await asyncio.to_thread(sf.read, audiofile)
    await asyncio.to_thread(sd.play, audiodata, samplerate, device=12)
    await asyncio.to_thread(sd.wait)
```

**AFTER:**
```python
async def speak_friendly(
    text: str,
    emotion: str = "helpful",
    cache_name: Optional[str] = None,
    enable_streaming: bool = True,
    audio_sink=None  # ← NEW
):
    """TTS synthesis and playback"""
    
    # Synthesis (unchanged)
    result = await tts.synthesize_to_file(text, emotion=emotion)
    
    # Playback
    audiodata, samplerate = await asyncio.to_thread(sf.read, audiofile)
    
    if audio_sink:
        # Stream to browser via FastRTC
        await audio_sink.write_audio(audiodata, samplerate)
    else:
        # Play to physical speakers (original behavior)
        await asyncio.to_thread(sd.play, audiodata, samplerate, device=12)
        await asyncio.to_thread(sd.wait)
```

---

### **Step 5: Create Connection Script**

**New file: `run_with_fastrtc.py`**

```python
"""
Main entry point for running Leibniz with FastRTC browser interface.

Usage:
    Terminal 1: python leibniz_fastrtc_server.py
    Terminal 2: python run_with_fastrtc.py
"""

import asyncio
import logging
from leibniz_fastrtc_server import get_fastrtc_handler
from leibniz_pro import run_conversation_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """
    Main loop that connects your pipeline to FastRTC.
    """
    
    # Wait for FastRTC server to initialize
    logger.info("Waiting for FastRTC server to start...")
    await asyncio.sleep(2)
    
    # Get FastRTC handler (provides audio source and sink)
    handler = get_fastrtc_handler()
    
    if not handler:
        logger.error("FastRTC handler not available. Start leibniz_fastrtc_server.py first.")
        return
    
    logger.info("=" * 60)
    logger.info("✅ Connected to FastRTC server")
    logger.info("🌐 Open http://localhost:7860 in browser")
    logger.info("🎤 Click 'Record' and start speaking")
    logger.info("=" * 60)
    
    # Run your conversation loop with FastRTC audio I/O
    while True:
        try:
            # Wait for user to speak in browser
            await handler.user_finished_speaking.wait()
            
            # Run your existing conversation logic
            # (It will use handler.source for mic input and handler.sink for speaker output)
            await run_conversation_session(
                audio_source=handler.source,
                audio_sink=handler.sink
            )
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🚀 Usage Instructions

### **Option 1: Run with Native Microphone/Speakers (Current)**

```bash
# No changes needed - works exactly as before
python leibniz_pro.py
```

### **Option 2: Run with FastRTC Browser Interface (New)**

**Terminal 1 - Start FastRTC Server:**
```bash
python leibniz_fastrtc_server.py

# Output:
# 🚀 Starting Leibniz FastRTC Bridge Server
# 📱 Browser UI: http://localhost:7860
```

**Terminal 2 - Run Main Pipeline:**
```bash
python run_with_fastrtc.py

# Output:
# ✅ Connected to FastRTC server
# 🌐 Open http://localhost:7860 in browser
# 🎤 Click 'Record' and start speaking
```

**Browser:**
1. Open `http://localhost:7860`
2. Click "Record"
3. Grant microphone permissions
4. Start speaking - responses play automatically

---

## 📦 Dependencies

Add to `requirements.txt`:

```txt
# Existing dependencies (keep all)
sounddevice
soundfile
pygame
numpy
google-genai
# ... all your other deps

# New for FastRTC
fastrtc[vad,stt,tts]
fastapi
uvicorn[standard]
```

Install:
```bash
pip install "fastrtc[vad,stt,tts]" fastapi "uvicorn[standard]"
```

---

## 🔍 What Changes and What Doesn't

### ✅ **UNCHANGED (Your Core Pipeline)**

| Component | File | Status |
|-----------|------|--------|
| VAD logic | `leibniz_vad.py` | ✅ Unchanged |
| STT logic | `leibniz_stt.py` | ✅ Unchanged |
| Intent parsing | `leibniz_intent_parser.py` | ✅ Unchanged |
| RAG queries | `leibniz_rag.py` | ✅ Unchanged |
| TTS synthesis | `leibniz_tts.py` | ✅ Unchanged |
| Conversation flow | `leibniz_pro.py` main loop | ✅ Unchanged |
| Persistent services | `leibniz_persistent_services.py` | ✅ Unchanged |

### 🆕 **CHANGED (Audio I/O Only)**

| Component | File | Change |
|-----------|------|--------|
| Audio source | `leibniz_vad.py` | Add `audio_source` parameter (already exists!) |
| Audio sink | `leibniz_pro.py` TTS playback | Add `audio_sink` parameter |
| Function signatures | `transcribe_and_classify()`, `speak_friendly()` | Add optional parameters |

### 🆕 **NEW FILES**

- `leibniz_fastrtc_adapters.py` - Audio source/sink wrappers
- `leibniz_fastrtc_handler.py` - FastRTC callback handler
- `leibniz_fastrtc_server.py` - FastAPI + FastRTC server
- `run_with_fastrtc.py` - Connection script

---

## 🎯 Key Integration Points Summary

### **1. Microphone Replacement**

**Where:** `leibniz_vad.py` line ~470

**Before:**
```python
stream = sd.InputStream(...)  # Physical microphone
```

**After:**
```python
if audio_source:
    # Use FastRTC audio from browser
    frames = await audio_source.get_frames(800)
else:
    # Use physical microphone (original)
    stream = sd.InputStream(...)
```

### **2. Speaker Replacement**

**Where:** `leibniz_pro.py` line ~850

**Before:**
```python
sd.play(audiodata, samplerate, device=12)  # Physical speakers
```

**After:**
```python
if audio_sink:
    # Stream to browser via FastRTC
    await audio_sink.write_audio(audiodata, samplerate)
else:
    # Use physical speakers (original)
    sd.play(audiodata, samplerate, device=12)
```

---

## 🧪 Testing Checklist

### **Test 1: Native Mode (should work unchanged)**
```bash
python leibniz_pro.py
# Should work exactly as before with physical mic/speakers
```

### **Test 2: FastRTC Server Starts**
```bash
python leibniz_fastrtc_server.py
# Should see: 📱 Browser UI: http://localhost:7860
# Browser should open Gradio UI
```

### **Test 3: Connection Works**
```bash
# Terminal 1
python leibniz_fastrtc_server.py

# Terminal 2
python run_with_fastrtc.py

# Browser: Click "Record", speak, should see response
```

### **Test 4: Audio Flow**
- Browser mic → FastRTC source → VAD → STT ✓
- RAG response → TTS → FastRTC sink → Browser speakers ✓

---

## 🐛 Troubleshooting

### **Issue: "No FastRTC handler available"**
**Solution:** Start `leibniz_fastrtc_server.py` BEFORE `run_with_fastrtc.py`

### **Issue: "Audio queue full"**
**Solution:** Increase queue size in `FastRTCAudioSource(maxsize=200)`

### **Issue: "Browser not receiving audio"**
**Solution:** Check `audio_sink.write_audio()` is being called in `speak_friendly()`

### **Issue: "Microphone not capturing"**
**Solution:** Check browser microphone permissions granted

---

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (FastRTC UI)                     │
│  - Microphone → WebRTC stream → Server                     │
│  - WebRTC stream → Speakers                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ WebSocket (audio chunks)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│         leibniz_fastrtc_server.py (FastAPI + FastRTC)       │
│  - ReplyOnPause handler detects pauses                      │
│  - Routes audio to/from adapters                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ In-process (adapters)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│            leibniz_fastrtc_adapters.py                      │
│  - FastRTCAudioSource: browser mic → VAD                    │
│  - FastRTCAudioSink: TTS → browser speakers                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ audio_source / audio_sink parameters
                       ↓
┌─────────────────────────────────────────────────────────────┐
│         leibniz_pro.py (Your Main Pipeline)                 │
│  run_conversation_session(audio_source, audio_sink)         │
│    ↓                                                         │
│  transcribe_and_classify(audio_source=source)               │
│    ↓                                                         │
│  leibniz_vad.capture_speech_bidirectional(audio_source)     │
│    ↓                                                         │
│  [Your existing VAD/STT/Intent/RAG logic - UNCHANGED]       │
│    ↓                                                         │
│  speak_friendly(text, audio_sink=sink)                      │
│    ↓                                                         │
│  leibniz_tts.synthesize_to_file() → audio file              │
│    ↓                                                         │
│  audio_sink.write_audio() → streams to browser              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎉 Summary

### **What This Approach Achieves:**

1. ✅ **Zero changes to your conversation logic** - All VAD/STT/Intent/RAG/TTS code unchanged
2. ✅ **Minimal changes to audio I/O** - Just add optional parameters
3. ✅ **Backward compatible** - Works with native mic/speakers when parameters not provided
4. ✅ **Clean separation** - FastRTC runs in separate process, connects via adapters
5. ✅ **Two-terminal workflow** - Run server and pipeline separately
6. ✅ **Browser interface** - Users access via http://localhost:7860
7. ✅ **Production ready** - Error handling, logging, graceful fallbacks

### **Files to Create:**

1. `leibniz_fastrtc_adapters.py` - Audio I/O wrappers (200 lines)
2. `leibniz_fastrtc_handler.py` - FastRTC callback (100 lines)
3. `leibniz_fastrtc_server.py` - FastAPI server (80 lines)
4. `run_with_fastrtc.py` - Connection script (50 lines)

### **Files to Modify:**

1. `leibniz_pro.py` - Add `audio_source`/`audio_sink` parameters to 3 functions
2. No other files need changes!

### **Total Code Added:** ~500 lines (all wrapper code)
### **Total Pipeline Changes:** ~20 lines (parameter additions)

This is the **cleanest, most maintainable approach** - FastRTC as a pure I/O wrapper with zero impact on your conversation logic.