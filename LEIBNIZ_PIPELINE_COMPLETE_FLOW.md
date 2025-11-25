# Leibniz Pro Pipeline - Complete Data Flow Documentation

**Version**: 1.0  
**Last Updated**: 2024  
**Purpose**: Complete trace of data flow from WebRTC/FastRTC browser integration through entire pipeline to TTS output

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [FastRTC/WebRTC Integration Layer](#fastrtcwebrtc-integration-layer)
3. [Audio Input Flow (Browser → STT)](#audio-input-flow-browser--stt)
4. [Intent Classification Flow](#intent-classification-flow)
5. [Response Generation Flow (RAG/Appointment)](#response-generation-flow-ragappointment)
6. [TTS Synthesis & Streaming Flow](#tts-synthesis--streaming-flow)
7. [Audio Output Flow (TTS → Browser)](#audio-output-flow-tts--browser)
8. [Conversation Loop & Cycle Management](#conversation-loop--cycle-management)
9. [Message Types & Data Structures](#message-types--data-structures)
10. [Error Handling & Fallbacks](#error-handling--fallbacks)
11. [Key Functions & Classes Reference](#key-functions--classes-reference)
12. [Performance Characteristics](#performance-characteristics)

---

## Architecture Overview

The Leibniz Pro pipeline is a voice-first conversational AI system that processes real-time audio streams from a browser via WebRTC/FastRTC, performs speech-to-text, intent classification, knowledge retrieval, and text-to-speech synthesis, then streams audio responses back to the browser.

### High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (WebRTC)                           │
│  - Microphone Capture (float32, 16kHz/48kHz)                 │
│  - Audio Playback (int16, 24kHz)                            │
└───────────────┬───────────────────────▲──────────────────────┘
                │                       │
                │ WebRTC Stream         │ WebRTC Stream
                │ (Audio Chunks)        │ (Audio Chunks)
                ▼                       │
┌─────────────────────────────────────────────────────────────┐
│         FastRTC Integration Layer                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ leibniz_fastrtc_server.py                             │  │
│  │  - Creates Gradio WebRTC interface                    │  │
│  │  - Manages FastRTC Stream with ReplyOnPause           │  │
│  │  - Port: 7860 (default)                              │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ leibniz_fastrtc_handler.py                            │  │
│  │  - LeibnizFastRTCHandler class                        │  │
│  │  - __call__() method (ReplyOnPause callback)          │  │
│  │  - Coordinates audio routing                            │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ leibniz_fastrtc_wrapper.py                           │  │
│  │  - LeibnizFastRTCStreamHandler (AsyncStreamHandler)   │  │
│  │  - Alternative implementation                         │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────┬───────────────────────▲──────────────────────┘
                │                       │
                │                       │
        ┌───────▼──────────┐   ┌───────┴──────────┐
        │ FastRTCAudioSource│   │ FastRTCAudioSink │
        │  (Input Adapter)  │   │ (Output Adapter) │
        │  - 16kHz          │   │  - 24kHz         │
        │  - asyncio.Queue  │   │  - asyncio.Queue │
        └───────┬───────────┘   └───────▲──────────┘
                │                       │
                │                       │
┌───────────────▼───────────────────────┴──────────────────────┐
│           Main Pipeline (leibniz_pro.py)                       │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ PHASE 1: Audio Capture & Transcription                  │  │
│  │                                                         │  │
│  │ capture_and_transcribe()                               │  │
│  │   ↓                                                     │  │
│  │ capture_leibniz_speech()                               │  │
│  │   ↓                                                     │  │
│  │ leibniz_vad.py: capture_speech_bidirectional()         │  │
│  │   ↓                                                     │  │
│  │ audio_source.get_frames() → Gemini Live VAD/STT        │  │
│  │   ↓                                                     │  │
│  │ Transcript String                                       │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                    │
│                          ▼                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ PHASE 2: Intent Classification                         │  │
│  │                                                         │  │
│  │ transcribe_and_classify()                              │  │
│  │   ↓                                                     │  │
│  │ extract_semantic_context() (<5ms)                       │  │
│  │   ↓                                                     │  │
│  │ classify_intent() (pattern <10ms | LLM 200-800ms)     │  │
│  │   ↓                                                     │  │
│  │ (TranscriptMessage, IntentMessage)                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                    │
│                          ▼                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ PHASE 3: Response Generation                          │  │
│  │                                                         │  │
│  │ Route based on intent:                                 │  │
│  │                                                         │  │
│  │ IF intent == "RAG_QUERY":                             │  │
│  │   handle_rag_query()                                   │  │
│  │     - Check cache (1-5ms)                             │  │
│  │     - Check speculative (10-50ms)                     │  │
│  │     - Execute RAG (800-2000ms)                        │  │
│  │     - Stream to TTS queue                             │  │
│  │     → RAGMessage                                      │  │
│  │                                                         │  │
│  │ IF intent == "APPOINTMENT_SCHEDULING":                │  │
│  │   handle_appointment_booking()                         │  │
│  │     - Create FSM                                       │  │
│  │     - Loop: Capture → Process → Speak                 │  │
│  │     → Booking Data                                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                    │
│                          ▼                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ PHASE 4: TTS Synthesis & Streaming                    │  │
│  │                                                         │  │
│  │ stream_rag_to_tts()                                    │  │
│  │   ↓                                                     │  │
│  │ _tts_streaming_queue (sentence queue)                  │  │
│  │   ↓                                                     │  │
│  │ consume_tts_streaming_queue()                          │  │
│  │   ↓                                                     │  │
│  │ 2-Slot Pipeline:                                       │  │
│  │   - Synthesize N+1 while playing N                     │  │
│  │   - LRU cache deduplication                            │  │
│  │   - Barge-in detection                                 │  │
│  │   ↓                                                     │  │
│  │ IF audio_sink provided:                                │  │
│  │   audio_sink.write_audio() → FastRTC                   │  │
│  │ ELSE:                                                  │  │
│  │   Local playback (pygame/sounddevice)                  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                │                       │
                │                       │
        ┌───────┴──────────┐   ┌───────┴──────────┐
        │ FastRTCAudioSink │   │ FastRTCAudioSink │
        │  .write_audio()  │   │ .stream_to_fastrtc()│
        └───────┬──────────┘   └───────▲──────────┘
                │                       │
                │                       │
        ┌───────▼───────────────────────┴──────────┐
        │   FastRTC Handler.emit() / yield()       │
        │   Streams audio chunks to browser        │
        └───────┬──────────────────────────────────┘
                │
                ▼
        ┌──────────────────┐
        │   Browser        │
        │   (Playback)     │
        └──────────────────┘
```

---

## FastRTC/WebRTC Integration Layer

### Entry Point: FastRTC Server

**File**: `leibniz_fastrtc_server.py`  
**Function**: `create_fastrtc_app()`  
**Line**: 173  
**Port**: 7860 (default)

#### Server Initialization Flow

```python
def create_fastrtc_app() -> FastAPI:
    """
    Creates FastAPI app with FastRTC WebRTC stream.
    
    Flow:
    1. Initialize FastRTC handler with audio adapters
    2. Create Stream with ReplyOnPause wrapper
    3. Configure Gradio UI
    4. Return app instance
    """
    # Create handler instance
    handler = LeibnizFastRTCHandler()
    
    # Create FastRTC stream
    stream = Stream(
        handler=ReplyOnPause(handler),
        modality="audio",
        mode="send-receive",
        ui_args={
            "title": "Leibniz University Customer Service",
            "description": "Voice conversation with AI assistant"
        }
    )
    
    return stream.ui
```

**Key Components**:
- `ReplyOnPause`: Wraps handler to trigger on user speech pause
- `Stream`: FastRTC stream manager
- `Gradio UI`: Browser interface for WebRTC connection

**Additional Functions**:
- `run_conversation_loop()` (Line 229): Main conversation coordination loop
- `play_intro_via_sink()` (Line 278): Plays intro greeting via audio sink

### FastRTC Handler

**File**: `leibniz_fastrtc_handler.py`  
**Class**: `LeibnizFastRTCHandler`  
**Line**: 22

#### Handler Initialization

**Method**: `__init__()` (Line 40)

```python
class LeibnizFastRTCHandler:
    def __init__(self):
        # Audio adapters
        self.source = FastRTCAudioSource(sample_rate=16000)
        self.sink = FastRTCAudioSink(sample_rate=24000)
        
        # Coordination events
        self.user_started_speaking = asyncio.Event()
        self.user_finished_speaking = asyncio.Event()
        
        # Session state
        self.is_conversation_active = False
```

#### Main Callback Method

**Method**: `__call__(self, audio: tuple[int, np.ndarray])`  
**Line**: 62

**Invoked by**: FastRTC `ReplyOnPause` when user pauses speaking

**Complete Flow**:

```python
def __call__(self, audio: tuple[int, np.ndarray]):
    """
    FastRTC callback - complete conversation turn handler.
    
    Args:
        audio: Tuple of (sample_rate: int, audio_array: np.ndarray)
               Audio from browser microphone (float32, normalized -1.0 to 1.0)
    
    Yields:
        Tuples of (sample_rate: int, audio_chunk: np.ndarray)
        Audio chunks for browser playback (int16, 24kHz)
    """
    try:
        # STEP 1: Receive browser audio
        if audio is not None:
            sample_rate, audio_array = audio
            
            # Normalize audio format
            audio_flat = audio_array.flatten()
            
            # Push to source adapter (for VAD pipeline)
            self.source.push_audio_from_fastrtc(audio_flat)
            logger.debug("Pushed browser audio to source buffer")
            
            # STEP 2: Signal main conversation loop
            # This triggers run_conversation_session() to process the audio
            self.user_finished_speaking.set()
            self.user_started_speaking.set()
            logger.debug("Signaled main loop to process audio")
        
        # STEP 3: Stream TTS response back to browser
        # The main loop will call sink.write_audio() when TTS completes
        # This generator yields audio chunks as they become available
        for sample_rate, audio_chunk in self.sink.stream_to_fastrtc():
            yield (sample_rate, audio_chunk)
            logger.debug(f"Yielded audio chunk: {len(audio_chunk)} samples")
        
        logger.debug("Turn completed successfully")
        
    except Exception as e:
        logger.error(f"FastRTC handler error: {e}")
        # Yield silence on error to prevent browser hang
        yield (24000, np.zeros(1024, dtype=np.float32))
    
    finally:
        # Clear buffers between turns
        self.source.clear()
        self.sink.clear()
```

**Key Points**:
- Receives audio as `(sample_rate, audio_array)` tuple
- Audio array is float32, normalized (-1.0 to 1.0)
- Pushes audio to `FastRTCAudioSource` queue
- Signals conversation loop via `asyncio.Event`
- Yields audio chunks from `FastRTCAudioSink` queue
- Handles errors gracefully with silence fallback

### FastRTC Wrapper (Alternative Implementation)

**File**: `leibniz_fastrtc_wrapper.py`  
**Class**: `LeibnizFastRTCStreamHandler`  
**Line**: 65

Alternative implementation using `AsyncStreamHandler` pattern:

**Initialization** (Line 75):
```python
def __init__(self):
    self.audio_source = FastRTCAudioSource(sample_rate=16000)
    self.audio_sink = FastRTCAudioSink(sample_rate=24000)
    self.is_conversation_active = False
```

**Receive Method** (Line 127):
```python
async def receive(self, audio: tuple) -> None:
    """
    Receive audio from browser and push to VAD pipeline.
    
    Called continuously as audio chunks arrive from browser.
    """
    sample_rate, audio_array = audio
    
    # Normalize format
    audio_array = audio_array.squeeze().astype(np.float32)
    
    # Normalize amplitude if needed
    max_val = np.max(np.abs(audio_array))
    if max_val > 1.0:
        audio_array = audio_array / 32767.0
    
    # Push to audio source
    await self.audio_source._async_push(audio_array)
```

**Emit Method** (Line 180):
```python
async def emit(self):
    """
    Emit TTS audio chunks to browser.
    
    Called continuously to stream audio to browser.
    Returns: (sample_rate, audio_chunk) tuple
    """
    try:
        # Get audio from sink queue
        audio_chunk = await self.audio_sink.output_queue.get()
        return (24000, audio_chunk)
    except asyncio.TimeoutError:
        # No audio available, return silence
        return (24000, np.zeros((1, 2400), dtype=np.int16))
```

**Differences from Handler**:
- Uses `AsyncStreamHandler` base class
- `receive()` called continuously (not just on pause)
- `emit()` called continuously (not generator-based)
- Better for real-time bidirectional streaming

**Additional Methods**:
- `start_up()` (Line 98): Initializes conversation session
- `_run_conversation_session()` (Line 243): Runs conversation with adapters
- `_consume_tts_for_fastrtc()` (Line 267): Consumes TTS queue for FastRTC

---

## Audio Input Flow (Browser → STT)

### Step 1: Browser Audio Capture

**Location**: Browser (WebRTC API)

**Process**:
1. Browser captures microphone audio via `getUserMedia()`
2. Formats audio as float32 normalized (-1.0 to 1.0)
3. Sample rate: 16kHz (input) or 48kHz (may be resampled by FastRTC)
4. Sends audio chunks via WebRTC to FastRTC server
5. Chunk size: Typically 100-200ms of audio

**Data Format**:
- Type: `Float32Array` or `np.ndarray`
- Range: -1.0 to 1.0 (normalized)
- Channels: Mono (1 channel)
- Sample Rate: 16kHz or 48kHz

### Step 2: FastRTC Audio Source Adapter

**File**: `leibniz_fastrtc_adapters.py`  
**Class**: `FastRTCAudioSource`  
**Line**: 34

#### Class Structure

```python
class FastRTCAudioSource:
    """
    Audio source adapter that receives browser audio from FastRTC
    and provides it to the VAD pipeline.
    
    Implements the audio_source interface expected by leibniz_vad.py.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize FastRTC audio source.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
        """
        self.sample_rate = sample_rate
        self.audio_queue = asyncio.Queue(maxsize=100)  # Thread-safe buffer
        self.is_active = False
        self._loop = None
```

#### Audio Injection Method

**Method**: `push_audio_from_fastrtc(audio_chunk: np.ndarray) -> None`  
**Line**: 57

**Called by**: FastRTC handler when browser audio arrives

**Flow**:

```python
def push_audio_from_fastrtc(self, audio_chunk: np.ndarray) -> None:
    """
    Inject browser audio from FastRTC into the source queue.
    
    Args:
        audio_chunk: Audio data as float32 numpy array (normalized -1.0 to 1.0)
    """
    try:
        # Try to get running event loop
        try:
            loop = asyncio.get_running_loop()
            # Schedule async push
            asyncio.create_task(self._async_push(audio_chunk))
        except RuntimeError:
            # No running loop - store for later
            if not hasattr(self, '_pending_chunks'):
                self._pending_chunks = []
            self._pending_chunks.append(audio_chunk)
    except Exception as e:
        logger.error(f"Error pushing audio to FastRTC source: {e}")

async def _async_push(self, audio_chunk: np.ndarray) -> None:
    """Internal async method to push audio to queue."""
    await self.audio_queue.put(audio_chunk)
    self.is_active = True
```

**Key Features**:
- Handles both sync and async contexts
- Stores pending chunks if no event loop available
- Thread-safe queue insertion
- Sets `is_active` flag when audio arrives

#### Audio Retrieval Method

**Method**: `async def get_frames(num_samples: int) -> np.ndarray`  
**Line**: 103

**Called by**: VAD module to retrieve audio frames

**Flow**:

```python
async def get_frames(self, num_samples: int) -> np.ndarray:
    """
    Called by VAD to retrieve audio frames.
    
    Args:
        num_samples: Number of samples to retrieve
        
    Returns:
        Audio frame as numpy array (float32, shape=(num_samples,))
    """
    # Process pending chunks first (if any)
    if hasattr(self, '_pending_chunks') and self._pending_chunks:
        for chunk in self._pending_chunks:
            await self.audio_queue.put(chunk)
        self._pending_chunks = []
    
    # Collect frames from queue
    frames = []
    while len(frames) < num_samples:
        try:
            chunk = await asyncio.wait_for(
                self.audio_queue.get(), timeout=1.0
            )
            frames.extend(chunk)
        except asyncio.TimeoutError:
            # No more audio available
            break
    
    # Return requested number of samples
    return np.array(frames[:num_samples], dtype=np.float32)
```

**Key Features**:
- Processes pending chunks first
- Collects frames from queue until `num_samples` reached
- Handles timeout if no audio available
- Returns float32 numpy array

**Clear Method** (Line 157):
```python
def clear(self) -> None:
    """Clear audio queue and reset state."""
    while not self.audio_queue.empty():
        try:
            self.audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    self.is_active = False
```

**Data Flow**:
```
Browser Audio Chunk
    ↓
FastRTC Handler.__call__()
    ↓
FastRTCAudioSource.push_audio_from_fastrtc()
    ↓
_async_push() → audio_queue.put()
    ↓
[Queue: asyncio.Queue(maxsize=100)]
    ↓
get_frames() → audio_queue.get()
    ↓
VAD Processing
```

### Step 3: VAD/STT Processing

**File**: `leibniz_pro.py`  
**Function**: `capture_and_transcribe()`  
**Line**: 2254

#### Function Signature

```python
async def capture_and_transcribe(
    streaming_callback: Optional[Callable] = None,
    context: Optional[Dict[str, Any]] = None,
    audio_source: Optional['FastRTCAudioSource'] = None
) -> Optional[str]:
    """
    Capture audio and transcribe with Gemini Live VAD.
    
    Args:
        streaming_callback: Optional callback(fragment: str, is_final: bool)
                            for real-time transcript display
        context: Optional conversation context dict with keys:
                 - conversation_context: str (greeting, decision, complex_query, etc.)
                 - attempt_count: int (number of retry attempts)
        audio_source: Optional FastRTC audio source for browser input
                     (None = use native microphone)
    
    Returns:
        Transcript string only (NO audio file) - matches SINDH/TARA pattern
    """
```

#### Implementation Flow

```python
async def capture_and_transcribe(...):
    try:
        logger.info(" Listening for your input...")
        print(" Starting audio capture with parallel processing...")
        
        # Capture speech with VAD (returns transcript only)
        transcript = await capture_leibniz_speech(
            streaming_callback=streaming_callback,
            context=context,
            audio_source=audio_source  # FastRTC source passed here
        )
        
        if transcript:
            logger.info(" Speech captured successfully")
            print(f" Original: '{transcript}'")
            return transcript
        else:
            return None
            
    except Exception as e:
        logger.error(f"Capture error: {e}", exc_info=True)
        return None
```

**Key Points**:
- Calls `capture_leibniz_speech()` from `leibniz_vad.py`
- Passes `audio_source` parameter through to VAD
- Returns transcript string directly (no audio file)
- Handles errors gracefully

**File**: `leibniz_vad.py`  
**Function**: `capture_speech_bidirectional()`  
**Method**: `capture_speech_bidirectional(audio_source: Optional['AudioSource'] = None)`

#### VAD Processing Flow

```python
async def capture_speech_bidirectional(
    self,
    streaming_callback: Optional[Callable] = None,
    audio_source: Optional['AudioSource'] = None
) -> Optional[str]:
    """
    Capture speech using bidirectional VAD with optional audio source.
    
    Flow:
    1. If audio_source provided, use it for audio retrieval
    2. Otherwise, use native microphone
    3. Process audio chunks through Gemini Live API
    4. Return transcript when speech detected
    """
    # Set dynamic timeout based on context
    timeout = self._get_dynamic_timeout(context)
    
    # Initialize Gemini Live session if needed
    if not self.session:
        await self._initialize_session()
    
    # Audio capture loop
    transcript_parts = []
    speech_detected = False
    
    while True:
        # Get audio frames
        if audio_source is not None:
            # Use FastRTC source
            frames = await audio_source.get_frames(800)  # 50ms at 16kHz
        else:
            # Use native microphone
            frames = await self._get_microphone_frames(800)
        
        # Process through Gemini Live VAD
        result = await self._process_audio_chunk(frames)
        
        # Check for speech
        if result.get('is_speech'):
            speech_detected = True
            partial_text = result.get('text', '')
            
            # Call streaming callback
            if streaming_callback:
                streaming_callback(partial_text, is_final=False)
            
            # Accumulate transcript
            if partial_text:
                transcript_parts.append(partial_text)
        
        # Check for final transcript
        if result.get('is_final') and speech_detected:
            final_transcript = ' '.join(transcript_parts)
            if streaming_callback:
                streaming_callback(final_transcript, is_final=True)
            return final_transcript
        
        # Check timeout
        if time.time() - start_time > timeout:
            return None if not speech_detected else ' '.join(transcript_parts)
```

**Key Features**:
- Supports both FastRTC source and native microphone
- Processes audio in chunks (800 samples = 50ms at 16kHz)
- Provides real-time partial transcripts via callback
- Returns final transcript when speech complete
- Handles timeouts gracefully

**Data Flow**:
```
FastRTCAudioSource.get_frames(800)
    ↓
[800 samples, float32, 16kHz]
    ↓
Gemini Live VAD API
    ↓
Partial Transcript (if speech detected)
    ↓
[Accumulate in transcript_parts]
    ↓
Final Transcript (when is_final=True)
    ↓
Return Transcript String
```

### Step 4: Transcript Normalization

**File**: `leibniz_stt.py`

The STT module normalizes transcripts:
- Removes filler words ("um", "uh", "like")
- Capitalizes properly
- Handles punctuation
- Returns clean transcript string

**Normalization Steps**:
1. Remove filler words
2. Capitalize first letter
3. Add punctuation if missing
4. Trim whitespace
5. Return normalized string

---

## Intent Classification Flow

### Step 1: Transcript & Classification Entry Point

**File**: `leibniz_pro.py`  
**Function**: `transcribe_and_classify()`  
**Line**: 2306

#### Function Signature

```python
async def transcribe_and_classify(
    streaming_callback: Optional[Callable] = None,
    context: Optional[Dict[str, Any]] = None,
    audio_source: Optional['FastRTCAudioSource'] = None
) -> Tuple[TranscriptMessage, IntentMessage]:
    """
    Complete flow: Capture → Transcribe → Classify Intent
    
    Returns:
        Tuple of (TranscriptMessage, IntentMessage)
    """
```

#### Complete Implementation Flow

```python
async def transcribe_and_classify(...):
    try:
        # STEP 1: Capture and transcribe
        transcript = await capture_and_transcribe(
            streaming_callback=streaming_callback,
            context=context,
            audio_source=audio_source
        )
        
        if not transcript:
            # Return empty messages
            return (
                TranscriptMessage(transcript="", confidence=0.0),
                IntentMessage(intent="UNCLEAR", confidence=0.0, ...)
            )
        
        # Create TranscriptMessage
        transcript_msg = TranscriptMessage(
            transcript=transcript,
            confidence=1.0  # Gemini Live provides high-quality transcription
        )
        
        # STEP 2: Extract semantic context (FAST, <5ms)
        from leibniz_semantic_extractor import extract_semantic_context
        
        context_gen_start = time.time()
        semantic_context = extract_semantic_context(transcript)
        context_gen_elapsed = time.time() - context_gen_start
        
        # Log semantic extraction
        print(f" Semantic extraction: {context_gen_elapsed*1000:.2f}ms")
        logger.info(f" Semantic context: '{semantic_context['user_goal']}'")
        
        # STEP 3: Classify intent with enriched context
        if services_manager and services_manager.intent_parser.parser:
            # Build enriched context
            enriched_context = {
                **(context or {}),
                'semantic_context': semantic_context,
                'user_goal': semantic_context['user_goal'],
                'key_entities': semantic_context['key_entities'],
                'extracted_meaning': semantic_context['extracted_meaning']
            }
            
            # Use Leibniz's intent parser
            if _leibniz_parser:
                intent_result = await _leibniz_parser.classify_intent(
                    text=semantic_context['extracted_meaning'],  # Use normalized meaning
                    context=enriched_context
                )
            else:
                # Fallback
                intent_result = {
                    'intent': 'RAG_QUERY',
                    'confidence': 0.5,
                    'context': semantic_context
                }
            
            classify_elapsed = time.time() - classify_start
            
            # Merge semantic context with intent result
            intent_context = {
                **semantic_context,
                **intent_result.get('context', {})
            }
            
            # Generate user_context string for RAG
            user_goal = intent_context.get('user_goal', '')
            extracted_meaning = intent_context.get('extracted_meaning', transcript)
            
            if user_goal and user_goal != extracted_meaning:
                user_context_transcript = f"{user_goal}: {extracted_meaning}"
            else:
                user_context_transcript = extracted_meaning
            
            # Create IntentMessage
            intent_msg = IntentMessage(
                intent=intent_result.get("intent", "UNCLEAR"),
                confidence=intent_result.get("confidence", 0.0),
                entities=intent_context,
                user_context=user_context_transcript,
                reasoning=intent_result.get("reasoning", ""),
                processing_time=classify_elapsed,
                timing_breakdown={
                    'classification_ms': (classify_elapsed - context_gen_elapsed) * 1000,
                    'context_generation_ms': context_gen_elapsed * 1000
                }
            )
        
        return transcript_msg, intent_msg
        
    except Exception as e:
        logger.error(f"Transcribe and classify error: {e}", exc_info=True)
        return (
            TranscriptMessage(transcript="", confidence=0.0),
            IntentMessage(intent="UNCLEAR", confidence=0.0, entities={}, reasoning=str(e))
        )
```

**Key Steps**:
1. Capture transcript via `capture_and_transcribe()`
2. Extract semantic context (fast pattern-based)
3. Classify intent with enriched context
4. Merge contexts and create messages
5. Return `(TranscriptMessage, IntentMessage)` tuple

### Step 2: Semantic Context Extraction

**File**: `leibniz_semantic_extractor.py`  
**Function**: `extract_semantic_context()`

#### Purpose

Fast pattern-based extraction (<5ms) that enriches raw transcript with structured context before intent classification.

#### Output Structure

```python
{
    'user_goal': str,              # e.g., "asking about CS program"
    'key_entities': Dict[str, str], # e.g., {"program": "computer science", "topic": "requirements"}
    'extracted_meaning': str,      # e.g., "computer science program admission requirements"
    'extraction_method': str       # 'pattern' or 'llm'
}
```

#### Extraction Process

1. **Pattern Matching** (<5ms)
   - Regex patterns for common entities
   - Department names, program names, topics
   - Extracts user goal from sentence structure

2. **LLM Fallback** (if pattern fails)
   - Uses Gemini 2.0 Flash
   - More accurate but slower (50-200ms)

#### Example

**Input**: "What are the requirements for the computer science program?"

**Output**:
```python
{
    'user_goal': 'asking about CS program requirements',
    'key_entities': {
        'program': 'computer science',
        'topic': 'requirements'
    },
    'extracted_meaning': 'computer science program admission requirements',
    'extraction_method': 'pattern'
}
```

### Step 3: Intent Classification

**File**: `leibniz_intent_parser.py`  
**Class**: `LeibnizIntentParser`  
**Method**: `classify_intent()`

#### Two-Tier Classification

**Tier 1: Fast Pattern Matching** (<10ms)

```python
# Pattern examples
APPOINTMENT_PATTERNS = [
    r'schedule|book|appointment|meeting',
    r'want to see|need to meet',
    r'available.*time|when.*available'
]

GREETING_PATTERNS = [
    r'hello|hi|hey|greetings',
    r'good morning|good afternoon|good evening'
]

EXIT_PATTERNS = [
    r'bye|goodbye|exit|quit|stop',
    r'thanks.*bye|thank you.*bye'
]
```

**Tier 2: LLM Fallback** (200-800ms)

If no pattern matches, uses Gemini 2.0 Flash:

```python
prompt = f"""
Classify the following user query into one of these intents:
- APPOINTMENT_SCHEDULING
- RAG_QUERY
- GREETING
- EXIT
- UNCLEAR

Query: {text}
Context: {context}

Return JSON: {{"intent": "...", "confidence": 0.0-1.0, "reasoning": "..."}}
"""
```

#### Intent Types

1. **APPOINTMENT_SCHEDULING**
   - User wants to schedule an appointment
   - Routes to Appointment FSM

2. **RAG_QUERY**
   - User asking about university information
   - Routes to RAG system

3. **GREETING**
   - User greeting the agent
   - Returns greeting response

4. **EXIT**
   - User wants to end conversation
   - Returns farewell and exits

5. **UNCLEAR**
   - Cannot determine intent
   - Returns clarification prompt

#### Output Format

```python
{
    'intent': str,           # One of the 5 intent types
    'confidence': float,     # 0.0-1.0
    'reasoning': str,        # Explanation
    'context': Dict[str, Any] # Additional context
}
```

---

## Response Generation Flow (RAG/Appointment)

### Step 1: Intent Routing

**File**: `leibniz_pro.py`  
**Function**: `run_conversation_session()`  
**Line**: 3764

#### Routing Logic

**Location**: Lines 4146-4194

```python
# Extract intent from IntentMessage
intent = intent_msg.intent
context = intent_msg.entities
user_context = intent_msg.user_context

# Route based on intent
if intent == "APPOINTMENT_SCHEDULING":
    # Route to Appointment FSM
    booking_data = await handle_appointment_booking(
        initial_input=transcript,
        audio_sink=audio_sink
    )
    last_interaction_type = "appointment"

elif intent == "RAG_QUERY":
    # Route to RAG system
    rag_msg = await handle_rag_query(
        text=user_context or transcript,
        context=context,
        enable_streaming=False,
        audio_sink=audio_sink
    )
    await speak_friendly(
        text=rag_msg.answer,
        audio_sink=audio_sink
    )
    last_interaction_type = "rag_query"

elif intent == "GREETING":
    await speak_friendly(
        dialogue_key='greetings.intro',
        audio_sink=audio_sink
    )
    last_interaction_type = "greeting"

elif intent == "EXIT":
    await speak_friendly(
        dialogue_key='farewells.exit',
        audio_sink=audio_sink
    )
    conversation_active = False
    break

else:  # UNCLEAR
    await speak_friendly(
        dialogue_key='prompts.clarify',
        audio_sink=audio_sink
    )
    last_interaction_type = "fallback"
```

### Step 2A: RAG Query Processing

**File**: `leibniz_pro.py`  
**Function**: `handle_rag_query()`  
**Line**: 2815

#### Complete RAG Flow

```python
async def handle_rag_query(
    text: str,
    context: Dict[str, Any],
    enable_streaming: bool = True,
    audio_sink: Optional[object] = None
) -> RAGMessage:
    """
    Comprehensive RAG handler with caching and streaming.
    
    Execution Flow:
    1. Check cache (1-5ms)
    2. Check speculative execution (10-50ms)
    3. Execute persistent RAG (800-2000ms)
    4. Stream to TTS queue
    5. Return RAGMessage
    """
    start_time = time.time()
    
    # Query deduplication lock
    query_hash = hash(text.strip().lower())
    query_lock = _rag_deduplication_locks.get(query_hash, asyncio.Lock())
    
    async with query_lock:
        # Start TTS consumer if streaming enabled
        consumer_task = None
        if enable_streaming:
            consumer_task = await start_tts_consumer(audio_sink=audio_sink)
        
        # STEP 1: Cache Check (1-5ms)
        cache_mgr = get_rag_cache_manager()
        cached_response = cache_mgr.get_query_response(text, language='english')
        
        if cached_response:
            logger.info(f" Cache HIT in {cache_check_time:.1f}ms")
            # Stream cached response to TTS
            if enable_streaming:
                await stream_rag_to_tts(cached_response, is_final=True)
            return RAGMessage(answer=cached_response, ...)
        
        # STEP 2: Speculative Execution Check (10-50ms)
        spec_coord = get_speculative_coordinator()
        speculative_result = spec_coord.get_best_result(text)
        
        if speculative_result:
            logger.info(f" Speculative HIT")
            # Stream speculative response
            if enable_streaming:
                await stream_rag_to_tts(speculative_result, is_final=True)
            return RAGMessage(answer=speculative_result, ...)
        
        # STEP 3: Persistent RAG Execution (800-2000ms)
        services = await get_leibniz_services_manager()
        
        # Define streaming callback for progressive TTS
        def rag_streaming_callback_sync(partial_text, is_final: bool = False):
            """Thread-safe streaming callback - queues sentence strings for TTS"""
            if enable_streaming:
                if partial_text.strip():
                    _tts_streaming_queue.put_nowait((partial_text, 1.0))
                if is_final:
                    _tts_streaming_queue.put_nowait(None)  # Sentinel
        
        # Call RAG system in thread
        result = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: services.rag_system.rag_system.process_rag_query(
                    context=context,
                    query=text,
                    streaming_callback=rag_streaming_callback_sync if enable_streaming else None
                )
            ),
            timeout=adaptive_timeout
        )
        
        # Extract result components
        raw_answer = result.get('response', '') or result.get('answer', '')
        sources = result.get('sources', [])
        timing_breakdown = result.get('timing_breakdown', {})
        
        # STEP 4: Process response for natural conversation
        processed_answer = process_rag_for_natural_conversation(
            raw_answer, text, intent_type
        )
        
        # STEP 5: Cache successful result
        cache_mgr.cache_query_response(text, processed_answer, language='english')
        
        # STEP 6: Wait for TTS consumer if streaming
        if consumer_task and enable_streaming:
            await consumer_task
        
        return RAGMessage(
            answer=processed_answer,
            sources=sources,
            confidence=compute_rag_confidence(timing_breakdown=timing_breakdown),
            processing_time_ms=(time.time() - start_time) * 1000,
            timing_breakdown={
                'method': 'persistent_rag',
                'rag_ms': rag_time,
                **timing_breakdown
            }
        )
```

#### RAG Streaming Callback

The `rag_streaming_callback_sync` function receives partial text chunks from RAG generation and queues them for progressive TTS:

**Location**: Line 3085

```python
def rag_streaming_callback_sync(partial_text, is_final: bool = False):
    """
    Thread-safe streaming callback.
    
    Called by RAG system as it generates response text.
    Queues sentence strings to TTS queue for progressive playback.
    """
    if enable_streaming:
        # Queue sentence string if non-empty
        if partial_text.strip():
            _tts_streaming_queue.put_nowait((partial_text, 1.0))
            logger.debug(f" Enqueued: '{partial_text[:50]}...'")
        
        # Send sentinel on final chunk
        if is_final:
            _tts_streaming_queue.put_nowait(None)
            logger.debug(" Sentinel sent from RAG callback")
```

**Key Features**:
- Receives partial text chunks as RAG generates
- Queues chunks immediately to TTS queue
- Enables progressive TTS playback (reduces perceived latency)
- Sends sentinel (None) when generation complete

### Step 2B: Appointment FSM Processing

**File**: `leibniz_pro.py`  
**Function**: `handle_appointment_booking()`  
**Line**: 3294

#### FSM Flow

```python
async def handle_appointment_booking(
    initial_input: Optional[str] = None,
    audio_sink: Optional[object] = None
) -> Optional[Dict[str, Any]]:
    """
    Handle appointment booking conversation flow.
    
    Flow:
    1. Create FSM instance
    2. Process initial input
    3. Loop: Capture → Process → Speak → Repeat
    4. Return booking data when complete
    """
    # Pause continuous VAD to prevent conflicts
    continuous_vad_was_running = False
    if _continuous_vad_enabled:
        await stop_leibniz_continuous_listening()
        continuous_vad_was_running = True
    
    try:
        # Create FSM instance
        fsm = create_appointment_fsm()
        
        # Process initial input
        result = await fsm.process_input(initial_input or "")
        await speak_friendly(result['response'], audio_sink=audio_sink)
        
        # FSM Loop
        while True:
            # Check completion
            if result.get('complete', False):
                booking_data = format_appointment_for_submission(fsm.data)
                await speak_friendly(
                    dialogue_key='appointment_confirm',
                    audio_sink=audio_sink
                )
                return booking_data
            
            # Check cancellation
            if result.get('state') == AppointmentState.CANCELLED.value:
                return None
            
            # Capture user input
            transcript = await capture_and_transcribe(audio_source=audio_source)
            
            # Handle empty transcript (confirmation states)
            if not transcript:
                current_state = result.get('state', '')
                if 'confirm' in current_state:
                    transcript = ""  # Pass empty to FSM
                else:
                    await speak_friendly(
                        dialogue_key='timeout',
                        audio_sink=audio_sink
                    )
                    continue
            
            # Multilingual support (optional)
            if transcript and enable_multilingual:
                semantic_context = await generate_semantic_context_for_fsm(
                    user_input=transcript,
                    fsm_state=result.get('state', '')
                )
                transcript = semantic_context['translated_text']
            
            # Process with FSM
            result = await fsm.process_input(transcript)
            await speak_friendly(result['response'], audio_sink=audio_sink)
    
    finally:
        # Resume continuous VAD if it was running
        if continuous_vad_was_running:
            await start_leibniz_continuous_listening()
```

#### FSM States

**File**: `leibniz_appointment_fsm.py`

**States**:
- `COLLECTING_NAME`: Collect user's name
- `COLLECTING_EMAIL`: Collect user's email
- `COLLECTING_DATE`: Collect appointment date
- `COLLECTING_TIME`: Collect appointment time
- `CONFIRMING`: Confirm booking details
- `COMPLETE`: Booking completed
- `CANCELLED`: User cancelled

**State Transitions**:
```
COLLECTING_NAME → COLLECTING_EMAIL → COLLECTING_DATE 
    → COLLECTING_TIME → CONFIRMING → COMPLETE
```

**Slot Filling**:
- Extracts entities from user input
- Validates format (email, date, time)
- Prompts for missing information
- Confirms before finalizing

---

## TTS Synthesis & Streaming Flow

### Step 1: TTS Queue Infrastructure

**File**: `leibniz_pro.py`  
**Global Variables** (Lines 733-767)

```python
# Global queue for sentence-level streaming from RAG to TTS
_tts_streaming_queue = asyncio.Queue(maxsize=15)  # Bounded queue

# Streaming state
_streaming_active = False  # Consumer running flag
_tts_consumer_task: Optional[asyncio.Task] = None  # Consumer task handle
_cancel_streaming = asyncio.Event()  # Cancellation signal for barge-in
_sentence_buffer = []  # Global sentence buffer for progressive streaming

# Sentinel tracking
_sentinel_sent = False  # End-of-stream marker
_sentinels_sent_count = 0  # Debug counter
_sentinels_received_count = 0  # Debug counter

# Deduplication
_synthesis_in_flight = set()  # In-flight synthesis hashes
_recent_synthesis_cache = OrderedDict()  # LRU cache (64 entries, 10s TTL)

# Playback guard
_playing_now = False  # Prevents overlapping playback
```

### Step 2: RAG to TTS Streaming

**File**: `leibniz_pro.py`  
**Function**: `stream_rag_to_tts()`  
**Line**: 926

#### Function Flow

```python
async def stream_rag_to_tts(
    rag_response: str,
    pace: float = 1.0,
    is_final: bool = True
):
    """
    Split RAG response into sentences and stream to TTS queue.
    
    Args:
        rag_response: RAG response text (can be partial or complete)
        pace: Speech pace for TTS (1.0 = normal)
        is_final: If False, don't send end-of-stream signal
                  If True, send end-of-stream signal
    """
    # Ensure consumer is running
    if streaming_enabled and (_tts_consumer_task is None or _tts_consumer_task.done()):
        await start_tts_consumer()
    
    try:
        # Accumulate text in buffer
        _sentence_buffer.append(rag_response)
        accumulated_text = ''.join(_sentence_buffer)
        
        # Split into sentences
        sentences = split_into_sentences(accumulated_text)
        
        # If not final, hold the last incomplete sentence
        if not is_final and len(sentences) > 0:
            incomplete_sentence = sentences[-1]
            complete_sentences = sentences[:-1]
            _sentence_buffer = [incomplete_sentence]
        else:
            complete_sentences = sentences
            _sentence_buffer = []
        
        # Queue complete sentences with backpressure handling
        for sentence in complete_sentences:
            try:
                _tts_streaming_queue.put_nowait((sentence, pace))
            except asyncio.QueueFull:
                # Backpressure: merge with last queued sentence
                logger.warning("TTS queue full, merging sentences")
                try:
                    last_item = _tts_streaming_queue.get_nowait()
                    if last_item is not None:
                        last_sentence, last_pace = last_item
                        merged = f"{last_sentence} {sentence}"
                        _tts_streaming_queue.put_nowait((merged, pace))
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    logger.warning("Dropped sentence due to queue pressure")
        
        # Send sentinel if final
        if is_final and not _sentinel_sent:
            await _tts_streaming_queue.put(None)
            _sentinel_sent = True
            logger.debug("Sentinel sent (stream_rag_to_tts)")
    
    except Exception as e:
        logger.error(f"Error in stream_rag_to_tts: {e}")
        # Clear buffer and send sentinel on error
        _sentence_buffer = []
        if not _sentinel_sent:
            await _tts_streaming_queue.put(None)
            _sentinel_sent = True
```

#### Sentence Splitting

**Function**: `split_into_sentences(text: str) -> List[str]`  
**Line**: 862

```python
def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences for streaming TTS.
    
    Preserves abbreviations by protecting periods.
    """
    # Protect abbreviations
    abbrev_map = {
        "Dr.": "Dr<PERIOD>",
        "Mr.": "Mr<PERIOD>",
        "Mrs.": "Mrs<PERIOD>",
        # ... etc
    }
    
    # Replace abbreviations
    for abbrev, placeholder in abbrev_map.items():
        text = text.replace(abbrev, placeholder)
    
    # Split on sentence delimiters
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    
    # Restore abbreviations
    for sentence in sentences:
        for abbrev, placeholder in abbrev_map.items():
            sentence = sentence.replace(placeholder, abbrev)
    
    # Filter empty and very short fragments
    valid_sentences = [s for s in sentences if s and len(s) > 3]
    
    # Fallback: split long text into ~50 char chunks
    if not valid_sentences and text.strip():
        words = text.split()
        chunk = []
        chunk_len = 0
        for word in words:
            chunk.append(word)
            chunk_len += len(word) + 1
            if chunk_len >= 50:
                valid_sentences.append(' '.join(chunk))
                chunk = []
                chunk_len = 0
        if chunk:
            valid_sentences.append(' '.join(chunk))
    
    return valid_sentences
```

### Step 3: TTS Consumer (Queue Processor)

**File**: `leibniz_pro.py`  
**Function**: `consume_tts_streaming_queue()`  
**Line**: 1157

#### 2-Slot Pipeline Pattern

**Concept**: Synthesize sentence N+1 while playing sentence N to minimize latency.

#### Complete Consumer Flow

```python
async def consume_tts_streaming_queue(
    audio_sink: Optional[object] = None
) -> bool:
    """
    Consumer: Read sentences from queue and synthesize/play.
    
    Features:
    - 2-slot pipeline: Synthesize N+1 while playing N
    - Barge-in detection
    - LRU cache for deduplication
    - FastRTC sink routing
    """
    global _streaming_active, _sentinel_sent
    
    # Prevent multiple concurrent consumers
    if _streaming_active:
        return False
    
    _streaming_active = True
    
    # Initialize state
    sentence_count = 0
    sentences_queued = 0
    sentences_played = 0
    sentences_failed = 0
    agent_speaking_set = False
    
    # Staging buffer for non-destructive peek
    stage = deque()
    
    # 2-slot pipeline state
    current_result = None  # Audio result for current sentence
    next_future = None  # Background synthesis task for N+1
    
    try:
        while True:
            # Check cancellation
            if _cancel_streaming.is_set():
                logger.debug("TTS consumer cancelled")
                break
            
            # Check barge-in
            if check_leibniz_barge_in():
                logger.info("User started speaking (barge-in), stopping TTS")
                await clear_tts_queue()
                break
            
            # Fill staging buffer (2 slots)
            while len(stage) < 2:
                try:
                    item = await asyncio.wait_for(
                        _tts_streaming_queue.get(), timeout=0.5
                    )
                    stage.append(item)
                    if item is None:  # Sentinel
                        break
                except asyncio.TimeoutError:
                    break
            
            # Get current sentence from staging buffer
            current_item = stage.popleft()
            
            if current_item is None:  # Sentinel
                logger.debug("Sentinel received, exiting consumer")
                break
            
            sentence, pace = current_item
            sentence_count += 1
            sentences_queued += 1
            
            logger.info(f"PLAYING: {sentence}")
            
            # Set agent speaking state on first sentence
            if not agent_speaking_set and sentence_count == 1:
                await set_leibniz_agent_speaking(True, "Streaming TTS - First Sentence")
                agent_speaking_set = True
            
            # Use prefetched audio if available
            if next_future:
                try:
                    current_result = await next_future
                    logger.debug(f"Using prefetched audio")
                except Exception as synth_error:
                    logger.error(f"Prefetch synthesis error: {synth_error}")
                    current_result = None
                next_future = None
            
            # Prefetch next sentence (N+1 synthesis)
            if stage:  # Has next sentence
                next_item = stage[0]  # Peek without removing
                
                if next_item is not None:  # Not sentinel
                    next_sentence, next_pace = next_item
                    
                    # Check LRU cache
                    should_skip, cached_path = check_recent_synthesis(next_sentence)
                    
                    if cached_path:
                        # Use cached file
                        async def _return_cached(path):
                            return {'audio_path': path, 'success': True, 'cached': True}
                        next_future = asyncio.create_task(_return_cached(cached_path))
                    elif not should_skip:
                        # Start N+1 synthesis in background
                        normalized_next = normalize_sentence_for_hash(next_sentence)
                        sentence_hash = hashlib.md5(normalized_next.encode()).hexdigest()
                        
                        if sentence_hash not in _synthesis_in_flight:
                            _synthesis_in_flight.add(sentence_hash)
                            
                            async def _synthesize_and_track(text, hash_key):
                                try:
                                    result = await _synthesize_only(text, emotion="helpful")
                                    return result
                                finally:
                                    _synthesis_in_flight.discard(hash_key)
                            
                            next_future = asyncio.create_task(
                                _synthesize_and_track(next_sentence, sentence_hash)
                            )
            
            # Synthesize current sentence if not prefetched
            if not current_result:
                # Check cache
                should_skip, cached_path = check_recent_synthesis(sentence)
                
                if cached_path:
                    current_result = {'audio_path': cached_path, 'success': True, 'cached': True}
                elif not should_skip:
                    # Synthesize now
                    normalized_sentence = normalize_sentence_for_hash(sentence)
                    sentence_hash = hashlib.md5(normalized_sentence.encode()).hexdigest()
                    
                    if sentence_hash not in _synthesis_in_flight:
                        _synthesis_in_flight.add(sentence_hash)
                        try:
                            current_result = await _synthesize_only(sentence, emotion="helpful")
                        finally:
                            _synthesis_in_flight.discard(sentence_hash)
            
            # Route audio to sink or local playback
            if current_result:
                audio_bytes = current_result.get('audio_bytes')
                sample_rate = current_result.get('sample_rate', 24000)
                
                if audio_sink is not None:
                    # FastRTC Mode: Route to sink
                    audio_data = np.frombuffer(
                        audio_bytes, dtype=np.int16
                    ).astype(np.float32) / 32767.0
                    
                    await audio_sink.write_audio(audio_data, sample_rate)
                    sentences_played += 1
                    logger.debug(f"✅ Routed {len(audio_data)} samples to FastRTC sink")
                else:
                    # Local Mode: Play with sounddevice/pygame
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                    await asyncio.to_thread(sd.play, audio_data, sample_rate, device=12)
                    await asyncio.to_thread(sd.wait)
                    sentences_played += 1
            
            # Update for next iteration
            current_result = None
    
    except asyncio.CancelledError:
        logger.debug("TTS consumer task cancelled")
        raise
    except Exception as e:
        logger.error(f"TTS streaming error: {e}")
    finally:
        _streaming_active = False
        
        # Clear agent speaking state
        if agent_speaking_set:
            await set_leibniz_agent_speaking(False, "TTS consumer finished")
        
        logger.info(f"TTS Streaming Complete: {sentences_queued} queued, {sentences_played} played")
```

#### Key Features

1. **2-Slot Pipeline**
   - Synthesizes next sentence while playing current
   - Reduces latency between sentences
   - Uses staging buffer for non-destructive peek

2. **LRU Cache**
   - Prevents duplicate synthesis within 10s TTL
   - Checks cache before synthesizing
   - 64-entry cache with LRU eviction

3. **Barge-In Support**
   - Checks `check_leibniz_barge_in()` flag
   - Stops playback immediately if user speaks
   - Clears queue and exits gracefully

4. **FastRTC Routing**
   - Routes to `audio_sink.write_audio()` if provided
   - Otherwise uses local playback
   - Handles both int16 and float32 formats

#### Supporting Functions

**Function**: `start_tts_consumer()` (Line 1021)
- Starts consumer task if not running
- Accepts `audio_sink` parameter
- Returns task handle

**Function**: `finalize_tts_streaming()` (Line 1092)
- Sends sentinel to queue
- Awaits consumer completion
- Clears queue and resets flags

**Function**: `_synthesize_only()` (Line 1596)
- Synthesizes audio without playing
- Used for prefetch pattern
- Returns TTS result dict

**Function**: `check_recent_synthesis()` (Line 770)
- Checks LRU cache for recent synthesis
- Returns `(should_skip, cached_path)` tuple
- Prevents duplicate synthesis within TTL

### Step 4: TTS Synthesis

**File**: `leibniz_tts.py`  
**Function**: `synthesize_to_file()`

#### Triple-Provider Architecture

```python
async def synthesize_to_file(
    text: str,
    emotion: str = "helpful",
    cache_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synthesize text to audio with triple-provider fallback.
    
    Provider Priority:
    1. ElevenLabs (Primary) - 300-800ms
    2. Google Cloud TTS (Fallback) - 400-1000ms
    3. Gemini TTS (Last Resort) - Unstable
    
    Returns:
        {
            'success': bool,
            'audio_file': str,
            'audio_bytes': bytes,
            'sample_rate': int,
            'duration': float
        }
    """
    # Try ElevenLabs first
    try:
        result = await elevenlabs_synthesize(text, emotion)
        return result
    except Exception as e1:
        logger.warning(f"ElevenLabs failed: {e1}")
        
        # Fallback to Google Cloud TTS
        try:
            result = await google_tts_synthesize(text, emotion)
            return result
        except Exception as e2:
            logger.warning(f"Google TTS failed: {e2}")
            
            # Last resort: Gemini TTS (unstable)
            try:
                result = await gemini_tts_synthesize(text, emotion)
                return result
            except Exception as e3:
                logger.error(f"All TTS providers failed: {e3}")
                return {'success': False, 'error': str(e3)}
```

---

## Audio Output Flow (TTS → Browser)

### Step 1: FastRTC Audio Sink

**File**: `leibniz_fastrtc_adapters.py`  
**Class**: `FastRTCAudioSink`  
**Line**: 167

#### Class Structure

```python
class FastRTCAudioSink:
    """
    Audio sink adapter that receives TTS output and streams it to browser.
    
    Provides interface for streaming TTS audio through FastRTC WebSocket.
    """
    
    def __init__(self, sample_rate: int = 24000):
        """
        Initialize FastRTC audio sink.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 24000)
        """
        self.sample_rate = sample_rate
        self.output_queue = asyncio.Queue()  # Unbounded output buffer
        self.is_streaming = False
```

#### Audio Writing Method

**Method**: `async def write_audio(audio_data: np.ndarray, sample_rate: int) -> None`  
**Line**: 189

**Called by**: TTS consumer when audio is synthesized

**Flow**:

```python
async def write_audio(
    self,
    audio_data: np.ndarray,
    sample_rate: int
) -> None:
    """
    Send TTS audio to the sink for streaming to browser.
    
    Called by TTS consumer to provide TTS output.
    Handles format conversion and queues audio for streaming.
    
    Args:
        audio_data: Audio data as numpy array (float32 or int16)
        sample_rate: Sample rate of the audio data
    """
    try:
        # Ensure float32 format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed (prevent clipping)
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        # Convert to int16 for FastRTC (matches test_tts_webrtc_stream.py pattern)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Queue for streaming
        await self.output_queue.put((sample_rate, audio_int16))
        self.is_streaming = True
        
        logger.debug(f"Queued {len(audio_int16)} samples to FastRTC sink")
        
    except Exception as e:
        logger.error(f"Error writing audio to FastRTC sink: {e}")
```

**Data Format Conversion**:
- Input: float32 numpy array (normalized -1.0 to 1.0)
- Normalize: Ensure max amplitude ≤ 1.0
- Convert: Multiply by 32767 and cast to int16
- Queue: Store as `(sample_rate, audio_int16)` tuple

#### Streaming Generator

**Method**: `def stream_to_fastrtc(self)`  
**Line**: 246

**Called by**: FastRTC handler to retrieve audio chunks

**Flow**:

```python
def stream_to_fastrtc(self):
    """
    Generator that yields audio chunks for FastRTC emit().
    
    Yields:
        (sample_rate, audio_chunk) tuples
        Audio chunks are int16 PCM, typically 2400 samples (100ms at 24kHz)
    """
    while True:
        try:
            # Get chunk from queue (with timeout)
            sample_rate, audio_chunk = await asyncio.wait_for(
                self.output_queue.get(), timeout=0.1
            )
            yield (sample_rate, audio_chunk)
            
        except asyncio.TimeoutError:
            # No more audio available
            # Yield silence to keep stream alive, then break
            yield (24000, np.zeros(2400, dtype=np.int16))
            break
```

**Key Features**:
- Generator pattern for continuous streaming
- Timeout prevents blocking if queue empty
- Yields silence when no audio available
- Breaks loop when timeout occurs

**Additional Methods**:
- `stop_streaming()` (Line 298): Stops streaming and clears queue
- `clear()` (Line 303): Clears output queue

### Step 2: FastRTC Emit

**File**: `leibniz_fastrtc_handler.py`  
**Method**: `__call__()` (yield statement)  
**Line**: 107

The FastRTC handler's `__call__()` method yields audio chunks from the sink:

```python
# In handler.__call__()
for sample_rate, audio_chunk in self.sink.stream_to_fastrtc():
    yield (sample_rate, audio_chunk)
    logger.debug(f"Yielded audio chunk: {len(audio_chunk)} samples at {sample_rate}Hz")
```

**Chunking**:
- Typical chunk size: 2400 samples (100ms at 24kHz)
- Format: int16 PCM
- Sample rate: 24kHz
- Yields continuously until queue empty

### Step 3: Browser Playback

**Location**: Browser (WebRTC API)

**Process**:
1. Receives audio chunks via WebRTC from FastRTC server
2. Converts int16 to float32 for browser audio API
3. Buffers chunks for smooth playback
4. Plays through browser audio API (`AudioContext`, `MediaStream`)
5. Handles buffering and timing automatically

**Data Format**:
- Input: int16 PCM chunks
- Conversion: int16 → float32 (normalized -1.0 to 1.0)
- Playback: Browser audio API

---

## Conversation Loop & Cycle Management

### Main Conversation Session

**File**: `leibniz_pro.py`  
**Function**: `run_conversation_session()`  
**Line**: 3764

#### Complete Session Flow

```python
async def run_conversation_session(
    audio_source=None,
    audio_sink=None,
    skip_intro=False
):
    """
    Main conversation loop with max 5 attempts.
    
    Flow:
    1. Initialize session
    2. Play intro (unless skipped)
    3. For each attempt (max 5):
       a. Determine context
       b. Set dynamic timeout
       c. Capture and classify
       d. Route to handler
       e. Generate response
       f. Speak response
       g. Pause 0.5s
    4. Cleanup
    """
    # Session initialization
    conversation_active = True
    await reset_leibniz_conversation()
    
    session_start = time.time()
    consecutive_no_input = 0
    last_interaction_type = ""
    max_attempts = 5
    
    # Play intro
    if not skip_intro:
        await play_natural_intro(audio_sink=audio_sink)
    
    # Main conversation loop
    for attempt in range(max_attempts):
        turn_start = time.time()
        
        # Determine conversation context
        if attempt == 0:
            current_context = "greeting"
        elif last_interaction_type == "appointment":
            current_context = "post_service"
        elif last_interaction_type == "rag_query":
            current_context = "complex_query"
        else:
            current_context = "decision"
        
        # Set dynamic timeout
        vad = get_leibniz_vad()
        vad.set_dynamic_timeout(
            attempt_count=attempt,
            conversation_context=current_context
        )
        
        # Reset speculative coordinator
        if SPECULATIVE_AVAILABLE:
            coordinator.reset_session(session_id=f"attempt_{attempt}")
        
        # Capture and classify
        print("\n Listening... SPEAK NOW!")
        capture_start_time = time.time()
        
        transcript_msg, intent_msg = await transcribe_and_classify(
            streaming_callback=display_streaming_transcript,
            context={'conversation_context': current_context},
            audio_source=audio_source
        )
        
        capture_duration = time.time() - capture_start_time
        
        transcript = transcript_msg.transcript
        intent = intent_msg.intent
        context = intent_msg.entities
        
        # Handle no input
        if not transcript:
            consecutive_no_input += 1
            if consecutive_no_input >= 5:
                break
            
            # Force VAD reset after 2 consecutive timeouts
            if consecutive_no_input >= 2:
                await reset_leibniz_conversation()
            
            # Speak timeout prompt
            await speak_friendly(
                dialogue_key='timeout',
                audio_sink=audio_sink
            )
            last_interaction_type = "fallback"
            continue
        
        # Reset no-input counter
        consecutive_no_input = 0
        
        # Check exit
        if is_exit_phrase(transcript):
            await speak_friendly(
                dialogue_key='farewells.exit',
                audio_sink=audio_sink
            )
            conversation_active = False
            break
        
        # Route based on intent
        if intent == "APPOINTMENT_SCHEDULING":
            booking_data = await handle_appointment_booking(
                initial_input=transcript,
                audio_sink=audio_sink
            )
            last_interaction_type = "appointment"
        
        elif intent == "RAG_QUERY":
            rag_msg = await handle_rag_query(
                text=intent_msg.user_context or transcript,
                context=context,
                enable_streaming=False,
                audio_sink=audio_sink
            )
            await speak_friendly(
                text=rag_msg.answer,
                audio_sink=audio_sink
            )
            last_interaction_type = "rag_query"
        
        elif intent == "GREETING":
            await speak_friendly(
                dialogue_key='greetings.intro',
                audio_sink=audio_sink
            )
            last_interaction_type = "greeting"
        
        elif intent == "EXIT":
            await speak_friendly(
                dialogue_key='farewells.exit',
                audio_sink=audio_sink
            )
            conversation_active = False
            break
        
        else:  # UNCLEAR
            await speak_friendly(
                dialogue_key='prompts.clarify',
                audio_sink=audio_sink
            )
            last_interaction_type = "fallback"
        
        # Inter-attempt pause
        await asyncio.sleep(0.5)
    
    # Cleanup
    conversation_active = False
    session_duration = time.time() - session_start
    logger.info(f"Session ended. Duration: {session_duration:.1f}s")
```

### Cycle Management Features

#### 1. Attempt Tracking

- **Max Attempts**: 5 attempts per session
- **Consecutive No-Input**: Tracks consecutive timeouts
- **Last Interaction Type**: Tracks interaction type for context determination
- **Session State**: `conversation_active` flag

#### 2. Dynamic Timeouts

**Context-Based Timeouts**:
- `greeting`: 12s (first attempt, user may be greeting)
- `decision`: 15s (user deciding what to ask)
- `complex_query`: 18s (after RAG, user processing answer)
- `post_service`: 8s (after appointment, quick follow-up)

**Implementation**:
```python
vad.set_dynamic_timeout(
    attempt_count=attempt,
    conversation_context=current_context
)
```

#### 3. VAD Health Checks

- **Reset After 2 Timeouts**: Forces VAD session reset
- **Session Health Monitoring**: Tracks consecutive timeouts
- **Automatic Recovery**: Resets VAD and continues

#### 4. Barge-In Support

**Detection**:
- Checks `check_leibniz_barge_in()` flag during TTS
- Stops playback immediately if user speaks
- Processes new input right away

**Implementation**:
```python
# In consume_tts_streaming_queue()
if check_leibniz_barge_in():
    logger.info("User started speaking (barge-in), stopping TTS")
    await clear_tts_queue()
    break
```

#### 5. Session Cleanup

**Cleanup Steps**:
1. Clear TTS queue
2. Reset streaming flags
3. Clear audio adapters
4. Reset VAD session
5. Log session metrics

---

## Message Types & Data Structures

### TranscriptMessage

**File**: `leibniz_messages.py`

```python
@dataclass
class TranscriptMessage:
    transcript: str  # Clean transcript text
    confidence: float  # Confidence score (0.0-1.0)
```

**Usage**: Returned by `capture_and_transcribe()`

**Example**:
```python
TranscriptMessage(
    transcript="I need to schedule an appointment",
    confidence=1.0
)
```

### IntentMessage

**File**: `leibniz_messages.py`

```python
@dataclass
class IntentMessage:
    intent: str  # APPOINTMENT_SCHEDULING, RAG_QUERY, etc.
    confidence: float  # Classification confidence (0.0-1.0)
    entities: Dict[str, Any]  # Full context dict
    user_context: str  # Enriched query for RAG
    reasoning: str  # Classification reasoning
    processing_time: float  # Total processing time (seconds)
    timing_breakdown: Dict[str, float]  # Detailed timing
```

**Usage**: Returned by `transcribe_and_classify()`

**Example**:
```python
IntentMessage(
    intent="APPOINTMENT_SCHEDULING",
    confidence=0.98,
    entities={
        'user_goal': 'schedule appointment',
        'key_entities': {'action': 'schedule'},
        'extracted_meaning': 'Schedule appointment request'
    },
    user_context="User wants to book a meeting",
    reasoning="Pattern match: 'schedule' + 'appointment'",
    processing_time=0.045,
    timing_breakdown={
        'classification_ms': 10.0,
        'context_generation_ms': 3.5
    }
)
```

### RAGMessage

**File**: `leibniz_messages.py`

```python
@dataclass
class RAGMessage:
    answer: str  # Generated response text
    sources: List[str]  # Source document IDs
    confidence: float  # Retrieval confidence (0.0-1.0)
    processing_time_ms: float  # Processing time (milliseconds)
    timing_breakdown: Dict[str, Any]  # Method, cache hits, etc.
```

**Usage**: Returned by `handle_rag_query()`

**Example**:
```python
RAGMessage(
    answer="Office hours are Monday through Friday 9 AM to 5 PM.",
    sources=["office_hours.md", "contact_info.md"],
    confidence=0.92,
    processing_time_ms=1250.0,
    timing_breakdown={
        'method': 'persistent_rag',
        'rag_ms': 1200.0,
        'cache_check_ms': 2.5
    }
)
```

### TTSMessage

**File**: `leibniz_messages.py`

```python
@dataclass
class TTSMessage:
    text: str  # Synthesized text
    audio_path: Optional[str]  # Path to audio file
    duration_ms: float  # Audio duration (milliseconds)
```

**Usage**: Returned by `speak_friendly()`

**Example**:
```python
TTSMessage(
    text="Hello! How can I help you today?",
    audio_path="/path/to/audio.wav",
    duration_ms=2500.0
)
```

---

## Error Handling & Fallbacks

### Audio Input Errors

#### 1. FastRTC Connection Loss

**Detection**: Handler detects disconnection  
**Handling**:
- Clears audio queues
- Logs error
- Continues with next attempt
- Returns None transcript

#### 2. VAD Timeout

**Detection**: No speech detected within timeout  
**Handling**:
- Returns None transcript
- Increments `consecutive_no_input`
- Speaks timeout prompt
- Resets VAD after 2 consecutive timeouts

#### 3. STT API Failure

**Detection**: Gemini Live API error  
**Handling**:
- Falls back to cached transcript (if available)
- Logs error
- Returns empty transcript
- Continues conversation

### Intent Classification Errors

#### 1. Parser Unavailable

**Detection**: `_leibniz_parser` is None  
**Handling**:
- Falls back to basic classification
- Returns UNCLEAR intent
- Continues with fallback response
- Logs warning

#### 2. Semantic Extraction Failure

**Detection**: `extract_semantic_context()` raises exception  
**Handling**:
- Uses raw transcript
- Logs warning
- Continues with classification
- May reduce accuracy

### RAG Query Errors

#### 1. Cache Miss

**Handling**:
- Proceeds to speculative check
- Falls back to persistent RAG
- No error - expected behavior

#### 2. RAG Timeout

**Detection**: `asyncio.TimeoutError` in RAG execution  
**Handling**:
- Returns timeout fallback message
- Logs warning
- Continues conversation
- Records timeout in metrics

#### 3. RAG Error

**Detection**: Exception in RAG execution  
**Handling**:
- Returns error fallback message
- Logs error with traceback
- Continues conversation
- Records error in metrics

### TTS Errors

#### 1. Synthesis Failure

**Detection**: All TTS providers fail  
**Handling**:
- Tries next provider (ElevenLabs → Google → Gemini)
- Falls back to text-only mode if all fail
- Logs error
- Continues conversation

#### 2. FastRTC Sink Error

**Detection**: `audio_sink.write_audio()` raises exception  
**Handling**:
- Falls back to local playback (if available)
- Logs error
- Continues conversation
- May result in no audio output

#### 3. Queue Full (Backpressure)

**Detection**: `asyncio.QueueFull` exception  
**Handling**:
- Merges sentences if possible
- Drops oldest sentence if merge fails
- Logs warning
- Continues streaming

### Barge-In Handling

#### 1. User Speaks During TTS

**Detection**: `check_leibniz_barge_in()` returns True  
**Handling**:
- Sets `_cancel_streaming` event
- Stops current playback immediately
- Clears TTS queue
- Processes new input right away
- Logs barge-in event

#### 2. Echo Cancellation

**Detection**: Speech detected immediately after TTS start  
**Handling**:
- Debounces barge-in for 0.8s after TTS start
- Prevents false positives from echo/feedback
- Ignores speech during debounce window
- Logs debug message

---

## Key Functions & Classes Reference

### FastRTC Integration

| Function/Class | File | Line | Purpose |
|---------------|------|------|---------|
| `create_fastrtc_app()` | `leibniz_fastrtc_server.py` | 173 | Creates FastAPI app with FastRTC stream |
| `LeibnizFastRTCHandler` | `leibniz_fastrtc_handler.py` | 22 | Main FastRTC callback handler |
| `LeibnizFastRTCStreamHandler` | `leibniz_fastrtc_wrapper.py` | 65 | Alternative AsyncStreamHandler implementation |
| `FastRTCAudioSource` | `leibniz_fastrtc_adapters.py` | 34 | Browser audio input adapter |
| `FastRTCAudioSink` | `leibniz_fastrtc_adapters.py` | 167 | Browser audio output adapter |

### Main Pipeline Functions

| Function | File | Line | Purpose |
|---------|------|------|---------|
| `run_conversation_session()` | `leibniz_pro.py` | 3764 | Main conversation loop |
| `capture_and_transcribe()` | `leibniz_pro.py` | 2254 | Capture audio and transcribe |
| `transcribe_and_classify()` | `leibniz_pro.py` | 2306 | Complete capture + classification |
| `handle_rag_query()` | `leibniz_pro.py` | 2815 | RAG query processing |
| `handle_appointment_booking()` | `leibniz_pro.py` | 3294 | Appointment FSM handling |
| `speak_friendly()` | `leibniz_pro.py` | 1761 | TTS synthesis and playback |

### TTS Streaming Functions

| Function | File | Line | Purpose |
|---------|------|------|---------|
| `stream_rag_to_tts()` | `leibniz_pro.py` | 926 | Stream RAG response to TTS queue |
| `start_tts_consumer()` | `leibniz_pro.py` | 1021 | Start TTS consumer task |
| `consume_tts_streaming_queue()` | `leibniz_pro.py` | 1157 | Process TTS queue and synthesize/play |
| `finalize_tts_streaming()` | `leibniz_pro.py` | 1092 | Finalize streaming and cleanup |
| `split_into_sentences()` | `leibniz_pro.py` | 862 | Split text into sentences |

### VAD/STT Functions

| Function | File | Purpose |
|---------|------|---------|
| `capture_leibniz_speech()` | `leibniz_vad.py` | Main VAD capture function |
| `capture_speech_bidirectional()` | `leibniz_vad.py` | Bidirectional VAD with audio_source support |

### Intent Classification Functions

| Function | File | Purpose |
|---------|------|---------|
| `extract_semantic_context()` | `leibniz_semantic_extractor.py` | Fast semantic context extraction |
| `classify_intent()` | `leibniz_intent_parser.py` | Intent classification with LLM fallback |

### RAG Functions

| Function | File | Line | Purpose |
|---------|------|------|---------|
| `process_rag_query()` | `leibniz_rag.py` | Main RAG query processing |
| `compute_rag_confidence()` | `leibniz_pro.py` | 2679 | Compute confidence score |
| `process_rag_for_natural_conversation()` | `leibniz_pro.py` | 2763 | Clean and format RAG response |

### TTS Functions

| Function | File | Purpose |
|---------|------|---------|
| `synthesize_to_file()` | `leibniz_tts.py` | TTS synthesis with triple-provider fallback |
| `_synthesize_only()` | `leibniz_pro.py` | 1596 | Synthesis without playback |

### Helper Functions

| Function | File | Line | Purpose |
|---------|------|------|---------|
| `is_exit_phrase()` | `leibniz_pro.py` | 3712 | Check if transcript contains exit phrase |
| `normalize_sentence_for_hash()` | `leibniz_pro.py` | 476 | Normalize sentence for deduplication |
| `check_recent_synthesis()` | `leibniz_pro.py` | 770 | Check LRU cache for recent synthesis |
| `clear_tts_queue()` | `leibniz_pro.py` | 833 | Clear TTS queue and reset state |

---

## Performance Characteristics

### Latency Breakdown

#### 1. Audio Input (Browser → VAD)
- FastRTC transmission: ~20-50ms
- Queue buffering: <5ms
- **Total: ~25-55ms**

#### 2. STT Processing
- VAD capture: 500-2000ms (depends on speech length)
- Gemini Live API: 200-800ms
- **Total: 700-2800ms**

#### 3. Intent Classification
- Semantic extraction: <5ms
- Pattern matching: <10ms (fast path)
- LLM fallback: 200-800ms
- **Total: 5-810ms**

#### 4. RAG Query
- Cache hit: 1-5ms
- Speculative hit: 10-50ms
- Persistent RAG: 800-2000ms
- **Total: 1-2000ms**

#### 5. TTS Synthesis
- ElevenLabs: 300-800ms
- Google Cloud: 400-1000ms
- **Total: 300-1000ms**

#### 6. Audio Output (TTS → Browser)
- Queue buffering: <5ms
- FastRTC transmission: ~20-50ms
- **Total: ~25-55ms**

**Total End-to-End Latency**: ~1.5-5 seconds (typical)

### Optimization Strategies

#### 1. Pre-warming
- **VAD Session**: Pre-warmed during TTS playback
- **RAG Models**: Pre-warmed on speech detection
- **Effect**: Reduces cold start latency by 200-500ms

#### 2. Streaming TTS
- **Progressive Playback**: Sentences play as they're synthesized
- **2-Slot Pipeline**: Synthesizes N+1 while playing N
- **Effect**: Reduces perceived latency by 50-70%

#### 3. Caching
- **RAG Cache**: 30-50% hit rate
- **TTS Cache**: LRU with 10s TTL
- **Intent Cache**: 40-60% hit rate
- **Effect**: Reduces latency by 80-95% on cache hits

#### 4. Deduplication
- **RAG Queries**: Prevents duplicate executions
- **TTS Synthesis**: Prevents duplicate synthesis
- **Effect**: Reduces redundant work by 15-20%

---

## Conclusion

This documentation provides a complete trace of data flow through the Leibniz Pro pipeline, from browser audio input via FastRTC/WebRTC to audio output back to the browser. The system is designed for low-latency, real-time conversation with robust error handling and fallback mechanisms.

### Key Architectural Decisions

1. **Message-Based Communication**: Clean separation of concerns with structured messages
2. **Adapter Pattern**: FastRTC adapters bridge browser to pipeline without modifying core logic
3. **Streaming Architecture**: Progressive TTS for reduced perceived latency
4. **Robust Error Handling**: Multiple fallback layers at every stage
5. **State Management**: Clear session and conversation state tracking
6. **2-Slot Pipeline**: Synthesizes next sentence while playing current for smooth playback

### Integration Points

- **FastRTC Server**: Entry point for browser WebRTC connections
- **Audio Adapters**: Bridge browser audio to/from pipeline
- **Main Pipeline**: Unchanged core logic, accepts adapters as parameters
- **Message Types**: Structured data flow between components

For implementation details, refer to the specific files mentioned in each section.

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Maintained By**: Leibniz Agent Development Team
