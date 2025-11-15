# VAD & STT Bug Fixes for Complete, Robust Real-time Speech Transcription

## Executive Summary

Both `leibniz_vad.py` and `leibniz_stt.py` have **critical bugs** preventing complete, gap-free transcription:

1. **No Gemini VAD configuration** - Using defaults that miss speech onset and cut off slow speakers
2. **Incomplete character handling** - Fragments may contain partial words causing gaps when joined
3. **Audio streaming race conditions** - Sync callbacks + async queues lead to dropped chunks
4. **Silence detection conflicts** - Custom logic fights with Gemini's built-in VAD

---

## 🔴 Critical Fix #1: Configure Gemini's Automatic Activity Detection

### Problem
Neither implementation configures Gemini Live API's `automatic_activity_detection` parameters. This causes:
- **Missed speech onset** (default 100ms padding may be insufficient)
- **Premature cutoff** of slow/thoughtful speakers (default 300ms silence threshold)
- **Unreliable VAD** due to default sensitivity settings

### Solution

**Add proper VAD configuration during session setup:**

```python
async def _get_session(client, model_name, config):
    """Get or create Live API session with proper VAD configuration."""
    
    # Configure VAD for robust speech capture
    vad_config = {
        "setup": {
            "model": f"models/{model_name}",
            "generation_config": {
                "response_modalities": ["TEXT"]  # We only need transcription
            },
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "disabled": False,  # Enable Gemini's VAD
                    
                    # CRITICAL: Prefix padding captures speech onset
                    # 200-300ms ensures we don't miss first syllables
                    "prefix_padding_ms": 250,
                    
                    # CRITICAL: Silence duration before end-of-speech
                    # 400-500ms balances responsiveness with natural pauses
                    "silence_duration_ms": 450,
                    
                    # Sensitivity tuning for reliable detection
                    "start_of_speech_sensitivity": "HIGH",  # Catch speech quickly
                    "end_of_speech_sensitivity": "MEDIUM"   # Don't cut off mid-thought
                }
            }
        }
    }
    
    session = client.aio.live.connect(model=model_name, config=vad_config)
    
    # Send setup message
    await session.send(vad_config)
    
    return session
```

### Implementation Changes

**For `leibniz_vad.py`:**

```python
class LeibnizVADConfig:
    def __init__(self):
        # ... existing config ...
        
        # ADD: Gemini VAD parameters
        self.vad_prefix_padding_ms: int = 250
        self.vad_silence_duration_ms: int = 450
        self.vad_start_sensitivity: str = "HIGH"
        self.vad_end_sensitivity: str = "MEDIUM"
```

**For session creation:**

```python
async def capture_speech_bidirectional(self, streaming_callback=None):
    # ... existing code ...
    
    # REPLACE the simple session.get() with configured setup
    session_config = {
        "setup": {
            "model": f"models/{self.config.model_name}",
            "generation_config": {"response_modalities": ["TEXT"]},
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "disabled": False,
                    "prefix_padding_ms": self.config.vad_prefix_padding_ms,
                    "silence_duration_ms": self.config.vad_silence_duration_ms,
                    "start_of_speech_sensitivity": self.config.vad_start_sensitivity,
                    "end_of_speech_sensitivity": self.config.vad_end_sensitivity
                }
            }
        }
    }
    
    session = await LeibnizPersistentSession.get_session(
        self.client, 
        self.config.model_name, 
        self.config,
        vad_config=session_config  # Pass VAD config
    )
```

---

## 🔴 Critical Fix #2: Handle Incomplete Characters and Word Boundaries

### Problem
Gemini Live API **does NOT guarantee word boundaries** in transcript fragments. You may receive:
- Fragment 1: `"Hello, my name is Al"`
- Fragment 2: `"ice and I need help"`

Joining with spaces creates: `"Hello, my name is Al ice and I need help"` ❌

### Solution: Smart Fragment Merging

```python
class TranscriptBuffer:
    """Handles incomplete word boundaries in streaming transcripts."""
    
    def __init__(self):
        self.fragments = []
        self.pending_partial = ""  # Buffer for incomplete words
    
    def add_fragment(self, text: str) -> str:
        """
        Add fragment with word boundary detection.
        
        Returns: Complete text ready for callback (may be empty if buffering)
        """
        if not text or not text.strip():
            return ""
        
        # Combine with any pending partial word
        combined = self.pending_partial + text
        self.pending_partial = ""
        
        # Check if fragment ends mid-word (no trailing space/punctuation)
        if not self._ends_complete_word(combined):
            # Buffer the last word for next fragment
            words = combined.rsplit(maxsplit=1)
            if len(words) == 2:
                complete_text, partial_word = words
                self.pending_partial = partial_word
                combined = complete_text
            else:
                # Entire fragment is one incomplete word
                self.pending_partial = combined
                return ""
        
        # Store complete fragment
        if combined.strip():
            self.fragments.append(combined)
        
        return combined
    
    def _ends_complete_word(self, text: str) -> bool:
        """Check if text ends with a complete word."""
        # Ends with punctuation or whitespace = complete
        if text[-1] in ' \t\n.,!?;:':
            return True
        
        # Check if last "word" is actually a word (not a fragment)
        last_word = text.split()[-1] if text.split() else ""
        
        # Very short = likely incomplete (except common words)
        if len(last_word) < 2 and last_word.lower() not in ['i', 'a']:
            return False
        
        # Check for common incomplete patterns
        incomplete_patterns = [
            r'^[a-z]$',           # Single letter (except I/A)
            r'-$',                # Ends with hyphen
            r'^\d{1,2}$'          # 1-2 digits (likely incomplete number)
        ]
        
        import re
        for pattern in incomplete_patterns:
            if re.match(pattern, last_word):
                return False
        
        return True
    
    def get_final_transcript(self) -> str:
        """Get complete transcript, including any buffered partial."""
        all_text = self.fragments.copy()
        if self.pending_partial:
            all_text.append(self.pending_partial)
        
        return ' '.join(all_text).strip()
```

### Integration into VAD/STT

**Replace fragment accumulation:**

```python
async def capture_speech_bidirectional(self, streaming_callback=None):
    # ... existing setup ...
    
    # REPLACE: fragments = []
    transcript_buffer = TranscriptBuffer()  # NEW
    
    async for response in session.receive():
        if response.server_content and response.server_content.input_transcription:
            text = response.server_content.input_transcription.text
            
            if text and text.strip():
                # REPLACE: fragments.append(text)
                complete_text = transcript_buffer.add_fragment(text)  # NEW
                
                # Only callback with complete text
                if complete_text and streaming_callback:
                    streaming_callback(complete_text, is_final=False)
    
    # At the end:
    # REPLACE: transcript_result = ' '.join(fragments).strip()
    transcript_result = transcript_buffer.get_final_transcript()  # NEW
```

---

## 🔴 Critical Fix #3: Proper Audio Streaming Without Race Conditions

### Problem
Both implementations use **synchronous audio callbacks** with **async queues**, causing:
- Race conditions when event loop changes
- Dropped chunks when queue is full
- No handling of device errors mid-stream

### Solution: Thread-Safe Audio Queue

```python
import threading
from queue import Queue  # Use threading.Queue, not asyncio.Queue

class RobustAudioStreamer:
    """Thread-safe audio streaming for Gemini Live API."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_queue = Queue(maxsize=100)  # Thread-safe queue
        self.pre_buffer = []
        self.is_streaming = False
        self.error_event = threading.Event()
        self.dropped_chunks = 0
        
    def audio_callback(self, indata, frames, time_info, status):
        """Sounddevice callback - runs in audio thread."""
        if status:
            print(f"⚠️ Audio status: {status}")
        
        try:
            # Convert to PCM16
            pcm_data = (indata.flatten() * 32767).astype(np.int16).tobytes()
            
            # Always add to pre-buffer (rolling window)
            self.pre_buffer.append(pcm_data)
            if len(self.pre_buffer) > 20:  # Keep last 1 second
                self.pre_buffer.pop(0)
            
            # Try to queue for streaming
            if self.is_streaming:
                try:
                    self.audio_queue.put_nowait(pcm_data)
                except:
                    self.dropped_chunks += 1
                    if self.dropped_chunks % 10 == 0:
                        print(f"⚠️ Dropped {self.dropped_chunks} audio chunks")
        
        except Exception as e:
            print(f"❌ Audio callback error: {e}")
            self.error_event.set()
    
    async def stream_audio_to_session(self, session):
        """Stream audio from queue to Gemini session."""
        self.is_streaming = True
        
        # Send pre-buffer first
        for buffered_chunk in self.pre_buffer:
            try:
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=buffered_chunk,
                        mime_type=f"audio/pcm;rate={self.sample_rate}"
                    )
                )
            except Exception as e:
                print(f"❌ Pre-buffer send error: {e}")
        
        # Stream real-time audio
        while self.is_streaming and not self.error_event.is_set():
            try:
                # Use run_in_executor to await thread-safe queue
                chunk = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.audio_queue.get, 
                    True,  # block
                    0.1    # timeout
                )
                
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=chunk,
                        mime_type=f"audio/pcm;rate={self.sample_rate}"
                    )
                )
            
            except Exception as e:
                # Timeout is expected, continue
                if "Empty" not in str(type(e)):
                    print(f"❌ Stream error: {e}")
                    break
        
        self.is_streaming = False
    
    def start_stream(self):
        """Start audio input stream."""
        import sounddevice as sd
        
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=800,  # 50ms at 16kHz
            callback=self.audio_callback
        )
        
        stream.start()
        return stream
```

### Integration

```python
async def capture_speech_bidirectional(self, streaming_callback=None):
    streamer = RobustAudioStreamer(self.config.sample_rate)
    
    with streamer.start_stream():
        # Start async audio streaming task
        send_task = asyncio.create_task(
            streamer.stream_audio_to_session(session)
        )
        
        try:
            # Process responses...
            async for response in session.receive():
                # ... handle transcription ...
                pass
        
        finally:
            # Clean shutdown
            streamer.is_streaming = False
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
```

---

## 🟡 Medium Priority Fix #4: Proper Turn Detection

### Problem
Custom silence detection conflicts with Gemini's `turn_complete` signal, causing:
- Premature finalization before user finishes speaking
- Missed transcription when timeout fires before natural end

### Solution

```python
async def capture_speech_bidirectional(self, streaming_callback=None):
    # ... setup ...
    
    turn_complete_received = False
    speech_detected = False
    
    async for response in session.receive():
        # Check for Gemini's turn_complete signal
        if hasattr(response, 'server_content') and response.server_content:
            if hasattr(response.server_content, 'turn_complete'):
                if response.server_content.turn_complete:
                    # TRUST GEMINI'S VAD
                    if speech_detected:
                        logger.info("✅ Gemini signaled turn_complete - finalizing")
                        turn_complete_received = True
                        break
                    else:
                        # Turn complete but no speech yet = user paused
                        logger.debug("Turn complete but no speech - continuing")
            
            # Handle transcription
            if response.server_content.input_transcription:
                text = response.server_content.input_transcription.text
                if text and text.strip():
                    speech_detected = True
                    # ... process fragment ...
        
        # ONLY use timeout as safety fallback
        elapsed = time.time() - start_time
        if elapsed >= self.config.start_timeouts:
            logger.warning(f"⚠️ Safety timeout after {elapsed}s")
            break
    
    # If turn_complete not received, send audioStreamEnd to flush
    if not turn_complete_received:
        try:
            await session.send({"client_content": {"audio_stream_end": {}}})
            logger.info("Sent audioStreamEnd to flush cached audio")
        except:
            pass
```

---

## 🟢 Additional Improvements

### 5. Deduplicate Repeated Fragments

Gemini may send duplicate fragments. Add deduplication:

```python
class TranscriptBuffer:
    def __init__(self):
        self.fragments = []
        self.last_fragment = ""
        self.pending_partial = ""
    
    def add_fragment(self, text: str) -> str:
        # Skip exact duplicates
        if text == self.last_fragment:
            return ""
        
        self.last_fragment = text
        
        # ... rest of logic ...
```

### 6. Validate Audio Quality Before Streaming

```python
def validate_audio_chunk(audio_data: np.ndarray) -> bool:
    """Check if audio chunk is valid before sending."""
    # Check for silence
    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms < 0.001:  # Essentially silent
        return False
    
    # Check for clipping
    clipped = np.sum(np.abs(audio_data) >= 0.99)
    if clipped / len(audio_data) > 0.05:  # >5% clipped
        logger.warning("⚠️ Audio clipping detected")
    
    return True
```

### 7. Handle Interruptions Properly

```python
async for response in session.receive():
    if response.server_content and response.server_content.interrupted:
        # Model was interrupted (user barged in)
        logger.info("🔄 Generation interrupted - discarding incomplete response")
        
        # Discard any buffered partial words
        transcript_buffer.pending_partial = ""
        
        # Stop any audio playback if implemented
        # stop_audio_playback()
```

---

## Complete Recommended Architecture

```python
# NEW: Robust transcript capture with all fixes
async def capture_complete_transcript(
    vad_config: dict,
    streaming_callback: Optional[Callable] = None
) -> str:
    """
    Capture complete, gap-free speech transcript.
    
    Returns: Complete transcript with no incomplete characters or gaps
    """
    
    # 1. Configure Gemini VAD properly
    session = await get_configured_session(vad_config)
    
    # 2. Use robust audio streamer
    streamer = RobustAudioStreamer(sample_rate=16000)
    
    # 3. Use smart transcript buffer
    transcript_buffer = TranscriptBuffer()
    
    with streamer.start_stream():
        send_task = asyncio.create_task(
            streamer.stream_audio_to_session(session)
        )
        
        try:
            speech_detected = False
            turn_complete = False
            
            async for response in session.receive():
                # Handle interruptions
                if response.server_content and response.server_content.interrupted:
                    transcript_buffer.pending_partial = ""
                    continue
                
                # Check for turn_complete
                if response.server_content and response.server_content.turn_complete:
                    if speech_detected:
                        turn_complete = True
                        break
                
                # Process transcription
                if response.server_content and response.server_content.input_transcription:
                    text = response.server_content.input_transcription.text
                    if text and text.strip():
                        speech_detected = True
                        
                        # Smart fragment merging
                        complete_text = transcript_buffer.add_fragment(text)
                        
                        if complete_text and streaming_callback:
                            streaming_callback(complete_text, is_final=False)
            
            # Get final transcript
            final_transcript = transcript_buffer.get_final_transcript()
            
            # Apply normalization AFTER all fragments merged
            final_transcript = normalize_transcript(final_transcript)
            
            # Final callback
            if streaming_callback:
                streaming_callback(final_transcript, is_final=True)
            
            return final_transcript
        
        finally:
            streamer.is_streaming = False
            send_task.cancel()
```

---

## Testing Checklist

✅ **Test slow/thoughtful speakers** - should not cut off mid-sentence  
✅ **Test rapid speech** - should capture all words  
✅ **Test with pauses** - should allow natural pauses without premature end  
✅ **Test word boundaries** - "Alice" should not become "Al ice"  
✅ **Test interruptions** - barge-in should discard incomplete text  
✅ **Test long audio** - no dropped chunks after minutes of streaming  
✅ **Test device errors** - graceful handling if microphone disconnects  

---

## Summary of Changes

| Issue | Current Behavior | Fixed Behavior |
|-------|-----------------|----------------|
| Missing VAD config | Uses Gemini defaults | Configures prefix_padding, silence_duration, sensitivity |
| Incomplete characters | Joins fragments blindly | Smart merging with word boundary detection |
| Audio race conditions | Sync callback + async queue | Thread-safe queue with proper executor |
| Silence conflicts | Custom logic fights Gemini | Trusts Gemini's turn_complete signal |
| No deduplication | May process same fragment twice | Deduplicates based on content |
| No validation | Sends all audio | Validates quality before streaming |
| Poor interruption handling | Keeps incomplete text | Discards buffered partials on interrupt |

These fixes will provide **complete, gap-free, robust real-time transcription** aligned with Gemini Live API best practices.
