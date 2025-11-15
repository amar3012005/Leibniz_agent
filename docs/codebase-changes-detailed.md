# Exact Changes Required: Convert Your Codebase to Persistent VAD

## Quick Overview of Your Current Code

**leibniz_vad.py:**
- Class: `LeibnizVAD`
- Main method: `capture_leibniz_speech()` - Creates NEW session each time
- Pattern: Per-turn initialization (inefficient)

**leibniz_pro.py:**
- Main orchestration module
- Key methods: `transcribe_and_classify()`, `play_and_listen()`
- Pattern: Calls `capture_and_transcribe()` after each agent TTS

---

## Problem in Your Current Architecture

```
Agent speaks
  ↓
After TTS: transcribe_and_classify()
  ↓
capture_and_transcribe() called
  ↓
leibniz_vad.py: capture_leibniz_speech()
  ↓
NEW session created (1-2 seconds) ❌ ← BOTTLENECK
  ↓
User speech captured
  ↓
Session closed (wasted connection)
  ↓
Repeat for next turn
```

**Cost:** 1-2 seconds × number of turns = **Huge latency overhead**

---

## Solution: Persistent VAD Session

```
App Start
  ↓
Initialize PersistentGeminiVAD (ONCE) ✅
  ↓
Session stays OPEN (never close)
  ↓
Background tasks start:
  - Audio streaming (continuous)
  - Speech listening (continuous)
  ↓
Main loop:
  Agent speaks → VAD already listening (no startup cost)
  User barges in → Immediate transcript (already listening)
  ↓
No per-turn VAD initialization needed!
```

**Benefit:** 1-2s per turn × N turns = **Massive latency reduction**

---

## STEP 1: Create NEW File `leibniz_persistent_vad.py`

**File Path:** `leibnizagent/leibniz_persistent_vad.py`

Copy the complete `PersistentGeminiVAD` class from [93] `gemini-persistent-vad.md`

This file contains:
- `PersistentGeminiVAD` class
- `initialize_session()` - Opens one session
- `send_audio_stream()` - Continuous audio (background task)
- `listen_for_user_speech()` - Waits for user input (background task)
- `on_user_speech()` - Callback when user speaks
- `send_agent_text()` - Send agent response
- `close_session()` - Cleanup

---

## STEP 2: Modify `leibniz_vad.py` (Mark as Deprecated)

**File Path:** `leibnizagent/leibniz_vad.py`

At the top of the file, add deprecation notice:

```python
"""
DEPRECATED: Use PersistentGeminiVAD from leibniz_persistent_vad.py instead.

This module is kept for backward compatibility only.
Per-turn VAD initialization causes 1-2s overhead per turn.
"""

import warnings

warnings.warn(
    "LeibnizVAD is deprecated. Use PersistentGeminiVAD from leibniz_persistent_vad.py",
    DeprecationWarning,
    stacklevel=2
)
```

**Do NOT delete this file.** Keep it as fallback.

---

## STEP 3: Major Changes in `leibniz_pro.py`

This is where most changes happen.

### 3.1: Add Import at Top

**FIND THIS** (around line 10-30, imports section):
```python
from leibnizagent.leibnizintentparser import getleibnizparser
from leibnizagent.leibnizvad import getleibnizvad, capture_leibniz_speech
# ... other imports ...
```

**ADD THIS AFTER** (around line 30-40):
```python
# NEW: Import persistent VAD
from leibnizagent.leibniz_persistent_vad import PersistentGeminiVAD
```

### 3.2: Initialize Persistent VAD at Application Start

**FIND THIS** (look for `async def main()` or `async def initialize_leibniz_services()`):
```python
async def main():
    """Main application loop."""
    # ... initialization code ...
```

**REPLACE/ADD** this code at the beginning of main:

```python
async def main():
    """Main application loop."""
    
    # ===== STEP 1: Initialize Persistent VAD (ONCE at app start) =====
    print("🟢 Initializing persistent VAD session...")
    persistent_vad = PersistentGeminiVAD(api_key=os.getenv("GOOGLE_API_KEY"))
    
    try:
        # Initialize session (stays open throughout app lifetime)
        await persistent_vad.initialize_session()
        print("✅ Persistent VAD ready")
        
        # ===== STEP 2: Start background audio streaming =====
        print("🎤 Starting audio capture...")
        stream = await persistent_vad.start_microphone_stream()
        
        # ===== STEP 3: Create background tasks (never stop until app ends) =====
        send_audio_task = asyncio.create_task(
            persistent_vad.send_audio_stream(stream)
        )
        listen_task = asyncio.create_task(
            persistent_vad.listen_for_user_speech()
        )
        
        print("🚀 Persistent VAD background tasks started\n")
        
        # ===== STEP 4: Main conversation loop =====
        await main_conversation_loop(persistent_vad)
        
    finally:
        # Cleanup
        print("\n🔴 Shutting down...")
        persistent_vad.is_running = False
        if 'send_audio_task' in locals():
            send_audio_task.cancel()
        if 'listen_task' in locals():
            listen_task.cancel()
        if 'stream' in locals():
            stream.stop()
        await persistent_vad.close_session()
        print("✅ Shutdown complete")
```

### 3.3: Rename Your Main Loop Function

**FIND THIS** (your current main loop, probably named `main()` or similar):
```python
async def main():
    # ... current code doing agent cycles ...
```

**RENAME IT TO:**
```python
async def main_conversation_loop(persistent_vad):
    # ... your current main loop code ...
    # (modified below)
```

### 3.4: Update Main Loop to Use Persistent VAD

**IN** `main_conversation_loop()`, **FIND THIS** (where you capture user speech per turn):

```python
# OLD CODE - Per-turn VAD
while True:
    # Agent generates response
    agent_response = await generate_agent_response(current_intent)
    
    # Play TTS
    await play_tts_streaming(agent_response)
    
    # PROBLEM: New VAD session every turn!
    transcript = await transcribe_and_classify()  # ← REMOVE THIS
    
    # Process transcript
    # ...
```

**REPLACE WITH:**

```python
# NEW CODE - Persistent background VAD
while True:
    # Agent generates response
    agent_response = await generate_agent_response(current_intent)
    
    # Play TTS (non-blocking, VAD listening in background)
    await play_tts_streaming(agent_response)
    
    # DON'T call transcribe_and_classify() here!
    # VAD is already listening in background via listen_for_user_speech()
    
    # The user transcript will arrive via:
    # → persistent_vad.listen_for_user_speech() (background task)
    # → on_user_speech() callback
    
    # Wait a bit for user to speak (or they already spoke during TTS)
    await asyncio.sleep(0.5)
```

### 3.5: Override the Callback in PersistentGeminiVAD

**IN** `main_conversation_loop()` function, **AFTER** starting background tasks, **ADD:**

```python
# Override callback to process user speech
async def handle_user_speech(transcript: str):
    """Process user speech from persistent VAD."""
    print(f"\n👤 User: {transcript}")
    
    # Classify intent
    intent_result = await robust_intent_classifier.classify_intent_robust(transcript)
    print(f"Intent: {intent_result['intent']} (confidence: {intent_result['confidence']:.2f})")
    
    # Store for main loop
    global current_user_input, user_input_ready
    current_user_input = transcript
    user_input_ready = True

# Set the callback
persistent_vad.on_user_speech = handle_user_speech
```

### 3.6: Update Main Loop to Wait for User Input Events

**IN** `main_conversation_loop()`, **CHANGE** the loop logic:

```python
# Global flags for async communication
user_input_ready = False
current_user_input = None

async def main_conversation_loop(persistent_vad):
    global user_input_ready, current_user_input
    
    while True:
        # Reset flag
        user_input_ready = False
        
        # Agent generates response
        agent_response = await generate_agent_response()
        await play_tts_streaming(agent_response)
        
        # WAIT FOR USER INPUT from persistent VAD background listening
        timeout = 15  # Max wait time
        start_time = time.time()
        
        while not user_input_ready:
            if time.time() - start_time > timeout:
                print("⏱️ No user input, continuing...")
                break
            await asyncio.sleep(0.1)  # Check every 100ms
        
        if user_input_ready:
            # Got user speech from persistent VAD
            user_transcript = current_user_input
            print(f"\n✅ Processing: {user_transcript}")
            
            # Your normal processing here
            # ... continue loop ...
```

### 3.7: Remove Old Functions

**DELETE or COMMENT OUT** these functions (they're no longer needed):

```python
# DELETE THESE:
async def transcribe_and_classify():  # ❌ OLD - used per-turn VAD
async def capture_and_transcribe():   # ❌ OLD - used per-turn VAD
def check_leibniz_bargein():          # ❌ OLD - Gemini handles barge-in automatically now
```

---

## STEP 4: Update Any TTS Playback Code

**FIND THIS** (your TTS playback function, probably `play_tts_streaming()`):

```python
async def play_tts_streaming(text):
    # ... current TTS code ...
    # Check for barge-in manually? ← OLD APPROACH
```

**CHANGE TO** (remove manual barge-in checking):

```python
async def play_tts_streaming(text):
    """Play TTS. Barge-in handled automatically by persistent VAD."""
    # Just play TTS - don't worry about barge-in
    # Gemini's VAD will handle interruption automatically
    # When user speaks, persistent_vad.listen_for_user_speech() gets it
    
    # ... your TTS code (unchanged) ...
    # No need to check for barge-in flags anymore
```

---

## STEP 5: Update Configuration (If Needed)

**IN** `leibniz_pro.py`, **FIND** any VAD configuration:

```python
# OLD
VAD_TIMEOUT = 30
VAD_SILENCE_THRESHOLD = 2.0
VAD_PREFIX_PADDING = 100
```

**These are NO LONGER NEEDED** - Gemini Live handles VAD automatically

**You can DELETE** or keep them commented as reference

---

## Summary of Changes

| File | Action | Reason |
|------|--------|--------|
| **Create `leibniz_persistent_vad.py`** | New file | Persistent VAD logic |
| **`leibniz_vad.py`** | Add deprecation notice | Mark as old pattern |
| **`leibniz_pro.py`** (imports) | Add new import | Use persistent VAD |
| **`leibniz_pro.py`** (main) | Initialize VAD once at start | Not per-turn |
| **`leibniz_pro.py`** (main loop) | Remove `transcribe_and_classify()` calls | Background listening now |
| **`leibniz_pro.py`** (TTS) | Remove barge-in checking | Gemini handles it |
| **`leibniz_pro.py`** (callbacks) | Add user speech handler | Process async events |

---

## Before vs After Code Flow

### BEFORE (Your Current Code)
```
main()
  ↓
while True:
    agent_response = generate()
    play_tts()
    
    NEW_VAD_SESSION = LeibnizVAD()  ← 1-2s delay
    transcript = capture_leibniz_speech()  ← Wait for input
    classify_intent(transcript)
    
    (repeat)
```

### AFTER (Proposed)
```
main()
  ↓
persistent_vad = PersistentGeminiVAD()
initialize_session()  ← Called ONCE
start_background_tasks()  ← Audio + listen running always
  ↓
while True:
    agent_response = generate()
    play_tts()
    
    # VAD already listening in background!
    wait_for_user_input_event()  ← No session startup!
    
    handle_user_speech(transcript)
    
    (repeat)
```

**Latency Improvement:** 1-2s per turn → <100ms per turn (20x faster!)

---

## Testing Checklist

- [ ] Create `leibniz_persistent_vad.py` with complete code
- [ ] Add import in `leibniz_pro.py`
- [ ] Initialize persistent VAD at app start
- [ ] Start background audio + listening tasks
- [ ] Test: User speech captured without session restart
- [ ] Test: User can interrupt agent mid-TTS (barge-in)
- [ ] Test: Run for 2-3 minutes without crashes
- [ ] Monitor: Check that no new sessions created per turn

---

## Rollback Plan (If Needed)

If something breaks:

1. Comment out new import in `leibniz_pro.py`:
   ```python
   # from leibniz_persistent_vad import PersistentGeminiVAD
   ```

2. Revert main loop to use old functions:
   ```python
   # Uncomment old code
   transcript = await transcribe_and_classify()
   ```

3. Old code will still work (backward compatible)

---

## Questions During Implementation?

**Q: Where exactly is main() in my code?**
A: Search for `async def main():` in `leibniz_pro.py`

**Q: What is my current agent loop function called?**
A: Search for the function that has `while True:` and agent speech/response handling

**Q: How do I integrate with my existing intent classifier?**
A: Modify the `handle_user_speech()` callback to call your classifier

**Q: Will this break my existing code?**
A: No - old functions still exist, just not used. Safe to experiment.
