# Specific Code Locations & Exact Line Changes

## Your Current Code Structure Analysis

Based on your codebase:

### `leibniz_vad.py` Current Structure
```
Line 1-50:    Imports
Line 50-100:  Config classes (@dataclass)
Line 100-200: LeibnizVAD class __init__
Line 200-300: initialize_client()
Line 300-400: capture_leibniz_speech() ← THIS IS THE BOTTLENECK
Line 400-500: capture_speech_bidirectional()
Line 500+:    Helper methods
```

**Problem Location:** `capture_leibniz_speech()` creates NEW session on every call

### `leibniz_pro.py` Current Structure
```
Line 1-50:    Imports
Line 50-100:  Configuration
Line 100-200: Helper functions
Line 200-300: Main orchestration logic
Line 300-400: transcribe_and_classify() ← CALLS capture_leibniz_speech() PER TURN
Line 400-500: play_and_listen()
Line 500+:    Main loop, async functions
```

**Problem Location:** Every agent turn calls `transcribe_and_classify()` which creates new VAD session

---

## EXACT CHANGES NEEDED

### Change 1: `leibniz_vad.py` - Add Deprecation

**Location:** Line 1-10 (top of file, after module docstring)

**FIND:**
```python
#!/usr/bin/env python3
"""
Leibniz Bidirectional VAD Integration Module
...
"""

import os
import asyncio
```

**ADD AFTER the module docstring, BEFORE imports:**
```python
"""
DEPRECATED: Per-turn VAD initialization.
Use PersistentGeminiVAD from leibniz_persistent_vad.py instead.

This module is kept for backward compatibility only.
Creating new VAD session per turn causes 1-2s overhead per turn.
"""

import warnings
warnings.warn(
    "LeibnizVAD pattern is deprecated. Use PersistentGeminiVAD instead.",
    DeprecationWarning,
    stacklevel=2
)
```

---

### Change 2: `leibniz_pro.py` - Add Persistent VAD Import

**Location:** Top of file, imports section (~line 20-40)

**FIND:**
```python
from leibnizagent.leibnizintentparser import getleibnizparser
from leibnizagent.leibnizvad import getleibnizvad
from leibnizagent.leibnizvad import LeibnizVAD
# ... other imports ...
```

**ADD AFTER:**
```python
# NEW: Persistent background VAD
from leibnizagent.leibniz_persistent_vad import PersistentGeminiVAD
```

**Also find and verify these imports exist:**
```python
import asyncio
import time
import os
from google import genai  # For accessing Gemini API
```

---

### Change 3: `leibniz_pro.py` - Initialize Persistent VAD

**Location:** In `main()` function, very beginning (~line 300-400 area, depending on file)

**FIND THE MAIN FUNCTION:**
```python
async def main():
    """Main application entry point."""
    logger.info("Starting Leibniz Agent")
    
    # ... existing initialization code ...
```

**ADD AT THE VERY BEGINNING (before any other code in main()):**
```python
async def main():
    """Main application entry point."""
    logger.info("Starting Leibniz Agent")
    
    # ========== NEW: Initialize Persistent VAD ==========
    logger.info("🟢 Initializing persistent VAD session...")
    persistent_vad = PersistentGeminiVAD(api_key=os.getenv("GOOGLE_API_KEY"))
    
    try:
        # Initialize session (stays open throughout app lifetime)
        await persistent_vad.initialize_session()
        logger.info("✅ Persistent VAD ready")
        
        # Start background audio streaming
        logger.info("🎤 Starting audio capture...")
        stream = await persistent_vad.start_microphone_stream()
        
        # Create background tasks (never stop until app ends)
        send_audio_task = asyncio.create_task(
            persistent_vad.send_audio_stream(stream)
        )
        listen_task = asyncio.create_task(
            persistent_vad.listen_for_user_speech()
        )
        
        logger.info("🚀 Persistent VAD background tasks started")
        
        # ========== CONTINUE WITH EXISTING CODE ==========
        # ... your existing initialization code ...
```

---

### Change 4: `leibniz_pro.py` - Remove Per-Turn VAD Calls

**Location:** In main conversation loop (~line 400-600 area)

**FIND THE SECTION** that looks like:
```python
while True:
    # Agent generates response
    agent_response = await generate_agent_response(...)
    
    # Play agent speech
    await play_and_listen(agent_response)
    
    # OR alternative:
    await play_tts_streaming(agent_response)
    
    # CAPTURE USER SPEECH - THIS IS THE OLD PATTERN
    transcript_result = await transcribe_and_classify()
    
    # Process transcript
    user_intent = transcript_result.get('intent', 'UNCLEAR')
```

**CHANGE TO:**
```python
while True:
    # Agent generates response
    agent_response = await generate_agent_response(...)
    
    # Play agent speech (non-blocking, VAD listening in background)
    await play_tts_streaming(agent_response)
    
    # NO NEED TO CALL transcribe_and_classify() ANYMORE!
    # VAD is already listening in background via listen_for_user_speech()
    
    # Wait for user input from background VAD
    # The user speech will arrive via persistent_vad.listen_for_user_speech()
    await asyncio.sleep(0.5)
```

---

### Change 5: `leibniz_pro.py` - Add User Speech Callback

**Location:** In main conversation loop, AFTER initializing persistent VAD tasks

**ADD THIS NEW FUNCTION:**
```python
async def main():
    """Main application entry point."""
    # ... VAD initialization code from Change 3 ...
    
    # ========== NEW: Setup user speech handler ==========
    async def handle_user_speech(transcript: str):
        """Process user speech from persistent VAD."""
        logger.info(f"👤 User: {transcript}")
        
        # Classify intent using your existing classifier
        intent_classifier = get_robust_intent_parser()
        try:
            intent_result = await intent_classifier.classify_intent_robust(transcript)
            user_intent = intent_result['intent']
            confidence = intent_result['confidence']
            context = intent_result.get('context', {})
            
            logger.info(f"Intent: {user_intent} (confidence: {confidence:.2f})")
            
            # Store for main loop to process
            global current_user_transcript, current_intent, user_input_received
            current_user_transcript = transcript
            current_intent = user_intent
            user_input_received = True
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            current_intent = "UNCLEAR"
            current_user_transcript = transcript
            user_input_received = True
    
    # Connect callback to persistent VAD
    persistent_vad.on_user_speech = handle_user_speech
    
    logger.info("✅ User speech handler configured")
    
    # ========== Continue with rest of initialization ==========
```

---

### Change 6: `leibniz_pro.py` - Update Main Loop to Use Events

**Location:** Main conversation loop while True block

**FIND:**
```python
while True:
    # Current loop that calls transcribe_and_classify()
    
    agent_response = ...
    await play_tts_streaming(...)
    transcript_result = await transcribe_and_classify()  # ← OLD
```

**CHANGE THE ENTIRE LOOP TO:**
```python
# Global variables for async event communication
current_user_transcript = None
current_intent = None
user_input_received = False

while True:
    global user_input_received, current_user_transcript, current_intent
    
    # Reset event flag
    user_input_received = False
    current_user_transcript = None
    current_intent = None
    
    # Generate agent response
    agent_response = await generate_agent_response(current_intent)
    
    # Play TTS (non-blocking, VAD listening in background)
    await play_tts_streaming(agent_response)
    
    # WAIT for user input from persistent VAD (max 15 seconds)
    timeout = 15
    start_time = time.time()
    
    while not user_input_received:
        if time.time() - start_time > timeout:
            logger.warning("⏱️ No user input received, timeout")
            break
        await asyncio.sleep(0.1)  # Check every 100ms
    
    if user_input_received and current_user_transcript:
        logger.info(f"✅ User input processed: {current_user_transcript}")
        
        # Continue main loop with current_intent already set
        # by handle_user_speech callback
    else:
        logger.warning("No user input, continuing to next turn")
```

---

### Change 7: `leibniz_pro.py` - Cleanup at App Shutdown

**Location:** At the END of main() function, in the finally block

**FIND:**
```python
async def main():
    try:
        # ... main code ...
    
    except Exception as e:
        logger.error(...)
    
    finally:
        logger.info("Shutting down")
        # ... cleanup code ...
```

**ADD IN THE FINALLY BLOCK:**
```python
    finally:
        logger.info("🔴 Shutting down Leibniz Agent")
        
        # ========== NEW: Cleanup persistent VAD ==========
        try:
            persistent_vad.is_running = False
            
            if 'send_audio_task' in locals() and send_audio_task:
                send_audio_task.cancel()
            
            if 'listen_task' in locals() and listen_task:
                listen_task.cancel()
            
            if 'stream' in locals() and stream:
                stream.stop()
            
            await persistent_vad.close_session()
            logger.info("✅ Persistent VAD closed")
        
        except Exception as e:
            logger.error(f"Error during VAD cleanup: {e}")
        
        # ========== EXISTING CLEANUP CODE ==========
        # ... your existing cleanup code ...
```

---

### Change 8: `leibniz_pro.py` - Remove Old Functions (Optional)

**If you want to completely remove old VAD patterns:**

**FIND AND DELETE or COMMENT OUT:**
```python
async def transcribe_and_classify():  # ❌ OLD - DELETE
    """OLD PATTERN: Per-turn VAD."""
    # ... this entire function, ~30-50 lines ...
    pass

async def capture_and_transcribe():  # ❌ OLD - DELETE
    """OLD PATTERN: Per-turn VAD."""
    # ... this entire function, ~20-30 lines ...
    pass

def check_leibniz_bargein():  # ❌ OLD - DELETE
    """OLD PATTERN: Manual barge-in checking."""
    # ... this entire function, ~10-20 lines ...
    pass
```

**IF YOU DELETE THESE, make sure they're not called elsewhere!**

Use search & replace to find any calls to these functions and update them.

---

## File Locations Quick Reference

| File | Location | Action |
|------|----------|--------|
| **Create** `leibniz_persistent_vad.py` | `leibnizagent/leibniz_persistent_vad.py` | New file (copy from [93]) |
| **Modify** `leibniz_vad.py` | Top of file (~line 15-20) | Add deprecation notice |
| **Modify** `leibniz_pro.py` (imports) | ~line 30-40 | Add PersistentGeminiVAD import |
| **Modify** `leibniz_pro.py` (init) | main() start | Initialize VAD once |
| **Modify** `leibniz_pro.py` (callback) | main() function | Add handle_user_speech() |
| **Modify** `leibniz_pro.py` (loop) | while True block | Change to event-based |
| **Modify** `leibniz_pro.py` (cleanup) | finally block | Add VAD cleanup |
| **Delete** `leibniz_pro.py` | Various | Remove old functions |

---

## Testing Checklist

After making changes, verify:

- [ ] Code compiles without syntax errors
- [ ] App starts without crashes
- [ ] `persistent_vad` initializes successfully
- [ ] Audio stream starts (check logs)
- [ ] Background listening task starts (check logs)
- [ ] User speech detected without new session (check logs)
- [ ] User can interrupt agent mid-TTS (barge-in works)
- [ ] No per-turn VAD startup delay
- [ ] App runs for 2-3 minutes without crashes
- [ ] Graceful shutdown (cleanup called)

---

## Common Issues & Solutions

### Issue 1: "ImportError: cannot import PersistentGeminiVAD"
**Solution:** Make sure you created `leibniz_persistent_vad.py` in the right location

### Issue 2: "current_user_transcript not defined"
**Solution:** Make sure you added the global variables section before the while loop

### Issue 3: "persistent_vad.on_user_speech not called"
**Solution:** Check that handle_user_speech() is assigned: `persistent_vad.on_user_speech = handle_user_speech`

### Issue 4: Audio not streaming
**Solution:** Check `send_audio_stream()` is running as background task

### Issue 5: User speech not received
**Solution:** Check `listen_for_user_speech()` is running as background task

---

## Line-by-Line Summary

| Change # | File | What | Why |
|----------|------|------|-----|
| 1 | leibniz_vad.py | Add deprecation notice | Mark old pattern |
| 2 | leibniz_pro.py | Add import | Use new persistent VAD |
| 3 | leibniz_pro.py | Initialize at app start | VAD stays open always |
| 4 | leibniz_pro.py | Remove per-turn calls | No more per-turn startup |
| 5 | leibniz_pro.py | Add callback handler | Process user speech |
| 6 | leibniz_pro.py | Update main loop | Use events instead of blocking |
| 7 | leibniz_pro.py | Add cleanup | Proper shutdown |
| 8 | leibniz_pro.py | Delete old functions | Clean up code |

---

**TOTAL CHANGES:** ~200-300 lines of code modifications
**TIME TO IMPLEMENT:** 1-2 hours
**BENEFIT:** 1-2 seconds per turn saved = 20-50x latency improvement!
