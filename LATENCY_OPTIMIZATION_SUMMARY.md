# Latency Optimization Implementation Summary

## Problem
Terminal was generating responses but taking 5-6 seconds for speech to start in browser.

## Root Causes Identified
1. **Sequential Processing**: RAG query waited for complete transcript before starting
2. **TTS Consumer Start Delay**: TTS consumer started AFTER RAG completed
3. **No Streaming**: RAG responses were non-streaming, waiting for complete answer before TTS

## Optimizations Implemented

### ✅ Optimization 1: Early TTS Consumer Start
**Location**: `handle_rag_query()` function
- **Before**: TTS consumer started after RAG query completed
- **After**: TTS consumer starts IMMEDIATELY before RAG query begins
- **Impact**: First audio can start playing as soon as first sentence arrives from RAG

**Code Change**:
```python
# OPTIMIZATION 3: Start TTS consumer IMMEDIATELY (before RAG starts)
consumer_task = await start_tts_consumer(audio_sink=audio_sink)
logger.info("⚡ TTS consumer started IMMEDIATELY (before RAG) for faster first audio")
```

### ✅ Optimization 2: Speculative RAG Execution
**Location**: `display_streaming_transcript()` callback and `handle_continuous_user_speech()`
- **Before**: RAG query started only after complete transcript received
- **After**: RAG query starts speculatively when user speaks 5+ words
- **Impact**: RAG processing begins while user is still speaking, saving 1-3 seconds

**Implementation**:
1. **Per-Turn Mode**: Starts speculative RAG in `display_streaming_transcript()` when fragment has 5+ words
2. **Continuous VAD Mode**: Starts speculative RAG in `handle_continuous_user_speech()` for RAG_QUERY intents
3. **Result Checking**: Main loop checks for speculative result before executing fresh query

**Code Changes**:
- Added `_speculative_rag_execution()` helper function
- Added speculative task tracking in per-turn mode
- Added speculative task tracking in continuous VAD mode
- Added result checking in RAG_QUERY handler

### ✅ Optimization 3: TTS Streaming with Prefetch (Already Implemented)
**Location**: `consume_tts_streaming_queue()` function
- **Status**: Already implemented with 2-slot pipeline
- **How it works**: 
  - Synthesizes next sentence (N+1) while playing current sentence (N)
  - Uses `current_result` and `next_future` for parallel synthesis/playback
  - Non-blocking prefetch ensures continuous audio flow

### ✅ Optimization 4: Enable Streaming for RAG Queries
**Location**: RAG_QUERY handler in `run_conversation_session()`
- **Before**: `enable_streaming=False` - waited for complete response
- **After**: `enable_streaming=True` - streams sentences as they're generated
- **Impact**: First sentence can start playing immediately when RAG generates it

**Code Change**:
```python
rag_msg = await handle_rag_query(
    text=query_text,
    context=context,
    enable_streaming=True,  # OPTIMIZATION 3: Enable streaming for faster first audio
    audio_sink=audio_sink
)
```

## Expected Latency Improvements

### Before Optimizations:
- RAG Query: 2-4 seconds
- TTS Synthesis: 1-2 seconds
- **Total: 5-6 seconds** before first audio

### After Optimizations:
- Speculative RAG starts: 0-1 seconds (while user speaks)
- First sentence TTS: 0.5-1 second (streaming)
- **Total: 0.5-2 seconds** before first audio

**Expected Improvement: 3-4 seconds faster**

## Testing Recommendations

1. **Test Speculative RAG**: Speak a question with 5+ words and verify RAG starts before you finish
2. **Test Streaming**: Verify first sentence plays while RAG is still generating
3. **Test Prefetch**: Verify smooth playback without gaps between sentences
4. **Monitor Logs**: Look for "⚡" emoji indicators showing optimizations activating

## Configuration

No configuration changes needed - optimizations are enabled by default.

## Files Modified

1. `leibniz_pro.py`:
   - Added `_speculative_rag_execution()` function
   - Modified `handle_rag_query()` to start TTS consumer early
   - Modified `display_streaming_transcript()` to start speculative RAG
   - Modified `handle_continuous_user_speech()` to start speculative RAG
   - Modified RAG_QUERY handler to enable streaming and check speculative results
   - Added global variable `_speculative_rag_task_continuous`

## Notes

- Speculative RAG may execute queries that don't match final transcript (wasted computation)
- This is acceptable trade-off for latency reduction
- Failed speculative queries are cancelled gracefully
- TTS prefetch already implemented - no changes needed

