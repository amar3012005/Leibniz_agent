#!/usr/bin/env python3
"""
Test Echo Prevention and State Machine Logic
============================================

Verifies that the ConversationStateMachine correctly manages:
1. Browser playback timing (Echo Prevention)
2. Conversation state transitions
3. Input acceptance rules
"""

import asyncio
import time
import logging
from leibniz_state_machine import get_state_machine, ConversationState, AudioState

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_echo_prevention():
    print("\n[INFO] Testing Echo Prevention Logic...")
    sm = get_state_machine()
    
    # 1. Initial State
    assert sm.conversation_state == ConversationState.IDLE
    assert sm.audio_state == AudioState.IDLE
    assert not sm.is_agent_speaking_in_browser()
    assert sm.can_accept_user_input()
    print("[OK] Initial state: IDLE, Input Allowed")
    
    # 2. Start Agent Turn
    await sm.transition_to(ConversationState.USER_SPEAKING)
    await sm.transition_to(ConversationState.PROCESSING)
    await sm.transition_to(ConversationState.AGENT_SPEAKING)
    assert sm.conversation_state == ConversationState.AGENT_SPEAKING
    
    # Strict Mode: Input blocked during AGENT_SPEAKING state to prevent race conditions
    # This ensures we don't process input while setting up audio
    assert not sm.can_accept_user_input() 
    print("[OK] Agent Speaking (Generating): Input Blocked (Strict State Check)")
    
    # 3. Start Audio Emission
    await sm.set_audio_state(AudioState.EMITTING)
    
    # Simulate emitting a 2-second chunk
    # This updates the predicted end time
    chunk_duration = 2.0
    await sm.update_browser_playback_end_time(chunk_duration)
    
    # Now we expect:
    # - is_agent_speaking_in_browser() -> True
    # - can_accept_user_input() -> False (Echo blocked)
    
    print(f"   Playback locked until: {sm._agent_playback_end_time:.2f} (Now: {time.time():.2f})")
    assert sm.is_agent_speaking_in_browser()
    assert not sm.can_accept_user_input()
    print("[OK] Agent Speaking (Playing): Input BLOCKED (Echo Prevention active)")
    
    # 4. Simulate Echo Input
    if not sm.can_accept_user_input():
        print("   [Simulated] Microphone input ignored: 'Echo of agent voice'")
    else:
        print("[FAIL] ERROR: Echo was not blocked!")
        return False

    # 5. Wait for Playback to Finish
    print("[INFO] Waiting for playback to finish (simulated)...")
    # We won't actually wait 2s in test, just check logic
    # But let's sleep slightly to ensure time moves forward
    await asyncio.sleep(0.1)
    
    # 6. Simulate End of Turn
    # Reset state manually for test (normally main loop does this)
    # But first, let's see if lock expires automatically
    
    # 7. Transition to IDLE
    await sm.transition_to(ConversationState.IDLE)
    
    # Even in IDLE, if audio is still playing, input should be blocked
    if sm.is_agent_speaking_in_browser():
        assert not sm.can_accept_user_input()
        print("[OK] IDLE state but Playing: Input still BLOCKED (Tail Echo Prevention)")
    
    print("[OK] Echo Prevention Test Passed")
    return True

async def main():
    try:
        await test_echo_prevention()
    except AssertionError as e:
        print(f"[FAIL] Test Assertion Failed: {e}")
    except Exception as e:
        print(f"[FAIL] Test Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())

