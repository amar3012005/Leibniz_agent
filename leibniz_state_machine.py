from enum import Enum
from typing import Optional, Callable, Dict, Any
import asyncio
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """Atomic conversation states - ONLY ONE active at a time"""
    IDLE = "idle"                      # No activity
    USER_SPEAKING = "user_speaking"    # User has floor
    PROCESSING = "processing"          # STT→Intent→RAG pipeline
    AGENT_SPEAKING = "agent_speaking"  # Agent has floor
    BARGE_IN = "barge_in"              # User interrupted agent

class AudioState(Enum):
    """Audio subsystem state - tracks TTS pipeline"""
    IDLE = "idle"
    SYNTHESIZING = "synthesizing"      # TTS model generating
    BUFFERING = "buffering"            # Audio chunks queued
    EMITTING = "emitting"              # Sending to browser
    PLAYING = "playing"                # Browser playback active

@dataclass
class TurnContext:
    """Per-turn metadata for state transitions"""
    turn_id: str
    started_at: float
    user_transcript: Optional[str] = None
    intent: Optional[str] = None
    agent_response: Optional[str] = None
    audio_duration_ms: float = 0.0
    browser_playback_end_time: float = 0.0  # Predicted end time
    
class ConversationStateMachine:
    """
    Unified state machine for conversation flow control.
    
    Based on Google ADK live streaming architecture and production voice AI patterns.
    
    Key principles:
    1. Single source of truth for ALL state
    2. Atomic state transitions with validation
    3. Precise timing for WebRTC browser playback
    4. Event-driven callbacks for subsystems
    """
    
    def __init__(self):
        # Core state
        self.conversation_state = ConversationState.IDLE
        self.audio_state = AudioState.IDLE
        
        # Turn management
        self.current_turn: Optional[TurnContext] = None
        self.turn_history: list[TurnContext] = []
        
        # Timing state (critical for echo prevention)
        self._agent_playback_end_time = 0.0
        self._last_state_change = time.time()
        
        # Locks for thread-safety
        self._state_lock = asyncio.Lock()
        self._timing_lock = asyncio.Lock()
        
        # Event callbacks (observer pattern)
        self._callbacks: Dict[str, list[Callable]] = {
            "state_change": [],
            "audio_state_change": [],
            "turn_start": [],
            "turn_end": [],
            "barge_in": []
        }
        
        # Configuration
        self._network_buffer_ms = 800  # Browser network jitter buffer (increased from 500ms)
        self._safety_buffer_ms = 300   # Additional safety margin (increased from 200ms)
        
        logger.info("✅ ConversationStateMachine initialized")
        
    async def transition_to(
        self, 
        new_state: ConversationState, 
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Atomic state transition with validation.
        
        Returns:
            True if transition successful, False if invalid transition
        """
        async with self._state_lock:
            old_state = self.conversation_state
            
            # Validate transition
            if not self._is_valid_transition(old_state, new_state):
                logger.warning(
                    f"❌ Invalid transition: {old_state.value} → {new_state.value} "
                    f"(reason: {reason})"
                )
                return False
            
            # Execute transition
            self.conversation_state = new_state
            self._last_state_change = time.time()
            
            logger.info(
                f"🔄 State: {old_state.value} → {new_state.value} "
                f"({reason})"
            )
            
            # Notify observers
            await self._notify_callbacks("state_change", {
                "old_state": old_state,
                "new_state": new_state,
                "reason": reason,
                "metadata": metadata or {}
            })
            
            return True
    
    def _is_valid_transition(
        self, 
        from_state: ConversationState, 
        to_state: ConversationState
    ) -> bool:
        """
        Validate state transition according to conversation flow rules.
        """
        # Transition rules matrix
        valid_transitions = {
            ConversationState.IDLE: {
                ConversationState.USER_SPEAKING,
                ConversationState.AGENT_SPEAKING
            },
            ConversationState.USER_SPEAKING: {
                ConversationState.PROCESSING,
                ConversationState.IDLE,  # User stopped without input
                ConversationState.AGENT_SPEAKING # Direct jump if processing is instant
            },
            ConversationState.PROCESSING: {
                ConversationState.AGENT_SPEAKING,
                ConversationState.USER_SPEAKING,  # Barge-in during processing
                ConversationState.IDLE  # Processing failed
            },
            ConversationState.AGENT_SPEAKING: {
                ConversationState.BARGE_IN,
                ConversationState.IDLE,  # Agent finished
                ConversationState.USER_SPEAKING  # Agent done, user starts
            },
            ConversationState.BARGE_IN: {
                ConversationState.USER_SPEAKING,  # User continues after barge-in
                ConversationState.IDLE,  # False barge-in
                ConversationState.PROCESSING # Recover and process new input
            }
        }
        
        # Allow self-transition (e.g., AGENT_SPEAKING -> AGENT_SPEAKING for sentence updates)
        if from_state == to_state:
            return True
            
        return to_state in valid_transitions.get(from_state, set())
    
    async def set_audio_state(
        self, 
        new_audio_state: AudioState,
        reason: str = ""
    ):
        """Update audio subsystem state (TTS pipeline tracking)"""
        async with self._state_lock:
            old_audio_state = self.audio_state
            self.audio_state = new_audio_state
            
            logger.debug(
                f"🎵 Audio: {old_audio_state.value} → {new_audio_state.value} "
                f"({reason})"
            )
            
            await self._notify_callbacks("audio_state_change", {
                "old_state": old_audio_state,
                "new_state": new_audio_state,
                "reason": reason
            })
    
    async def update_browser_playback_end_time(
        self, 
        audio_chunk_duration_ms: float
    ):
        """
        Update predicted browser playback end time.
        
        CRITICAL for echo prevention - tracks EXACT browser playback timing
        including network latency and buffering.
        
        Args:
            audio_chunk_duration_ms: Duration of audio chunk being sent to browser
        """
        async with self._timing_lock:
            current_time = time.time()
            
            # Calculate new end time
            # If no playback active, start from now + network buffer
            if self._agent_playback_end_time <= current_time:
                self._agent_playback_end_time = (
                    current_time + 
                    (self._network_buffer_ms / 1000.0) +
                    (audio_chunk_duration_ms / 1000.0) +
                    (self._safety_buffer_ms / 1000.0)
                )
            else:
                # Extend existing playback
                self._agent_playback_end_time += (audio_chunk_duration_ms / 1000.0)
            
            # Update turn context
            if self.current_turn:
                self.current_turn.browser_playback_end_time = self._agent_playback_end_time
            
            logger.debug(
                f"🔒 VAD lock extended to {self._agent_playback_end_time:.2f} "
                f"(+{audio_chunk_duration_ms:.0f}ms)"
            )
    
    def is_agent_speaking_in_browser(self) -> bool:
        """
        Check if agent audio is ACTUALLY playing in browser.
        
        This is the ONLY reliable way to prevent echo loops.
        
        Returns:
            True if browser is currently playing agent audio
        """
        return time.time() < self._agent_playback_end_time
    
    def can_accept_user_input(self) -> bool:
        """
        Check if system can accept user speech input.
        
        Blocks input during agent playback to prevent self-hearing.
        """
        # Block if agent is speaking in browser
        if self.is_agent_speaking_in_browser():
            return False
        
        # Block if in wrong conversation state
        if self.conversation_state in [
            ConversationState.AGENT_SPEAKING,
            ConversationState.PROCESSING
        ] and self.conversation_state != ConversationState.BARGE_IN:
            return False
        
        return True
    
    async def start_turn(self, turn_id: str):
        """Begin new conversation turn"""
        self.current_turn = TurnContext(
            turn_id=turn_id,
            started_at=time.time()
        )
        await self._notify_callbacks("turn_start", {"turn_id": turn_id})
    
    async def end_turn(self):
        """Complete current conversation turn"""
        if self.current_turn:
            self.turn_history.append(self.current_turn)
            turn_id = self.current_turn.turn_id
            self.current_turn = None
            await self._notify_callbacks("turn_end", {"turn_id": turn_id})
    
    def register_callback(self, event: str, callback: Callable):
        """Register observer callback for state changes"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    async def _notify_callbacks(self, event: str, data: Dict[str, Any]):
        """Notify all registered callbacks for an event"""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state snapshot for debugging"""
        return {
            "conversation_state": self.conversation_state.value,
            "audio_state": self.audio_state.value,
            "agent_speaking_in_browser": self.is_agent_speaking_in_browser(),
            "can_accept_input": self.can_accept_user_input(),
            "current_turn": self.current_turn.turn_id if self.current_turn else None,
            "playback_end_time": self._agent_playback_end_time,
            "time_until_playback_end": max(0, self._agent_playback_end_time - time.time())
        }

# Global singleton instance
_state_machine: Optional[ConversationStateMachine] = None

def get_state_machine() -> ConversationStateMachine:
    """Get global state machine instance"""
    global _state_machine
    if _state_machine is None:
        _state_machine = ConversationStateMachine()
    return _state_machine

