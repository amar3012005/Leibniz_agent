"""
Leibniz Agent - WebRTC I/O Adapters
====================================

Pluggable audio source/sink interfaces for WebRTC integration.
Abstracts audio I/O to enable seamless switching between local devices and WebRTC streams.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class AudioSource(ABC):
    """Abstract base class for audio input sources."""
    
    @abstractmethod
    async def get_frames(self, num_frames: int) -> np.ndarray:
        """
        Get audio frames from the source.
        
        Args:
            num_frames: Number of frames to read (at source sample rate)
            
        Returns:
            np.ndarray: Float32 mono audio frames
        """
        pass
    
    @abstractmethod
    async def close(self):
        """Close the audio source."""
        pass


class AudioSink(ABC):
    """Abstract base class for audio output sinks."""
    
    @abstractmethod
    async def put_frames(self, frames: np.ndarray):
        """
        Send audio frames to the sink.
        
        Args:
            frames: Float32 mono audio frames to output
        """
        pass
    
    @abstractmethod
    async def close(self):
        """Close the audio sink."""
        pass


class WebRTCSource(AudioSource):
    """WebRTC-based audio source that reads from FastRTC stream."""
    
    def __init__(self, webrtc_id: str, sample_rate: int = 16000):
        self.webrtc_id = webrtc_id
        self.sample_rate = sample_rate
        self._buffer = deque(maxlen=10000)  # Buffer for incoming frames
        self._closed = False
        
    async def get_frames(self, num_frames: int) -> np.ndarray:
        """Get frames from WebRTC buffer."""
        if self._closed:
            return np.zeros(num_frames, dtype=np.float32)
            
        # Wait for frames if buffer is empty
        while len(self._buffer) < num_frames and not self._closed:
            await asyncio.sleep(0.01)
            
        if self._closed or len(self._buffer) < num_frames:
            return np.zeros(num_frames, dtype=np.float32)
            
        # Extract frames
        frames = []
        for _ in range(num_frames):
            frames.append(self._buffer.popleft())
            
        return np.array(frames, dtype=np.float32)
    
    async def receive_audio(self, audio_data: np.ndarray, sample_rate: int):
        """Receive audio data from WebRTC handler and buffer it."""
        # Ensure audio is float32 mono
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        if audio_data.ndim == 2:
            audio_data = audio_data.mean(axis=1)  # Convert to mono
            
        # Resample if needed
        if sample_rate != self.sample_rate:
            # Simple resampling (could be improved)
            ratio = self.sample_rate / sample_rate
            new_length = int(len(audio_data) * ratio)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), new_length),
                np.arange(len(audio_data)),
                audio_data
            )
        
        # Add to buffer
        for frame in audio_data:
            self._buffer.append(frame)
    
    async def close(self):
        """Close the WebRTC source."""
        self._closed = True
        self._buffer.clear()


class WebRTCSink(AudioSink):
    """WebRTC-based audio sink that streams to FastRTC peer."""
    
    def __init__(self, webrtc_id: str, sample_rate: int = 16000, chunk_size: int = 800):  # 50ms at 16kHz
        self.webrtc_id = webrtc_id
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._buffer = deque()
        self._chunks = asyncio.Queue()  # Queue for completed chunks
        self._closed = False
        self._task: Optional[asyncio.Task] = None
        
    async def put_frames(self, frames: np.ndarray):
        """Buffer frames for streaming."""
        if self._closed:
            return
            
        # Add frames to buffer
        for frame in frames:
            self._buffer.append(frame)
            
        # Process complete chunks
        while len(self._buffer) >= self.chunk_size:
            # Extract chunk
            chunk = []
            for _ in range(self.chunk_size):
                chunk.append(self._buffer.popleft())
                
            chunk_array = np.array(chunk, dtype=np.float32)
            
            # Queue chunk for retrieval
            await self._chunks.put(chunk_array)
    
    async def get_audio_chunks(self):
        """Generator that yields audio chunks for WebRTC transmission."""
        try:
            while not self._closed:
                try:
                    # Wait for next chunk with timeout
                    chunk = await asyncio.wait_for(self._chunks.get(), timeout=0.1)
                    yield chunk
                except asyncio.TimeoutError:
                    # No more chunks available
                    break
        except Exception as e:
            logger.error(f"Error in get_audio_chunks: {e}")
    
    async def close(self):
        """Close the WebRTC sink."""
        self._closed = True
        self._buffer.clear()
        # Clear remaining chunks
        while not self._chunks.empty():
            try:
                self._chunks.get_nowait()
            except asyncio.QueueEmpty:
                break


class WebRTCSessionRegistry:
    """Registry for managing WebRTC sessions."""
    
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        
    def register_session(self, webrtc_id: str, source: WebRTCSource, sink: WebRTCSink, 
                        conversation_task: Optional[asyncio.Task] = None):
        """Register a new WebRTC session."""
        self._sessions[webrtc_id] = {
            'source': source,
            'sink': sink,
            'conversation_task': conversation_task,
            'created_at': asyncio.get_event_loop().time()
        }
        logger.info(f"Registered WebRTC session: {webrtc_id}")
    
    def get_session(self, webrtc_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by WebRTC ID."""
        return self._sessions.get(webrtc_id)
    
    def unregister_session(self, webrtc_id: str):
        """Unregister and cleanup a WebRTC session."""
        session = self._sessions.pop(webrtc_id, None)
        if session:
            # Close source and sink
            asyncio.create_task(session['source'].close())
            asyncio.create_task(session['sink'].close())
            
            # Cancel conversation task
            if session['conversation_task']:
                session['conversation_task'].cancel()
                
            logger.info(f"Unregistered WebRTC session: {webrtc_id}")
    
    def get_or_create_session(self, webrtc_id: str, sample_rate: int = 16000) -> tuple[WebRTCSource, WebRTCSink]:
        """Get existing session or create new one."""
        session = self.get_session(webrtc_id)
        if session:
            return session['source'], session['sink']
        
        # Create new session
        source = WebRTCSource(webrtc_id, sample_rate)
        sink = WebRTCSink(webrtc_id, sample_rate)
        
        self.register_session(webrtc_id, source, sink)
        return source, sink


# Global registry instance
_webrtc_registry = WebRTCSessionRegistry()


def get_webrtc_registry() -> WebRTCSessionRegistry:
    """Get the global WebRTC session registry."""
    return _webrtc_registry