"""
Leibniz Agent - WebRTC Conversation Handler
Integrates existing VAD, Intent, RAG, and TTS pipeline with FastRTC streaming
"""

import asyncio
import logging
import os
import numpy as np
from typing import AsyncGenerator, Optional, Dict, Any, Tuple
from fastrtc import ReplyOnPause
import time

# Import existing Leibniz components
from leibniz_persistent_services import get_leibniz_services_manager
from leibniz_intent_parser import LeibnizIntentParser
from leibniz_appointment_fsm import LeibnizAppointmentFSM
from leibniz_stt import LeibnizSTT
from leibniz_tts import LeibnizTTS
# Import WebRTC I/O adapters
from leibniz_webrtc_io import WebRTCSessionRegistry, WebRTCSource, WebRTCSink

logger = logging.getLogger(__name__)


class AudioFormatConverter:
    """Utilities for audio format conversion between different sample rates"""
    
    @staticmethod
    def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio from original sample rate to target sample rate
        
        Args:
            audio_data: Input audio samples (numpy array)
            orig_sr: Original sample rate (e.g., 24000 for LemonFox)
            target_sr: Target sample rate (e.g., 16000 for WebRTC)
            
        Returns:
            Resampled audio as numpy array
        """
        if orig_sr == target_sr:
            return audio_data
        
        try:
            from scipy import signal
            # Calculate resampling ratio
            num_samples = int(len(audio_data) * target_sr / orig_sr)
            resampled = signal.resample(audio_data, num_samples)
            return resampled.astype(np.float32)
        except ImportError:
            logger.warning("scipy not available, using simple decimation")
            # Fallback: simple decimation (lower quality)
            step = orig_sr // target_sr
            return audio_data[::step]
    
    @staticmethod
    def ensure_float32(audio_data: np.ndarray) -> np.ndarray:
        """Ensure audio is in float32 format normalized to [-1, 1]"""
        if audio_data.dtype == np.int16:
            return audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            return audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.float64:
            return audio_data.astype(np.float32)
        return audio_data
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            return audio_data / max_val
        return audio_data


class LeibnizConversationHandler(ReplyOnPause):
    """
    WebRTC conversation handler integrating Leibniz Agent pipeline
    Extends FastRTC's ReplyOnPause for automatic turn-taking
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        enable_logging: bool = True,
        session_timeout: int = 1800  # 30 minutes
    ):
        """
        Initialize Leibniz WebRTC handler
        
        Args:
            sample_rate: Audio sample rate (FastRTC uses 16kHz)
            enable_logging: Enable detailed conversation logging
            session_timeout: Session timeout in seconds
        """
        # Initialize parent with our run method as the handler function
        super().__init__(self.run)
        
        self.sample_rate = sample_rate
        self.enable_logging = enable_logging
        self.session_timeout = session_timeout
        
        # Session registry for WebRTC adapters
        self.session_registry = WebRTCSessionRegistry()
        
        # Initialize Leibniz services
        self.services_manager = None
        self.stt = None
        self.tts = None
        self.intent_parser = None
        self.current_session_id = None
        self.current_fsm = None  # Appointment FSM instance
        
        # Session state tracking
        self.conversation_history = []
        self.session_start_time = time.time()
        self.greeted_sessions = set()  # Track which sessions have received greeting
        
        # Audio format converter
        self.audio_converter = AudioFormatConverter()
        
        logger.info(" LeibnizConversationHandler initialized")
    
    async def initialize_services(self):
        """Initialize Leibniz services (VAD, TTS, RAG, Intent)"""
        try:
            # Get persistent services manager
            self.services_manager = await get_leibniz_services_manager()
            
            # Initialize STT and TTS
            self.stt = LeibnizSTT()
            self.tts = LeibnizTTS()
            
            # Initialize intent parser
            self.intent_parser = LeibnizIntentParser()
            
            # Generate session ID
            import uuid
            self.current_session_id = str(uuid.uuid4())
            
            logger.info(f" Services initialized for session {self.current_session_id}")
            
        except Exception as e:
            logger.error(f" Failed to initialize services: {e}", exc_info=True)
            raise
    
    async def run(
        self,
        audio_input: tuple[int, np.ndarray],
        webrtc_id: str = None
    ) -> AsyncGenerator[tuple[int, np.ndarray], None]:
        """
        Main conversation handler - receives user audio and yields agent audio
        
        Args:
            audio_input: Tuple of (sample_rate, audio_data) from user
            webrtc_id: WebRTC session identifier
            
        Yields:
            Tuples of (sample_rate, audio_data) for agent response
        """
        # Initialize services on first run
        if self.services_manager is None:
            await self.initialize_services()
        
        # Extract audio from input
        input_sr, input_audio = audio_input
        
        if self.enable_logging:
            logger.info(f" Received audio: {len(input_audio)} samples at {input_sr}Hz (session: {webrtc_id})")
        
        try:
            # Use provided webrtc_id or default
            if not webrtc_id:
                webrtc_id = "default_webrtc_session"
            
            # Create/get WebRTC adapters for this session
            source, sink = self.session_registry.get_or_create_session(webrtc_id)
            
            # Import leibniz_pro functions
            from leibniz_agent import leibniz_pro
            
            # Send greeting on first interaction for this session
            if webrtc_id not in self.greeted_sessions:
                self.greeted_sessions.add(webrtc_id)
                logger.info(f" Sending greeting for new session: {webrtc_id}")
                
                # Send greeting through WebRTC
                await leibniz_pro.speak_friendly(
                    text="Hello! I'm your Leibniz University assistant. How can I help you today?",
                    sink=sink
                )
                
                # Yield greeting audio chunks
                async for audio_chunk in sink.get_audio_chunks():
                    yield (self.sample_rate, audio_chunk)
                
                # Save greeting turn
                self._save_conversation_turn("", "Greeting sent via WebRTC", "GREETING")
            
            # Feed incoming audio to the source
            await source.receive_audio(input_audio, input_sr)
            
            # Use the existing conversation flow but with WebRTC adapters
            # This mimics the run_conversation_session() logic but adapted for WebRTC
            
            # Step 1: Capture speech using WebRTC source
            transcript = await leibniz_pro.capture_leibniz_speech(
                audio_source=source,  # Use WebRTC source instead of microphone
                context={'webrtc_session': True, 'session_id': webrtc_id}
            )
            
            if not transcript or len(transcript.strip()) < 2:
                logger.info(" Empty or too short transcript, skipping")
                return
            
            if self.enable_logging:
                logger.info(f" User: {transcript}")
            
            # Step 2: Classify intent using existing logic
            # Extract semantic context
            from leibniz_semantic_extractor import extract_semantic_context
            semantic_context = extract_semantic_context(transcript)
            
            # Classify intent
            if leibniz_pro._leibniz_parser:
                intent_result = await leibniz_pro._leibniz_parser.classify_intent(
                    text=semantic_context['extracted_meaning'],
                    context={
                        'semantic_context': semantic_context,
                        'user_goal': semantic_context['user_goal'],
                        'key_entities': semantic_context['key_entities'],
                        'extracted_meaning': semantic_context['extracted_meaning']
                    }
                )
            else:
                intent_result = {
                    'intent': 'RAG_QUERY',
                    'confidence': 0.5,
                    'context': semantic_context
                }
            
            intent = intent_result.get('intent', 'UNCLEAR')
            user_context = semantic_context['extracted_meaning']
            
            if self.enable_logging:
                logger.info(f" Intent: {intent} (confidence: {intent_result.get('confidence', 0):.2f})")
            
            # Step 3: Generate response based on intent
            if intent == 'APPOINTMENT_SCHEDULING':
                # Handle appointment booking
                booking_data = await leibniz_pro.handle_appointment_booking(initial_input=transcript)
                if booking_data:
                    await leibniz_pro.speak_friendly(
                        text="Appointment scheduled successfully!",
                        sink=sink
                    )
                else:
                    await leibniz_pro.speak_friendly(
                        text="Let's schedule your appointment. When would you like to meet?",
                        sink=sink
                    )
                    
            elif intent == 'RAG_QUERY':
                # Handle RAG query with streaming TTS
                rag_result = await leibniz_pro.handle_rag_query(
                    text=user_context,
                    context={'last_intent': {'intent': intent, 'user_context': user_context}},
                    enable_streaming=True,
                    user_id=webrtc_id
                )
                # Stream RAG response using consume_tts_streaming_queue with sink
                await leibniz_pro.stream_rag_to_tts(rag_result.answer, pace=1.0, is_final=True)
                await leibniz_pro.consume_tts_streaming_queue(sink=sink)
                
            elif intent == 'GREETING':
                await leibniz_pro.speak_friendly(
                    text="Hello! I'm your Leibniz University assistant. How can I help you today?",
                    sink=sink
                )
                
            elif intent == 'EXIT':
                await leibniz_pro.speak_friendly(
                    text="Thank you for using Leibniz Assistant. Have a great day!",
                    sink=sink
                )
                
            else:  # UNCLEAR or other
                await leibniz_pro.speak_friendly(
                    text="I didn't quite catch that. Could you rephrase your question?",
                    sink=sink
                )
            
            # Yield audio chunks from the sink
            async for audio_chunk in sink.get_audio_chunks():
                yield (self.sample_rate, audio_chunk)
            
            # Save conversation turn
            self._save_conversation_turn(transcript, "Agent response sent via WebRTC", intent)
            
        except Exception as e:
            logger.error(f" Error in conversation handler: {e}", exc_info=True)
            
            # Send error message through sink
            try:
                error_message = "I apologize, but I encountered an error. Please try again."
                await leibniz_pro.speak_friendly(text=error_message, sink=sink)
                # Yield any error audio
                async for audio_chunk in sink.get_audio_chunks():
                    yield (self.sample_rate, audio_chunk)
            except:
                # Fallback: yield silence
                yield (self.sample_rate, np.zeros(800, dtype=np.float32))
    
    async def _transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Transcribe audio using Leibniz STT
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate of audio
            
        Returns:
            Transcript text
        """
        try:
            import tempfile
            import soundfile as sf
            
            # Ensure audio is in correct format
            audio_data = self.audio_converter.ensure_float32(audio_data)
            
            # Resample if needed (WebRTC uses 16kHz, but input might differ)
            if sample_rate != self.sample_rate:
                audio_data = self.audio_converter.resample_audio(
                    audio_data, sample_rate, self.sample_rate
                )
            
            # Save to temporary file for STT
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, audio_data, self.sample_rate)
            
            try:
                # Use Leibniz STT service to transcribe file
                result = await self.stt.transcribe_file(tmp_path, validate_english=False)
                transcript = result.get('text', '')
                return transcript
            finally:
                # Clean up temp file
                import os
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
        except Exception as e:
            logger.error(f" STT error: {e}", exc_info=True)
            return ""
    
    async def _classify_intent(self, transcript: str) -> Dict[str, Any]:
        """
        Classify user intent using Leibniz Intent Parser
        
        Args:
            transcript: User transcript text
            
        Returns:
            Intent classification result with intent, confidence, entities, etc.
        """
        try:
            context = {
                'session_id': self.current_session_id,
                'conversation_history': self.conversation_history[-3:]  # Last 3 turns
            }
            
            intent_result = await self.intent_parser.classify_intent(
                text=transcript,
                context=context
            )
            
            return intent_result
            
        except Exception as e:
            logger.error(f" Intent classification error: {e}", exc_info=True)
            return {
                'intent': 'UNCLEAR',
                'confidence': 0.0,
                'user_context': transcript
            }
    
    async def _generate_response(
        self,
        intent: str,
        transcript: str,
        user_context: str,
        intent_result: Dict[str, Any]
    ) -> str:
        """
        Generate agent response based on classified intent
        
        Args:
            intent: Classified intent (APPOINTMENT_SCHEDULING, RAG_QUERY, etc.)
            transcript: Original user transcript
            user_context: Enriched user query from intent parser
            intent_result: Full intent classification result
            
        Returns:
            Agent response text
        """
        try:
            if intent == 'APPOINTMENT_SCHEDULING':
                return await self._handle_appointment(transcript, intent_result)
            
            elif intent == 'RAG_QUERY':
                return await self._handle_rag_query(user_context, transcript)
            
            elif intent == 'GREETING':
                return self._handle_greeting()
            
            elif intent == 'EXIT':
                return self._handle_exit()
            
            else:  # UNCLEAR or other
                return await self._handle_unclear(transcript)
                
        except Exception as e:
            logger.error(f" Response generation error: {e}", exc_info=True)
            return "I apologize, but I'm having trouble processing your request. Could you rephrase that?"
    
    async def _handle_appointment(self, transcript: str, intent_result: Dict[str, Any]) -> str:
        """Handle appointment scheduling using FSM"""
        try:
            # Initialize FSM if needed
            if self.current_fsm is None:
                self.current_fsm = LeibnizAppointmentFSM(
                    session_id=self.current_session_id
                )
            
            # Process user input through FSM
            fsm_result = self.current_fsm.process_user_input(transcript)
            
            # Get next prompt from FSM
            next_prompt = fsm_result.get('next_prompt', '')
            
            # Check if appointment is complete
            if fsm_result.get('is_complete', False):
                # Reset FSM for next appointment
                self.current_fsm = None
            
            return next_prompt
            
        except Exception as e:
            logger.error(f" Appointment FSM error: {e}", exc_info=True)
            return "I can help you schedule an appointment. When would you like to meet?"
    
    async def _handle_rag_query(self, user_context: str, original_query: str) -> str:
        """Handle RAG knowledge base queries"""
        try:
            # Process RAG query using services manager
            answer = await self.services_manager.process_rag_query(
                text=user_context,
                context={'session_id': self.current_session_id}
            )
            
            if not answer:
                return "I couldn't find information about that. Could you rephrase your question?"
            
            return answer
            
        except Exception as e:
            logger.error(f" RAG query error: {e}", exc_info=True)
            return "I'm having trouble accessing the knowledge base right now. Please try again."
    
    def _handle_greeting(self) -> str:
        """Handle greeting intent"""
        return "Hello! I'm your Leibniz University assistant. How can I help you today?"
    
    def _handle_exit(self) -> str:
        """Handle exit intent"""
        return "Thank you for using Leibniz Assistant. Have a great day!"
    
    async def _handle_unclear(self, transcript: str) -> str:
        """Handle unclear intent - fallback to RAG"""
        logger.info(f" Unclear intent, trying RAG fallback: {transcript}")
        return await self._handle_rag_query(transcript, transcript)
    
    async def _synthesize_and_stream(
        self,
        text: str
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Synthesize text to speech and stream audio chunks
        
        Args:
            text: Text to synthesize
            
        Yields:
            Tuples of (sample_rate, audio_chunk) for streaming
        """
        try:
            import tempfile
            import soundfile as sf
            
            # Synthesize audio to temporary file (LemonFox returns 24kHz WAV)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Synthesize using LeibnizTTS
                result = await self.tts.synthesize_to_file(
                    text=text,
                    output_filepath=tmp_path,
                    emotion="neutral",
                    cache_name=None  # No caching for fresh audio
                )
                
                if not result:
                    logger.error(" TTS synthesis returned empty result")
                    return
                
                # Read the audio file
                audio_data, orig_sr = sf.read(tmp_path)
                
                # Convert to numpy array if needed
                if not isinstance(audio_data, np.ndarray):
                    audio_data = np.array(audio_data, dtype=np.float32)
                
                # Ensure correct format
                audio_data = self.audio_converter.ensure_float32(audio_data)
                
                # Resample to WebRTC sample rate (16kHz) if needed
                if orig_sr != self.sample_rate:
                    logger.info(f" Resampling audio from {orig_sr}Hz to {self.sample_rate}Hz")
                    audio_data = self.audio_converter.resample_audio(
                        audio_data, orig_sr, self.sample_rate
                    )
                
                # Normalize audio
                audio_data = self.audio_converter.normalize_audio(audio_data)
                
                # Stream in chunks (50ms chunks = 800 samples at 16kHz)
                chunk_size = int(self.sample_rate * 0.05)  # 50ms
                
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    
                    # Pad last chunk if needed
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                    
                    yield (self.sample_rate, chunk)
                    
                    # Small delay to simulate streaming
                    await asyncio.sleep(0.001)
                
                logger.info(f" Streamed {len(audio_data)} samples in {len(audio_data) // chunk_size} chunks")
                
            finally:
                # Clean up temp file
                import os
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
        except Exception as e:
            logger.error(f" TTS streaming error: {e}", exc_info=True)
    
    def _save_conversation_turn(self, user_text: str, agent_text: str, intent: str):
        """Save conversation turn to history"""
        turn = {
            'timestamp': time.time(),
            'user': user_text,
            'agent': agent_text,
            'intent': intent
        }
        self.conversation_history.append(turn)
        
        # Keep only last 20 turns
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        return {
            'session_id': self.current_session_id,
            'start_time': self.session_start_time,
            'uptime_seconds': time.time() - self.session_start_time,
            'conversation_turns': len(self.conversation_history),
            'current_fsm_state': self.current_fsm.state if self.current_fsm else None
        }
