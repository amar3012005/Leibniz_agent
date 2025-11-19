"""
STT/VAD Microservice FastAPI Application

WebSocket-based speech transcription service using Gemini Live API.

Endpoints:
    WebSocket /api/v1/transcribe/stream - Real-time speech transcription
    GET /health - Health check
    GET /metrics - Service metrics
    POST /admin/reset_session - Force reset Gemini session

Reference:
    leibniz_agent/docs/Cloud Transformation.md - Phase 2
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from leibniz_agent.services.stt_vad.config import VADConfig
from leibniz_agent.services.stt_vad.vad_manager import VADManager
from leibniz_agent.services.stt_vad.utils import validate_audio_chunk, format_transcript_fragment
from leibniz_agent.services.stt_vad.gemini_client import GeminiLiveSession
from leibniz_agent.services.shared.redis_client import get_redis_client, close_redis_client, ping_redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
vad_manager: VADManager = None
redis_client = None
active_sessions: Dict[str, Any] = {}
app_start_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for application startup/shutdown"""
    global vad_manager, redis_client
    
    logger.info(" Starting STT/VAD microservice...")
    
    # Load configuration
    config = VADConfig.from_env()
    
    # Initialize Redis client
    redis_client = await get_redis_client()
    
    # Initialize VAD manager
    vad_manager = VADManager(config, redis_client)
    
    logger.info(" STT/VAD microservice initialized")
    
    yield
    
    # Shutdown
    logger.info(" Shutting down STT/VAD microservice...")
    
    # Close Gemini session
    await GeminiLiveSession.close_session()
    
    # Close Redis
    await close_redis_client(redis_client)
    
    logger.info(" STT/VAD microservice stopped")


# Initialize FastAPI app
app = FastAPI(
    title="Leibniz STT/VAD Service",
    description="Speech transcription service using Gemini Live API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.websocket("/api/v1/transcribe/stream")
async def transcribe_stream(websocket: WebSocket, session_id: str = Query(...)):
    """
    WebSocket endpoint for real-time speech transcription.
    
    Args:
        websocket: WebSocket connection
        session_id: Session identifier
    """
    await websocket.accept()
    
    # Send welcome message
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id
    })
    
    # Create per-connection audio queue
    audio_queue = asyncio.Queue()
    active_sessions[session_id] = {
        "audio_queue": audio_queue,
        "websocket": websocket,
        "last_activity": time.time()
    }
    
    # Define streaming callback
    def streaming_callback(text: str, is_final: bool):
        """Send transcript fragments to client"""
        async def send_fragment():
            fragment = format_transcript_fragment(text, is_final)
            try:
                await websocket.send_json(fragment)
            except Exception as e:
                logger.warning(f"Failed to send fragment to client: {e}")
        
        asyncio.create_task(send_fragment())
    
    try:
        while True:
            # Receive message with timeout
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), 
                    timeout=30.0  # 30s idle timeout
                )
            except asyncio.TimeoutError:
                # Send timeout message and close
                await websocket.send_json({
                    "type": "timeout",
                    "text": "",
                    "session_id": session_id,
                    "timestamp": time.time()
                })
                break
            
            if "bytes" in message:
                # Validate audio chunk
                audio_data = message["bytes"]
                is_valid, errors = validate_audio_chunk(audio_data)
                
                if not is_valid:
                    await websocket.send_json({
                        "type": "error",
                        "errors": errors,
                        "session_id": session_id,
                        "timestamp": time.time()
                    })
                    continue
                
                # Add to queue
                await audio_queue.put(audio_data)
                active_sessions[session_id]["last_activity"] = time.time()
                
            elif "text" in message:
                # Parse JSON command
                import json
                try:
                    command = json.loads(message["text"])
                    
                    if command.get("type") == "start_capture":
                        # Start capture
                        await vad_manager.capture_speech_streaming(
                            session_id,
                            audio_queue,
                            streaming_callback
                        )
                        
                    elif command.get("type") == "stop_capture":
                        # Stop capture (implementation depends on VADManager)
                        pass
                        
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "text": "Invalid JSON command",
                        "session_id": session_id,
                        "timestamp": time.time()
                    })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    
    finally:
        # Cleanup
        if session_id in active_sessions:
            del active_sessions[session_id]


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Service health status
    """
    uptime_seconds = time.time() - app_start_time
    
    # Get Gemini stats
    gemini_stats = GeminiLiveSession.get_session_stats()
    
    # Check Redis
    redis_connected = False
    if redis_client:
        try:
            await ping_redis(redis_client)
            redis_connected = True
        except Exception:
            redis_connected = False
    
    # Get VAD metrics
    vad_metrics = vad_manager.get_performance_metrics() if vad_manager else {}
    
    # Determine status
    status = "healthy"
    if not redis_connected or not gemini_stats.get("active", False):
        status = "degraded"
    
    return {
        "status": status,
        "service": "stt-vad",
        "uptime_seconds": uptime_seconds,
        "active_sessions": len(active_sessions),
        "gemini_session": "active" if gemini_stats.get("active", False) else "inactive",
        "redis_connected": redis_connected,
        "total_captures": vad_metrics.get("total_captures", 0),
        "avg_capture_time_ms": vad_metrics.get("avg_capture_time_ms", 0)
    }


@app.get("/metrics")
async def get_metrics():
    """
    Get service performance metrics.
    
    Returns:
        dict: Performance metrics
    """
    if not vad_manager:
        return {"error": "VAD manager not initialized"}
    
    metrics = vad_manager.get_performance_metrics()
    metrics["active_sessions"] = len(active_sessions)
    metrics["gemini_stats"] = GeminiLiveSession.get_session_stats()
    
    return metrics


@app.post("/admin/reset_session")
async def reset_session():
    """
    Force reset Gemini session (admin endpoint).
    
    Returns:
        dict: Reset confirmation
    """
    await GeminiLiveSession.close_session()
    
    if vad_manager:
        vad_manager.consecutive_timeouts = 0
    
    return {"status": "success", "message": "Session reset"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=False
    )
