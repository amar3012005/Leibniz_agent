"""
STT/VAD Microservice - FastAPI Application

WebSocket-based speech transcription service using Gemini Live API.

Endpoints:
    WebSocket /api/v1/transcribe/stream - Real-time speech transcription
    GET /health - Health check
    GET /metrics - Service metrics
    POST /admin/reset - Force reset Gemini session

Reference:
    leibniz_agent/docs/Cloud Transformation.md - Phase 2
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from leibniz_agent.services.stt_vad.config import VADConfig
from leibniz_agent.services.stt_vad.vad_manager import VADManager
from leibniz_agent.services.stt_vad.gemini_client import GeminiLiveSession
from leibniz_agent.services.shared.redis_client import get_redis_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
vad_manager: Optional[VADManager] = None
redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for application startup/shutdown"""
    global vad_manager, redis_client
    
    logger.info("🚀 Starting STT/VAD microservice...")
    
    # Load configuration
    config = VADConfig.from_env()
    logger.info(f"✅ Config loaded: model={config.model_name}, timeout={config.initial_timeout_s}s")
    
    # Initialize Redis (optional) with timeout
    try:
        logger.info("Connecting to Redis...")
        redis_client = await asyncio.wait_for(get_redis_client(), timeout=5.0)
        await asyncio.wait_for(redis_client.ping(), timeout=2.0)
        logger.info("✅ Redis connected")
    except asyncio.TimeoutError:
        logger.warning("⚠️ Redis connection timeout. Running without state persistence.")
        redis_client = None
    except Exception as e:
        logger.warning(f"⚠️ Redis unavailable: {e}. Running without state persistence.")
        redis_client = None
    
    # Initialize VAD manager
    vad_manager = VADManager(config, redis_client)
    logger.info("✅ VADManager initialized")
    
    # Prewarm Gemini session (skip for faster startup)
    logger.info("⚡ Skipping Gemini prewarm for faster startup (will connect on first request)")
    
    logger.info("🎉 STT/VAD microservice ready!")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down STT/VAD microservice...")
    
    # Close Gemini session
    try:
        await GeminiLiveSession.close_session()
        logger.info("✅ Gemini session closed")
    except Exception as e:
        logger.error(f"❌ Gemini session close error: {e}")
    
    # Close Redis
    if redis_client:
        try:
            await redis_client.close()
            logger.info("✅ Redis connection closed")
        except Exception as e:
            logger.error(f"❌ Redis close error: {e}")
    
    logger.info("👋 STT/VAD microservice stopped")


# Initialize FastAPI app
app = FastAPI(
    title="STT/VAD Microservice",
    description="Speech transcription service using Gemini Live API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Service health status
    """
    health_status = {
        "status": "healthy",
        "service": "stt-vad",
        "vad_manager": vad_manager is not None,
        "redis": redis_client is not None
    }
    
    # Check Gemini session
    session_stats = GeminiLiveSession.get_session_stats()
    health_status["gemini_session"] = session_stats
    
    # Check Redis connectivity
    if redis_client:
        try:
            await redis_client.ping()
            health_status["redis_healthy"] = True
        except Exception as e:
            health_status["redis_healthy"] = False
            health_status["redis_error"] = str(e)
    
    return JSONResponse(content=health_status, status_code=200)


@app.get("/metrics")
async def get_metrics():
    """
    Get service performance metrics.
    
    Returns:
        dict: Performance metrics
    """
    if not vad_manager:
        raise HTTPException(status_code=503, detail="VAD manager not initialized")
    
    metrics = vad_manager.get_performance_metrics()
    return JSONResponse(content=metrics, status_code=200)


@app.post("/admin/reset")
async def reset_session():
    """
    Force reset Gemini session (admin endpoint).
    
    Returns:
        dict: Reset confirmation
    """
    try:
        await GeminiLiveSession.close_session()
        logger.info("🔄 Gemini session reset via admin endpoint")
        return {"status": "success", "message": "Session reset"}
    except Exception as e:
        logger.error(f"❌ Session reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.websocket("/api/v1/transcribe/stream")
async def transcribe_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time speech transcription.
    
    Protocol:
        Client -> Server: Binary PCM audio chunks (16-bit, 16kHz, mono)
        Server -> Client: JSON messages with transcript fragments
        
    Message Format:
        {
            "type": "partial" | "final" | "error" | "timeout",
            "text": str,
            "session_id": str,
            "timestamp": float
        }
    
    Args:
        websocket: WebSocket connection
    """
    await websocket.accept()
    session_id = f"ws_{id(websocket)}"
    audio_queue = asyncio.Queue()
    
    logger.info(f"🔌 [{session_id}] WebSocket connected")
    
    if not vad_manager:
        await websocket.send_json({
            "type": "error",
            "text": "VAD manager not initialized",
            "session_id": session_id
        })
        await websocket.close()
        return
    
    try:
        # Streaming callback for transcript fragments
        def streaming_callback(text: str, is_final: bool):
            """Send transcript fragments to client"""
            asyncio.create_task(
                websocket.send_json({
                    "type": "final" if is_final else "partial",
                    "text": text,
                    "session_id": session_id,
                    "timestamp": asyncio.get_event_loop().time()
                })
            )
        
        # Background task to receive audio from client
        async def receive_audio_task():
            """Receive audio chunks from WebSocket"""
            try:
                audio_chunk_count = 0
                total_bytes_received = 0
                
                while True:
                    # Receive binary audio data
                    audio_data = await websocket.receive_bytes()
                    
                    # Observability logging (debug level only)
                    audio_chunk_count += 1
                    total_bytes_received += len(audio_data)
                    queue_size = audio_queue.qsize()
                    
                    # Log every 50 chunks to avoid spam
                    if audio_chunk_count % 50 == 0:
                        logger.debug(
                            f"📊 [{session_id}] Audio ingress: "
                            f"chunk #{audio_chunk_count}, "
                            f"{len(audio_data)} bytes, "
                            f"queue size: {queue_size}, "
                            f"total: {total_bytes_received} bytes"
                        )
                    
                    # Add to queue for VAD processing
                    await audio_queue.put(audio_data)
                    
            except WebSocketDisconnect:
                logger.info(
                    f"🔌 [{session_id}] Client disconnected "
                    f"(received {audio_chunk_count} chunks, {total_bytes_received} bytes total)"
                )
            except Exception as e:
                logger.error(f"❌ [{session_id}] Audio receive error: {e}")
        
        # Start receiving audio in background
        receive_task = asyncio.create_task(receive_audio_task())
        
        try:
            # Capture speech with streaming
            final_transcript = await vad_manager.capture_speech_streaming(
                session_id=session_id,
                audio_queue=audio_queue,
                streaming_callback=streaming_callback
            )
            
            if final_transcript:
                # Send final transcript
                await websocket.send_json({
                    "type": "final",
                    "text": final_transcript,
                    "session_id": session_id,
                    "timestamp": asyncio.get_event_loop().time()
                })
                logger.info(f"✅ [{session_id}] Transcription complete: {final_transcript}")
            else:
                # Timeout or error
                await websocket.send_json({
                    "type": "timeout",
                    "text": "",
                    "session_id": session_id,
                    "timestamp": asyncio.get_event_loop().time()
                })
                logger.warning(f"⏱️ [{session_id}] Transcription timeout")
        
        finally:
            # Cancel audio receiving task
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
    
    except WebSocketDisconnect:
        logger.info(f"🔌 [{session_id}] WebSocket disconnected")
    
    except Exception as e:
        logger.error(f"❌ [{session_id}] WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "text": str(e),
                "session_id": session_id,
                "timestamp": asyncio.get_event_loop().time()
            })
        except:
            pass
    
    finally:
        try:
            await websocket.close()
        except:
            pass
        logger.info(f"🔌 [{session_id}] WebSocket closed")


if __name__ == "__main__":
    # Run with uvicorn
    import os
    
    host = os.getenv("LEIBNIZ_STT_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("LEIBNIZ_STT_SERVICE_PORT", "8001"))
    workers = int(os.getenv("LEIBNIZ_STT_SERVICE_WORKERS", "2"))
    
    logger.info(f"🚀 Starting STT/VAD service on {host}:{port} (workers={workers})")
    
    uvicorn.run(
        "leibniz_agent.services.stt_vad.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        reload=False
    )
