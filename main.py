"""
FastAPI app for translating audio (JA->EN) using openai-whisper.

M1 requirements:
- POST /translate accepts audio file (WAV/MP3)
- Loads Whisper "small" model
- Uses task="translate" to translate Japanese -> English
- Returns JSON: {"translation": "..."}
- No external APIs, only local openai-whisper

Run example:
  curl -X POST -F "file=@sample.wav" http://localhost:8000/translate

Start server:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
import os
import asyncio
import subprocess
import logging
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

import whisper
import torch

# Load environment variables from .env file
load_dotenv()

# Configure logging
def _setup_logging():
    """Configure logging based on LOG_LEVEL environment variable.

    Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    Default: INFO
    """
    valid_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    log_level_str = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    log_level = valid_levels.get(log_level_str, logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(message)s",
        force=True  # Override any existing configuration
    )

    if log_level_str not in valid_levels:
        logging.warning(f"Invalid LOG_LEVEL '{log_level_str}'. Using default 'INFO'.")
        logging.warning(f"Valid options: {', '.join(valid_levels.keys())}")

_setup_logging()
logger = logging.getLogger(__name__)


app = FastAPI(title="Whisper Translate (JA->EN)")


def _detect_device() -> str:
    """Return 'cuda' if available else 'cpu'."""
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_whisper_model_name() -> str:
    """Get Whisper model name from environment variable with validation.

    Returns:
        Valid Whisper model name. Defaults to 'small' if .env is missing,
        unreadable, or contains invalid value.

    Valid models: tiny, base, small, medium, large, large-v2, large-v3
    """
    # Valid Whisper model names
    valid_models = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
    default_model = "small"

    # Try to read from environment variable
    model_name = os.getenv("WHISPER_MODEL", "").strip().lower()

    # Validate and return
    if model_name in valid_models:
        return model_name
    elif model_name:
        # Invalid value provided - warn and use default
        logger.warning(f"Invalid WHISPER_MODEL '{model_name}'. Using default '{default_model}'.")
        logger.warning(f"Valid options: {', '.join(sorted(valid_models))}")
        return default_model
    else:
        # No value or .env not loaded - use default silently
        return default_model


@app.on_event("startup")
def load_model_on_startup():
    """Load and cache the Whisper model at server startup.

    Storing the model in app.state prevents reloading on each request.
    """
    device = _detect_device()
    # Get model name from environment variable (defaults to "small")
    # Options: tiny (fastest) -> base -> small (default) -> medium -> large (most accurate)
    model_name = _get_whisper_model_name()
    logger.info(f"Loading Whisper model: {model_name}")

    # Load Whisper model. If running on CPU, fp16 will be disabled at inference time.
    model = whisper.load_model(model_name, device=device)
    app.state.model = model
    app.state.device = device
    app.state.model_name = model_name
    logger.info(f"Whisper model '{model_name}' loaded successfully on {device}")


async def _transcribe_bytes_async(audio_bytes: bytes, *, suffix: str = ".webm") -> str:
    """Transcribe given audio bytes and return English translation.

    This runs the blocking Whisper call in a thread to avoid blocking the event loop.

    Parameters:
        audio_bytes: raw audio data (e.g., 2s chunk from client). Default assumes webm/ogg-like container.
        suffix: temp file suffix to help ffmpeg detect format. MediaRecorder often yields webm/ogg.
    Returns:
        English translation text (possibly empty string if nothing recognized).
    """

    # Access model/device from app.state
    model = getattr(app.state, "model", None)
    device = getattr(app.state, "device", "cpu")
    if model is None:
        # Fallback safety (should not happen because we load on startup)
        device = _detect_device()
        model_name = _get_whisper_model_name()
        model = whisper.load_model(model_name, device=device)
        app.state.model = model
        app.state.device = device
        app.state.model_name = model_name

    def _ffmpeg_convert_to_wav(src_path: str) -> str:
        """Convert input audio file to mono 16k WAV using ffmpeg and return output path."""
        out_tmp: Optional[str] = None
        try:
            with NamedTemporaryFile(delete=False, suffix=".wav") as out_f:
                out_tmp = out_f.name
            # -y to overwrite, -loglevel error for concise errors
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", src_path,
                "-ac", "1", "-ar", "16000", "-f", "wav", out_tmp,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                err = proc.stderr.strip() or proc.stdout.strip()
                raise RuntimeError(f"ffmpeg convert failed: {err}")
            return out_tmp
        except Exception:
            # cleanup if conversion failed
            if out_tmp and os.path.exists(out_tmp):
                try:
                    os.remove(out_tmp)
                except Exception:
                    pass
            raise

    def _blocking_transcribe() -> str:
        # Check if audio data is too small (likely invalid)
        # Reduced threshold to 256 bytes to accommodate various audio formats
        MIN_AUDIO_SIZE = 256  # 256 bytes minimum
        logger.debug(f"Audio bytes length: {len(audio_bytes)}")
        if len(audio_bytes) < MIN_AUDIO_SIZE:
            logger.debug(f"Audio too small ({len(audio_bytes)} < {MIN_AUDIO_SIZE}), skipping")
            return ""  # Return empty string for too-small chunks

        tmp_path = None
        wav_path = None
        try:
            with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            logger.debug(f"Wrote audio to temp file: {tmp_path}")

            # Try to convert to WAV, but skip if conversion fails (invalid header)
            try:
                logger.debug("Starting ffmpeg conversion...")
                wav_path = _ffmpeg_convert_to_wav(tmp_path)
                audio_file = wav_path
                logger.debug(f"FFmpeg conversion successful: {wav_path}")
            except Exception as conv_err:
                # If conversion fails, return empty (invalid audio chunk)
                logger.debug(f"FFmpeg conversion failed: {conv_err}")
                return ""

            try:
                # Get English translation
                logger.debug("Starting Whisper transcription...")
                result = model.transcribe(
                    audio_file,
                    task="translate",
                    language="ja",
                    fp16=(device == "cuda"),
                )
                english_text = (result.get("text") or "").strip()
                logger.debug(f"Whisper raw result: {result}")
                logger.debug(f"Whisper extracted text: '{english_text}'")
                return english_text
            except Exception as transcribe_err:
                # If Whisper fails, return empty string
                logger.debug(f"Whisper transcription failed: {transcribe_err}")
                return ""
            finally:
                if wav_path and os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except Exception:
                        pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # Offload CPU-bound work to a thread
    return await asyncio.to_thread(_blocking_transcribe)


def _ext_from_content_type(content_type: Optional[str]) -> str:
    """Infer file extension from MIME type for temp file saving."""
    if not content_type:
        return ""
    ct = content_type.lower()
    if ct in {"audio/wav", "audio/x-wav", "audio/wave"}:
        return ".wav"
    if ct in {"audio/mpeg", "audio/mp3"}:
        return ".mp3"
    if ct in {"audio/webm"}:
        return ".webm"
    if ct in {"audio/ogg"}:
        return ".ogg"
    return ""


@app.post("/translate")
async def translate(file: UploadFile = File(...)):
    """Translate uploaded audio to English using Whisper.

    - Accepts WAV/MP3 (and some common audio types) as UploadFile
    - Saves to a temporary file so Whisper can read it
    - Uses task="translate" with language="ja" to force JA->EN
    """
    # Basic validation of content type
    allowed_types = {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/webm",
        "audio/ogg",
    }
    if file.content_type not in allowed_types:
        # We still try to proceed as long as extension looks OK, but warn via 415 if clearly unsupported
        pass

    # Decide a temp file suffix (extension) for ffmpeg/whisper decoding
    suffix = _ext_from_content_type(file.content_type)
    if not suffix:
        # Fallback to original filename's extension if available
        _, ext = os.path.splitext(file.filename or "")
        suffix = ext if ext else ".wav"  # default to .wav

    tmp_path = None
    try:
        # Read entire file body (acceptable for typical short audio). For very large files, consider chunked write.
        data = await file.read()
        if not data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file uploaded")

        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        # Retrieve cached model and device
        model = getattr(app.state, "model", None)
        device = getattr(app.state, "device", "cpu")
        if model is None:
            # Fallback: lazily load if startup hook didn't run (e.g., during certain testing scenarios)
            device = _detect_device()
            model_name = _get_whisper_model_name()
            model = whisper.load_model(model_name, device=device)
            app.state.model = model
            app.state.device = device
            app.state.model_name = model_name

        # Whisper inference; for CPU, ensure fp16=False to avoid errors
        result = model.transcribe(
            tmp_path,
            task="translate",      # force translation to English
            language="ja",         # source language is Japanese
            fp16=(device == "cuda"),
        )

        translation = (result.get("text") or "").strip()
        return JSONResponse(content={"translation": translation})

    except HTTPException:
        raise
    except Exception as e:
        # Convert any unexpected error into 500 with safe message
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Translation failed: {e}")
    finally:
        # Cleanup temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ------------------------------
# M2: WebSocket for realtime translation
# ------------------------------
@app.websocket("/ws")
async def ws_translate(websocket: WebSocket):
    """WebSocket endpoint for realtime JA->EN translation.

    Client flow (example):
    - Connect to /ws
    - Every ~10s, send a binary message containing recorded audio bytes (e.g., MediaRecorder chunks)
    - Receive JSON {"english": "translated text"} per chunk

    Notes:
    - This implementation returns only English translation (no Japanese transcription).
    - For best compatibility with MediaRecorder, we assume WEBM/OGG container by default.
    - If your client sends WAV/MP3, adjust the suffix or send a small text control message first with the mime.
    """
    await websocket.accept()

    # Optional: you could accept an initial text message specifying mime, e.g., {"mime":"audio/webm"}
    # Per-connection state
    mime_suffix = ".webm"  # default; can be updated via a text control message from client

    # Keep only the most recent chunk instead of accumulating all audio
    # This significantly improves processing speed
    current_chunk = bytearray()

    try:
        while True:
            # Wait for the next message (binary expected every ~2s)
            message = await websocket.receive()

            # If the client closes the socket gracefully
            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                chunk: bytes = message["bytes"]
                logger.debug(f"Received audio chunk: {len(chunk)} bytes")
                if not chunk:
                    await websocket.send_json({"english": ""})
                    continue

                # Replace buffer with the latest chunk (no accumulation)
                # This trades context for speed
                current_chunk = bytearray(chunk)

                try:
                    # Send processing acknowledgement to keep connection alive
                    if websocket.client_state.name == "CONNECTED":
                        await websocket.send_json({"status": "processing"})
                        logger.debug("Sent processing status")

                    logger.debug(f"Starting transcription of {len(current_chunk)} bytes")
                    result = await _transcribe_bytes_async(bytes(current_chunk), suffix=mime_suffix)
                    logger.debug(f"Transcription result: '{result}'")

                    # Check if WebSocket is still open before sending
                    if websocket.client_state.name == "CONNECTED":
                        await websocket.send_json({
                            "english": result
                        })
                        logger.debug("Sent translation result")
                except WebSocketDisconnect:
                    # Client disconnected, exit gracefully
                    logger.debug("WebSocket disconnected")
                    break
                except Exception as e:
                    # Only send error if connection is still open
                    logger.error(f"Error during transcription: {e}")
                    if websocket.client_state.name == "CONNECTED":
                        await websocket.send_json({"error": f"transcription_failed: {e}"})

            elif "text" in message and message["text"] is not None:
                # Handle simple control messages from client
                raw_txt = (message["text"] or "").strip()
                txt_low = raw_txt.lower()
                if txt_low in {"close", "quit", "stop"}:
                    await websocket.close(code=1000)
                    break
                elif txt_low == "reset":
                    current_chunk.clear()
                    await websocket.send_json({"info": "buffer_reset"})
                else:
                    # Try to parse MIME hint as JSON or simple 'mime: ...'
                    updated = False
                    try:
                        import json as _json
                        obj = _json.loads(raw_txt)
                        mime = (obj.get("mime") or obj.get("contentType") or "").lower()
                        if mime:
                            if "ogg" in mime:
                                mime_suffix = ".ogg"
                                updated = True
                            elif "webm" in mime:
                                mime_suffix = ".webm"
                                updated = True
                            elif "wav" in mime:
                                mime_suffix = ".wav"
                                updated = True
                            elif "mp3" in mime or "mpeg" in mime:
                                mime_suffix = ".mp3"
                                updated = True
                    except Exception:
                        pass

                    if not updated and raw_txt.lower().startswith("mime:"):
                        mime = raw_txt.split(":", 1)[1].strip().lower()
                        if "ogg" in mime:
                            mime_suffix = ".ogg"; updated = True
                        elif "webm" in mime:
                            mime_suffix = ".webm"; updated = True
                        elif "wav" in mime:
                            mime_suffix = ".wav"; updated = True
                        elif "mp3" in mime or "mpeg" in mime:
                            mime_suffix = ".mp3"; updated = True

                    # Acknowledge or echo message
                    if updated:
                        await websocket.send_json({"info": f"mime_set:{mime_suffix}"})
                    else:
                        await websocket.send_json({"info": f"ignored_text_message: {raw_txt}"})

    except WebSocketDisconnect:
        # Client disconnected unexpectedly
        return


@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "device": getattr(app.state, "device", None),
        "model": getattr(app.state, "model_name", None)
    }


@app.get("/")
def serve_index():
    """Serve index.html for convenience (M3 frontend)."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({"detail": "index.html not found"}, status_code=404)


@app.get("/favicon.ico")
def serve_favicon():
    """Return 204 No Content for favicon requests to suppress 404 errors."""
    from fastapi.responses import Response
    return Response(status_code=204)


if __name__ == "__main__":
    # Allow running as: python main.py
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
