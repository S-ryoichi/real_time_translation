# Real-time Translation (M1)

Minimal FastAPI app that translates Japanese speech to English text using openai-whisper (local, no external APIs).

## Setup

1) Create and activate a Python 3.10+ environment.
2) Install dependencies:

```
pip install -r requirements.txt
```

Whisper uses ffmpeg to decode audio. On Linux, install it via your package manager if missing:

```
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## Run

```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

If you see "Form data requires python-multipart" install error, ensure `python-multipart` is installed (it is already listed in requirements.txt). If connection is refused, verify the server is running in another terminal and that nothing else is using port 8000.

## Try it

```
curl -X POST -F "file=@sample.wav" http://localhost:8000/translate
```

Response example:

```json
{"translation": "translated text"}
```

## Notes
- Model: `small` is loaded on startup. GPU is used if available.
- Endpoint `/health` is available for a quick check.
