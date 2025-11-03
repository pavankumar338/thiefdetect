# Thief Detection — Web UI

This project runs a thief-detection model on uploaded videos and returns an annotated output video.

How to run (development):

1. Create a Python environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start the server:

```powershell
python run.py
```

3. Open http://localhost:8501 in your browser. Upload a video and run detection.

Notes:
- The server calls `detect_video` in `app.py` which attempts to load `best(new).pt` by default.
- If your checkpoint is a plain state_dict you'll need to reconstruct the model; prefer ultralytics YOLOv8 or a TorchScript model for easy loading.
