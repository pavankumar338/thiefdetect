"""
Flask server to serve the HTML frontend and run detection.

Endpoints:
- GET / -> web/index.html
- GET /static/<path> -> static files (styles.js)
- POST /process -> accepts multipart form with 'video' file and parameters; returns annotated video file

Run:
    python run.py

Note: This is a minimal example. For production, run with a WSGI server and secure file handling.
"""

from pathlib import Path
import tempfile
import shutil
import threading
from flask import Flask, send_from_directory, request, abort, send_file, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import time

from app import detect_video

app = Flask(__name__, static_folder='web')

# Directory to publish output videos so the browser can request them normally.
# Use a temp location outside the project tree to avoid the debug reloader
# picking up file changes.
OUTPUT_DIR = Path(tempfile.gettempdir()) / 'dlproject_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.route('/')
def index():
    return send_from_directory('web', 'index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    # serve files from web folder
    return send_from_directory('web', filename)


@app.route('/process', methods=['POST'])
def process():
    # Validate file
    if 'video' not in request.files:
        return 'No video file provided', 400

    f = request.files['video']
    if f.filename == '':
        return 'Empty filename', 400

    model = request.form.get('model', 'best(new).pt')
    device = request.form.get('device', 'cpu')
    conf = float(request.form.get('conf', 0.4))
    max_frames = int(request.form.get('max_frames', 0))
    max_frames = None if max_frames == 0 else max_frames

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        filename = secure_filename(f.filename)
        src_path = tmp_dir / filename
        f.save(str(src_path))

        out_path = tmp_dir / f"annotated_{int(time.time())}.mp4"

        # Run detection (blocking). Exceptions will return 500.
        detect_video(str(model), str(src_path), str(out_path), device=device, conf_thr=conf, max_frames=max_frames)

        if not out_path.exists():
            return 'Processing failed: output not created', 500

        # Move the produced file to OUTPUT_DIR and return a JSON URL where the
        # browser can request it normally. Serving the file via a regular URL
        # (instead of streaming a blob over XHR) lets the browser handle
        # progressive playback and range requests more reliably.

        # Use the actual output filename produced in the tmp dir so the
        # returned URL matches the moved file exactly (avoid new timestamp
        # which can cause mismatches and 404s).
        final_name = out_path.name
        final_path = OUTPUT_DIR / final_name
        shutil.move(str(out_path), str(final_path))
        print(f"[run.py] moved output -> {final_path}")

        # schedule removal of the published file after some minutes
        def _delayed_remove(path, delay=300):
            try:
                time.sleep(delay)
                os.remove(path)
            except Exception:
                pass

        threading.Thread(target=_delayed_remove, args=(str(final_path),), kwargs={'delay': 300}, daemon=True).start()

        # also remove temporary working dir shortly
        def _delayed_cleanup(path, delay=30):
            try:
                time.sleep(delay)
                shutil.rmtree(path)
            except Exception:
                pass

        threading.Thread(target=_delayed_cleanup, args=(str(tmp_dir), 30), daemon=True).start()

        # Return JSON with the URL to the output file
        return jsonify({'url': f'/outputs/{final_name}'})

    except Exception as e:
        # On error, clean up immediately
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass
        return f'Processing error: {e}', 500


if __name__ == '__main__':
    # Run a simple dev server. For production, use gunicorn/uwsgi.
    app.run(host='0.0.0.0', port=8501, debug=True)


@app.route('/outputs/<path:filename>')
def outputs(filename):
    # Serve the file from OUTPUT_DIR. We intentionally do not put outputs
    # into the project tree to avoid triggering the debug reloader.
    safe = secure_filename(filename)
    return send_from_directory(str(OUTPUT_DIR), safe, as_attachment=False)
