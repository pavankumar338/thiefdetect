"""
Video thief-detection runner

Loads a model checkpoint (default: ./best(new).pt) and runs inference on a video file,
writing an annotated output video with bounding boxes, labels and confidences.

Usage (example):
	python app.py --model best(new).pt --source input.mp4 --output output.mp4 --device cpu

Notes:
- This tries to support common YOLO flavors (ultralytics YOLOv8, YOLOv5 via torch.hub).
- Requires: torch, opencv-python, numpy. Optionally: ultralytics for YOLOv8.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple
import os
import smtplib
from email.message import EmailMessage
import tempfile

try:
	import cv2
	import numpy as np
except Exception as e:
	print("Missing dependency: please install opencv-python and numpy. Error:", e)
	raise

try:
	import torch
except Exception as e:
	print("Missing dependency: please install torch. Error:", e)
	raise


def load_model(model_path: str, device: str = "cpu"):
	"""Try loading a model checkpoint using common loaders.

	Returns an object `model` that is callable: results = model(frame)
	and (optionally) provides `.names` mapping for class ids.
	"""
	mp = Path(model_path)
	if not mp.exists():
		raise FileNotFoundError(f"Model file not found: {model_path}")

	# Try ultralytics YOLO (v8+)
	try:
		from ultralytics import YOLO

		print(f"Loading model with ultralytics.YOLO from {model_path} ...")
		model = YOLO(str(mp))
		model.to(device)
		return model
	except Exception:
		pass

	# Try yolov5 via torch.hub (requires internet to fetch repo on first run)
	try:
		print(f"Trying to load model via torch.hub (YOLOv5-style) from {model_path} ...")
		model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(mp), force_reload=False)
		if device != 'cpu':
			model.to(device)
		return model
	except Exception:
		pass

	# Try loading a scripted/traced model
	try:
		print(f"Trying torch.jit.load for {model_path} ...")
		model = torch.jit.load(str(mp), map_location=device)
		model.eval()
		return model
	except Exception:
		pass

	# As a last attempt try torch.load (may return state_dict)
	try:
		print(f"Trying torch.load for {model_path} ...")
		data = torch.load(str(mp), map_location=device)
		# If it's a model object already
		if hasattr(data, 'eval'):
			data.eval()
			return data
		# Otherwise we cannot reconstruct architecture here
		raise RuntimeError("torch.load returned an object that isn't directly runnable. Provide a scripted model or use ultralytics/yolov5 formats.")
	except Exception as e:
		raise RuntimeError(f"Could not load model with supported loaders: {e}")


def _extract_detections(results) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
	"""Normalize result object into (boxes, scores, classes, names).

	boxes: (N,4) xyxy
	scores: (N,)
	classes: (N,) int
	names: optional dict mapping id->name
	"""
	# ultralytics YOLOv8 Results: results[0].boxes -> .xyxy, .conf, .cls
	try:
		res0 = results[0]
		if hasattr(res0, 'boxes'):
			boxes = res0.boxes.xyxy.cpu().numpy()
			scores = res0.boxes.conf.cpu().numpy()
			classes = res0.boxes.cls.cpu().numpy().astype(int)
			names = getattr(results.model, 'names', None) if hasattr(results, 'model') else None
			return boxes, scores, classes, names
	except Exception:
		pass

	# yolov5 (torch.hub) results: results.xyxy[0] -> (N,6) x1,y1,x2,y2,conf,class
	try:
		if hasattr(results, 'xyxy'):
			xy = results.xyxy[0].cpu().numpy()
			boxes = xy[:, :4]
			scores = xy[:, 4]
			classes = xy[:, 5].astype(int)
			names = getattr(results, 'names', None)
			return boxes, scores, classes, names
	except Exception:
		pass

	# If results is a raw tensor or unknown format, try to be defensive
	raise RuntimeError('Unsupported results object returned by model; cannot extract detections')


def send_gmail_alert(user: str, app_password: str, to_email: str, subject: str, body: str, attachment_path: Optional[str] = None):
	"""Send a simple email via Gmail SMTP using an app password.

	Credentials are intentionally read from environment by caller; this helper
	does a best-effort send and prints errors instead of raising.
	"""
	try:
		msg = EmailMessage()
		msg['From'] = user
		msg['To'] = to_email
		msg['Subject'] = subject
		msg.set_content(body)

		if attachment_path:
			try:
				with open(attachment_path, 'rb') as f:
					data = f.read()
				import mimetypes
				ctype, _ = mimetypes.guess_type(attachment_path)
				if ctype:
					maintype, subtype = ctype.split('/', 1)
				else:
					maintype, subtype = 'application', 'octet-stream'
				msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=Path(attachment_path).name)
			except Exception as e:
				print(f"[send_gmail_alert] warning: couldn't attach file: {e}")

		# Connect and send
		try:
			with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
				s.login(user, app_password)
				s.send_message(msg)
			print(f"[send_gmail_alert] sent alert to {to_email}")
		except Exception as e:
			print(f"[send_gmail_alert] send failed: {e}")
	except Exception as e:
		print(f"[send_gmail_alert] failed building message: {e}")


def draw_boxes(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, names: Optional[dict], conf_thr: float = 0.4):
	"""Draw boxes and labels on frame (inplace)"""
	h, w = frame.shape[:2]
	for box, score, cls in zip(boxes, scores, classes):
		if score < conf_thr:
			continue
		x1, y1, x2, y2 = map(int, box)
		color = (0, 0, 255)  # red for thief
		cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
		label = f"{names.get(int(cls), str(int(cls))) if names else int(cls)} {score:.2f}"
		# label background
		(tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
		cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
		cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def detect_video(model_path: str, source_path: str, output_path: str, device: str = "cpu", conf_thr: float = 0.4, max_frames: Optional[int] = None):
	model = load_model(model_path, device=device)

	# Open video
	cap = cv2.VideoCapture(str(source_path))
	if not cap.isOpened():
		raise RuntimeError(f"Could not open video source: {source_path}")

	fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

	frame_idx = 0
	print(f"Starting detection: {source_path} -> {output_path}, device={device}, conf={conf_thr}")

	# Email alerting settings via env vars (optional)
	GMAIL_USER = os.environ.get('GMAIL_USER')
	GMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD')
	ALERT_TO = os.environ.get('ALERT_TO')  # admin email
	alert_sent = False
	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			frame_idx += 1
			if max_frames and frame_idx > max_frames:
				break

			# Model inference — many models accept BGR numpy arrays directly
			try:
				results = model(frame)
			except Exception:
				# Try converting to RGB array
				try:
					rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					results = model(rgb)
				except Exception as e:
					raise RuntimeError(f"Model inference failed on frame {frame_idx}: {e}")

			try:
				boxes, scores, classes, names = _extract_detections(results)
			except Exception as e:
				raise RuntimeError(f"Failed to parse model results on frame {frame_idx}: {e}")

			# Draw boxes
			draw_boxes(frame, boxes, scores, classes, names if names else getattr(model, 'names', None), conf_thr=conf_thr)

			# If we have an environment-configured admin email and credentials,
			# send a single alert when a 'thief' class is detected above threshold.
			if not alert_sent and GMAIL_USER and GMAIL_APP_PASSWORD and ALERT_TO:
				label_map = names if names else (getattr(model, 'names') if hasattr(model, 'names') else None)
				for box, score, cls in zip(boxes, scores, classes):
					if score < conf_thr:
						continue
					label = label_map.get(int(cls), str(int(cls))) if label_map else str(int(cls))
					if 'thief' in str(label).lower():
						# Save a snapshot to a temporary file and email it
						tmpfd, tmpjpg = tempfile.mkstemp(suffix='.jpg')
						os.close(tmpfd)
						x1, y1, x2, y2 = map(int, box)
						# clip coords
						h, w = frame.shape[:2]
						x1 = max(0, min(w-1, x1))
						y1 = max(0, min(h-1, y1))
						x2 = max(0, min(w, x2))
						y2 = max(0, min(h, y2))
						crop = frame[y1:y2, x1:x2] if (y2>y1 and x2>x1) else frame
						try:
							import cv2 as _cv2
							_cv2.imwrite(tmpjpg, crop)
						except Exception:
							# fallback: write full frame
							try:
								import cv2 as _cv2
								_cv2.imwrite(tmpjpg, frame)
							except Exception:
								pass
						subject = f"ALERT: thief detected in {Path(source_path).name}"
						body = f"A detection labeled '{label}' with confidence {score:.2f} was found in {source_path} at frame {frame_idx}. See attached snapshot."
						try:
							send_gmail_alert(GMAIL_USER, GMAIL_APP_PASSWORD, ALERT_TO, subject, body, attachment_path=tmpjpg)
							alert_sent = True
						except Exception as e:
							print(f"[alert] send failed: {e}")
						try:
							os.remove(tmpjpg)
						except Exception:
							pass
						break

			out.write(frame)

			if frame_idx % 50 == 0:
				print(f"Processed {frame_idx} frames...")
	finally:
		cap.release()
		out.release()

	print(f"Done — processed {frame_idx} frames. Output saved to {output_path}")


def parse_args():
	p = argparse.ArgumentParser(description='Run thief detection on a video and save annotated output')
	p.add_argument('--model', type=str, default='best(new).pt', help='Path to model checkpoint (.pt)')
	p.add_argument('--source', type=str, required=True, help='Input video file path')
	p.add_argument('--output', type=str, required=True, help='Output video file path')
	p.add_argument('--device', type=str, default='cpu', help='Device to use: cpu or cuda')
	p.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
	p.add_argument('--max-frames', type=int, default=0, help='Optional: process at most N frames (0 = all)')
	return p.parse_args()


if __name__ == '__main__':
	args = parse_args()
	max_frames = args.max_frames if args.max_frames > 0 else None
	try:
		detect_video(args.model, args.source, args.output, device=args.device, conf_thr=args.conf, max_frames=max_frames)
	except Exception as e:
		print('Error during detection:', e)
		sys.exit(1)

