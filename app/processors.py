"""
processors.py — WebRTC stream processors for real-time emotion detection.

Classes:
    EmotionProcessor       — VideoProcessorBase: detects faces, runs emotion
                             inference and head pose per frame.
    AudioEmotionProcessor  — AudioProcessorBase: buffers mic PCM and runs
                             audio emotion inference every AUDIO_WINDOW seconds.
"""

import threading
import time

import av
import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks.python import vision
from streamlit_webrtc import VideoProcessorBase, AudioProcessorBase
import librosa

from paths import (
    MEDIAPIPE_MODEL_PATH,
    EMOTIONS, SAMPLE_RATE, AUDIO_WINDOW,
)
from models import (
    face_model, face_landmarker,
    preprocess_face, predict_audio_emotion,
)
from scoring_engine import rotation_matrix_to_euler, compute_head_engagement_tick


# ===============================
# VIDEO PROCESSOR
# ===============================

class EmotionProcessor(VideoProcessorBase):
    """
    Processes each webcam frame:
      - Detects faces via MediaPipe FaceDetector (VIDEO mode)
      - Runs ResNet18 emotion inference on each crop
      - Estimates head pose via FaceLandmarker and draws overlay
    Thread-safe: results stored under self._lock.
    """

    def __init__(self):
        self._start_ms   = int(time.time() * 1000)
        self._lock       = threading.Lock()
        self.last_results = []   # [(emotion, confidence, probs, bbox), ...]
        self.last_head    = {    # latest head pose reading
            "score": 0.0, "status": "No Face",
            "yaw": None, "pitch": None, "roll": None, "face_present": False,
        }

        options = vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.5,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(time.time() * 1000) - self._start_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self._detector.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        detections = []
        if result.detections:
            for detection in result.detections:
                bbox = detection.bounding_box
                x = max(0, int(bbox.origin_x))
                y = max(0, int(bbox.origin_y))
                w, h = int(bbox.width), int(bbox.height)

                face_crop = img[y:y+h, x:x+w]
                if face_crop.size == 0 or face_crop.shape[0] < 8 or face_crop.shape[1] < 8:
                    continue

                try:
                    if face_model is None:
                        emotion, confidence, probs = "No Model", 0.0, [1/7]*7
                    else:
                        with torch.no_grad():
                            logits     = face_model(preprocess_face(face_crop))
                            probs      = torch.softmax(logits, dim=1)[0].tolist()
                            idx        = int(np.argmax(probs))
                            emotion    = EMOTIONS[idx]
                            confidence = float(probs[idx]) * 100
                except Exception:
                    emotion, confidence, probs = "Unknown", 0.0, [1/7]*7

                detections.append((emotion, confidence, probs, (x, y, w, h)))

                # Draw bounding box + label
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 120), 2)
                label = f"{emotion}  {confidence:.0f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.rectangle(img, (x, y - th - 14), (x + tw + 10, y), (0, 255, 120), -1)
                cv2.putText(img, label, (x + 5, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        # ── Head pose via FaceLandmarker ──────────────────────────────────
        head_data = {"score": 0.0, "status": "No Face",
                     "yaw": None, "pitch": None, "roll": None, "face_present": False}
        if face_landmarker is not None:
            try:
                mp_img_lm = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                lm_result = face_landmarker.detect(mp_img_lm)
                if lm_result.facial_transformation_matrixes:
                    mat = np.array(lm_result.facial_transformation_matrixes[0])
                    R   = mat[:3, :3]
                    pitch, yaw, roll = rotation_matrix_to_euler(R)
                    score, status, _ = compute_head_engagement_tick(
                        yaw, pitch, roll, face_present=True
                    )
                    head_data = {
                        "score": score, "status": status, "face_present": True,
                        "yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1),
                    }
                    # Draw head pose overlay
                    h_img, w_img = img.shape[:2]
                    cx, cy = w_img // 2, 30
                    status_color = (
                        (0, 220, 80)  if status == "Engaged" else
                        (0, 180, 255) if status == "Partial" else
                        (0, 60, 255)
                    )
                    pose_txt = f"Y:{yaw:+.0f} P:{pitch:+.0f} R:{roll:+.0f}  [{status}]"
                    (ptw, pth), _ = cv2.getTextSize(pose_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(img,
                                  (cx - ptw//2 - 6, cy - pth - 6),
                                  (cx + ptw//2 + 6, cy + 6), (0, 0, 0), -1)
                    cv2.putText(img, pose_txt, (cx - ptw//2, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)
                    # Engagement score bar at bottom
                    bar_w = int(w_img * score / 100)
                    cv2.rectangle(img, (0, h_img - 6), (w_img, h_img), (30, 30, 30), -1)
                    cv2.rectangle(img, (0, h_img - 6), (bar_w, h_img), status_color, -1)
            except Exception:
                pass

        with self._lock:
            self.last_results = detections
            self.last_head    = head_data

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ===============================
# AUDIO PROCESSOR
# ===============================

class AudioEmotionProcessor(AudioProcessorBase):
    """
    Buffers microphone PCM and runs audio emotion inference every AUDIO_WINDOW
    seconds. Uses recv_queued() to drain the full frame queue at once and avoid
    dropped-frame warnings.
    """

    def __init__(self):
        self._lock        = threading.Lock()
        self._buffer      = np.array([], dtype=np.float32)
        self.last_emotion = None    # (label, probs_list, updated_at) or None

    def recv_queued(self, frames: list) -> av.AudioFrame:
        pcm_parts = []
        sr = SAMPLE_RATE
        for frame in frames:
            pcm = frame.to_ndarray()
            if pcm.ndim > 1:
                pcm = pcm.mean(axis=0)
            pcm = pcm.astype(np.float32)
            sr  = frame.sample_rate or SAMPLE_RATE
            if sr != SAMPLE_RATE:
                pcm = librosa.resample(pcm, orig_sr=sr, target_sr=SAMPLE_RATE)
            pcm_parts.append(pcm)

        if not pcm_parts:
            return frames[-1] if frames else None

        combined      = np.concatenate(pcm_parts)
        run_inference = False
        audio_chunk   = None

        with self._lock:
            self._buffer = np.concatenate([self._buffer, combined])
            if len(self._buffer) / SAMPLE_RATE >= AUDIO_WINDOW:
                audio_chunk   = self._buffer.copy()
                self._buffer  = np.array([], dtype=np.float32)
                run_inference = True

        if run_inference:
            label, probs = predict_audio_emotion(audio_chunk)
            with self._lock:
                # Store result with a timestamp so callers can detect stale results
                self.last_emotion = (label, probs, time.time()) if label else None

        return frames[-1]   # required by streamlit-webrtc