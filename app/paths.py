import os

# ===============================
# PATHS
# ===============================

_HERE = os.path.dirname(os.path.abspath(__file__))

# Move one level up from /app → root → models
_MODELS_DIR = os.path.join(_HERE, "..", "models")

FACE_MODEL_PATH       = os.path.join(_MODELS_DIR, "face_emotion_model.pth")
AUDIO_MODEL_PATH      = os.path.join(_MODELS_DIR, "audio_emotion_model.keras")
MEDIAPIPE_MODEL_PATH  = os.path.join(_MODELS_DIR, "blaze_face_short_range.tflite")
LANDMARKER_MODEL_PATH = os.path.join(_MODELS_DIR, "face_landmarker.task")

# ===============================
# EMOTION LABELS & MAPPINGS
# ===============================

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

EMOTION_COLORS = {
    "Angry":    "#FF4B4B",
    "Disgust":  "#A855F7",
    "Fear":     "#F59E0B",
    "Happy":    "#22C55E",
    "Neutral":  "#60A5FA",
    "Sad":      "#6B7280",
    "Surprise": "#F97316",
}

EMOTION_EMOJI = {
    "Angry":    "😠",
    "Disgust":  "🤢",
    "Fear":     "😨",
    "Happy":    "😄",
    "Neutral":  "😐",
    "Sad":      "😢",
    "Surprise": "😲",
}

# Scoring weights: how each emotion contributes to engagement
ENGAGEMENT_WEIGHT = {
    "Happy":    0.9,
    "Surprise": 1.0,
    "Angry":    0.8,
    "Fear":     0.7,
    "Disgust":  0.6,
    "Sad":      0.4,
    "Neutral":  0.1,
}

# Fusion weights
FACE_WEIGHT  = 0.60
AUDIO_WEIGHT = 0.40

# Audio config
SAMPLE_RATE  = 16000
N_MFCC       = 39
# Model was trained on 215 MFCC frames × 512 hop / 16000 sr ≈ 6.88 s
# AUDIO_WINDOW controls how much PCM the live processor buffers per inference.
# extract_mfcc() always pads/truncates to 215 frames so any window works,
# but using the training window gives the most accurate live predictions.
AUDIO_WINDOW = 6.88
HOP_LENGTH   = 512
N_FFT        = 2048

# Head pose engagement thresholds (degrees)
YAW_ENGAGED    = 15   # within ±15° = fully facing forward
YAW_PARTIAL    = 30   # ±15–30° = partial attention
YAW_AWAY       = 30   # beyond ±30° = looking away
PITCH_UP_MAX   = 20   # looking up beyond 20° = distracted
PITCH_DOWN_MAX = 25   # looking down beyond 25° = distracted
ROLL_MAX       = 20   # head tilt beyond ±20° = inattentive

# Key MediaPipe landmark indices (478-point map)
NOSE  = 1
CHIN  = 152
L_EYE = 33
R_EYE = 263
L_EAR = 234
R_EAR = 454