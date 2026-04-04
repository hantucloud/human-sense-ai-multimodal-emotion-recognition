"""
models.py — Model architecture definitions, loaders, and inference preprocessing.

Exports:
    load_face_model()        — cached ResNet18 face emotion model
    load_audio_model()       — cached AudioNet Keras model
    load_face_landmarker()   — cached MediaPipe FaceLandmarker
    preprocess_face()        — BGR crop → normalised torch tensor
    extract_mfcc()           — raw PCM → MFCC matrix
    predict_audio_emotion()  — raw PCM → (label, probs)
    face_model               — module-level loaded face model
    audio_model              — module-level loaded audio model
    face_landmarker          — module-level loaded landmarker
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
import librosa
import tensorflow as tf
from tensorflow import keras
from torchvision import models
from keras.layers import (
    Layer, Activation, Conv1D, SpatialDropout1D,
    Add, GlobalAveragePooling1D, BatchNormalization,
    Dense, Input,
)
from keras.models import Model
import mediapipe as mp
from mediapipe.tasks.python import vision
import streamlit as st

from paths import (
    FACE_MODEL_PATH, AUDIO_MODEL_PATH, LANDMARKER_MODEL_PATH,
    EMOTIONS, SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH,
)

# ===============================
# AUDIO MODEL ARCHITECTURE
# ===============================

class ReverseLayer(Layer):
    def call(self, x):
        return tf.reverse(x, axis=[1])


class SigmoidGateLayer(Layer):
    def call(self, inputs):
        original_x, gated = inputs
        return tf.multiply(original_x, tf.sigmoid(gated))


class ExpandDimsLayer(Layer):
    def call(self, x):
        return tf.expand_dims(x, axis=1)


class ConcatLayer(Layer):
    def call(self, inputs):
        return tf.concat(inputs, axis=-2)


def Temporal_Aware_Block(x, i, activation, nb_filters, kernel_size, dropout_rate=0.1):
    original_x = x
    out = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                 dilation_rate=i, padding='causal')(x)
    out = BatchNormalization()(out)
    out = Activation(activation)(out)
    out = SpatialDropout1D(dropout_rate)(out)
    out = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                 dilation_rate=i, padding='causal')(out)
    out = BatchNormalization()(out)
    out = Activation(activation)(out)
    out = SpatialDropout1D(dropout_rate)(out)
    if original_x.shape[-1] != nb_filters:
        original_x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(original_x)
    return SigmoidGateLayer()([original_x, out])


def build_audionet(input_shape, num_classes,
                   nb_filters=39, kernel_size=2, nb_stacks=1,
                   dilations=8, activation='relu', dropout_rate=0.1,
                   lr=0.001, beta1=0.93, beta2=0.98):
    inputs   = Input(shape=input_shape, name='mfcc_input')
    forward  = inputs
    backward = ReverseLayer(name='reverse')(inputs)
    forward  = Conv1D(nb_filters, kernel_size=1, padding='causal', name='proj_forward')(forward)
    backward = Conv1D(nb_filters, kernel_size=1, padding='causal', name='proj_backward')(backward)
    skip_connections = []
    for stack in range(nb_stacks):
        for d_idx, d in enumerate([2**i for i in range(dilations)]):
            forward  = Temporal_Aware_Block(forward,  d, activation, nb_filters, kernel_size, dropout_rate)
            backward = Temporal_Aware_Block(backward, d, activation, nb_filters, kernel_size, dropout_rate)
            merged   = Add(name=f'add_scale_{stack}_{d_idx}')([forward, backward])
            pooled   = GlobalAveragePooling1D()(merged)
            expanded = ExpandDimsLayer()(pooled)
            skip_connections.append(expanded)
    multi_scale = skip_connections[0] if len(skip_connections) == 1 else ConcatLayer()(skip_connections)
    transposed  = tf.keras.layers.Permute((2, 1))(multi_scale)
    weighted    = Dense(1, use_bias=False, name='scale_weights')(transposed)
    squeezed    = tf.keras.layers.Reshape((nb_filters,))(weighted)
    outputs     = Dense(num_classes, activation='softmax', name='emotion_output')(squeezed)
    model = Model(inputs=inputs, outputs=outputs, name='AudioNet')
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2, epsilon=1e-8),
        metrics=['accuracy'])
    return model


# ===============================
# MODEL LOADERS
# ===============================

@st.cache_resource
def load_face_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    checkpoint = torch.load(FACE_MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@st.cache_resource
def load_audio_model():
    custom_objects = {
        'ReverseLayer':     ReverseLayer,
        'SigmoidGateLayer': SigmoidGateLayer,
        'ExpandDimsLayer':  ExpandDimsLayer,
        'ConcatLayer':      ConcatLayer,
    }
    try:
        return tf.keras.models.load_model(AUDIO_MODEL_PATH, custom_objects=custom_objects)
    except Exception:
        return None


@st.cache_resource
def load_face_landmarker():
    try:
        FaceLandmarker        = vision.FaceLandmarker
        FaceLandmarkerOptions = vision.FaceLandmarkerOptions
        options = FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        return FaceLandmarker.create_from_options(options)
    except Exception:
        return None


# Module-level singletons (loaded once on import)
# Wrapped in try-except to handle missing model files gracefully
try:
    face_model = load_face_model()
except Exception as e:
    print(f"Warning: Failed to load face model: {e}")
    face_model = None

try:
    audio_model = load_audio_model()
except Exception as e:
    print(f"Warning: Failed to load audio model: {e}")
    audio_model = None

try:
    face_landmarker = load_face_landmarker()
except Exception as e:
    print(f"Warning: Failed to load face landmarker: {e}")
    face_landmarker = None

# ImageNet normalisation tensors
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ===============================
# PREPROCESSING & INFERENCE
# ===============================

def preprocess_face(face_bgr):
    """BGR crop → normalised (1, 3, 224, 224) torch tensor."""
    if face_model is None:
        raise RuntimeError("Face model not loaded")
    face = cv2.resize(face_bgr, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = torch.tensor(face).permute(2, 0, 1)
    face = (face - _IMAGENET_MEAN) / _IMAGENET_STD
    return face.unsqueeze(0)


# Exact number of MFCC time-frames the saved AudioNet model was trained on.
_AUDIO_MODEL_FRAMES = 215


def extract_mfcc(audio_pcm):
    """Raw PCM array → (215, N_MFCC) float32 MFCC matrix.
    Output is always padded or truncated to _AUDIO_MODEL_FRAMES so the
    fixed-input AudioNet accepts it regardless of chunk length.
    """
    mfcc = librosa.feature.mfcc(
        y=audio_pcm, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    frames = mfcc.T.astype(np.float32)          # (T, 39)
    T = frames.shape[0]
    if T >= _AUDIO_MODEL_FRAMES:
        frames = frames[:_AUDIO_MODEL_FRAMES]   # truncate
    else:
        pad = np.zeros((_AUDIO_MODEL_FRAMES - T, N_MFCC), dtype=np.float32)
        frames = np.concatenate([frames, pad], axis=0)  # zero-pad
    return frames                               # (215, 39)


def predict_audio_emotion(audio_pcm):
    """Raw PCM → (emotion_label, probs_list) or (None, None).
    Raises on inference errors so callers can surface the exact reason.
    """
    if audio_model is None:
        return None, None
    mfcc  = extract_mfcc(audio_pcm)   # always (215, 39)
    probs = audio_model.predict(mfcc[np.newaxis], verbose=0)[0]
    idx   = int(np.argmax(probs))
    return EMOTIONS[idx], probs.tolist()