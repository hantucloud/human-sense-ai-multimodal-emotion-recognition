"""
app.py — Emotion AI · Streamlit entry point.

Responsibility: page config, tab layout, and wiring together all modules.

Module map
----------
paths.py          — file paths, constants, emotion labels & weights
models.py         — AudioNet architecture, model loaders, preprocessing
processors.py     — WebRTC EmotionProcessor / AudioEmotionProcessor
scoring_engine.py — fusion, head pose scoring, session score computation
ui_components.py  — CSS styles and render_scores()
"""

import time
import tempfile
import os

import streamlit as st
import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import librosa

# ── Local modules ─────────────────────────────────────────────────────────────
from paths import (
    MEDIAPIPE_MODEL_PATH,
    EMOTIONS, EMOTION_COLORS, EMOTION_EMOJI,
    SAMPLE_RATE, AUDIO_WINDOW, HOP_LENGTH,
)
from models import (
    face_model, audio_model, face_landmarker,
    preprocess_face, predict_audio_emotion,
)
from processors import EmotionProcessor, AudioEmotionProcessor
from scoring_engine import (
    rotation_matrix_to_euler,
    compute_head_engagement_tick,
    fuse_predictions,
    compute_session_scores,
    build_records_from_lists,
)
from ui_components import apply_styles, render_scores

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Emotion AI", layout="wide", page_icon="🧠")
apply_styles()

st.markdown("<h1>🧠 Emotion AI</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Real-time emotion detection &amp; analysis · '
    'Face (ResNet18) + Voice (AudioNet)</p>',
    unsafe_allow_html=True,
)

# ── RTC config (shared by all tabs) ──────────────────────────────────────────
_RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📹  Live Detection", "📊  Fusion Analysis", "📁  File Analysis", "🎙  Audio Analysis"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live real-time face + audio
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    col_cam1, col_res1 = st.columns([3, 2], gap="large")

    with col_cam1:
        st.markdown("#### Webcam Feed")
        ctx1 = webrtc_streamer(
            key="emotion-tab1",
            video_processor_factory=EmotionProcessor,
            audio_processor_factory=AudioEmotionProcessor,
            media_stream_constraints={"video": True, "audio": True},
            async_processing=True,
            rtc_configuration=_RTC_CONFIG,
        )
        if audio_model is None:
            st.warning("⚠️ Audio model not loaded — face only.")

    with col_res1:
        st.markdown("#### Live Results")
        face_ph  = st.empty()
        audio_ph = st.empty()

    if ctx1.state.playing and ctx1.video_processor:
        while True:
            vp = ctx1.video_processor
            ap = ctx1.audio_processor
            if vp is None:
                break

            with vp._lock:
                face_detections = list(vp.last_results)

            audio_result = None
            if ap is not None:
                with ap._lock:
                    ar = ap.last_emotion
                if ar:
                    _label, _probs, _updated_at = ar
                    if time.time() - _updated_at <= AUDIO_WINDOW:
                        audio_result = (_label, _probs)

            # Face detection cards
            with face_ph.container():
                st.markdown('<div class="section-label">👁 Face Detection</div>',
                            unsafe_allow_html=True)
                if not face_detections:
                    st.markdown('<div class="no-signal">SCANNING FOR FACES …</div>',
                                unsafe_allow_html=True)
                else:
                    for i, (emotion, confidence, probs, _bbox) in enumerate(face_detections):
                        color = EMOTION_COLORS.get(emotion, "#60A5FA")
                        emoji = EMOTION_EMOJI.get(emotion, "🙂")
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="emotion-emoji">{emoji}</div>
                            <div style="flex:1">
                                <div class="emotion-label" style="color:{color}">{emotion}</div>
                                <div class="emotion-conf">Face #{i+1} · {confidence:.1f}% confidence</div>
                                <div class="bar-bg">
                                    <div class="bar-fill" style="width:{confidence:.0f}%;background:{color}"></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.metric("Faces", len(face_detections))
                    m2.metric("Top", face_detections[0][0])

            # Head + audio cards
            with vp._lock:
                head_data = dict(vp.last_head)

            with audio_ph.container():
                st.markdown('<div class="divider-row">🧭 HEAD ENGAGEMENT</div>',
                            unsafe_allow_html=True)
                hstatus  = head_data.get("status", "No Face")
                hscore   = head_data.get("score", 0)
                hyaw     = head_data.get("yaw")
                hpitch   = head_data.get("pitch")
                hroll    = head_data.get("roll")
                hcol     = ("#22C55E" if hstatus == "Engaged" else
                            "#F97316" if hstatus == "Partial" else "#FF4B4B")
                hemj     = ("🟢" if hstatus == "Engaged" else
                            "🟡" if hstatus == "Partial" else "🔴")
                pose_str = (f"Y:{hyaw:+.0f}° P:{hpitch:+.0f}° R:{hroll:+.0f}°"
                            if hyaw is not None else "No face detected")
                st.markdown(f"""
                <div class="result-card" style="border-color:{hcol}33;">
                    <div class="emotion-emoji">{hemj}</div>
                    <div style="flex:1">
                        <div class="emotion-label" style="color:{hcol}">{hstatus}</div>
                        <div class="emotion-conf">{pose_str} · {hscore:.0f}/100</div>
                        <div class="bar-bg">
                            <div class="bar-fill" style="width:{hscore:.0f}%;background:{hcol}"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="divider-row">🎙 VOICE DETECTION</div>',
                            unsafe_allow_html=True)
                if audio_model is None:
                    st.markdown('<div class="no-signal">AUDIO MODEL NOT LOADED</div>',
                                unsafe_allow_html=True)
                elif audio_result is None:
                    st.markdown(
                        f'<div class="no-signal">BUFFERING MIC … ({AUDIO_WINDOW:.0f}s window)</div>',
                        unsafe_allow_html=True)
                else:
                    label, probs = audio_result
                    if label:
                        color = EMOTION_COLORS.get(label, "#a78bfa")
                        emoji = EMOTION_EMOJI.get(label, "🎙")
                        conf  = float(max(probs)) * 100
                        st.markdown(f"""
                        <div class="result-card audio-card">
                            <div class="emotion-emoji">{emoji}</div>
                            <div style="flex:1">
                                <div class="emotion-label" style="color:{color}">{label}</div>
                                <div class="emotion-conf">Voice · {conf:.1f}% confidence</div>
                                <div class="bar-bg">
                                    <div class="bar-fill" style="width:{conf:.0f}%;background:{color}"></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            time.sleep(0.15)
    else:
        with face_ph.container():
            st.markdown('<div class="no-signal">▶ START THE WEBCAM ABOVE TO BEGIN</div>',
                        unsafe_allow_html=True)
        with audio_ph.container():
            st.markdown('<div class="no-signal">🎙 MIC ACTIVATES WITH WEBCAM</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Fusion Analysis + Scoring Engine
# ══════════════════════════════════════════════════════════════════════════════

with tab2:

    # Session state init
    for key, default in [
        ("session_running",  False),
        ("session_records",  []),
        ("session_scores",   None),
        ("session_duration", 15),
        ("session_start_ts", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    col_ctrl, col_score = st.columns([3, 2], gap="large")

    with col_ctrl:
        st.markdown("#### Fusion Analysis Session")
        ctx2 = webrtc_streamer(
            key="emotion-tab2",
            video_processor_factory=EmotionProcessor,
            audio_processor_factory=AudioEmotionProcessor,
            media_stream_constraints={"video": True, "audio": True},
            async_processing=True,
            rtc_configuration=_RTC_CONFIG,
        )
        if audio_model is None:
            st.warning("⚠️ Audio model not loaded — face only.")
        else:
            st.info(f"💡 Audio requires {AUDIO_WINDOW:.1f}s buffering. Use 15s+ sessions for voice detection (first detection ~7s in).")
        st.markdown("---")

        # Duration selector with more options
        dur_col1, dur_col2, dur_col3, dur_col4 = st.columns([1, 1, 1, 2])
        with dur_col1:
            if st.button("⏱ 15s", use_container_width=True,
                         disabled=st.session_state.session_running):
                st.session_state.session_duration = 15
        with dur_col2:
            if st.button("⏱ 30s", use_container_width=True,
                         disabled=st.session_state.session_running):
                st.session_state.session_duration = 30
        with dur_col3:
            if st.button("⏱ 45s", use_container_width=True,
                         disabled=st.session_state.session_running):
                st.session_state.session_duration = 45
        with dur_col4:
            chosen = st.session_state.session_duration
            st.markdown(
                f"<div style='padding:8px 0;color:#a78bfa;font-family:Space Mono,monospace;"
                f"font-size:0.82rem;'>Session: <b>{chosen}s</b> selected</div>",
                unsafe_allow_html=True,
            )

        start_col, stop_col = st.columns(2)
        with start_col:
            start_clicked = st.button(
                "▶ Start Session",
                use_container_width=True,
                disabled=st.session_state.session_running or not ctx2.state.playing,
                type="primary",
            )
        with stop_col:
            stop_clicked = st.button(
                "⏹ Stop Early",
                use_container_width=True,
                disabled=not st.session_state.session_running,
            )

        if start_clicked:
            st.session_state.session_running  = True
            st.session_state.session_records  = []
            st.session_state.session_scores   = None
            st.session_state.session_start_ts = time.time()

        if stop_clicked:
            st.session_state.session_running = False
            st.session_state.session_scores  = compute_session_scores(
                st.session_state.session_records
            )
            st.rerun()  # Force UI update to show detailed breakdown immediately

        if not ctx2.state.playing:
            st.info("▶ Start the webcam above first, then launch a session.")

        progress_ph = st.empty()
        fused_ph    = st.empty()
        timeline_ph = st.empty()

    with col_score:
        st.markdown("#### Session Scores")
        scores_ph = st.empty()

    # Render final / idle scores in right column
    if not st.session_state.session_running:
        with scores_ph.container():
            render_scores(st.session_state.session_scores)

    # ══════════════════════════════════════════════════════════════════════
    # DETAILED BREAKDOWN - Appears ABOVE scores when session ends
    # ══════════════════════════════════════════════════════════════════════
    if not st.session_state.session_running and st.session_state.session_records:
        st.markdown('<div style="margin: 30px 0 20px 0; border-top: 2px solid #1e1e30;"></div>', 
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h3 style="font-family: 'Space Mono', monospace; font-size: 1.5rem; 
                 background: linear-gradient(135deg, #a78bfa, #38bdf8);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                 margin: 0;">
                📊 Detailed Session Breakdown
            </h3>
            <p style="color: #6b7280; font-size: 0.85rem; margin-top: 8px;">
                Frame-by-frame emotions · Audio chunks · Fusion metrics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        records_snap = st.session_state.session_records
        t0           = records_snap[0]["timestamp"]
        total_ticks  = len(records_snap)

        # ── Three-column layout: Face Frames | Audio Chunks | Fusion Report ──
        bd_left, bd_center, bd_right = st.columns([1, 1, 1], gap="medium")

        # ══════════════════════════════════════════════════════════════════
        # LEFT COLUMN: FRAME-BY-FRAME EMOTIONS
        # ══════════════════════════════════════════════════════════════════
        with bd_left:
            st.markdown(
                '<div class="section-label">😶 FRAME-BY-FRAME EMOTIONS</div>',
                unsafe_allow_html=True)
            face_recs = [r for r in records_snap if r.get("face_emotion")]
            if face_recs:
                # Show up to 12 evenly spaced frames
                step = max(1, len(face_recs) // 12)
                for r in face_recs[::step][:12]:
                    fe  = r["face_emotion"]
                    ts  = r["timestamp"] - t0
                    c   = EMOTION_COLORS.get(fe, "#60A5FA")
                    emj = EMOTION_EMOJI.get(fe, "🙂")
                    hst = r.get("head_status", "—")
                    hsc = r.get("head_score", 0)
                    hcl = ("#22C55E" if hst == "Engaged" else
                           "#F97316" if hst == "Partial" else "#FF4B4B")
                    # Face confidence from fused_probs
                    fp_list = r.get("fused_probs", [1/7]*7)
                    fe_idx  = EMOTIONS.index(fe) if fe in EMOTIONS else 0
                    fe_conf = fp_list[fe_idx] * 100
                    st.markdown(f"""
                    <div class="result-card" style="padding:10px 14px;margin-bottom:6px;">
                        <div style="font-size:1.6rem">{emj}</div>
                        <div style="flex:1">
                            <div style="display:flex;justify-content:space-between;
                                 align-items:baseline;">
                                <span class="emotion-label" style="color:{c};
                                     font-size:0.95rem">{fe}</span>
                                <span style="font-family:Space Mono,monospace;
                                     font-size:0.7rem;color:#6b7280">{ts:.1f}s</span>
                            </div>
                            <div style="font-size:0.72rem;color:{hcl};margin-top:2px;">
                                Head: {hst} ({hsc:.0f}/100)
                            </div>
                            <div class="bar-bg" style="margin-top:4px;">
                                <div class="bar-fill"
                                     style="width:{fe_conf:.0f}%;background:{c}"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="no-signal" style="padding:20px;">NO FACE DETECTIONS</div>',
                    unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════
        # CENTER COLUMN: AUDIO CHUNKS
        # ══════════════════════════════════════════════════════════════════
        with bd_center:
            st.markdown(
                '<div class="section-label">🎙 AUDIO CHUNKS</div>',
                unsafe_allow_html=True)

            # Build deduplicated chunk list with timestamps + confidence
            audio_chunks = []
            last_ae = None
            for r in records_snap:
                ae = r.get("audio_emotion")
                if ae and ae != last_ae:
                    # confidence = audio emotion's weight in fused_probs
                    fp_list = r.get("fused_probs", [1/7]*7)
                    ae_idx  = EMOTIONS.index(ae) if ae in EMOTIONS else 0
                    # back-calculate audio-only confidence from fused probs
                    # fused = face*0.6 + audio*0.4, so audio ≈ fused/0.4
                    # instead just use the raw fused prob as proxy
                    conf = min(100.0, fp_list[ae_idx] / 0.4 * 100)
                    audio_chunks.append({
                        "label": ae,
                        "conf":  conf,
                        "ts":    r["timestamp"] - t0,
                        "chunk": len(audio_chunks) + 1,
                    })
                    last_ae = ae

            if audio_chunks:
                for ch in audio_chunks:
                    lbl  = ch["label"]
                    conf = ch["conf"]
                    c    = EMOTION_COLORS.get(lbl, "#a78bfa")
                    emj  = EMOTION_EMOJI.get(lbl, "🎙")
                    st.markdown(f"""
                    <div class="result-card audio-card"
                         style="padding:10px 14px;margin-bottom:6px;">
                        <div style="font-size:1.6rem">{emj}</div>
                        <div style="flex:1">
                            <div style="display:flex;justify-content:space-between;
                                 align-items:baseline;">
                                <span class="emotion-label" style="color:{c};
                                     font-size:0.95rem">{lbl}</span>
                                <span style="font-family:Space Mono,monospace;
                                     font-size:0.7rem;color:#6b7280">
                                    chunk {ch["chunk"]} · {ch["ts"]:.1f}s
                                </span>
                            </div>
                            <div class="bar-bg" style="margin-top:4px;">
                                <div class="bar-fill"
                                     style="width:{conf:.0f}%;background:{c}"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Check if ANY audio_emotion exists in records
                has_any_audio = any(r.get("audio_emotion") for r in records_snap)
                audio_model_loaded = audio_model is not None
                
                if not audio_model_loaded:
                    msg = "AUDIO MODEL NOT LOADED"
                    submsg = "Audio emotion detection requires the model file"
                elif not has_any_audio:
                    msg = "NO AUDIO DETECTED"
                    submsg = f"Session may be too short (needs {AUDIO_WINDOW:.0f}s+ for first detection) or mic may not be active"
                else:
                    msg = "NO AUDIO CHUNKS"
                    submsg = "Audio detected but no distinct emotion chunks found"
                
                st.markdown(
                    f'<div class="no-signal" style="padding:20px;">'
                    f'{msg}<br>'
                    f'<span style="font-size:0.72rem;color:#6b7280;">'
                    f'{submsg}</span></div>',
                    unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════
        # RIGHT COLUMN: FUSION REPORT
        # ══════════════════════════════════════════════════════════════════
        with bd_right:
            st.markdown(
                '<div class="section-label">⚡ FUSION REPORT</div>',
                unsafe_allow_html=True)

            # ── Model source summary cards ────────────────────────────
            face_hits  = sum(1 for r in records_snap if r.get("face_emotion"))
            audio_hits = sum(1 for r in records_snap if r.get("audio_emotion"))
            
            # Debug: Count unique audio emotions detected
            unique_audio = set(r.get("audio_emotion") for r in records_snap if r.get("audio_emotion"))
            audio_debug = f"{len(unique_audio)} unique" if unique_audio else "none"

            # Face Model Card
            fc   = "#22C55E" if face_hits > 0 else "#FF4B4B"
            fp   = face_hits / total_ticks * 100 if total_ticks else 0
            st.markdown(f"""
            <div class="score-card" style="border-color:{fc}33;padding:10px 14px;margin-bottom:8px;">
                <div style="font-family:Space Mono,monospace;font-size:0.6rem;
                     letter-spacing:2px;color:#4b5563;">FACE MODEL</div>
            <div style="font-family:Space Mono,monospace;font-size:1.3rem;
                     font-weight:700;color:{fc};margin-top:4px;">
                    {'✅' if face_hits > 0 else '❌'} {face_hits}/{total_ticks}
                </div>
                <div style="font-size:0.72rem;color:#6b7280;margin-top:2px;">
                    ticks with face detected ({fp:.0f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Audio Model Card
            ac   = "#22C55E" if audio_hits > 0 else "#FF4B4B"
            ap_p = audio_hits / total_ticks * 100 if total_ticks else 0
            st.markdown(f"""
            <div class="score-card" style="border-color:{ac}33;padding:10px 14px;margin-bottom:12px;">
                <div style="font-family:Space Mono,monospace;font-size:0.6rem;
                     letter-spacing:2px;color:#4b5563;">AUDIO MODEL</div>
                <div style="font-family:Space Mono,monospace;font-size:1.3rem;
                     font-weight:700;color:{ac};margin-top:4px;">
                    {'✅' if audio_hits > 0 else '❌'} {audio_hits}/{total_ticks}
                </div>
                <div style="font-size:0.72rem;color:#6b7280;margin-top:2px;">
                    ticks with voice detected ({ap_p:.0f}%)
                </div>
                <div style="font-size:0.65rem;color:#4b5563;margin-top:4px;">
                    Emotions: {audio_debug}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Session scores (compact version for right column)
            scores = st.session_state.session_scores
            if scores:
                dom   = scores["dominant_emotion"]
                col   = EMOTION_COLORS.get(dom, "#60A5FA")
                emj   = EMOTION_EMOJI.get(dom, "🙂")
                eng   = scores["engagement"]
                ecol  = "#F97316"
                agr   = scores["agreement_rate"]

                st.markdown(f"""
                <div class="score-card" style="border-color:{col}33;padding:10px 14px;margin-bottom:8px;">
                    <div style="font-family:Space Mono,monospace;font-size:0.6rem;
                         letter-spacing:2px;color:#4b5563;margin-bottom:4px;">DOMINANT</div>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <div style="font-size:2rem;line-height:1;">{emj}</div>
                        <div style="flex:1;">
                            <div style="font-family:Space Mono,monospace;font-size:1.1rem;
                                 font-weight:700;color:{col};">{dom}</div>
                            <div style="font-size:0.7rem;color:#6b7280;">
                                {scores["dominant_pct"]:.1f}% of session
                            </div>
                        </div>
                    </div>
                </div>

                <div class="score-card" style="border-color:{ecol}33;padding:10px 14px;margin-bottom:8px;">
                    <div style="font-family:Space Mono,monospace;font-size:0.6rem;
                         letter-spacing:2px;color:#4b5563;margin-bottom:4px;">ENGAGEMENT</div>
                    <div style="font-family:Space Mono,monospace;font-size:1.4rem;
                         font-weight:700;color:{ecol};">{eng:.0f}</div>
                    <div class="bar-bg" style="margin-top:4px;">
                        <div class="bar-fill" style="width:{eng:.0f}%;background:{ecol}"></div>
                    </div>
                </div>

                <div class="score-card" style="padding:10px 14px;">
                    <div style="font-family:Space Mono,monospace;font-size:0.6rem;
                         letter-spacing:2px;color:#4b5563;margin-bottom:4px;">FACE–VOICE AGREEMENT</div>
                    <div style="font-family:Space Mono,monospace;font-size:1.4rem;
                         font-weight:700;color:#38bdf8;">{agr:.0f}%</div>
                    <div class="bar-bg" style="margin-top:4px;">
                        <div class="bar-fill" style="width:{agr:.0f}%;background:#38bdf8"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="no-signal" style="padding:20px;font-size:0.75rem;">'
                    'NOT ENOUGH DATA</div>',
                    unsafe_allow_html=True)

    # Session recording loop
    if st.session_state.session_running and ctx2.state.playing and ctx2.video_processor:

        duration = st.session_state.session_duration
        start_ts = st.session_state.session_start_ts
        records  = st.session_state.session_records

        while st.session_state.session_running:
            elapsed = time.time() - start_ts

            if elapsed > duration:  # Changed from >= to > to ensure full duration
                st.session_state.session_running = False
                st.session_state.session_scores  = compute_session_scores(records)
                st.rerun()  # Force UI update to show detailed breakdown
                break

            vp = ctx2.video_processor
            ap = ctx2.audio_processor

            face_emotion = face_probs = audio_emotion = audio_probs = None
            head_tick = {"score": 0.0, "status": "No Face",
                         "yaw": None, "pitch": None, "roll": None, "face_present": False}

            if vp:
                with vp._lock:
                    fd        = list(vp.last_results)
                    head_tick = dict(vp.last_head)
                if fd:
                    face_emotion = fd[0][0]
                    face_probs   = fd[0][2]

            audio_fresh = False
            if ap:
                with ap._lock:
                    ar = ap.last_emotion
                # Allow result to stay valid for 2× the audio window
                if ar:
                    _label, _probs, _updated_at = ar
                    age = time.time() - _updated_at
                    if age <= AUDIO_WINDOW * 2:
                        audio_emotion, audio_probs = _label, _probs
                        audio_fresh = True

            if face_probs is not None:
                fused_emotion, fused_conf, fused_probs = fuse_predictions(face_probs, audio_probs)
            else:
                fused_emotion = face_emotion or "Neutral"
                fused_conf    = 0.0
                fused_probs   = [1/7] * 7

            records.append({
                'timestamp':     time.time(),
                'face_emotion':  face_emotion,
                'audio_emotion': audio_emotion,
                'fused_emotion': fused_emotion,
                'fused_probs':   fused_probs,
                'head_score':    head_tick["score"],
                'head_status':   head_tick["status"],
                'head_yaw':      head_tick["yaw"],
                'head_pitch':    head_tick["pitch"],
                'head_roll':     head_tick["roll"],
                'head_present':  head_tick["face_present"],
            })

            # Progress bar
            pct = min(100, elapsed / duration * 100)
            with progress_ph.container():
                st.markdown(
                    f"<div style='font-family:Space Mono,monospace;font-size:0.75rem;"
                    f"color:#6b7280;margin-bottom:4px;'>"
                    f"⏱ {elapsed:.1f}s / {duration}s — {len(records)} samples collected</div>",
                    unsafe_allow_html=True,
                )
                st.progress(int(pct))

            # Live fused emotion card
            color = EMOTION_COLORS.get(fused_emotion, "#60A5FA")
            emoji = EMOTION_EMOJI.get(fused_emotion, "🙂")

            face_status_html  = (
                f'<span style="color:{EMOTION_COLORS.get(face_emotion,"#22C55E")};">●</span> '
                f'Face: <b>{face_emotion}</b>'
                if face_emotion else
                '<span style="color:#FF4B4B;">●</span> Face: <i>no detection</i>'
            )
            audio_status_html = (
                f'<span style="color:{EMOTION_COLORS.get(audio_emotion,"#F97316")};">●</span> '
                f'Voice: <b>{audio_emotion}</b>'
                if audio_fresh and audio_emotion else
                '<span style="color:#6b7280;">●</span> Voice: <i>buffering ({AUDIO_WINDOW:.0f}s window)…</i>'
            )

            with fused_ph.container():
                st.markdown('<div class="section-label">⚡ FUSED EMOTION (LIVE)</div>',
                            unsafe_allow_html=True)
                st.markdown(f"""
                <div class="result-card fused-card">
                    <div class="emotion-emoji">{emoji}</div>
                    <div style="flex:1">
                        <div class="emotion-label" style="color:{color}">{fused_emotion}</div>
                        <div class="emotion-conf" style="display:flex;gap:18px;flex-wrap:wrap;margin-top:4px;">
                            <span>{face_status_html}</span>
                            <span>{audio_status_html}</span>
                        </div>
                        <div style="font-family:Space Mono,monospace;font-size:0.7rem;
                             color:#6b7280;margin-top:4px;">
                            Confidence: {fused_conf:.1f}%
                            {"&nbsp;·&nbsp; fusion: face+voice" if audio_fresh and audio_emotion else "&nbsp;·&nbsp; face only"}
                        </div>
                        <div class="bar-bg">
                            <div class="bar-fill" style="width:{fused_conf:.0f}%;background:{color}"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Rolling timeline (last 20 ticks)
            dots_html = '<div class="timeline-wrap">'
            for r in records[-20:]:
                e = r['fused_emotion']
                c = EMOTION_COLORS.get(e, "#60A5FA")
                m = EMOTION_EMOJI.get(e, "·")
                dots_html += f'<div class="tl-dot" style="background:{c}" title="{e}">{m}</div>'
            dots_html += '</div>'
            with timeline_ph.container():
                st.markdown('<div class="section-label">RECENT EMOTION TRAIL</div>',
                            unsafe_allow_html=True)
                st.markdown(dots_html, unsafe_allow_html=True)

            # Interim scores every 3 seconds
            if len(records) % 6 == 0 and len(records) > 0:
                interim = compute_session_scores(records)
                if interim:
                    with scores_ph.container():
                        render_scores(interim, interim=True)

            time.sleep(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — File Upload Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab3:

    # ── Offline face detector (IMAGE mode) ───────────────────────────────────
    @st.cache_resource
    def _get_offline_detector():
        opts = vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.3,   # lower from 0.5 — video frames can be blurry/dark
        )
        return vision.FaceDetector.create_from_options(opts)

    _offline_detector = _get_offline_detector()

    def analyse_frame_offline(bgr_frame):
        """
        Run face detection, emotion inference, and head pose on a single BGR frame.
        Returns (face_emotion, face_probs, head_tick, bbox, error_str) where
        bbox = (x, y, w, h) or None if no face detected.
        """
        rgb          = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        # MediaPipe requires a C-contiguous uint8 array
        rgb          = np.ascontiguousarray(rgb, dtype=np.uint8)
        face_emotion = None
        face_probs   = None
        bbox         = None
        face_error   = None
        head_tick    = {"score": 0.0, "status": "No Face",
                        "yaw": None, "pitch": None, "roll": None, "face_present": False}

        try:
            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            det_res = _offline_detector.detect(mp_img)
            if det_res.detections:
                det  = det_res.detections[0]
                bb   = det.bounding_box
                x = max(0, int(bb.origin_x))
                y = max(0, int(bb.origin_y))
                w, h = int(bb.width), int(bb.height)
                # Clamp to frame bounds
                x2 = min(bgr_frame.shape[1], x + w)
                y2 = min(bgr_frame.shape[0], y + h)
                w, h = x2 - x, y2 - y
                bbox = (x, y, w, h)
                crop = bgr_frame[y:y2, x:x2]
                if crop.size > 0 and crop.shape[0] >= 8 and crop.shape[1] >= 8:
                    with torch.no_grad():
                        logits       = face_model(preprocess_face(crop))
                        probs        = torch.softmax(logits, dim=1)[0].tolist()
                        idx          = int(np.argmax(probs))
                        face_emotion = EMOTIONS[idx]
                        face_probs   = probs
            else:
                face_error = "no_detection"
        except Exception as ex:
            face_error = str(ex)

        if face_landmarker is not None:
            try:
                mp_img_lm = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                lm_res    = face_landmarker.detect(mp_img_lm)
                if lm_res.facial_transformation_matrixes:
                    mat   = np.array(lm_res.facial_transformation_matrixes[0])
                    R     = mat[:3, :3]
                    pitch, yaw, roll = rotation_matrix_to_euler(R)
                    score, status, _ = compute_head_engagement_tick(yaw, pitch, roll, True)
                    head_tick = {"score": score, "status": status, "face_present": True,
                                 "yaw": round(yaw, 1), "pitch": round(pitch, 1),
                                 "roll": round(roll, 1)}
            except Exception:
                pass

        return face_emotion, face_probs, head_tick, bbox, face_error


    def draw_frame_overlay(frame, emotion, head_tick, bbox, confidence=None):
        """
        Draw the full rich annotation overlay onto a BGR frame in-place.
        Mirrors the visual style of EmotionProcessor.recv().
          - Bounding box + emotion label badge
          - Head pose text (yaw / pitch / roll + status)
          - Engagement score bar at the bottom
          - Timestamp is handled by the caller
        Returns the annotated frame.
        """
        ann      = frame.copy()
        h_img, w_img = ann.shape[:2]

        # ── Bounding box + emotion badge ─────────────────────────────────
        if bbox is not None and emotion is not None:
            x, y, w, h = bbox
            conf = confidence if confidence is not None else 0.0
            cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 255, 120), 2)

            label = f"{emotion}  {conf:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            # Badge background — clamp so it never goes above the frame
            badge_y0 = max(0, y - th - 14)
            badge_y1 = max(th + 14, y)
            cv2.rectangle(ann, (x, badge_y0), (x + tw + 10, badge_y1), (0, 255, 120), -1)
            cv2.putText(ann, label, (x + 5, badge_y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        # ── Head pose overlay (top-centre) ───────────────────────────────
        status      = head_tick.get("status", "No Face")
        score       = head_tick.get("score", 0.0)
        yaw         = head_tick.get("yaw")
        pitch       = head_tick.get("pitch")
        roll        = head_tick.get("roll")
        face_present = head_tick.get("face_present", False)

        status_color = ((0, 220,  80) if status == "Engaged" else
                        (0, 180, 255) if status == "Partial"  else
                        (0,  60, 255))

        if face_present and yaw is not None:
            pose_txt = f"Y:{yaw:+.0f} P:{pitch:+.0f} R:{roll:+.0f}  [{status}]"
            cx, cy   = w_img // 2, 30
            (ptw, pth), _ = cv2.getTextSize(pose_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(ann,
                          (cx - ptw // 2 - 6, cy - pth - 6),
                          (cx + ptw // 2 + 6, cy + 6),
                          (0, 0, 0), -1)
            cv2.putText(ann, pose_txt, (cx - ptw // 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)

        # ── Engagement score bar (bottom of frame) ───────────────────────
        bar_w = int(w_img * score / 100)
        cv2.rectangle(ann, (0, h_img - 8), (w_img, h_img), (20, 20, 20), -1)
        if bar_w > 0:
            cv2.rectangle(ann, (0, h_img - 8), (bar_w, h_img), status_color, -1)
        # Score label at left edge of bar
        score_lbl = f"{status}  {score:.0f}/100"
        cv2.putText(ann, score_lbl, (6, h_img - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color, 1)

        return ann

    # ── UI ────────────────────────────────────────────────────────────────────
    st.markdown("#### Upload a File for Emotion Analysis")
    st.markdown(
        '<p style="color:#6b7280;font-size:0.85rem;">Supports video (.mp4 .mov .avi), '
        'audio (.wav .mp3 .flac), and images (.jpg .png .webp). '
        'For video, both face frames and audio track are analysed.</p>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop your file here",
        type=["mp4", "mov", "avi", "wav", "mp3", "flac", "jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    sample_fps = st.select_slider(
        "Video frame sample rate (frames per second analysed)",
        options=[1, 2, 4, 8],
        value=2,
        help="Higher = more accurate but slower. 2 fps is a good balance.",
    )

    analyse_btn        = st.button("🔍 Analyse File", type="primary",
                                    disabled=(uploaded is None))
    
    # Initialize session state for Tab 3 analysis results
    if "tab3_analysis_results" not in st.session_state:
        st.session_state.tab3_analysis_results = None
    if "tab3_current_file" not in st.session_state:
        st.session_state.tab3_current_file = None
    
    # Reset analysis if file changed
    if uploaded and st.session_state.tab3_current_file != uploaded.name:
        st.session_state.tab3_analysis_results = None
        st.session_state.tab3_current_file = uploaded.name
        # Also reset chat history for new file
        file_key = f"chat_history_{uploaded.name}"
        if file_key in st.session_state:
            del st.session_state[file_key]
    
    result_placeholder = st.empty()

    # ── Off-topic keyword filter (runs BEFORE Ollama — Python-level hard block) ─
    _OFF_TOPIC_KW = [
        "code", "program", "function", "script", "algorithm", "debug", "syntax",
        "java", "python", "javascript", "typescript", "c++", "c#", "html", "css",
        "sql", "php", "ruby", "swift", "kotlin", "golang", "rust", "bash",
        "math", "calculat", "equation", "formula", "solve", "integral", "derivative",
        "recipe", "cook", "weather", "news", "history", "science", "geography",
        "how do i", "how to", "what is", "who is", "tell me about", "explain",
        "write a", "create a", "build a", "make a", "generate a", "give me",
        "translate", "summarize", "summarise", "essay", "story", "poem",
    ]

    def _is_off_topic(msg: str) -> bool:
        m = msg.lower()
        return any(kw in m for kw in _OFF_TOPIC_KW)

    if analyse_btn and uploaded is not None:

        file_ext = uploaded.name.rsplit(".", 1)[-1].lower()
        is_video = file_ext in ("mp4", "mov", "avi")
        is_audio = file_ext in ("wav", "mp3", "flac")
        is_image = file_ext in ("jpg", "jpeg", "png", "webp")

        with result_placeholder.container():
            prog_bar   = st.progress(0)
            status_txt = st.empty()

        face_emotions    = []
        face_probs_list  = []
        head_ticks       = []
        timestamps       = []
        audio_results    = []
        annotated_frames = []
        out_video_path   = None

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            # ── Image ─────────────────────────────────────────────────────
            if is_image:
                status_txt.markdown('<div class="section-label">ANALYSING IMAGE …</div>',
                                    unsafe_allow_html=True)
                img_bgr = cv2.imread(tmp_path)
                if img_bgr is None:
                    arr     = np.frombuffer(open(tmp_path, "rb").read(), np.uint8)
                    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                fe, fp, ht, bbox, _ferr = analyse_frame_offline(img_bgr)
                face_emotions.append(fe)
                face_probs_list.append(fp)
                head_ticks.append(ht)
                timestamps.append(0.0)

                conf = float(max(fp)) * 100 if fp else 0.0
                ann  = draw_frame_overlay(img_bgr, fe, ht, bbox, conf)
                annotated_frames.append(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
                prog_bar.progress(100)

            # ── Audio ─────────────────────────────────────────────────────
            elif is_audio:
                status_txt.markdown('<div class="section-label">LOADING AUDIO …</div>',
                                    unsafe_allow_html=True)
                pcm, _     = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
                total_secs = len(pcm) / SAMPLE_RATE
                chunk_secs = 3.0
                n_chunks   = max(1, int(total_secs / chunk_secs))

                for ci in range(n_chunks):
                    s     = int(ci * chunk_secs * SAMPLE_RATE)
                    e     = int(min(len(pcm), s + chunk_secs * SAMPLE_RATE))
                    label, probs = predict_audio_emotion(pcm[s:e])
                    if label:
                        audio_results.append((label, probs))
                        face_emotions.append(None)
                        face_probs_list.append(None)
                        head_ticks.append({})
                        timestamps.append(ci * chunk_secs)
                    prog_bar.progress(int((ci + 1) / n_chunks * 100))

                status_txt.markdown(
                    f'<div class="section-label">ANALYSED {n_chunks} AUDIO CHUNKS</div>',
                    unsafe_allow_html=True)

            # ── Video ─────────────────────────────────────────────────────
            elif is_video:
                cap            = cv2.VideoCapture(tmp_path)
                total_fps      = cap.get(cv2.CAP_PROP_FPS) or 25
                total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration_secs  = total_frames / total_fps
                frame_interval = max(1, int(total_fps / sample_fps))
                frames_to_proc = total_frames // frame_interval

                status_txt.markdown(
                    f'<div class="section-label">ANALYSING VIDEO — '
                    f'{duration_secs:.1f}s · {frames_to_proc} frames @ {sample_fps}fps</div>',
                    unsafe_allow_html=True)

                # Output annotated video (mp4v → .mp4 for browser playback)
                out_video_path = tmp_path.replace(f".{file_ext}", "_annotated.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vwriter = cv2.VideoWriter(out_video_path, fourcc, total_fps, (frame_w, frame_h))

                frame_idx = processed = 0
                # Carry-forward state for non-sampled frames
                last_fe   = None
                last_fp   = None
                last_ht   = {"score": 0.0, "status": "No Face",
                             "yaw": None, "pitch": None, "roll": None, "face_present": False}
                last_bbox = None
                last_conf = 0.0

                # Extract audio track from video
                audio_extract_error = None
                try:
                    import subprocess
                    from models import _AUDIO_MODEL_FRAMES
                    audio_chunk_secs = (_AUDIO_MODEL_FRAMES * 512) / SAMPLE_RATE  # ≈ 6.88s
                    audio_tmp = tmp_path.replace(f".{file_ext}", "_audio.wav")
                    result_ff = subprocess.run(
                        ["ffmpeg", "-y", "-i", tmp_path, "-vn",
                         "-ar", str(SAMPLE_RATE), "-ac", "1", audio_tmp],
                        capture_output=True, timeout=60,
                    )
                    if result_ff.returncode == 0 and os.path.exists(audio_tmp):
                        pcm_vid, _ = librosa.load(audio_tmp, sr=SAMPLE_RATE, mono=True)
                        import math as _math
                        n_chunks = max(1, _math.ceil(len(pcm_vid) / SAMPLE_RATE / audio_chunk_secs))
                        for ci in range(n_chunks):
                            s = int(ci * audio_chunk_secs * SAMPLE_RATE)
                            e = int(min(len(pcm_vid), s + audio_chunk_secs * SAMPLE_RATE))
                            lbl, prbs = predict_audio_emotion(pcm_vid[s:e])
                            if lbl:
                                audio_results.append((lbl, prbs))
                        os.remove(audio_tmp)
                    else:
                        audio_extract_error = f"ffmpeg exited with code {result_ff.returncode}"
                except Exception as ex:
                    audio_extract_error = str(ex)

                face_errors = []   # collect per-frame detection errors

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # ── Run inference on sampled frames only ──────────────
                    if frame_idx % frame_interval == 0:
                        ts  = frame_idx / total_fps
                        fe, fp, ht, bbox, ferr = analyse_frame_offline(frame)
                        if ferr and ferr != "no_detection":
                            face_errors.append(f"frame {frame_idx}: {ferr}")
                        face_emotions.append(fe)
                        face_probs_list.append(fp)
                        head_ticks.append(ht)
                        timestamps.append(ts)

                        # Update carry-forward state
                        last_fe   = fe
                        last_fp   = fp
                        last_ht   = ht
                        last_bbox = bbox
                        last_conf = float(max(fp)) * 100 if fp else 0.0

                        processed += 1
                        prog_bar.progress(min(95, int(processed / max(1, frames_to_proc) * 100)))

                    # Draw full overlay using current (or carried-forward) state
                    ts_str = f"{frame_idx / total_fps:.2f}s"
                    ann    = draw_frame_overlay(frame, last_fe, last_ht, last_bbox, last_conf)

                    # Timestamp watermark (bottom-right, above the bar)
                    h_img, w_img = ann.shape[:2]
                    cv2.putText(ann, ts_str, (w_img - 80, h_img - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

                    vwriter.write(ann)

                    # Store annotated frame for the strip (sampled frames only)
                    if frame_idx % frame_interval == 0:
                        annotated_frames.append(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))

                    frame_idx += 1

                cap.release()
                vwriter.release()

                # Re-encode with ffmpeg for web-compatible H.264 if available
                try:
                    import subprocess
                    out_h264 = out_video_path.replace("_annotated.mp4", "_annotated_h264.mp4")
                    r = subprocess.run(
                        ["ffmpeg", "-y", "-i", out_video_path,
                         "-vcodec", "libx264", "-crf", "23",
                         "-preset", "fast", "-pix_fmt", "yuv420p", out_h264],
                        capture_output=True, timeout=120,
                    )
                    if r.returncode == 0 and os.path.exists(out_h264):
                        os.remove(out_video_path)
                        out_video_path = out_h264
                except Exception:
                    pass   # keep mp4v file if ffmpeg unavailable

                prog_bar.progress(100)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # ── Build records + compute scores ────────────────────────────────
        if timestamps:
            status_txt.markdown('<div class="section-label">COMPUTING SCORES …</div>',
                                unsafe_allow_html=True)
            records = build_records_from_lists(
                face_emotions, face_probs_list,
                audio_results if audio_results else None,
                head_ticks, timestamps,
            )
            scores = compute_session_scores(records)
        else:
            scores  = None
            records = []

        # Store results in session state for persistence
        st.session_state.tab3_analysis_results = {
            "scores": scores,
            "face_emotions": face_emotions,
            "face_probs_list": face_probs_list,
            "audio_results": audio_results,
            "head_ticks": head_ticks,
            "timestamps": timestamps,
            "annotated_frames": annotated_frames,
            "out_video_path": out_video_path,
            "is_video": is_video,
            "is_audio": is_audio,
            "is_image": is_image,
            "uploaded_name": uploaded.name,
        }

        prog_bar.empty()
        status_txt.empty()

        # ── Display results ───────────────────────────────────────────────
        with result_placeholder.container():
            res_col, score_col = st.columns([2, 3], gap="large")

            with res_col:
                st.markdown("#### Preview")

                if is_image and annotated_frames:
                    st.image(annotated_frames[0], use_container_width=True,
                             caption="Analysed image with emotion + head pose overlay")

                elif is_video and out_video_path and os.path.exists(out_video_path):
                    st.markdown(
                        '<div class="section-label">🎬 ANNOTATED VIDEO</div>',
                        unsafe_allow_html=True)
                    with open(out_video_path, "rb") as vf:
                        video_bytes = vf.read()
                    st.video(video_bytes)
                    st.download_button(
                        label="⬇️ Download Annotated Video",
                        data=video_bytes,
                        file_name=f"annotated_{uploaded.name.rsplit('.', 1)[0]}.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                    # Cleanup output video after reading into memory
                    os.remove(out_video_path)

                    # Frame strip below the player
                    if annotated_frames:
                        st.markdown(
                            f'<div class="section-label" style="margin-top:12px;">'
                            f'SAMPLED FRAMES — {len(annotated_frames)} @ {sample_fps}fps</div>',
                            unsafe_allow_html=True)
                        n_show     = min(8, len(annotated_frames))
                        indices    = np.linspace(0, len(annotated_frames)-1, n_show, dtype=int)
                        cols_strip = st.columns(n_show)
                        for ci, (col_s, frm) in enumerate(
                                zip(cols_strip, [annotated_frames[i] for i in indices])):
                            with col_s:
                                st.image(frm, use_container_width=True,
                                         caption=f"{timestamps[indices[ci]]:.1f}s")

                elif is_audio:
                    st.markdown(
                        f'<div class="no-signal">🎙 AUDIO FILE<br>'
                        f'{len(audio_results)} × 3s CHUNKS ANALYSED</div>',
                        unsafe_allow_html=True)

                # Per-frame emotion cards (up to 6)
                if face_emotions and any(e for e in face_emotions):
                    st.markdown(
                        '<div class="section-label" style="margin-top:14px;">'
                        'FRAME-BY-FRAME EMOTIONS</div>',
                        unsafe_allow_html=True)
                    show_n = min(6, len(face_emotions))
                    step   = max(1, len(face_emotions) // show_n)
                    for i in range(0, min(len(face_emotions), show_n * step), step):
                        fe  = face_emotions[i]
                        ht  = head_ticks[i] if i < len(head_ticks) else {}
                        ts  = timestamps[i]  if i < len(timestamps) else 0
                        if fe is None:
                            continue
                        c   = EMOTION_COLORS.get(fe, "#60A5FA")
                        emj = EMOTION_EMOJI.get(fe, "🙂")
                        hsc = ht.get("score", 0)
                        hst = ht.get("status", "—")
                        hcl = ("#22C55E" if hst == "Engaged" else
                               "#F97316" if hst == "Partial" else "#FF4B4B")
                        ts_label = f"{ts:.1f}s" if is_video else "frame"
                        st.markdown(f"""
                        <div class="result-card" style="padding:10px 14px;margin-bottom:6px;">
                            <div style="font-size:1.6rem">{emj}</div>
                            <div style="flex:1">
                                <div style="display:flex;justify-content:space-between;">
                                    <span class="emotion-label" style="color:{c};font-size:0.95rem">{fe}</span>
                                    <span style="font-family:Space Mono,monospace;font-size:0.7rem;color:#6b7280">{ts_label}</span>
                                </div>
                                <div style="font-size:0.72rem;color:{hcl};margin-top:2px;">
                                    Head: {hst} ({hsc:.0f}/100)
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Audio chunk results
                if audio_results:
                    st.markdown('<div class="divider-row">🎙 AUDIO CHUNKS</div>',
                                unsafe_allow_html=True)
                    for ci, (lbl, prbs) in enumerate(audio_results[:6]):
                        c    = EMOTION_COLORS.get(lbl, "#a78bfa")
                        emj  = EMOTION_EMOJI.get(lbl, "🎙")
                        conf = float(max(prbs)) * 100
                        st.markdown(f"""
                        <div class="result-card audio-card" style="padding:10px 14px;margin-bottom:6px;">
                            <div style="font-size:1.6rem">{emj}</div>
                            <div style="flex:1">
                                <div style="display:flex;justify-content:space-between;">
                                    <span class="emotion-label" style="color:{c};font-size:0.95rem">{lbl}</span>
                                    <span style="font-family:Space Mono,monospace;font-size:0.7rem;color:#6b7280">chunk {ci+1}</span>
                                </div>
                                <div class="bar-bg" style="margin-top:4px;">
                                    <div class="bar-fill" style="width:{conf:.0f}%;background:{c}"></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            with score_col:
                st.markdown("#### Fusion Report")

                # ── Model source summary ──────────────────────────────────
                face_detected  = sum(1 for e in face_emotions if e is not None)
                audio_detected = len(audio_results)
                total_frames   = len(face_emotions)

                fa_col, au_col = st.columns(2)
                with fa_col:
                    face_pct = face_detected / total_frames * 100 if total_frames else 0
                    fc = "#22C55E" if face_detected > 0 else "#FF4B4B"
                    face_err_note = ""
                    if face_detected == 0:
                        errs = [e for e in face_errors] if 'face_errors' in dir() and face_errors else []
                        if errs:
                            face_err_note = f'<div style="font-size:0.65rem;color:#FF4B4B;margin-top:3px;word-break:break-all;">{errs[0]}</div>'
                        else:
                            face_err_note = '<div style="font-size:0.65rem;color:#6b7280;margin-top:3px;">no faces found — try lowering detection threshold or check video quality</div>'
                    st.markdown(f"""
                    <div class="score-card" style="border-color:{fc}33;padding:10px 14px;">
                        <div style="font-family:Space Mono,monospace;font-size:0.6rem;
                             letter-spacing:2px;color:#4b5563;">FACE MODEL</div>
                        <div style="font-family:Space Mono,monospace;font-size:1.3rem;
                             font-weight:700;color:{fc};margin-top:4px;">
                            {'✅' if face_detected > 0 else '❌'} {face_detected}/{total_frames}
                        </div>
                        <div style="font-size:0.72rem;color:#6b7280;margin-top:2px;">
                            frames with face detected ({face_pct:.0f}%)
                        </div>
                        {face_err_note}
                    </div>
                    """, unsafe_allow_html=True)

                with au_col:
                    ac = "#22C55E" if audio_detected > 0 else "#FF4B4B"
                    err_note = ""
                    if audio_detected == 0 and 'audio_extract_error' in dir() and audio_extract_error:
                        err_note = f'<div style="font-size:0.65rem;color:#FF4B4B;margin-top:3px;word-break:break-all;">{audio_extract_error}</div>'
                    elif audio_detected == 0 and not is_audio:
                        err_note = '<div style="font-size:0.65rem;color:#6b7280;margin-top:3px;">no audio track found</div>'
                    st.markdown(f"""
                    <div class="score-card" style="border-color:{ac}33;padding:10px 14px;">
                        <div style="font-family:Space Mono,monospace;font-size:0.6rem;
                             letter-spacing:2px;color:#4b5563;">AUDIO MODEL</div>
                        <div style="font-family:Space Mono,monospace;font-size:1.3rem;
                             font-weight:700;color:{ac};margin-top:4px;">
                            {'✅' if audio_detected > 0 else '❌'} {audio_detected} chunks
                        </div>
                        <div style="font-size:0.72rem;color:#6b7280;margin-top:2px;">
                            audio chunks analysed
                        </div>
                        {err_note}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                if scores:
                    render_scores(scores, interim=False)
                else:
                    st.markdown(
                        '<div class="no-signal">NOT ENOUGH DATA TO SCORE<br>'
                        'Try a longer video or audio file.</div>',
                        unsafe_allow_html=True)

            # ══════════════════════════════════════════════════════════════════
            # EMOTION-AWARE CHATBOT (Ollama + Gemma 1B)
            # ══════════════════════════════════════════════════════════════════
            if scores:
                st.markdown('<div style="margin: 30px 0 20px 0; border-top: 2px solid #1e1e30;"></div>', 
                            unsafe_allow_html=True)
                st.markdown("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h3 style="font-family: 'Space Mono', monospace; font-size: 1.3rem; 
                         background: linear-gradient(135deg, #a78bfa, #38bdf8);
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                         margin: 0;">
                        🤖 Emotion-Aware AI Assistant
                    </h3>
                    <p style="color: #6b7280; font-size: 0.85rem; margin-top: 8px;">
                        Powered by Ollama Gemma 1B · Emotionally intelligent responses
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Initialize chat history in session state with file-specific key
                file_key = f"chat_history_{uploaded.name}" if uploaded else "chat_history_default"
                if file_key not in st.session_state:
                    st.session_state[file_key] = []
                
                # Get dominant emotion context
                dominant_emotion = scores["dominant_emotion"]
                dominant_pct = scores["dominant_pct"]
                engagement = scores["engagement"]
                
                # System prompt with strict emotion-aware engineering
                system_prompt = f"""You are an Emotion Support Assistant. Your ONLY purpose is helping users understand and process the emotional content detected in their media file.

YOUR IDENTITY:
- You are NOT a general assistant, coding helper, math tutor, or information tool
- You ONLY discuss emotions, mental wellness, emotional patterns, and the analysis results below
- You have NO ability to help with coding, math, general knowledge, writing, or any unrelated topic

ANALYSIS CONTEXT:
- File: {uploaded.name if uploaded else 'Unknown'}
- Dominant Emotion: {dominant_emotion} ({dominant_pct:.1f}% of content)
- Engagement Level: {engagement:.0f}/100

ALLOWED TOPICS — respond ONLY to these:
1. The emotion analysis results shown above
2. What the detected emotions might mean or reflect about the person
3. Emotional wellness tips, coping strategies, and self-reflection prompts
4. Empathetic follow-up questions about how the person is feeling
5. Encouragement and supportive responses related to the detected emotions

STRICT OFF-TOPIC REFUSAL RULE:
If the user asks about ANYTHING outside the allowed topics — coding, programming, math,
science, trivia, writing, recipes, news, history, or anything unrelated to emotions —
respond with ONLY this exact message:
"I'm here to support you emotionally and help you understand your analysis results. I'm not able to help with that, but I'd love to explore what the {dominant_emotion} emotion in your session might mean for you. Would you like to talk about that? 💙"
Do NOT answer off-topic questions even partially. No exceptions.

EMOTIONAL RESPONSE TONE:
- Negative emotion (Sad, Angry, Fear, Disgust): warm, validating, solution-focused
- Positive emotion (Happy, Surprise): celebratory and encouraging
- Neutral: calm, curious, gently exploratory
- Keep responses to 2-4 sentences unless the user wants more depth
- Never fabricate details — only reference what the analysis above provides
- Always end with a gentle open-ended question"""

                # Chat container
                chat_container = st.container()
                with chat_container:
                    # Display chat history
                    for msg in st.session_state[file_key]:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                
                # Chat input
                if prompt := st.chat_input("Ask about the emotional analysis...", key=f"chat_input_{file_key}"):
                    # Add user message to history
                    st.session_state[file_key].append({"role": "user", "content": prompt})

                    # Display user message
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(prompt)

                    # Call Ollama API
                    with chat_container:
                        with st.chat_message("assistant"):
                            message_placeholder = st.empty()
                            full_response = ""

                            # ── Python-level hard block — never reaches Ollama ──
                            if _is_off_topic(prompt):
                                full_response = (
                                    f"I'm here to support you emotionally and help you understand "
                                    f"your analysis results. I'm not able to help with that — but "
                                    f"I'd love to explore what the **{dominant_emotion}** emotion "
                                    f"in your session might mean for you. "
                                    f"Would you like to talk about that? 💙"
                                )
                                message_placeholder.markdown(full_response)
                            else:
                                try:
                                    import requests, json
                                    messages = [{"role": "system", "content": system_prompt}]
                                    for msg in st.session_state[file_key][-10:]:
                                        messages.append({"role": msg["role"], "content": msg["content"]})
                                    response = requests.post(
                                        "http://localhost:11434/api/chat",
                                        json={"model": "gemma3:4b", "messages": messages, "stream": True},
                                        stream=True, timeout=30
                                    )
                                    if response.status_code == 200:
                                        for line in response.iter_lines():
                                            if line:
                                                chunk = json.loads(line)
                                                if "message" in chunk:
                                                    content = chunk["message"].get("content", "")
                                                    full_response += content
                                                    message_placeholder.markdown(full_response + "▌")
                                        message_placeholder.markdown(full_response)
                                    else:
                                        full_response = f"⚠️ Ollama API error: {response.status_code}. Make sure Ollama is running with `ollama serve` and gemma3:4b is installed (`ollama pull gemma3:4b`)."
                                        message_placeholder.markdown(full_response)
                                except requests.exceptions.ConnectionError:
                                    full_response = "⚠️ Cannot connect to Ollama. Please start Ollama with `ollama serve`."
                                    message_placeholder.markdown(full_response)
                                except Exception as e:
                                    full_response = f"⚠️ Error: {str(e)}"
                                    message_placeholder.markdown(full_response)

                            # Add assistant response to history
                            st.session_state[file_key].append({"role": "assistant", "content": full_response})

                # Clear chat button
                if st.session_state[file_key]:
                    if st.button("🗑️ Clear Chat", key=f"clear_chat_{file_key}"):
                        st.session_state[file_key] = []
                        st.rerun()

    # Display results from session state (persists across reruns for chat)
    elif st.session_state.tab3_analysis_results is not None and uploaded is not None:
        results = st.session_state.tab3_analysis_results
        scores = results["scores"]
        
        with result_placeholder.container():
            res_col, score_col = st.columns([2, 3], gap="large")

            with res_col:
                st.markdown("#### Preview")
                
                if results["is_image"] and results["annotated_frames"]:
                    st.image(results["annotated_frames"][0], use_container_width=True,
                             caption="Analysed image with emotion + head pose overlay")
                
                elif results["is_video"] and results["out_video_path"] and os.path.exists(results["out_video_path"]):
                    st.markdown('<div class="section-label">🎬 ANNOTATED VIDEO</div>',
                                unsafe_allow_html=True)
                    with open(results["out_video_path"], "rb") as vf:
                        video_bytes = vf.read()
                    st.video(video_bytes)
                
                elif results["is_audio"]:
                    st.markdown(f'<div class="no-signal">🎙 AUDIO FILE<br>{len(results["audio_results"])} × 3s CHUNKS ANALYSED</div>',
                                unsafe_allow_html=True)

            with score_col:
                st.markdown("#### Fusion Report")
                
                # Model source summary
                face_detected = sum(1 for e in results["face_emotions"] if e is not None)
                audio_detected = len(results["audio_results"])
                total_frames = len(results["face_emotions"])
                
                fa_col, au_col = st.columns(2)
                with fa_col:
                    face_pct = face_detected / total_frames * 100 if total_frames else 0
                    fc = "#22C55E" if face_detected > 0 else "#FF4B4B"
                    st.markdown(f"""
                    <div class="score-card" style="border-color:{fc}33;padding:10px 14px;">
                        <div style="font-family:Space Mono,monospace;font-size:0.6rem;
                             letter-spacing:2px;color:#4b5563;">FACE MODEL</div>
                        <div style="font-family:Space Mono,monospace;font-size:1.3rem;
                             font-weight:700;color:{fc};margin-top:4px;">
                            {'✅' if face_detected > 0 else '❌'} {face_detected}/{total_frames}
                        </div>
                        <div style="font-size:0.72rem;color:#6b7280;margin-top:2px;">
                            frames with face detected ({face_pct:.0f}%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with au_col:
                    ac = "#22C55E" if audio_detected > 0 else "#FF4B4B"
                    st.markdown(f"""
                    <div class="score-card" style="border-color:{ac}33;padding:10px 14px;">
                        <div style="font-family:Space Mono,monospace;font-size:0.6rem;
                             letter-spacing:2px;color:#4b5563;">AUDIO MODEL</div>
                        <div style="font-family:Space Mono,monospace;font-size:1.3rem;
                             font-weight:700;color:{ac};margin-top:4px;">
                            {'✅' if audio_detected > 0 else '❌'} {audio_detected} chunks
                        </div>
                        <div style="font-size:0.72rem;color:#6b7280;margin-top:2px;">
                            audio chunks analysed
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                if scores:
                    render_scores(scores, interim=False)
                else:
                    st.markdown('<div class="no-signal">NOT ENOUGH DATA TO SCORE</div>',
                                unsafe_allow_html=True)

            # CHATBOT SECTION (persistent from session state)
            if scores:
                st.markdown('<div style="margin: 30px 0 20px 0; border-top: 2px solid #1e1e30;"></div>', 
                            unsafe_allow_html=True)
                st.markdown("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h3 style="font-family: 'Space Mono', monospace; font-size: 1.3rem; 
                         background: linear-gradient(135deg, #a78bfa, #38bdf8);
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                         margin: 0;">
                        🤖 Emotion-Aware AI Assistant
                    </h3>
                    <p style="color: #6b7280; font-size: 0.85rem; margin-top: 8px;">
                        Powered by Ollama Gemma 1B · Emotionally intelligent responses
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Initialize chat history
                file_key = f"chat_history_{results['uploaded_name']}"
                if file_key not in st.session_state:
                    st.session_state[file_key] = []
                
                # Get emotion context
                dominant_emotion = scores["dominant_emotion"]
                dominant_pct = scores["dominant_pct"]
                engagement = scores["engagement"]
                
                # System prompt
                system_prompt = f"""You are an Emotion Support Assistant. Your ONLY purpose is helping users understand and process the emotional content detected in their media file.

YOUR IDENTITY:
- You are NOT a general assistant, coding helper, math tutor, or information tool
- You ONLY discuss emotions, mental wellness, emotional patterns, and the analysis results below
- You have NO ability to help with coding, math, general knowledge, writing, or any unrelated topic

ANALYSIS CONTEXT:
- File: {results['uploaded_name']}
- Dominant Emotion: {dominant_emotion} ({dominant_pct:.1f}% of content)
- Engagement Level: {engagement:.0f}/100

ALLOWED TOPICS — respond ONLY to these:
1. The emotion analysis results shown above
2. What the detected emotions might mean or reflect about the person
3. Emotional wellness tips, coping strategies, and self-reflection prompts
4. Empathetic follow-up questions about how the person is feeling
5. Encouragement and supportive responses related to the detected emotions

STRICT OFF-TOPIC REFUSAL RULE:
If the user asks about ANYTHING outside the allowed topics — coding, programming, math,
science, trivia, writing, recipes, news, history, or anything unrelated to emotions —
respond with ONLY this exact message:
"I'm here to support you emotionally and help you understand your analysis results. I'm not able to help with that, but I'd love to explore what the {dominant_emotion} emotion in your session might mean for you. Would you like to talk about that? 💙"
Do NOT answer off-topic questions even partially. No exceptions.

EMOTIONAL RESPONSE TONE:
- Negative emotion (Sad, Angry, Fear, Disgust): warm, validating, solution-focused
- Positive emotion (Happy, Surprise): celebratory and encouraging
- Neutral: calm, curious, gently exploratory
- Keep responses to 2-4 sentences unless the user wants more depth
- Never fabricate details — only reference what the analysis above provides
- Always end with a gentle open-ended question"""

                # Display chat history
                for msg in st.session_state[file_key]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                
                # Chat input
                if prompt := st.chat_input("Ask about the emotional analysis...", key=f"chat_input_{file_key}"):
                    # Add user message
                    st.session_state[file_key].append({"role": "user", "content": prompt})

                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Call Ollama
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response = ""

                        # ── Python-level hard block — never reaches Ollama ──
                        if _is_off_topic(prompt):
                            full_response = (
                                f"I'm here to support you emotionally and help you understand "
                                f"your analysis results. I'm not able to help with that — but "
                                f"I'd love to explore what the **{dominant_emotion}** emotion "
                                f"in your session might mean for you. "
                                f"Would you like to talk about that? 💙"
                            )
                            message_placeholder.markdown(full_response)
                        else:
                            try:
                                import requests, json
                                messages = [{"role": "system", "content": system_prompt}]
                                for msg in st.session_state[file_key][-10:]:
                                    messages.append({"role": msg["role"], "content": msg["content"]})
                                response = requests.post(
                                    "http://localhost:11434/api/chat",
                                    json={"model": "gemma3:4b", "messages": messages, "stream": True},
                                    stream=True, timeout=30
                                )
                                if response.status_code == 200:
                                    for line in response.iter_lines():
                                        if line:
                                            chunk = json.loads(line)
                                            if "message" in chunk:
                                                content = chunk["message"].get("content", "")
                                                full_response += content
                                                message_placeholder.markdown(full_response + "▌")
                                    message_placeholder.markdown(full_response)
                                else:
                                    full_response = f"⚠️ Ollama API error: {response.status_code}. Make sure Ollama is running with `ollama serve` and gemma3:4b is installed (`ollama pull gemma3:4b`)."
                                    message_placeholder.markdown(full_response)
                            except requests.exceptions.ConnectionError:
                                full_response = "⚠️ Cannot connect to Ollama. Please start Ollama with `ollama serve`."
                                message_placeholder.markdown(full_response)
                            except Exception as e:
                                full_response = f"⚠️ Error: {str(e)}"
                                message_placeholder.markdown(full_response)

                        # Add to history
                        st.session_state[file_key].append({"role": "assistant", "content": full_response})

                # Clear chat button
                if st.session_state[file_key]:
                    if st.button("🗑️ Clear Chat", key=f"clear_chat_{file_key}"):
                        st.session_state[file_key] = []
                        st.rerun()

    elif uploaded is None:
        with result_placeholder.container():
            st.markdown("""
            <div class="no-signal" style="padding:40px;">
                📁 UPLOAD A FILE TO BEGIN ANALYSIS<br><br>
                <span style="font-size:0.75rem;color:#374151;">
                Video → frame emotions + audio track + head pose<br>
                Audio → voice emotion across 3s chunks<br>
                Image → single frame emotion + head pose
                </span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Audio Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab4:

    import subprocess as _sp

    st.markdown("#### Audio Emotion Analysis")
    st.markdown(
        '<p style="color:#6b7280;font-size:0.85rem;">'
        'Upload an audio file or record directly from your microphone. '
        'The model analyses every 3-second chunk and shows per-chunk emotion, '
        'confidence, a waveform preview, and a full timeline.</p>',
        unsafe_allow_html=True,
    )

    if audio_model is None:
        st.error("⚠️ Audio model not loaded — cannot run audio analysis.")
    else:

        # ── Input method ─────────────────────────────────────────────────
        input_method = st.radio(
            "Input source",
            ["📂 Upload file", "🎙 Record from mic"],
            horizontal=True,
            label_visibility="collapsed",
        )

        audio_bytes  = None
        audio_source = None   # filename or "microphone"

        if input_method == "📂 Upload file":
            aud_upload = st.file_uploader(
                "Drop an audio file",
                type=["wav", "mp3", "flac", "ogg", "m4a"],
                label_visibility="collapsed",
                key="tab4_uploader",
            )
            if aud_upload:
                audio_bytes  = aud_upload.read()
                audio_source = aud_upload.name

        else:
            st.markdown(
                '<p style="color:#6b7280;font-size:0.82rem;margin-bottom:6px;">'
                'Record audio below, then click <b>Analyse</b>.</p>',
                unsafe_allow_html=True,
            )
            recorded = st.audio_input("Record audio", key="tab4_recorder")
            if recorded:
                audio_bytes  = recorded.read()
                audio_source = "microphone"

        chunk_secs = st.select_slider(
            "Chunk size (seconds per analysis window)",
            options=[1, 2, 3, 5],
            value=3,
            help="Smaller chunks = more data points but lower accuracy per chunk.",
            key="tab4_chunk",
        )

        analyse_audio = st.button(
            "🔍 Analyse Audio",
            type="primary",
            disabled=(audio_bytes is None),
            key="tab4_analyse",
        )

        audio_result_ph = st.empty()

        if analyse_audio and audio_bytes:

            # Guard: re-check at analysis time since model loads at startup
            if audio_model is None:
                st.error("⚠️ Audio model is not loaded. Check that the model file exists at the path defined in paths.py.")
            else:
                # ── Save to temp file ─────────────────────────────────────
                ext = audio_source.rsplit(".", 1)[-1].lower() if "." in (audio_source or "") else "wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as atmp:
                    atmp.write(audio_bytes)
                    atmp_path = atmp.name

                try:
                    with audio_result_ph.container():
                        aud_prog   = st.progress(0)
                        aud_status = st.empty()

                    # ── Load & resample ───────────────────────────────────
                    aud_status.markdown(
                        '<div class="section-label">LOADING AUDIO …</div>',
                        unsafe_allow_html=True)
                    try:
                        pcm, _ = librosa.load(atmp_path, sr=SAMPLE_RATE, mono=True)
                    except Exception:
                        wav_tmp = atmp_path + "_converted.wav"
                        _sp.run(
                            ["ffmpeg", "-y", "-i", atmp_path,
                             "-ar", str(SAMPLE_RATE), "-ac", "1", wav_tmp],
                            capture_output=True, timeout=60,
                        )
                        pcm, _ = librosa.load(wav_tmp, sr=SAMPLE_RATE, mono=True)
                        os.remove(wav_tmp)

                    total_secs = len(pcm) / SAMPLE_RATE
                    import math as _math
                    n_chunks   = max(1, _math.ceil(total_secs / chunk_secs))

                    # ── Per-chunk inference ───────────────────────────────
                    aud_status.markdown(
                        f'<div class="section-label">ANALYSING {n_chunks} CHUNKS …</div>',
                        unsafe_allow_html=True)

                    chunk_results = []
                    errors        = []
                    for ci in range(n_chunks):
                        s     = int(ci * chunk_secs * SAMPLE_RATE)
                        e     = int(min(len(pcm), s + chunk_secs * SAMPLE_RATE))
                        chunk = pcm[s:e]
                        try:
                            lbl, prbs = predict_audio_emotion(chunk)
                            if lbl and prbs:
                                chunk_results.append({
                                    "start": ci * chunk_secs,
                                    "end":   min(total_secs, (ci + 1) * chunk_secs),
                                    "label": lbl,
                                    "probs": prbs,
                                    "conf":  float(max(prbs)) * 100,
                                })
                            else:
                                mfcc_frames = int(len(chunk) / HOP_LENGTH)
                                errors.append(
                                    f"Chunk {ci+1} ({ci*chunk_secs:.1f}s–"
                                    f"{min(total_secs,(ci+1)*chunk_secs):.1f}s): "
                                    f"returned None — {len(chunk)/SAMPLE_RATE:.2f}s / "
                                    f"{mfcc_frames} MFCC frames / "
                                    f"dtype={chunk.dtype} "
                                    f"min={chunk.min():.3f} max={chunk.max():.3f}"
                                )
                        except Exception as ex:
                            import traceback
                            errors.append(
                                f"Chunk {ci+1}: {type(ex).__name__}: {ex} | "
                                f"{traceback.format_exc()}"
                            )
                        aud_prog.progress(int((ci + 1) / n_chunks * 100))

                    aud_prog.empty()
                    aud_status.empty()

                finally:
                    if os.path.exists(atmp_path):
                        os.remove(atmp_path)

                # ── Render results ────────────────────────────────────────
                with audio_result_ph.container():

                    if not chunk_results:
                        err_detail = ""
                        if errors:
                            err_detail = "<br><br>" + "<br>".join(
                                f'<span style="font-size:0.68rem;color:#6b7280;'
                                f'font-family:monospace;">{err}</span>'
                                for err in errors
                            )
                        st.markdown(
                            f'<div class="no-signal" style="text-align:left;padding:24px;">'
                            f'⚠️ NO EMOTION DETECTED<br><br>'
                            f'<span style="font-size:0.75rem;color:#6b7280;">'
                            f'Duration: {total_secs:.1f}s &nbsp;·&nbsp; '
                            f'Chunks tried: {n_chunks} &nbsp;·&nbsp; '
                            f'Chunk size: {chunk_secs}s</span>'
                            f'{err_detail}'
                            f'</div>',
                            unsafe_allow_html=True)
                    else:
                        # ── Waveform + chunk overlay ──────────────────────
                        st.markdown(
                            '<div class="section-label">🌊 WAVEFORM + EMOTION TIMELINE</div>',
                            unsafe_allow_html=True)

                        pcm_down  = pcm[::max(1, len(pcm) // 800)]
                        pcm_norm  = pcm_down / (np.max(np.abs(pcm_down)) + 1e-9)
                        bar_width = 100 / len(pcm_norm)
                        waveform_bars = ""
                        for i, amp in enumerate(pcm_norm):
                            t      = i / len(pcm_norm) * total_secs
                            ci_w   = min(int(t / chunk_secs), len(chunk_results) - 1)
                            color  = EMOTION_COLORS.get(chunk_results[ci_w]["label"], "#60A5FA")
                            height = max(2, abs(float(amp)) * 48)
                            waveform_bars += (
                                f'<div style="display:inline-block;width:{bar_width:.3f}%;'
                                f'height:{height:.1f}px;background:{color};'
                                f'vertical-align:middle;margin:0;border-radius:1px;"></div>'
                            )
                        st.markdown(
                            f'<div style="background:#0d0d18;border-radius:12px;padding:12px 8px;'
                            f'line-height:0;overflow:hidden;display:flex;align-items:center;'
                            f'height:72px;">{waveform_bars}</div>',
                            unsafe_allow_html=True,
                        )

                        # ── Emotion timeline dots ─────────────────────────
                        tl_html = '<div class="timeline-wrap" style="margin-top:10px;">'
                        for cr in chunk_results:
                            c   = EMOTION_COLORS.get(cr["label"], "#60A5FA")
                            emj = EMOTION_EMOJI.get(cr["label"], "·")
                            tip = f'{cr["start"]:.0f}s–{cr["end"]:.0f}s: {cr["label"]} {cr["conf"]:.0f}%'
                            tl_html += (
                                f'<div class="tl-dot" style="background:{c};width:28px;height:28px;" '
                                f'title="{tip}">{emj}</div>'
                            )
                        tl_html += '</div>'
                        st.markdown(tl_html, unsafe_allow_html=True)

                        st.markdown("---")

                        # ── Summary + chunk cards ─────────────────────────
                        sum_col, chunks_col = st.columns([1, 2], gap="large")

                        with sum_col:
                            st.markdown("##### Summary")
                            label_counts = {}
                            for cr in chunk_results:
                                label_counts[cr["label"]] = label_counts.get(cr["label"], 0) + 1
                            dominant_lbl = max(label_counts, key=label_counts.get)
                            dominant_pct = label_counts[dominant_lbl] / len(chunk_results) * 100
                            dom_col  = EMOTION_COLORS.get(dominant_lbl, "#60A5FA")
                            dom_emj  = EMOTION_EMOJI.get(dominant_lbl, "🙂")
                            avg_conf = sum(cr["conf"] for cr in chunk_results) / len(chunk_results)

                            st.markdown(f"""
                            <div class="score-card" style="border-color:{dom_col}33;
                                 background:linear-gradient(135deg,#13131f,#0a0a1a);text-align:center;">
                                <div style="font-family:Space Mono,monospace;font-size:0.6rem;
                                     letter-spacing:2px;color:#4b5563;margin-bottom:6px;">
                                    DOMINANT EMOTION
                                </div>
                                <div style="font-size:2.8rem;line-height:1;">{dom_emj}</div>
                                <div style="font-family:Space Mono,monospace;font-size:1.4rem;
                                     font-weight:700;color:{dom_col};margin-top:4px;">
                                    {dominant_lbl}
                                </div>
                                <div style="font-size:0.78rem;color:#6b7280;margin-top:4px;">
                                    {dominant_pct:.0f}% of chunks · avg conf {avg_conf:.0f}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown(
                                '<div class="section-label" style="margin-top:14px;">'
                                'EMOTION DISTRIBUTION</div>',
                                unsafe_allow_html=True)
                            sorted_dist = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
                            dist_html = ""
                            for lbl, cnt in sorted_dist:
                                pct_d = cnt / len(chunk_results) * 100
                                c     = EMOTION_COLORS.get(lbl, "#60A5FA")
                                emj   = EMOTION_EMOJI.get(lbl, "")
                                dist_html += f"""
                                <div style="margin-bottom:9px;">
                                    <div style="display:flex;justify-content:space-between;
                                         font-size:0.82rem;margin-bottom:3px;">
                                        <span>{emj} {lbl}</span>
                                        <span style="font-family:Space Mono,monospace;color:{c};">
                                            {pct_d:.0f}%
                                        </span>
                                    </div>
                                    <div class="bar-bg">
                                        <div class="bar-fill" style="width:{pct_d:.0f}%;background:{c}">
                                        </div>
                                    </div>
                                </div>"""
                            st.markdown(dist_html, unsafe_allow_html=True)

                            st.markdown(f"""
                            <div class="score-card" style="margin-top:8px;">
                                <div class="section-label">📄 FILE INFO</div>
                                <div style="font-family:Space Mono,monospace;font-size:0.75rem;
                                     color:#a0aec0;line-height:1.8;">
                                    Source: {audio_source}<br>
                                    Duration: {total_secs:.1f}s<br>
                                    Chunks: {len(chunk_results)} × {chunk_secs}s<br>
                                    Sample rate: {SAMPLE_RATE} Hz
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        with chunks_col:
                            st.markdown("##### Per-chunk Results")
                            for ci, cr in enumerate(chunk_results):
                                c    = EMOTION_COLORS.get(cr["label"], "#60A5FA")
                                emj  = EMOTION_EMOJI.get(cr["label"], "🙂")
                                prob_bars = ""
                                for emo, prob in sorted(zip(EMOTIONS, cr["probs"]),
                                                        key=lambda x: x[1], reverse=True):
                                    ec  = EMOTION_COLORS.get(emo, "#60A5FA")
                                    pct = prob * 100
                                    prob_bars += f"""
                                    <div style="display:flex;align-items:center;gap:8px;
                                         margin-bottom:3px;font-size:0.7rem;">
                                        <span style="width:58px;color:#9ca3af;">{emo}</span>
                                        <div style="flex:1;background:#1e1e30;border-radius:99px;
                                             height:4px;overflow:hidden;">
                                            <div style="width:{pct:.0f}%;height:4px;
                                                 background:{ec};border-radius:99px;"></div>
                                        </div>
                                        <span style="width:32px;text-align:right;
                                             font-family:Space Mono,monospace;color:{ec};">
                                            {pct:.0f}%
                                        </span>
                                    </div>"""
                                card_top = f"""
                                <div class="result-card audio-card" style="
                                     align-items:flex-start;padding:14px 18px;margin-bottom:8px;">
                                    <div style="min-width:36px;text-align:center;">
                                        <div style="font-size:1.8rem;line-height:1;">{emj}</div>
                                        <div style="font-family:Space Mono,monospace;font-size:0.55rem;
                                             color:#4b5563;margin-top:3px;">#{ci+1}</div>
                                    </div>
                                    <div style="flex:1;">
                                        <div style="display:flex;justify-content:space-between;
                                             align-items:baseline;margin-bottom:6px;">
                                            <span class="emotion-label" style="color:{c};">
                                                {cr["label"]}
                                            </span>
                                            <span style="font-family:Space Mono,monospace;
                                                 font-size:0.7rem;color:#6b7280;">
                                                {cr["start"]:.1f}s \u2013 {cr["end"]:.1f}s
                                                &nbsp;&middot;&nbsp; {cr["conf"]:.0f}% conf
                                            </span>
                                        </div>"""
                                card_bottom = "</div></div>"
                                st.markdown(
                                    card_top + prob_bars + card_bottom,
                                    unsafe_allow_html=True)

        elif audio_bytes is None:
            with audio_result_ph.container():
                st.markdown(
                    '<div class="no-signal" style="padding:36px;">'
                    '🎙 UPLOAD OR RECORD AUDIO TO BEGIN</div>',
                    unsafe_allow_html=True,
                )