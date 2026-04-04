import math
import numpy as np
from paths import (
    EMOTIONS, ENGAGEMENT_WEIGHT,
    FACE_WEIGHT, AUDIO_WEIGHT,
    YAW_ENGAGED, YAW_AWAY, PITCH_UP_MAX, PITCH_DOWN_MAX, ROLL_MAX,
)

# ===============================
# FUSION ENGINE
# ===============================

def fuse_predictions(face_probs, audio_probs):
    """
    Weighted average fusion of face and audio probability vectors.
    face_probs / audio_probs: list of 7 floats (one per emotion class)
    Returns (emotion_label, confidence_pct, full_probs_list)
    """
    fp = np.array(face_probs, dtype=np.float32)
    if audio_probs is not None:
        ap = np.array(audio_probs, dtype=np.float32)
        fused = FACE_WEIGHT * fp + AUDIO_WEIGHT * ap
    else:
        fused = fp
    fused /= fused.sum()   # renormalise
    idx = int(np.argmax(fused))
    return EMOTIONS[idx], float(fused[idx]) * 100, fused.tolist()


# ===============================
# HEAD POSE HELPERS
# ===============================

def rotation_matrix_to_euler(R):
    """
    Decompose a 3x3 rotation matrix into Euler angles (degrees).
    Returns (pitch, yaw, roll).
      pitch = up/down tilt  (positive = looking up)
      yaw   = left/right    (positive = turning right)
      roll  = head tilt     (positive = tilting right)
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2( R[2, 1], R[2, 2])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = math.atan2( R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0
    return (math.degrees(pitch),
            math.degrees(yaw),
            math.degrees(roll))


def compute_head_engagement_tick(yaw, pitch, roll, face_present):
    """
    Score 0–100 for a single frame based on head pose angles.
    Deductions are applied for each axis of inattention.
    Returns (score, status_label, details_dict)
    """
    if not face_present:
        return 0.0, "No Face", {"yaw": None, "pitch": None, "roll": None}

    score = 100.0

    # Yaw penalty (most important — are they facing forward?)
    abs_yaw = abs(yaw)
    if abs_yaw <= YAW_ENGAGED:
        yaw_penalty = 0
    elif abs_yaw <= YAW_AWAY:
        yaw_penalty = (abs_yaw - YAW_ENGAGED) / (YAW_AWAY - YAW_ENGAGED) * 40
    else:
        yaw_penalty = 40 + min(40, (abs_yaw - YAW_AWAY) * 2)
    score -= yaw_penalty

    # Pitch penalty (looking too far up/down)
    if pitch > PITCH_UP_MAX:
        score -= min(30, (pitch - PITCH_UP_MAX) * 1.5)
    elif pitch < -PITCH_DOWN_MAX:
        score -= min(30, (-pitch - PITCH_DOWN_MAX) * 1.5)

    # Roll penalty (head tilt)
    abs_roll = abs(roll)
    if abs_roll > ROLL_MAX:
        score -= min(20, (abs_roll - ROLL_MAX) * 1.0)

    score = max(0.0, min(100.0, score))

    if score >= 75:
        status = "Engaged"
    elif score >= 45:
        status = "Partial"
    else:
        status = "Distracted"

    return score, status, {"yaw": yaw, "pitch": pitch, "roll": roll}


def compute_head_engagement_session(head_records):
    """
    head_records: list of dicts with keys:
        'timestamp', 'score', 'status', 'yaw', 'pitch', 'roll', 'face_present'
    Returns aggregated head engagement metrics dict.
    """
    if not head_records:
        return None

    n = len(head_records)
    scores       = [r['score'] for r in head_records]
    face_present = [r['face_present'] for r in head_records]

    mean_score   = float(np.mean(scores))
    presence_pct = sum(face_present) / n * 100

    yaws    = [r['yaw']   for r in head_records if r['yaw']   is not None]
    pitches = [r['pitch'] for r in head_records if r['pitch'] is not None]
    rolls   = [r['roll']  for r in head_records if r['roll']  is not None]

    mean_yaw   = float(np.mean(np.abs(yaws)))  if yaws   else 0
    mean_pitch = float(np.mean(pitches))       if pitches else 0
    mean_roll  = float(np.mean(np.abs(rolls))) if rolls  else 0

    status_counts = {}
    for r in head_records:
        s = r['status']
        status_counts[s] = status_counts.get(s, 0) + 1
    status_pcts = {k: v / n * 100 for k, v in status_counts.items()}

    consistency = max(0, 100 - float(np.std(scores)) * 2)

    head_timeline = []
    if head_records:
        t0 = head_records[0]['timestamp']
        tN = head_records[-1]['timestamp']
        duration = max(1, int(tN - t0) + 1)
        for sec in range(duration):
            sec_recs = [r for r in head_records
                        if int(r['timestamp'] - t0) == sec]
            if sec_recs:
                sec_score = np.mean([r['score'] for r in sec_recs])
                if sec_score >= 75:   head_timeline.append("Engaged")
                elif sec_score >= 45: head_timeline.append("Partial")
                else:                 head_timeline.append("Distracted")
            else:
                head_timeline.append(head_timeline[-1] if head_timeline else "Partial")

    return {
        "mean_score":    mean_score,
        "presence_pct":  presence_pct,
        "consistency":   consistency,
        "mean_yaw":      mean_yaw,
        "mean_pitch":    mean_pitch,
        "mean_roll":     mean_roll,
        "status_pcts":   status_pcts,
        "head_timeline": head_timeline,
        "total_ticks":   n,
    }


# ===============================
# SCORING ENGINE
# ===============================

def compute_session_scores(records):
    """
    records: list of dicts with keys:
        'timestamp', 'fused_probs' (list of 7 floats),
        'face_emotion', 'audio_emotion', 'fused_emotion'
    Returns a rich score dict.
    """
    if not records:
        return None

    n = len(records)
    all_probs  = np.array([r['fused_probs'] for r in records])  # (n, 7)
    mean_probs = all_probs.mean(axis=0)                          # (7,)

    dominant_idx     = int(np.argmax(mean_probs))
    dominant_emotion = EMOTIONS[dominant_idx]
    dominant_pct     = float(mean_probs[dominant_idx]) * 100

    tick_labels = [EMOTIONS[int(np.argmax(p))] for p in all_probs]

    head_scores = [r["head_score"] for r in records if "head_score" in r]
    if head_scores:
        engagement = float(np.mean(head_scores))
    else:
        raw_engagement = sum(ENGAGEMENT_WEIGHT[e] * mean_probs[i]
                             for i, e in enumerate(EMOTIONS))
        engagement = max(0, min(100, raw_engagement * 100))

    head_records_for_session = [
        {"timestamp": r["timestamp"], "score": r.get("head_score", 0),
         "status": r.get("head_status", "No Face"),
         "yaw": r.get("head_yaw"), "pitch": r.get("head_pitch"),
         "roll": r.get("head_roll"), "face_present": r.get("head_present", False)}
        for r in records if "head_score" in r
    ]
    head_metrics = compute_head_engagement_session(head_records_for_session)

    timeline = []
    duration = int(records[-1]['timestamp'] - records[0]['timestamp']) + 1
    for sec in range(duration):
        sec_records = [r for r in records
                       if int(r['timestamp'] - records[0]['timestamp']) == sec]
        if sec_records:
            sec_probs = np.array([r['fused_probs'] for r in sec_records]).mean(axis=0)
            timeline.append(EMOTIONS[int(np.argmax(sec_probs))])
        else:
            timeline.append(timeline[-1] if timeline else "Neutral")

    agreements = sum(1 for r in records
                     if r['face_emotion'] and r['audio_emotion']
                     and r['face_emotion'] == r['audio_emotion'])
    total_paired = sum(1 for r in records
                       if r['face_emotion'] and r['audio_emotion'])
    agreement_rate = (agreements / total_paired * 100) if total_paired > 0 else 0

    return {
        "dominant_emotion": dominant_emotion,
        "dominant_pct":     dominant_pct,
        "engagement":       engagement,
        "timeline":         timeline,
        "agreement_rate":   agreement_rate,
        "total_ticks":      n,
        "tick_labels":      tick_labels,
        "head_metrics":     head_metrics,
    }


# ===============================
# RECORD BUILDER (helper used by both Tab 2 and Tab 3)
# ===============================

def build_records_from_lists(face_emotions, face_probs_list,
                              audio_results, head_ticks, timestamps):
    """
    Zips per-frame lists into the record dicts expected by compute_session_scores.
    audio_results: list of (label, probs) tuples aligned to 3-second chunks, or None.

    Audio matching uses the frame timestamp to select the correct 3-second chunk.
    Frames that fall outside the range of available audio chunks get audio_emotion=None,
    so they are correctly excluded from the FACE–VOICE AGREEMENT calculation.
    """
    # Must match the chunk size used when audio_results were produced.
    # AudioNet expects 215 MFCC frames → 215*512/16000 ≈ 6.88 s per chunk.
    AUDIO_CHUNK_SECS = (215 * 512) / 16000   # ≈ 6.88 s
    records = []
    n = len(timestamps)
    for i in range(n):
        fe = face_emotions[i]
        fp = face_probs_list[i]
        ht = head_ticks[i] if head_ticks and i < len(head_ticks) else {}

        # Map this frame's timestamp to its audio chunk index
        ae, ap = None, None
        if audio_results:
            chunk_idx = int(timestamps[i] / AUDIO_CHUNK_SECS)
            if 0 <= chunk_idx < len(audio_results):
                ae, ap = audio_results[chunk_idx]
            # else: frame timestamp exceeds available audio — leave ae/ap as None

        if fp is None:
            fp = [1 / 7] * 7
        _, _, fused_probs = fuse_predictions(fp, ap)

        rec = {
            "timestamp":     timestamps[i],
            "face_emotion":  fe,
            "audio_emotion": ae,
            "fused_probs":   fused_probs,
            "fused_emotion": EMOTIONS[int(np.argmax(fused_probs))],
        }
        if ht.get("face_present"):
            rec.update({
                "head_score":   ht.get("score", 0),
                "head_status":  ht.get("status", "No Face"),
                "head_yaw":     ht.get("yaw"),
                "head_pitch":   ht.get("pitch"),
                "head_roll":    ht.get("roll"),
                "head_present": True,
            })
        records.append(rec)
    return records