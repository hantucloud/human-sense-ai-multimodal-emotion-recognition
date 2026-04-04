"""
ui_components.py — Reusable Streamlit UI components and CSS styles.

Exports:
    apply_styles()     — inject global CSS into the Streamlit page
    render_scores()    — render a full session score report (gauges, timelines,
                         head engagement breakdown)
"""

import streamlit as st
from paths import EMOTION_COLORS, EMOTION_EMOJI


# ===============================
# GLOBAL CSS
# ===============================

def apply_styles():
    """Inject the app-wide CSS into the Streamlit page."""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}
.main { background-color: #0a0a0f; }
h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem !important;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0 !important;
}
.subtitle { font-size: 0.95rem; color: #6b7280; margin-top: 2px; margin-bottom: 18px; }

/* --- result cards --- */
.result-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.audio-card  { border-color: #2a1f3d !important; background: #0f0f1c !important; }
.fused-card  { border-color: #1a2f1a !important; background: #0a1a0a !important; }
.score-card  {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.emotion-emoji  { font-size: 2rem; line-height: 1; }
.emotion-label  { font-family: 'Space Mono', monospace; font-size: 1.15rem; font-weight: 700; }
.emotion-conf   { font-size: 0.78rem; color: #6b7280; margin-top: 2px; }
.bar-bg  { background: #1e1e30; border-radius: 999px; height: 5px; margin-top: 7px; overflow: hidden; width: 100%; }
.bar-fill { height: 5px; border-radius: 999px; }
.no-signal {
    background: #13131f;
    border: 1px dashed #2a2a3d;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    color: #4b5563;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

/* --- section dividers --- */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem; letter-spacing: 2px;
    text-transform: uppercase; color: #4b5563; margin-bottom: 8px;
}
.divider-row {
    display: flex; align-items: center; gap: 10px;
    margin: 16px 0 10px;
    color: #374151; font-size: 0.68rem;
    font-family: 'Space Mono', monospace; letter-spacing: 1px;
}
.divider-row::before, .divider-row::after { content:''; flex:1; height:1px; background:#1e1e30; }

/* --- score gauge --- */
.gauge-wrap { text-align: center; padding: 4px 0; }
.gauge-val  {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem; font-weight: 700; line-height: 1;
}
.gauge-label { font-size: 0.72rem; color: #6b7280; letter-spacing: 1px; margin-top: 4px; }
.gauge-bar-bg  { background: #1e1e30; border-radius: 999px; height: 8px; margin-top: 10px; overflow: hidden; }
.gauge-bar-fill { height: 8px; border-radius: 999px; }

/* --- timeline --- */
.timeline-wrap { display: flex; gap: 3px; flex-wrap: wrap; margin-top: 8px; }
.tl-dot {
    width: 22px; height: 22px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.6rem; font-family: 'Space Mono', monospace;
    color: white; font-weight: 700; flex-shrink: 0;
}

/* --- session controls --- */
.session-btn {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 1px;
}

/* ── Pin chat input to bottom of viewport ────────────────────────────── */
div[data-testid="stChatInput"] {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 999;
    background-color: #0a0a0f;
    padding: 12px 24px 16px 24px;
    border-top: 1px solid #1e1e30;
    box-shadow: 0 -8px 24px rgba(0, 0, 0, 0.4);
}

/* Prevent chat messages from hiding behind the fixed input bar */
section[data-testid="stMain"] > div:first-child {
    padding-bottom: 100px !important;
}

/* Style the chat input field itself to match the dark theme */
div[data-testid="stChatInput"] textarea {
    background-color: #13131f !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.15) !important;
}

/* Send button inside chat input */
div[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #a78bfa, #38bdf8) !important;
    border: none !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ===============================
# SCORE REPORT RENDERER
# ===============================

# Status colours used across head-engagement displays
HEAD_STATUS_COLORS = {
    "Engaged":    "#22C55E",
    "Partial":    "#F97316",
    "Distracted": "#FF4B4B",
    "No Face":    "#6B7280",
}


def render_scores(scores, interim=False):
    """
    Render a complete session score report into the current Streamlit context.

    Parameters
    ----------
    scores : dict | None
        Output of ``compute_session_scores()``.  If None, shows a placeholder.
    interim : bool
        If True, labels the banner "INTERIM" instead of "FINAL".
    """
    if scores is None:
        st.markdown('<div class="no-signal">RUN A SESSION TO SEE SCORES</div>',
                    unsafe_allow_html=True)
        return

    label = "INTERIM" if interim else "FINAL"
    dom   = scores["dominant_emotion"]
    col   = EMOTION_COLORS.get(dom, "#60A5FA")
    emj   = EMOTION_EMOJI.get(dom, "🙂")

    # ── Dominant emotion banner ───────────────────────────────────────────
    st.markdown(f"""
    <div class="score-card" style="border-color:{col}33;background:linear-gradient(135deg,#13131f,#0a0a1a);text-align:center;">
        <div style="font-family:Space Mono,monospace;font-size:0.62rem;letter-spacing:2px;color:#4b5563;margin-bottom:6px;">
            {label} DOMINANT EMOTION
        </div>
        <div style="font-size:3rem;line-height:1;">{emj}</div>
        <div style="font-family:Space Mono,monospace;font-size:1.6rem;font-weight:700;color:{col};margin-top:4px;">
            {dom}
        </div>
        <div style="font-size:0.8rem;color:#6b7280;margin-top:2px;">
            {scores["dominant_pct"]:.1f}% of session · {scores["total_ticks"]} samples
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Engagement gauge ─────────────────────────────────────────────────
    eng  = scores["engagement"]
    ecol = "#F97316"
    st.markdown(f"""
    <div class="gauge-wrap score-card" style="border-color:{ecol}33;">
        <div class="gauge-val" style="color:{ecol}">{eng:.0f}</div>
        <div class="gauge-label">ENGAGEMENT</div>
        <div class="gauge-bar-bg">
            <div class="gauge-bar-fill" style="width:{eng:.0f}%;background:{ecol}"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Face–Audio agreement ──────────────────────────────────────────────
    st.markdown(f"""
    <div class="score-card" style="margin-top:4px;">
        <div class="section-label">🤝 FACE–VOICE AGREEMENT</div>
        <div style="font-family:Space Mono,monospace;font-size:1.4rem;font-weight:700;color:#38bdf8;">
            {scores["agreement_rate"]:.0f}%
        </div>
        <div class="bar-bg">
            <div class="bar-fill" style="width:{scores['agreement_rate']:.0f}%;background:#38bdf8"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Emotion timeline ──────────────────────────────────────────────────
    if scores["timeline"]:
        st.markdown('<div class="section-label" style="margin-top:16px;">⏱ EMOTION TIMELINE</div>',
                    unsafe_allow_html=True)
        tl_html = '<div class="timeline-wrap">'
        for sec, emo in enumerate(scores["timeline"]):
            c   = EMOTION_COLORS.get(emo, "#60A5FA")
            e   = EMOTION_EMOJI.get(emo, "·")
            tl_html += f'<div class="tl-dot" style="background:{c}" title="s{sec+1}:{emo}">{e}</div>'
        tl_html += '</div>'
        st.markdown(tl_html, unsafe_allow_html=True)

    # ── Head Engagement Section ───────────────────────────────────────────
    hm = scores.get("head_metrics")
    if hm:
        st.markdown('<div class="divider-row">🧭 HEAD ENGAGEMENT ANALYSIS</div>',
                    unsafe_allow_html=True)

        hcol_score = (
            "#22C55E" if hm["mean_score"] >= 75 else
            "#F97316" if hm["mean_score"] >= 45 else
            "#FF4B4B"
        )
        h1, h2, h3 = st.columns(3)
        with h1:
            st.markdown(f"""
            <div class="gauge-wrap score-card" style="border-color:{hcol_score}33;">
                <div class="gauge-val" style="color:{hcol_score}">{hm["mean_score"]:.0f}</div>
                <div class="gauge-label">HEAD SCORE</div>
                <div class="gauge-bar-bg">
                    <div class="gauge-bar-fill" style="width:{hm["mean_score"]:.0f}%;background:{hcol_score}"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        with h2:
            st.markdown(f"""
            <div class="gauge-wrap score-card" style="border-color:#38bdf833;">
                <div class="gauge-val" style="color:#38bdf8">{hm["presence_pct"]:.0f}</div>
                <div class="gauge-label">FACE PRESENT %</div>
                <div class="gauge-bar-bg">
                    <div class="gauge-bar-fill" style="width:{hm["presence_pct"]:.0f}%;background:#38bdf8"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        with h3:
            st.markdown(f"""
            <div class="gauge-wrap score-card" style="border-color:#a78bfa33;">
                <div class="gauge-val" style="color:#a78bfa">{hm["consistency"]:.0f}</div>
                <div class="gauge-label">CONSISTENCY</div>
                <div class="gauge-bar-bg">
                    <div class="gauge-bar-fill" style="width:{hm["consistency"]:.0f}%;background:#a78bfa"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Pose angles summary
        st.markdown(f"""
        <div class="score-card" style="margin-top:4px;">
            <div class="section-label">📐 MEAN HEAD ANGLES</div>
            <div style="display:flex;gap:24px;margin-top:8px;font-family:Space Mono,monospace;">
                <div>
                    <div style="font-size:1.2rem;font-weight:700;color:#F97316">{hm["mean_yaw"]:.1f}°</div>
                    <div style="font-size:0.7rem;color:#6b7280;">AVG YAW<br>(Left/Right)</div>
                </div>
                <div>
                    <div style="font-size:1.2rem;font-weight:700;color:#38bdf8">{hm["mean_pitch"]:.1f}°</div>
                    <div style="font-size:0.7rem;color:#6b7280;">AVG PITCH<br>(Up/Down)</div>
                </div>
                <div>
                    <div style="font-size:1.2rem;font-weight:700;color:#a78bfa">{hm["mean_roll"]:.1f}°</div>
                    <div style="font-size:0.7rem;color:#6b7280;">AVG ROLL<br>(Tilt)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Attention breakdown
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">ATTENTION BREAKDOWN</div>', unsafe_allow_html=True)
        breakdown_h = ""
        for st_name in ["Engaged", "Partial", "Distracted", "No Face"]:
            pct = hm["status_pcts"].get(st_name, 0)
            if pct > 0:
                c = HEAD_STATUS_COLORS[st_name]
                breakdown_h += f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:3px;">
                        <span>{st_name}</span>
                        <span style="font-family:Space Mono,monospace;color:{c};">{pct:.1f}%</span>
                    </div>
                    <div class="bar-bg">
                        <div class="bar-fill" style="width:{pct:.0f}%;background:{c}"></div>
                    </div>
                </div>"""
        st.markdown(breakdown_h + "</div>", unsafe_allow_html=True)

        # Head engagement timeline
        if hm["head_timeline"]:
            st.markdown('<div class="section-label" style="margin-top:12px;">🧭 HEAD ENGAGEMENT TIMELINE</div>',
                        unsafe_allow_html=True)
            htl = '<div class="timeline-wrap">'
            for sec, st_val in enumerate(hm["head_timeline"]):
                c    = HEAD_STATUS_COLORS.get(st_val, "#6B7280")
                icon = "✓" if st_val == "Engaged" else "~" if st_val == "Partial" else "✗"
                htl += f'<div class="tl-dot" style="background:{c}" title="s{sec+1}:{st_val}">{icon}</div>'
            htl += '</div>'
            st.markdown(htl, unsafe_allow_html=True)