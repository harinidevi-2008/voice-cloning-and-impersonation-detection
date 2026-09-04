"""
Voice Integrity Security Layer — Streamlit Dashboard (Member 2)

A thin HTTP client over the FastAPI backend built in member2_backend/app/.
This file talks to the backend ONLY through /enroll, /users, and /analyze —
it never imports backend code directly, so backend and dashboard can be
demoed, deployed, or swapped independently.

Two screens (as tabs, per the "no unnecessary pages" instruction):
  1. Speaker Enrollment
  2. Call Simulation / Analysis

DASHBOARD REFACTOR (this version): manual transaction-amount, urgency, and
known-contact entry have been REMOVED from the UI — these are now
auto-derived server-side from the call transcript and speaker similarity
(see app/routers/analyze.py, app/services/entity_extraction.py,
app/services/urgency_detector.py). The /analyze request still accepts
these fields if explicitly sent (backward compatible), it's just that this
dashboard no longer asks for them.

Run with:
    streamlit run dashboard/streamlit_app.py
(see README.md for full setup instructions)
"""

import os
import time
import requests
import streamlit as st
from streamlit_mic_recorder import mic_recorder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_API_BASE_URL = os.environ.get("VISL_API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECS = 30  # transcription can take a few seconds longer than a bare API call
ALLOWED_AUDIO_TYPES = ["wav", "mp3", "m4a", "aac", "flac", "ogg", "mp4"]

# ---------------------------------------------------------------------------
# Theme (Task 7) — professional cybersecurity palette
# ---------------------------------------------------------------------------
THEME = {
    "background": "#07111F",
    "surface": "#0F1B2D",
    "primary": "#2563EB",
    "success": "#16A34A",
    "warning": "#F59E0B",
    "danger": "#DC2626",
    "text": "#F8FAFC",
    "text_muted": "#94A3B8",
    "border": "#1E293B",
}

# Dashboard-level display tiers, derived from impersonation_risk.
# NOTE: the backend's own verdict has 3 buckets (LOW/MEDIUM/HIGH, thresholds
# in app/config.py: 0.40 and 0.70). The dashboard adds one extra split at
# 0.85 for a 4th visual severity level (CRITICAL) for display and
# recommended-action purposes only -- it does not change or reinterpret
# the backend's own verdict, which is always shown alongside it for full
# transparency. This is a presentation-layer choice, not a backend change.
RISK_TIERS = [
    (0.85, "CRITICAL", THEME["danger"], "rgba(220, 38, 38, 0.15)"),
    (0.70, "HIGH", "#F97316", "rgba(249, 115, 22, 0.15)"),
    (0.40, "MEDIUM", THEME["warning"], "rgba(245, 158, 11, 0.15)"),
    (0.00, "LOW", THEME["success"], "rgba(22, 163, 74, 0.15)"),
]

RECOMMENDED_ACTIONS = {
    "LOW": "Continue -- normal verification only.",
    "MEDIUM": "Additional verification recommended before proceeding (e.g. confirm one extra detail with the caller).",
    "HIGH": "Require secondary verification (callback on a known number, OTP, or supervisor approval) before proceeding.",
    "CRITICAL": "Block / escalate the transaction immediately. Do not proceed without a full security review.",
}


def get_risk_tier(score: float):
    """Returns (tier_name, color_hex, translucent_bg) for a risk score in [0, 1]."""
    for threshold, name, color, bg in RISK_TIERS:
        if score >= threshold:
            return name, color, bg
    return RISK_TIERS[-1][1], RISK_TIERS[-1][2], RISK_TIERS[-1][3]


# ---------------------------------------------------------------------------
# Backend HTTP client helpers
# ---------------------------------------------------------------------------
def api_base_url() -> str:
    return st.session_state.get("api_base_url", DEFAULT_API_BASE_URL)


def check_backend_health():
    """
    Returns (is_healthy, message, ai_backend_or_None). ai_backend reflects
    the backend's ACTUAL VISL_AI_BACKEND setting (via the root endpoint's
    "ai_backend" field) -- never hardcoded here, so the sidebar can't claim
    REAL when the backend is actually running MOCK or vice versa.
    """
    try:
        resp = requests.get(api_base_url() + "/", timeout=5)
        if resp.status_code == 200:
            ai_backend = None
            try:
                ai_backend = resp.json().get("ai_backend")
            except ValueError:
                pass
            return True, "Connected", ai_backend
        return False, f"Backend responded with HTTP {resp.status_code}", None
    except requests.exceptions.ConnectionError:
        return False, "Cannot reach backend (is uvicorn running?)", None
    except requests.exceptions.Timeout:
        return False, "Backend connection timed out", None
    except Exception as exc:  # noqa: BLE001
        return False, f"Unexpected error: {exc}", None


def fetch_users():
    """Returns (users_list_or_None, error_message_or_None)."""
    try:
        resp = requests.get(f"{api_base_url()}/users", timeout=REQUEST_TIMEOUT_SECS)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"GET /users failed: HTTP {resp.status_code} -- {resp.text}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach backend (is uvicorn running?)"
    except Exception as exc:  # noqa: BLE001
        return None, f"Unexpected error fetching users: {exc}"


def fetch_recent_analyses(limit: int = 10):
    """Task 6: 'Recent Analyses' dashboard section. Returns
    (list_or_None, error_message_or_None)."""
    try:
        resp = requests.get(
            f"{api_base_url()}/analysis/recent", params={"limit": limit}, timeout=REQUEST_TIMEOUT_SECS
        )
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"GET /analysis/recent failed: HTTP {resp.status_code} -- {resp.text}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach backend (is uvicorn running?)"
    except Exception as exc:  # noqa: BLE001
        return None, f"Unexpected error fetching recent analyses: {exc}"


def enroll_speaker(name: str, role: str, audio_bytes: bytes, filename: str, content_type: str):
    """Returns (response_json_or_None, error_message_or_None)."""
    try:
        files = {"audio_file": (filename, audio_bytes, content_type)}
        data = {"name": name, "role": role}
        resp = requests.post(
            f"{api_base_url()}/enroll", data=data, files=files, timeout=REQUEST_TIMEOUT_SECS
        )
        if resp.status_code == 200:
            return resp.json(), None
        return None, _extract_error(resp)
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach backend (is uvicorn running?)"
    except Exception as exc:  # noqa: BLE001
        return None, f"Unexpected error during enrollment: {exc}"


def analyze_call(audio_bytes: bytes, filename: str, content_type: str, claimed_user_id):
    """
    Returns (response_json_or_None, error_message_or_None).

    Deliberately does NOT send transaction_value/urgency/caller_known --
    the whole point of this refactor is that they're auto-derived
    server-side from the transcript and speaker similarity. The endpoint
    still accepts them if a future caller wants to override (see
    app/routers/analyze.py), this dashboard just never does.
    """
    try:
        files = {"audio_file": (filename, audio_bytes, content_type)}
        data = {}
        if claimed_user_id is not None:
            data["claimed_user_id"] = str(claimed_user_id)

        resp = requests.post(
            f"{api_base_url()}/analyze", data=data, files=files, timeout=REQUEST_TIMEOUT_SECS
        )
        if resp.status_code == 200:
            result = resp.json()
            result["_processing_time_ms"] = resp.headers.get("X-Processing-Time-Ms")
            return result, None
        return None, _extract_error(resp)
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach backend (is uvicorn running?)"
    except Exception as exc:  # noqa: BLE001
        return None, f"Unexpected error during analysis: {exc}"


def _extract_error(resp: requests.Response) -> str:
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:  # noqa: BLE001
        detail = resp.text
    if isinstance(detail, list):
        detail = "; ".join(
            f"{'.'.join(str(p) for p in d.get('loc', []))}: {d.get('msg', '')}" for d in detail
        )
    return f"HTTP {resp.status_code}: {detail}"


# ---------------------------------------------------------------------------
# Audio input widget (Task 1: wider formats + Task 2: real mic recording)
# ---------------------------------------------------------------------------
def audio_input_widget(key_prefix: str):
    """
    Renders an Upload File / Speak Now toggle.

    Returns a tuple (bytes, filename, content_type, is_fresh_recording) or
    None if no audio is available yet. is_fresh_recording is True only on
    the exact script run where a NEW recording just finished -- callers use
    this to auto-trigger analysis immediately on stop (Task 2: "Automatically
    send it to FastAPI for analysis"), without needing a separate submit
    click, while still requiring an explicit submit for uploaded files.
    """
    mode = st.radio(
        "Audio input method",
        ["Upload File", "Speak Now"],
        horizontal=True,
        key=f"{key_prefix}_mode",
        label_visibility="collapsed",
    )

    if mode == "Speak Now":
        st.caption("Click to start recording (5-30 seconds recommended). Click again to stop.")
        audio = mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop Recording",
            just_once=True,
            format="webm",
            key=f"{key_prefix}_mic",
        )
        if audio is not None:
            st.success("Recording captured -- sending for analysis...")
            # Task 1: "recording_<timestamp>" naming pattern. Extension
            # matches the ACTUAL encoding the browser produced (webm/Opus —
            # browsers' native MediaRecorder API doesn't produce wav
            # directly); the existing server-side ffmpeg conversion
            # (app/services/audio_conversion.py) still normalizes it to
            # mono 16kHz PCM WAV before any model sees it, same as any
            # other uploaded format.
            timestamp = int(time.time())
            filename = f"recording_{timestamp}.{audio['format']}"
            return audio["bytes"], filename, f"audio/{audio['format']}", True
        return None

    uploaded = st.file_uploader(
        "Upload an audio file",
        type=ALLOWED_AUDIO_TYPES,
        key=f"{key_prefix}_uploader",
    )
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name, uploaded.type or "audio/wav", False
    return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def inject_theme_css():
    """Task 7: dark cybersecurity theme -- rounded cards, soft shadows, blue
    active tabs, applied via CSS injection (Streamlit doesn't expose most
    of this through its own theming API)."""
    t = THEME
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {t['background']};
            color: {t['text']};
        }}
        [data-testid="stSidebar"] {{
            background-color: {t['surface']};
            border-right: 1px solid {t['border']};
        }}
        h1, h2, h3, h4, p, span, label, .stMarkdown {{
            color: {t['text']};
        }}
        .stCaption, [data-testid="stCaptionContainer"] {{
            color: {t['text_muted']} !important;
        }}
        div[data-testid="stForm"], .visl-card {{
            background-color: {t['surface']};
            border: 1px solid {t['border']};
            border-radius: 14px;
            padding: 20px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {t['primary']} !important;
            color: {t['text']} !important;
            border-radius: 8px 8px 0 0;
        }}
        .stButton button, .stFormSubmitButton button {{
            background-color: {t['primary']};
            color: {t['text']};
            border-radius: 8px;
            border: none;
        }}
        .stButton button:hover, .stFormSubmitButton button:hover {{
            background-color: #1D4ED8;
        }}
        input, textarea, .stSelectbox div[data-baseweb="select"] {{
            background-color: {t['background']} !important;
            color: {t['text']} !important;
            border-radius: 8px !important;
        }}
        [data-testid="stDataFrame"] {{
            border-radius: 10px;
            overflow: hidden;
        }}
        /* Metric cards (Task 8) -- Streamlit's built-in st.metric doesn't
        pick up the card styling above by default */
        [data-testid="stMetric"] {{
            background-color: {t['surface']};
            border: 1px solid {t['border']};
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
        }}
        [data-testid="stMetricValue"] {{
            color: {t['text']};
        }}
        [data-testid="stMetricLabel"] {{
            color: {t['text_muted']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_card_open(extra_style: str = ""):
    st.markdown(f'<div class="visl-card" style="{extra_style}">', unsafe_allow_html=True)


def render_card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def render_meter(label: str, value_0_1: float, color: str, caption: str = ""):
    """A colored horizontal meter with a percentage label."""
    pct = max(0.0, min(1.0, value_0_1)) * 100
    t = THEME
    st.markdown(
        f"""
        <div style="margin-bottom: 14px;">
            <div style="display:flex; justify-content:space-between; font-size:0.92rem; font-weight:600; color:{t['text']};">
                <span>{label}</span>
                <span>{pct:.1f}%</span>
            </div>
            <div style="background:{t['border']}; border-radius:8px; height:14px; width:100%; overflow:hidden; margin-top:4px;">
                <div style="background:{color}; width:{pct:.1f}%; height:100%; border-radius:8px; transition: width 0.3s ease;"></div>
            </div>
            {f'<div style="font-size:0.78rem; color:{t["text_muted"]}; margin-top:2px;">{caption}</div>' if caption else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value_display: str, color: str, subtitle: str = "", pct=None):
    """Task 9: risk dashboard metric card (AI Spoof / Speaker Match / Fraud Risk)."""
    t = THEME
    bar_html = ""
    if pct is not None:
        bar_html = f"""
        <div style="background:{t['border']}; border-radius:6px; height:8px; width:100%; overflow:hidden; margin-top:10px;">
            <div style="background:{color}; width:{max(0,min(100,pct)):.1f}%; height:100%; border-radius:6px;"></div>
        </div>
        """
    st.markdown(
        f"""
        <div class="visl-card" style="text-align:center; padding:18px;">
            <div style="font-size:0.82rem; color:{t['text_muted']}; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">
                {title}
            </div>
            <div style="font-size:2rem; font-weight:800; color:{color}; margin-top:6px;">
                {value_display}
            </div>
            {f'<div style="font-size:0.8rem; color:{t["text_muted"]}; margin-top:4px;">{subtitle}</div>' if subtitle else ''}
            {bar_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_verdict_banner(tier_name: str, color: str, bg: str, backend_verdict: str, risk_score: float):
    t = THEME
    st.markdown(
        f"""
        <div style="
            background:{bg};
            border: 3px solid {color};
            border-radius: 14px;
            padding: 22px 28px;
            text-align:center;
            margin: 10px 0 18px 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.35);
        ">
            <div style="font-size:1rem; color:{t['text_muted']}; font-weight:600; letter-spacing:0.05em; text-transform:uppercase;">
                Impersonation Risk Verdict
            </div>
            <div style="font-size:2.4rem; font-weight:800; color:{color}; line-height:1.2; margin: 4px 0;">
                {tier_name}
            </div>
            <div style="font-size:1.05rem; color:{t['text']};">
                Impersonation Risk Score: <strong>{risk_score:.1%}</strong>
            </div>
            <div style="font-size:0.85rem; color:{t['text_muted']}; margin-top:6px;">
                Backend verdict code: <code>{backend_verdict}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommended_action(tier_name: str, color: str, bg: str):
    t = THEME
    action_text = RECOMMENDED_ACTIONS.get(tier_name, "Review manually.")
    st.markdown(
        f"""
        <div style="
            background:{bg};
            border-left: 6px solid {color};
            border-radius: 8px;
            padding: 14px 18px;
            margin-top: 8px;
        ">
            <div style="font-size:0.85rem; color:{t['text_muted']}; font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">
                Recommended Action
            </div>
            <div style="font-size:1.05rem; color:{t['text']}; margin-top:4px;">
                {action_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_urgency_badge(urgency: str):
    t = THEME
    colors = {"high": t["danger"], "medium": t["warning"], "low": t["success"]}
    color = colors.get((urgency or "low").lower(), t["text_muted"])
    st.markdown(
        f"""
        <span style="
            background:{color}22; color:{color}; border:1px solid {color};
            padding:3px 12px; border-radius:999px; font-weight:700; font-size:0.85rem;
            text-transform:uppercase; letter-spacing:0.04em;
        ">{(urgency or "low").upper()}</span>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Voice Integrity Security Layer",
    page_icon=":shield:",
    layout="wide",
)

inject_theme_css()

if "api_base_url" not in st.session_state:
    st.session_state["api_base_url"] = DEFAULT_API_BASE_URL

with st.sidebar:
    st.markdown("### Voice Integrity Security Layer")
    st.caption("Real-time voice fraud detection")
    st.text_input("Backend API URL", key="api_base_url")

    healthy, health_msg, ai_backend = check_backend_health()
    if healthy:
        st.success(f"Backend: {health_msg}")
    else:
        st.error(f"Backend: {health_msg}")
        st.caption("Start it with: `uvicorn app.main:app --reload --port 8000`")

    st.divider()
    if ai_backend == "real":
        st.success("AI Backend: **REAL** (XLS-R+AASIST, ECAPA-TDNN)")
    elif ai_backend == "mock":
        st.warning(
            "AI Backend: **MOCK** -- deterministic hash-based stand-ins, "
            "not real spoof/speaker models. See `app/services/mock_ai_service.py`."
        )
    else:
        st.caption("AI Backend: unknown (backend unreachable or not reporting status)")

st.title("Voice Integrity Security Layer")
st.caption("Detects possible AI voice impersonation during high-risk calls or transactions.")

tab_enroll, tab_analyze = st.tabs(["Speaker Enrollment", "Call Simulation & Analysis"])

# ---------------------------------------------------------------------------
# SCREEN 1 -- Speaker Enrollment
# ---------------------------------------------------------------------------
with tab_enroll:
    st.subheader("Enroll a New Speaker")
    st.caption("Register a reference voice sample so future calls can be checked against it.")

    left, right = st.columns([1.1, 1])

    with left:
        name = st.text_input("Name", placeholder="e.g. Alice Sharma", key="enroll_name")
        role = st.text_input("Role", placeholder="e.g. customer, employee, executive", key="enroll_role")

        st.markdown("**Reference voice sample**")
        enroll_audio = audio_input_widget("enroll")

        is_fresh_recording = bool(enroll_audio) and len(enroll_audio) == 4 and enroll_audio[3]
        manual_submit = st.button("Enroll Speaker", use_container_width=True, key="enroll_submit_btn")

        if is_fresh_recording or manual_submit:
            if not name.strip():
                st.error("Please enter a name.")
            elif not role.strip():
                st.error("Please enter a role.")
            elif enroll_audio is None:
                st.error("Please upload or record a voice sample.")
            else:
                audio_bytes, filename, content_type, _ = enroll_audio
                with st.spinner("Enrolling speaker..."):
                    result, error = enroll_speaker(
                        name.strip(), role.strip(), audio_bytes, filename, content_type
                    )
                if error:
                    st.error(f"Enrollment failed: {error}")
                else:
                    st.success(
                        f"Speaker enrolled successfully! Assigned User ID: {result['user_id']}"
                    )
                    st.balloons()

    with right:
        st.markdown("**Currently Enrolled Speakers**")
        if st.button("Refresh list", key="refresh_users_enroll_tab"):
            st.rerun()

        users, error = fetch_users()
        if error:
            st.warning(error)
        elif not users:
            st.info("No speakers enrolled yet.")
        else:
            st.dataframe(
                users,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "user_id": "ID",
                    "name": "Name",
                    "role": "Role",
                    "enrolled_at": "Enrolled At (UTC)",
                },
            )

# ---------------------------------------------------------------------------
# SCREEN 2 -- Call Simulation / Analysis
# ---------------------------------------------------------------------------
with tab_analyze:
    st.subheader("Simulate & Analyze a Call")
    st.caption(
        "Transaction amount, urgency, and known-contact status are detected "
        "automatically from the call audio -- nothing to type."
    )

    users, users_error = fetch_users()
    if users_error:
        st.warning(users_error)
        users = []

    UNKNOWN_OPTION = "Unknown / No Claimed Identity"
    user_options = [UNKNOWN_OPTION] + [
        f"{u['user_id']} -- {u['name']} ({u['role']})" for u in (users or [])
    ]

    claimed_choice = st.selectbox("Claimed Caller Identity", user_options, key="analyze_claimed")

    st.markdown("**Call audio sample**")
    analyze_audio = audio_input_widget("analyze")

    is_fresh_recording = bool(analyze_audio) and len(analyze_audio) == 4 and analyze_audio[3]
    manual_submit = st.button("Analyze Call", use_container_width=True, key="analyze_submit_btn")

    if is_fresh_recording or manual_submit:
        if analyze_audio is None:
            st.error("Please upload or record the call's audio sample.")
        else:
            audio_bytes, filename, content_type, _ = analyze_audio
            claimed_user_id = None
            if claimed_choice != UNKNOWN_OPTION:
                claimed_user_id = int(claimed_choice.split(" -- ")[0])

            with st.spinner("Analyzing call -- transcribing, detecting spoofing, checking identity..."):
                result, error = analyze_call(
                    audio_bytes=audio_bytes,
                    filename=filename,
                    content_type=content_type,
                    claimed_user_id=claimed_user_id,
                )

            if error:
                st.error(f"Analysis failed: {error}")
            else:
                st.session_state["last_analysis_result"] = result

    # --- Results view (persists across reruns) ---
    result = st.session_state.get("last_analysis_result")
    if result:
        st.divider()

        spoof_score = result["spoof_score"]
        speaker_similarity = result["speaker_similarity"]
        impersonation_risk = result["impersonation_risk"]
        backend_verdict = result["verdict"]
        transcript = result.get("transcript")
        detected_amount = result.get("detected_amount")
        detected_urgency = result.get("detected_urgency") or "low"
        urgency_confidence = result.get("urgency_confidence")
        urgency_keywords = result.get("urgency_keywords") or []
        known_contact = result.get("known_contact")

        tier_name, tier_color, tier_bg = get_risk_tier(impersonation_risk)

        render_verdict_banner(tier_name, tier_color, tier_bg, backend_verdict, impersonation_risk)

        processing_ms = result.get("_processing_time_ms")
        if processing_ms:
            st.caption(f"Analyzed in {float(processing_ms):.0f} ms (server-side processing time)")

        st.markdown("### Live Conversation")
        render_card_open()
        if transcript:
            st.markdown(f'*"{transcript}"*')
        else:
            st.caption("No transcript available for this call.")
        render_card_close()

        st.markdown("")
        st.markdown("### Extracted Details")
        # Task 7 layout: Amount, Urgency, Speaker Match, Spoof Score, Risk
        d1, d2, d3, d4, d5 = st.columns(5)
        with d1:
            amount_display = f"\u20b9{int(detected_amount):,}" if detected_amount else "Not detected"
            st.metric("Amount", amount_display, help="Detected automatically from speech")
        with d2:
            st.markdown("**Urgency**")
            render_urgency_badge(detected_urgency)
            if urgency_confidence is not None:
                st.caption(f"Confidence: {urgency_confidence:.0%}")
            if urgency_keywords:
                st.caption(f"Detected keywords: {', '.join(urgency_keywords)}")
        with d3:
            st.markdown("**Speaker Match**")
            if known_contact:
                st.markdown(f"<span style='color:{THEME['success']}; font-weight:700;'>Recognized Speaker</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:{THEME['danger']}; font-weight:700;'>Unknown Speaker</span>", unsafe_allow_html=True)
        with d4:
            st.metric("Spoof Score", f"{spoof_score:.0%}")
        with d5:
            st.metric("Risk", tier_name)

        st.markdown("")
        st.markdown("### Risk Dashboard")
        c1, c2, c3 = st.columns(3)
        with c1:
            spoof_tier_name, spoof_color, _ = get_risk_tier(spoof_score)
            render_metric_card("AI Spoof Probability", f"{spoof_score:.0%}", spoof_color, pct=spoof_score * 100)
        with c2:
            if speaker_similarity is None:
                render_metric_card("Speaker Match", "N/A", THEME["text_muted"], "No claimed identity")
            else:
                _, match_color, _ = get_risk_tier(1 - speaker_similarity)
                render_metric_card("Speaker Match", f"{speaker_similarity:.0%}", match_color, pct=speaker_similarity * 100)
        with c3:
            render_metric_card("Fraud Risk", tier_name, tier_color, f"{impersonation_risk:.0%} score")

        with st.expander("Component score detail"):
            m1, m2 = st.columns(2)
            with m1:
                render_meter(
                    "Spoof Score (AI-generated voice likelihood)",
                    spoof_score, spoof_color,
                    "Higher = more likely synthetic/spoofed audio.",
                )
                if speaker_similarity is None:
                    render_meter(
                        "Speaker Similarity", 0.0, THEME["text_muted"],
                        "No claimed identity provided -- similarity could not be checked.",
                    )
                else:
                    identity_mismatch = 1 - speaker_similarity
                    _, mismatch_color, _ = get_risk_tier(identity_mismatch)
                    render_meter(
                        "Speaker Similarity (match to claimed identity)",
                        speaker_similarity, mismatch_color,
                        "Higher = more likely the claimed speaker.",
                    )
            with m2:
                context_risk = result["context_risk"]
                ctx_tier_name, ctx_color, _ = get_risk_tier(context_risk)
                render_meter(
                    "Context Risk", context_risk, ctx_color,
                    "Based on caller familiarity, transaction amount, urgency, and call time -- all auto-detected.",
                )
                render_meter(
                    "Overall Impersonation Risk", impersonation_risk, tier_color,
                    "0.5 x Spoof Risk + 0.3 x Identity Mismatch Risk + 0.2 x Context Risk",
                )

        render_recommended_action(tier_name, tier_color, tier_bg)

        with st.expander("Raw API response (for debugging / demo transparency)"):
            api_only = {k: v for k, v in result.items() if k != "_processing_time_ms"}
            st.json(api_only)

    # --- Task 6: Recent Analyses ---
    st.divider()
    st.markdown("### Recent Analyses")
    recent, recent_error = fetch_recent_analyses(limit=10)
    if recent_error:
        st.warning(recent_error)
    elif not recent:
        st.info("No calls analyzed yet.")
    else:
        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True,
            column_config={
                "call_id": None,  # hide the raw UUID, not useful to scan visually
                "timestamp": "Time (UTC)",
                "speaker_name": "Speaker",
                "transcript": st.column_config.TextColumn("Transcript", width="large"),
                "amount": st.column_config.NumberColumn("Amount", format="\u20b9%d"),
                "urgency": "Urgency",
                "spoof_score": st.column_config.NumberColumn("Spoof Score", format="%.0%%"),
                "similarity": st.column_config.NumberColumn("Similarity", format="%.0%%"),
                "risk": "Risk Verdict",
            },
        )
