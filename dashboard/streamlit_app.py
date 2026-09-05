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
import html
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
# Fixed enterprise light theme. Metric evidence colors remain intentionally
# semantic; the surrounding interface is strictly monochrome.
# ---------------------------------------------------------------------------
THEME = {
    "background": "#F5F6F8",
    "surface": "#FFFFFF",
    "card": "#FFFFFF",
    "primary": "#111111",
    "success": "#059669",
    "warning": "#D97706",
    "danger": "#DC2626",
    "text": "#111111",
    "text_muted": "#6B7280",
    "border": "#E5E7EB",
    "shadow": "rgba(17,17,17,0.08)",
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


def request_dashboard_refresh() -> None:
    """Explicit refresh callback used by the three dashboard data views."""
    st.session_state["_dashboard_refresh_at"] = time.time()


def check_backend_health():
    """
    Returns (is_healthy, latency_ms_or_None, message, ai_backend_or_None). ai_backend reflects
    the backend's ACTUAL VISL_AI_BACKEND setting (via the root endpoint's
    "ai_backend" field) -- never hardcoded here, so the sidebar can't claim
    REAL when the backend is actually running MOCK or vice versa.
    """
    started_at = time.perf_counter()
    try:
        resp = requests.get(api_base_url() + "/", timeout=5)
        latency_ms = (time.perf_counter() - started_at) * 1000
        if resp.status_code == 200:
            ai_backend = None
            try:
                ai_backend = resp.json().get("ai_backend")
            except ValueError:
                pass
            return True, latency_ms, "Connected", ai_backend
        return False, latency_ms, f"Backend responded with HTTP {resp.status_code}", None
    except requests.exceptions.ConnectionError:
        return False, None, "Cannot reach backend (is uvicorn running?)", None
    except requests.exceptions.Timeout:
        return False, None, "Backend connection timed out", None
    except Exception as exc:  # noqa: BLE001
        return False, None, f"Unexpected error: {exc}", None


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


def fetch_enrolled_speakers():
    """Fetch only profiles that the backend has verified have a voiceprint."""
    try:
        resp = requests.get(f"{api_base_url()}/enroll/speakers", timeout=REQUEST_TIMEOUT_SECS)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"GET /enroll/speakers failed: HTTP {resp.status_code} -- {resp.text}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach backend (is uvicorn running?)"
    except Exception as exc:  # noqa: BLE001
        return None, f"Unexpected error fetching enrolled speakers: {exc}"


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
        # Keep multipart field names aligned with FastAPI's established contract.
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
    """Apply the fixed enterprise light interface."""
    t = THEME
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {t['background']};
            color: {t['text']};
        }}
        [data-testid="stSidebar"] {{
            background: #111111;
            border-right: 1px solid #111111;
        }}
        h1, h2, h3, h4, p, span, label, .stMarkdown {{
            color: {t['text']};
        }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown {{ color: #FFFFFF !important; }}
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color: #D1D5DB !important; }}
        [data-testid="stSidebar"] .visl-card {{
            background: #1B1B1B;
            border-color: #3F3F46;
            box-shadow: none;
        }}
        .stCaption, [data-testid="stCaptionContainer"] {{
            color: {t['text_muted']} !important;
        }}
        div[data-testid="stForm"], .visl-card {{
            background: {t['card']};
            border: 1px solid {t['border']};
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 10px 28px {t['shadow']};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{ color: {t['text_muted']} !important; }}
        .stTabs [aria-selected="true"] {{
            background-color: transparent !important;
            color: #111111 !important;
            border-bottom: 3px solid #111111 !important;
        }}
        .stButton button, .stFormSubmitButton button {{
            background-color: #FFFFFF !important;
            color: #111111 !important;
            border-radius: 10px;
            border: 1px solid #111111 !important;
        }}
        .stButton button:hover, .stFormSubmitButton button:hover {{
            background-color: #F3F4F6 !important;
            color: #111111 !important;
            border-color: #111111 !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background: #FFFFFF !important;
            border: 1px solid #111111 !important;
            color: #111111 !important;
            border-radius: 10px !important;
        }}
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] * {{
            color: #111111 !important;
        }}
        [data-testid="stFileUploader"] button {{
            background: #FFFFFF !important;
            color: #111111 !important;
            border: 1px solid #111111 !important;
            border-radius: 10px !important;
        }}
        [data-testid="stFileUploader"] button:hover {{
            background: #F3F4F6 !important;
            color: #111111 !important;
        }}
        [data-testid="stFileUploader"] ::selection {{
            color: #111111 !important;
            background: #E5E7EB !important;
        }}
        [data-testid="stFileUploader"] ::-moz-selection {{
            color: #111111 !important;
            background: #E5E7EB !important;
        }}
        input, textarea, .stSelectbox div[data-baseweb="select"] {{
            background-color: {t['surface']} !important;
            color: {t['text']} !important;
            border: 1px solid {t['border']} !important;
            border-radius: 10px !important;
        }}
        input:focus, textarea:focus {{ border-color: #111111 !important; }}
        input::placeholder, textarea::placeholder {{
            color: #111111 !important;
            opacity: 1 !important;
        }}
        [data-testid="stDataFrame"] {{
            background: #FFFFFF;
            border: 1px solid {t['border']};
            box-shadow: 0 8px 24px {t['shadow']};
            border-radius: 10px;
            overflow: hidden;
        }}
        [data-testid="stDataFrame"] [role="row"]:nth-child(even) {{ background: #F9FAFB; }}
        [data-testid="stExpander"] {{
            background: #FFFFFF;
            border: 1px solid {t['border']};
            border-radius: 16px;
            box-shadow: 0 8px 24px {t['shadow']};
            margin-bottom: 12px;
        }}
        /* Metric cards (Task 8) -- Streamlit's built-in st.metric doesn't
        pick up the card styling above by default */
        [data-testid="stMetric"] {{
            background: linear-gradient(135deg, {t['card']}, {t['surface']});
            border: 1px solid {t['border']};
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: 0 10px 28px {t['shadow']};
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


def render_metric_card(title: str, value_display: str, color: str, subtitle: str = "", pct=None, icon: str = "●"):
    """Professional KPI card with a colored, animated evidence bar."""
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
        <div class="visl-card" style="padding:20px;">
            <div style="font-size:0.82rem; color:{t['text_muted']}; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">
                <span style="font-size:1.1rem;">{icon}</span> &nbsp;{title}
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


def spoof_display(score: float, label: str | None = None):
    """Return the calibrated label and color used consistently by result cards."""
    score = max(0.0, min(1.0, float(score)))
    if score <= 0.25:
        return label or "Genuine (Very High Confidence)", THEME["success"]
    if score <= 0.45:
        return label or "Probably Genuine", "#EAB308"  # yellow
    if score <= 0.65:
        return label or "Suspicious", "#F97316"  # orange
    return label or ("Likely AI Generated" if score <= 0.85 else "Highly Likely AI Generated"), THEME["danger"]


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


def generate_preventive_actions(result: dict) -> list[str]:
    """Return concise, deduplicated safeguards for the current analysis."""
    actions_by_risk = {
        "LOW": [
            "Proceed with the transaction.",
            "Log the interaction.",
            "No further verification required.",
        ],
        "MEDIUM": [
            "Verify one personal identity detail.",
            "Confirm payment using the registered number.",
            "Do not share OTP or PIN.",
            "Record the interaction.",
        ],
        "HIGH": [
            "Pause the transaction immediately.",
            "Verify identity via another channel.",
            "Contact the organization directly.",
            "Escalate to fraud monitoring.",
        ],
        "CRITICAL": [
            "Block the transaction.",
            "Disconnect the call.",
            "Freeze payment authorization.",
            "Preserve recording as evidence.",
            "Notify cybersecurity response.",
        ],
    }
    final_risk = (result.get("verdict") or result.get("final_risk") or "LOW").upper()
    actions = list(actions_by_risk.get(final_risk, actions_by_risk["MEDIUM"]))

    # Use the evidence itself, not a static risk label, for additions.
    amount = result.get("detected_amount") or result.get("amount") or 0
    urgency = (result.get("detected_urgency") or result.get("urgency") or "low").upper()
    spoof_score = result.get("spoof_score") or 0
    additions = []
    if amount > 100_000:
        additions.append("Require dual approval before payment.")
    if urgency == "HIGH":
        additions.append("Ignore pressure tactics requesting immediate action.")
    if spoof_score > 0.70:
        additions.append("Treat the voice as potentially AI-generated.")
    for action in additions:
        if action not in actions and len(actions) >= 5:
            actions.pop()
        if action not in actions:
            actions.append(action)
    return actions[:5]


def render_recommended_action(result: dict, tier_name: str, color: str, bg: str):
    t = THEME
    actions_html = "".join(f"<li>{action}</li>" for action in generate_preventive_actions(result))
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
            <ul style="font-size:1rem; color:{t['text']}; margin:8px 0 0 0; padding-left:20px; line-height:1.7;">
                {actions_html}
            </ul>
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


@st.fragment(run_every=4)
def render_backend_monitor():
    """Poll the live backend on an independent four-second refresh cycle."""
    healthy, latency_ms, health_msg, ai_backend = check_backend_health()
    if healthy:
        st.markdown(
            f"<div class='visl-card' style='padding:14px; margin:8px 0;'>"
            f"<span style='color:{THEME['text_muted']}'>Backend Status</span><br>"
            f"<b style='color:{THEME['success']}'>● Online</b><br>"
            f"<span style='color:{THEME['text_muted']}'>Latency</span> "
            f"<b>{latency_ms:.0f} ms</b> &nbsp; "
            f"<span style='color:{THEME['text_muted']}'>AI Backend</span> "
            f"<b>{(ai_backend or 'unknown').upper()}</b></div>",
            unsafe_allow_html=True,
        )
    else:
        st.error("🔴 Backend Offline")
        st.caption(health_msg)
        if st.button("Retry", key="backend_health_retry", use_container_width=True):
            st.rerun(scope="fragment")


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Voice Integrity Security Layer",
    page_icon=":shield:",
    layout="wide",
)

if "api_base_url" not in st.session_state:
    st.session_state["api_base_url"] = DEFAULT_API_BASE_URL
inject_theme_css()

with st.sidebar:
    st.markdown("## Voice Integrity")
    st.caption("Real-time Voice Fraud Detection")
    st.markdown("---")
    st.text_input("Backend API URL", key="api_base_url")
    render_backend_monitor()

st.title("Voice Integrity Security Layer")
st.caption("Detects possible AI voice impersonation during high-risk calls or transactions.")

tab_enroll, tab_analyze, tab_recent = st.tabs(
    ["Speaker Enrollment", "Call Simulation & Analysis", "Recent Analyses"]
)

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
                        f"Speaker enrolled successfully! Assigned User ID: {result.get('user_id', 'assigned')}"
                    )
                    st.success("✓ Voice embedding stored successfully")
                    st.caption(
                        f"Embedding dimension: {result.get('embedding_dimension', 192)}  ·  "
                        f"Status: {result.get('verification_status', 'Ready for verification')}"
                    )
                    st.balloons()

    with right:
        st.markdown("**Currently Enrolled Speakers**")
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

    speakers, speakers_error = fetch_enrolled_speakers()
    if speakers_error:
        st.warning(speakers_error)
        speakers = []

    UNKNOWN_OPTION = "Unknown / No Claimed Identity"
    user_options = [UNKNOWN_OPTION] + [
        f"{speaker['id']} -- {speaker['name']} ({speaker['role']})" for speaker in (speakers or [])
    ]

    if not speakers:
        st.info("No enrolled speakers available. Enroll a speaker first.")
    claimed_choice = st.selectbox(
        "Claimed Caller Identity",
        user_options,
        key="analyze_claimed",
        disabled=not bool(speakers),
    )

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
                st.error("Analysis failed")
                st.caption(f"Reason: {error}")
            else:
                st.session_state["last_analysis_result"] = result

    # --- Results view (persists across reruns) ---
    result = st.session_state.get("last_analysis_result")
    if result:
        st.divider()

        spoof_score = float(result.get("spoof_score") or 0.0)
        speaker_similarity = result.get("speaker_similarity")
        impersonation_risk = float(result.get("impersonation_risk") or 0.0)
        backend_verdict = result.get("verdict") or "UNKNOWN"
        transcript = result.get("transcript")
        detected_amount = result.get("detected_amount")
        detected_urgency = result.get("detected_urgency") or "low"
        urgency_confidence = result.get("urgency_confidence")
        urgency_keywords = result.get("urgency_keywords") or []
        known_contact = result.get("known_contact")

        # The backend's centralized risk engine is authoritative.  The
        # dashboard only maps its category to presentation colors.
        tier_name = backend_verdict
        tier_color, tier_bg = {
            "LOW": (THEME["success"], "rgba(22, 163, 74, 0.15)"),
            "MEDIUM": (THEME["warning"], "rgba(245, 158, 11, 0.15)"),
            "HIGH": ("#F97316", "rgba(249, 115, 22, 0.15)"),
            "CRITICAL": (THEME["danger"], "rgba(220, 38, 38, 0.15)"),
        }.get(tier_name, (THEME["text_muted"], "rgba(148, 163, 184, 0.15)"))

        render_verdict_banner(tier_name, tier_color, tier_bg, backend_verdict, impersonation_risk)

        processing_ms = result.get("_processing_time_ms")
        if processing_ms:
            st.caption(f"Analyzed in {float(processing_ms):.0f} ms (server-side processing time)")

        st.markdown("### Live Conversation")
        render_card_open()
        if transcript:
            st.markdown(f'*"{transcript}"*')
        else:
            st.error("Transcription failed")
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
            speaker_status = result.get("speaker_status") or (
                "Verified Identity" if known_contact else "Needs Verification"
            )
            status_color = {
                "VERIFIED IDENTITY": THEME["success"],
                "LIKELY MATCH": "#EAB308",
                "NEEDS VERIFICATION": "#F97316",
                "NO MATCH": THEME["danger"],
            }.get(speaker_status.upper(), THEME["text_muted"])
            st.markdown(
                f"<span style='color:{status_color}; font-weight:700;'>{speaker_status.title()}</span>",
                unsafe_allow_html=True,
            )
            if speaker_similarity is not None:
                render_meter(
                    "Confidence", speaker_similarity, status_color,
                    "Moderate confidence — verify one additional identity factor."
                    if 0.45 <= speaker_similarity < 0.65 else "",
                )
        with d4:
            spoof_label, spoof_color = spoof_display(spoof_score, result.get("spoof_label"))
            st.metric("AI Voice Authenticity", spoof_label)
        with d5:
            st.metric("Risk", tier_name)

        st.markdown("")
        st.markdown("### Risk Dashboard")
        c1, c2, c3 = st.columns(3)
        with c1:
            spoof_label, spoof_color = spoof_display(
                spoof_score, result.get("spoof_label") or result.get("spoof_category")
            )
            render_metric_card(
                "AI Voice Authenticity", spoof_label, spoof_color,
                f"Calibrated spoof evidence: {spoof_score:.0%}", pct=spoof_score * 100, icon="🛡"
            )
        with c2:
            if speaker_similarity is None:
                render_metric_card("Speaker Match", "N/A", THEME["text_muted"], "No claimed identity", icon="👤")
            else:
                speaker_status = result.get("speaker_status") or "Needs Verification"
                match_color = {
                    "VERIFIED IDENTITY": THEME["success"],
                    "LIKELY MATCH": "#EAB308",
                    "NEEDS VERIFICATION": "#F97316",
                    "NO MATCH": THEME["danger"],
                }.get(speaker_status.upper(), THEME["text_muted"])
                interpretation = (
                    "Moderate Confidence" if 0.45 <= speaker_similarity < 0.65
                    else speaker_status.title()
                )
                render_metric_card(
                    "Speaker Match", f"{speaker_similarity:.0%}", match_color,
                    interpretation, pct=speaker_similarity * 100, icon="👤"
                )
        with c3:
            render_metric_card(
                "Fraud Risk", tier_name, tier_color,
                "Requires Verification" if tier_name in {"MEDIUM", "HIGH"} else f"{impersonation_risk:.0%} score",
                pct=impersonation_risk * 100, icon="⚠"
            )

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
                context_risk = float(result.get("context_risk") or 0.0)
                ctx_tier_name, ctx_color, _ = get_risk_tier(context_risk)
                render_meter(
                    "Context Risk", context_risk, ctx_color,
                    "Based on caller familiarity, transaction amount, urgency, and call time -- all auto-detected.",
                )
                render_meter(
                    "Overall Impersonation Risk", impersonation_risk, tier_color,
                    "0.40 x Speaker Mismatch + 0.35 x AI Spoof + 0.15 x Amount + 0.10 x Urgency",
                )

        render_recommended_action(result, tier_name, tier_color, tier_bg)


# ---------------------------------------------------------------------------
# SCREEN 3 -- Persistent analysis history
# ---------------------------------------------------------------------------
with tab_recent:
    st.subheader("Recent Analyses")
    st.caption("Saved automatically after each successful call analysis.")

    filter_left, filter_middle, filter_right = st.columns([2, 1, 0.7])
    with filter_left:
        speaker_query = st.text_input(
            "Search by speaker", placeholder="Type a speaker name", key="recent_speaker_search"
        )
    with filter_middle:
        risk_filter = st.selectbox(
            "Filter by risk", ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"], key="recent_risk_filter"
        )
    recent, recent_error = fetch_recent_analyses(limit=100)
    if recent_error:
        st.warning(recent_error)
    else:
        normalized_query = speaker_query.strip().casefold()
        filtered_recent = [
            entry for entry in (recent or [])
            if (not normalized_query or normalized_query in (entry.get("speaker_name") or "").casefold())
            and (risk_filter == "All" or (entry.get("risk") or "").upper() == risk_filter)
        ]

        if not recent:
            st.info("No analyses available.")
        elif not filtered_recent:
            st.info("No saved analyses match the current filters.")
        else:
            st.caption(f"Showing {len(filtered_recent)} saved analysis record(s).")
            for entry in filtered_recent:
                risk = (entry.get("risk") or "UNKNOWN").upper()
                speaker = entry.get("speaker_name") or "Unknown speaker"
                amount = entry.get("amount")
                amount_display = f"₹{amount:,.0f}" if amount is not None else "No amount detected"
                label = f"{entry.get('timestamp', 'Unknown time')}  |  {speaker}  |  {risk}"
                risk_color = {
                    "LOW": THEME["success"], "MEDIUM": THEME["warning"],
                    "HIGH": "#F97316", "CRITICAL": THEME["danger"],
                }.get(risk, THEME["text_muted"])
                with st.expander(label):
                    summary_left, summary_mid, summary_right = st.columns(3)
                    summary_left.metric("Amount", amount_display)
                    summary_mid.metric("Spoof score", f"{(entry.get('spoof_score') or 0):.0%}")
                    similarity = entry.get("similarity")
                    summary_right.metric(
                        "Speaker similarity", f"{similarity:.0%}" if similarity is not None else "N/A"
                    )
                    st.markdown(
                        f"<span style='background:{risk_color}22; color:{risk_color}; "
                        f"border-radius:999px; padding:4px 10px; font-weight:700;'>{risk}</span>"
                        f" &nbsp; <b>Urgency:</b> {(entry.get('urgency') or 'unknown').upper()}",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Transcript**")
                    st.markdown(
                        "<div style='background:#F3F4F6; border-radius:10px; padding:12px; "
                        "color:#111111;'>"
                        f"{html.escape(entry.get('transcript') or 'Transcription failed')}"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Preventive actions**")
                    for action in generate_preventive_actions(entry):
                        st.markdown(f"- {action}")
