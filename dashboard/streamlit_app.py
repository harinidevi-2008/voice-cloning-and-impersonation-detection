"""Streamlit client for the Voice Integrity Security Layer API."""

import hashlib
import os
import time

import requests
import streamlit as st
from streamlit_mic_recorder import mic_recorder

DEFAULT_API_BASE_URL = os.environ.get("VISL_API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECS = 120
ALLOWED_AUDIO_TYPES = ["wav", "mp3", "m4a", "aac", "flac", "ogg", "mp4", "webm"]

THEME = {"ink": "#111111", "muted": "#666666", "border": "#D8D8D8", "surface": "#FFFFFF", "wash": "#F5F5F5"}
RISK_TIERS = [(0.85, "CRITICAL", "#B91C1C"), (0.70, "HIGH", "#DC2626"), (0.40, "MEDIUM", "#D97706"), (0.0, "LOW", "#15803D")]


def get_risk_tier(score):
    for threshold, label, color in RISK_TIERS:
        if score >= threshold:
            return label, color
    return "LOW", "#15803D"


def api_base_url():
    return st.session_state.get("api_base_url", DEFAULT_API_BASE_URL).rstrip("/")


def _extract_error(response):
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return f"HTTP {response.status_code}: {detail}"


def check_backend_health():
    try:
        response = requests.get(f"{api_base_url()}/", timeout=5)
        if response.ok:
            return True, "Connected", response.json().get("ai_backend")
        return False, f"HTTP {response.status_code}", None
    except requests.RequestException as exc:
        return False, str(exc), None


def fetch_users():
    try:
        response = requests.get(f"{api_base_url()}/users", timeout=15)
        return (response.json(), None) if response.ok else (None, _extract_error(response))
    except requests.RequestException as exc:
        return None, str(exc)


def fetch_recent_analyses():
    try:
        response = requests.get(f"{api_base_url()}/analysis/recent", params={"limit": 100}, timeout=15)
        return (response.json(), None) if response.ok else (None, _extract_error(response))
    except requests.RequestException as exc:
        return None, str(exc)


def send_audio(endpoint, audio, data):
    files = {"audio_file": (audio["filename"], audio["bytes"], audio["content_type"])}
    try:
        response = requests.post(f"{api_base_url()}{endpoint}", data=data, files=files, timeout=REQUEST_TIMEOUT_SECS)
        if response.ok:
            result = response.json()
            result["_processing_time_ms"] = response.headers.get("X-Processing-Time-Ms")
            return result, None
        return None, _extract_error(response)
    except requests.Timeout:
        return None, "The request exceeded the 120-second demo timeout. Check backend model logs."
    except requests.RequestException as exc:
        return None, str(exc)


def _store_audio(key, audio_bytes, filename, content_type):
    fingerprint = hashlib.sha256(audio_bytes).hexdigest()
    stored = st.session_state.get(key)
    if not stored or stored["fingerprint"] != fingerprint:
        st.session_state[key] = {"bytes": audio_bytes, "filename": filename, "content_type": content_type, "fingerprint": fingerprint}
    return st.session_state[key]


def audio_input_widget(prefix):
    """Persist uploaded/mic audio; recording never submits an API request itself."""
    storage_key = f"{prefix}_captured_audio"
    mode = st.radio("Audio input", ["Upload File", "Speak Now"], horizontal=True, key=f"{prefix}_mode")
    if mode == "Upload File":
        uploaded = st.file_uploader("Upload an audio file", type=ALLOWED_AUDIO_TYPES, key=f"{prefix}_upload")
        if uploaded is not None:
            return _store_audio(storage_key, uploaded.getvalue(), uploaded.name, uploaded.type or "audio/wav")
    else:
        st.caption("Click Start Recording, speak, then click Stop Recording.")
        recording = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", just_once=True, format="webm", key=f"{prefix}_mic")
        if recording:
            audio = _store_audio(storage_key, recording["bytes"], f"recording_{int(time.time())}.{recording['format']}", f"audio/{recording['format']}")
            st.success("Recording captured. Ready for analysis.")
            return audio
    audio = st.session_state.get(storage_key)
    if audio:
        ready_col, clear_col = st.columns([4, 1])
        ready_col.caption(f"Ready: {audio['filename']}")
        if clear_col.button("Clear", key=f"{prefix}_clear_audio"):
            st.session_state.pop(storage_key, None)
            st.rerun()
    return audio


def inject_css():
    st.markdown("""<style>
    .stApp { background: #fff; color: #111; }
    [data-testid="stSidebar"] { background:#111; }
    [data-testid="stSidebar"] * { color:#fff !important; }
    [data-testid="stSidebar"] .stCaption { color:#bbb !important; }
    h1,h2,h3,p,label,.stMarkdown { color:#111; }
    .stTabs [data-baseweb="tab-list"] { gap:28px; border-bottom:1px solid #ddd; }
    .stTabs [data-baseweb="tab"] { background:white; color:#111; padding:10px 2px; }
    .stTabs [aria-selected="true"] { border-bottom:3px solid #111; font-weight:700; }
    .stButton button, .stFormSubmitButton button { background:#111; color:#fff; border:1px solid #111; border-radius:6px; }
    .stButton button:hover, .stFormSubmitButton button:hover { background:#444; border-color:#444; }
    input, textarea, [data-baseweb="select"] > div { background:#fff !important; color:#111 !important; border-color:#555 !important; }
    .visl-card { background:#fff; border:1px solid #d8d8d8; border-radius:10px; padding:16px; margin:8px 0; }
    </style>""", unsafe_allow_html=True)


def show_result(result):
    risk = result["impersonation_risk"]
    tier, color = get_risk_tier(risk)
    st.markdown(f"<div class='visl-card' style='border-left:6px solid {color}'><h2 style='color:{color}'>{tier} RISK</h2><b>Impersonation risk:</b> {risk:.1%}<br><b>Verdict:</b> {result['verdict']}</div>", unsafe_allow_html=True)
    ms = result.get("_processing_time_ms") or result.get("processing_time_ms")
    if ms:
        st.caption(f"Server processing time: {float(ms):.0f} ms")
    a, b, c = st.columns(3)
    a.metric("Spoof score", f"{result['spoof_score']:.1%}")
    b.metric("Speaker similarity", "Not checked" if result.get("speaker_similarity") is None else f"{result['speaker_similarity']:.1%}")
    c.metric("Context risk", f"{result['context_risk']:.1%}")
    if result.get("transcript"):
        st.markdown("**Transcript**")
        st.write(result["transcript"])
    with st.expander("Analysis details"):
        st.json({key: value for key, value in result.items() if not key.startswith("_")})


st.set_page_config(page_title="Voice Integrity Security Layer", page_icon="🛡️", layout="wide")
inject_css()
st.session_state.setdefault("api_base_url", DEFAULT_API_BASE_URL)

with st.sidebar:
    st.markdown("## Voice Integrity")
    st.caption("Real-time Voice Fraud Detection")
    st.text_input("Backend API URL", key="api_base_url")
    healthy, message, backend = check_backend_health()
    st.markdown(f"**Backend Status**  \\n+{'Connected' if healthy else 'Unavailable'}")
    st.caption(f"{message} · AI: {(backend or 'unknown').upper()}")

st.title("Voice Integrity Security Layer")
st.caption("Detects possible AI voice impersonation during high-risk calls or transactions.")
tab_enroll, tab_analyze, tab_history = st.tabs(["Speaker Enrollment", "Call Simulation & Analysis", "Recent Analyses"])

with tab_enroll:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Enroll a New Speaker")
        name = st.text_input("Name", key="enroll_name")
        role = st.text_input("Role", key="enroll_role")
        st.markdown("**Reference voice sample**")
        audio = audio_input_widget("enroll")
        if st.button("Enroll Speaker", use_container_width=True):
            if not name.strip() or not role.strip() or not audio:
                st.error("Enter a name, role, and reference voice sample.")
            else:
                with st.spinner("Enrolling speaker — generating a voiceprint..."):
                    result, error = send_audio("/enroll", audio, {"name": name.strip(), "role": role.strip()})
                if error:
                    st.error(f"Enrollment failed: {error}")
                else:
                    st.success(f"Speaker enrolled with user ID {result['user_id']}.")
                    st.session_state.pop("enroll_captured_audio", None)
    with right:
        st.subheader("Currently Enrolled Speakers")
        users, error = fetch_users()
        if error:
            st.warning(error)
        elif users:
            st.dataframe(users, use_container_width=True, hide_index=True, column_config={"user_id": "ID", "enrolled_at": "Enrolled At"})
        else:
            st.info("No speakers enrolled yet.")

with tab_analyze:
    st.subheader("Call Simulation & Analysis")
    users, error = fetch_users()
    if error:
        st.warning(error)
        users = []
    choices = {"Unknown / No Claimed Identity": None}
    choices.update({f"{user['user_id']} — {user['name']} ({user['role']})": user["user_id"] for user in users})
    selection = st.selectbox("Claimed Caller Identity", list(choices), key="claimed_identity")
    st.markdown("**Call audio sample**")
    audio = audio_input_widget("analyze")
    if st.button("Analyze Call", use_container_width=True):
        if not audio:
            st.error("Upload or capture a call recording first.")
        else:
            with st.spinner("Analyzing audio — AASIST spoof detection, speaker verification, transcription and risk analysis..."):
                data = {"claimed_user_id": choices[selection]} if choices[selection] is not None else {}
                result, analysis_error = send_audio("/analyze", audio, data)
            if analysis_error:
                st.error(f"Analysis failed: {analysis_error}")
            else:
                st.session_state["last_analysis_result"] = result
                st.session_state.pop("analyze_captured_audio", None)
    if st.session_state.get("last_analysis_result"):
        show_result(st.session_state["last_analysis_result"])

with tab_history:
    st.subheader("Recent Analyses")
    st.caption("Saved automatically after each successful call analysis.")
    query_col, risk_col = st.columns([2, 1])
    with query_col:
        query = st.text_input("Search by speaker", placeholder="Speaker name or user ID")
    with risk_col:
        risk_filter = st.selectbox("Filter by risk", ["All", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
    records, error = fetch_recent_analyses()
    if error:
        st.warning(error)
    else:
        query = query.strip().lower()
        for record in records or []:
            risk = record.get("risk") or "LOW"
            tier = "CRITICAL" if "CRITICAL" in risk else "HIGH" if "HIGH" in risk else "MEDIUM" if "MEDIUM" in risk else "LOW"
            searchable = f"{record.get('speaker_name') or ''} {record.get('speaker_user_id') or ''}".lower()
            if query and query not in searchable:
                continue
            if risk_filter != "All" and tier != risk_filter:
                continue
            speaker = record.get("speaker_name") or "Unclaimed caller"
            with st.expander(f"{record.get('timestamp', '')}  |  {speaker}  |  {tier}"):
                st.write(f"**Speaker:** {speaker}")
                if record.get("speaker_user_id") is not None:
                    st.write(f"**User ID:** {record['speaker_user_id']}")
                st.write(f"**Transcript:** {record.get('transcript') or 'Not available'}")
                st.write(f"**Amount:** {record.get('amount') or 0} · **Urgency:** {record.get('urgency') or 'Not available'}")
                similarity = "Not checked" if record.get("similarity") is None else f"{record['similarity']:.1%}"
                st.write(f"**Spoof score:** {record.get('spoof_score') or 0:.1%} · **Speaker similarity:** {similarity}")
                st.write(f"**Risk verdict:** {risk}")
