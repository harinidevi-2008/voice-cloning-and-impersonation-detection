"""
Voice Integrity Security Layer — Streamlit Dashboard (Member 2)

A thin HTTP client over the FastAPI backend built in member2_backend/app/.
This file talks to the backend ONLY through /enroll, /users, and /analyze —
it never imports backend code directly, so backend and dashboard can be
demoed, deployed, or swapped independently.

Two screens (as tabs, per the "no unnecessary pages" instruction):
  1. Speaker Enrollment
  2. Call Simulation / Analysis

Run with:
    streamlit run dashboard/streamlit_app.py
(see README.md for full setup instructions)
"""

import os
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_API_BASE_URL = os.environ.get("VISL_API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECS = 15
ALLOWED_AUDIO_TYPES = ["wav", "mp3", "m4a", "flac", "ogg"]

# Dashboard-level display tiers, derived from impersonation_risk.
# NOTE: the backend's own verdict has 3 buckets (LOW/MEDIUM/HIGH, thresholds
# in app/config.py: 0.40 and 0.70). The requested UI has 4 visual severity
# levels (LOW/MEDIUM/HIGH/CRITICAL), so this dashboard adds one extra split
# at 0.85 *for display and recommended-action purposes only* — it does not
# change or reinterpret the backend's own verdict, which is always shown
# alongside it for full transparency. This is a presentation-layer choice,
# not a backend change (per "do not rewrite the backend").
RISK_TIERS = [
    # (min_score_inclusive, tier_name, color_hex, bg_hex)
    (0.85, "CRITICAL", "#dc2626", "#fee2e2"),
    (0.70, "HIGH", "#f97316", "#ffedd5"),
    (0.40, "MEDIUM", "#f59e0b", "#fef3c7"),
    (0.00, "LOW", "#16a34a", "#dcfce7"),
]

RECOMMENDED_ACTIONS = {
    "LOW": "✅ Continue — normal verification only.",
    "MEDIUM": "⚠️ Additional verification recommended before proceeding (e.g. confirm one extra detail with the caller).",
    "HIGH": "🔐 Require secondary verification (callback on a known number, OTP, or supervisor approval) before proceeding.",
    "CRITICAL": "⛔ Block / escalate the transaction immediately. Do not proceed without a full security review.",
}


def get_risk_tier(score: float):
    """Returns (tier_name, color_hex, bg_hex) for a risk score in [0, 1]."""
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
    "ai_backend" field) — never hardcoded here, so the sidebar can't claim
    REAL when the backend is actually running MOCK or vice versa.
    """
    try:
        resp = requests.get(api_base_url() + "/", timeout=5)
        if resp.status_code == 200:
            ai_backend = None
            try:
                ai_backend = resp.json().get("ai_backend")
            except ValueError:
                pass  # non-JSON response; health is still "connected"
            return True, "Connected", ai_backend
        return False, f"Backend responded with HTTP {resp.status_code}", None
    except requests.exceptions.ConnectionError:
        return False, "Cannot reach backend (is uvicorn running?)", None
    except requests.exceptions.Timeout:
        return False, "Backend connection timed out", None
    except Exception as exc:  # noqa: BLE001 — surfaced to the user, not swallowed
        return False, f"Unexpected error: {exc}", None


def fetch_users():
    """Returns (users_list_or_None, error_message_or_None)."""
    try:
        resp = requests.get(f"{api_base_url()}/users", timeout=REQUEST_TIMEOUT_SECS)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"GET /users failed: HTTP {resp.status_code} — {resp.text}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach backend (is uvicorn running?)"
    except Exception as exc:  # noqa: BLE001
        return None, f"Unexpected error fetching users: {exc}"


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


def analyze_call(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    claimed_user_id,
    transaction_value: float,
    urgency: str,
    caller_known: bool,
):
    """Returns (response_json_or_None, error_message_or_None)."""
    try:
        files = {"audio_file": (filename, audio_bytes, content_type)}
        data = {
            "transaction_value": str(transaction_value),
            "urgency": urgency,
            "caller_known": "true" if caller_known else "false",
        }
        if claimed_user_id is not None:
            data["claimed_user_id"] = str(claimed_user_id)

        resp = requests.post(
            f"{api_base_url()}/analyze", data=data, files=files, timeout=REQUEST_TIMEOUT_SECS
        )
        if resp.status_code == 200:
            result = resp.json()
            # Server-measured "audio in -> risk score out" latency, sent as a
            # response header rather than a JSON field so the frozen API
            # contract (Section 3) stays untouched. Attached here purely for
            # the dashboard's own display — not part of the API response.
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
    if isinstance(detail, list):  # FastAPI/Pydantic validation error format
        detail = "; ".join(
            f"{'.'.join(str(p) for p in d.get('loc', []))}: {d.get('msg', '')}" for d in detail
        )
    return f"HTTP {resp.status_code}: {detail}"


# ---------------------------------------------------------------------------
# Audio input widget (upload OR in-browser recording)
# ---------------------------------------------------------------------------
def audio_input_widget(key_prefix: str):
    """
    Renders an upload/record toggle and returns (bytes, filename, content_type)
    or None if no audio has been provided yet.
    """
    supports_recording = hasattr(st, "audio_input")
    options = ["Upload a file"] + (["🎙️ Speak Now"] if supports_recording else [])

    mode = st.radio(
        "Audio input method",
        options,
        horizontal=True,
        key=f"{key_prefix}_mode",
        label_visibility="collapsed",
    )

    if mode == "🎙️ Speak Now" and supports_recording:
        recorded = st.audio_input("Speak now to record a voice sample", key=f"{key_prefix}_recorder")
        if recorded is not None:
            return recorded.getvalue(), "recorded_audio.wav", "audio/wav"
        return None

    uploaded = st.file_uploader(
        "Upload an audio file",
        type=ALLOWED_AUDIO_TYPES,
        key=f"{key_prefix}_uploader",
    )
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name, uploaded.type or "audio/wav"
    return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_meter(label: str, value_0_1: float, color: str, caption: str = ""):
    """A colored horizontal meter with a percentage label — clearer at a
    glance than Streamlit's default single-color st.progress bar."""
    pct = max(0.0, min(1.0, value_0_1)) * 100
    st.markdown(
        f"""
        <div style="margin-bottom: 6px;">
            <div style="display:flex; justify-content:space-between; font-size:0.92rem; font-weight:600; color:#1f2937;">
                <span>{label}</span>
                <span>{pct:.1f}%</span>
            </div>
            <div style="background:#e5e7eb; border-radius:8px; height:14px; width:100%; overflow:hidden;">
                <div style="background:{color}; width:{pct:.1f}%; height:100%; border-radius:8px; transition: width 0.3s ease;"></div>
            </div>
            {f'<div style="font-size:0.78rem; color:#6b7280; margin-top:2px;">{caption}</div>' if caption else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_verdict_banner(tier_name: str, color: str, bg: str, backend_verdict: str, risk_score: float):
    st.markdown(
        f"""
        <div style="
            background:{bg};
            border: 3px solid {color};
            border-radius: 14px;
            padding: 22px 28px;
            text-align:center;
            margin: 10px 0 18px 0;
        ">
            <div style="font-size:1rem; color:#374151; font-weight:600; letter-spacing:0.05em; text-transform:uppercase;">
                Impersonation Risk Verdict
            </div>
            <div style="font-size:2.4rem; font-weight:800; color:{color}; line-height:1.2; margin: 4px 0;">
                {tier_name}
            </div>
            <div style="font-size:1.05rem; color:#374151;">
                Impersonation Risk Score: <strong>{risk_score:.1%}</strong>
            </div>
            <div style="font-size:0.85rem; color:#6b7280; margin-top:6px;">
                Backend verdict code: <code>{backend_verdict}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommended_action(tier_name: str, color: str, bg: str):
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
            <div style="font-size:0.85rem; color:#374151; font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">
                Recommended Action
            </div>
            <div style="font-size:1.05rem; color:#111827; margin-top:4px;">
                {action_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Voice Integrity Security Layer",
    page_icon="🛡️",
    layout="wide",
)

if "api_base_url" not in st.session_state:
    st.session_state["api_base_url"] = DEFAULT_API_BASE_URL

with st.sidebar:
    st.markdown("### 🛡️ Voice Integrity Security Layer")
    st.caption("Member 2 — Backend + Dashboard")
    st.text_input("Backend API URL", key="api_base_url")

    healthy, health_msg, ai_backend = check_backend_health()
    if healthy:
        st.success(f"● Backend: {health_msg}")
    else:
        st.error(f"● Backend: {health_msg}")
        st.caption("Start it with: `uvicorn app.main:app --reload --port 8000`")

    st.divider()
    if ai_backend == "real":
        st.success("🤖 AI Backend: **REAL** (XLS-R+AASIST, ECAPA-TDNN)")
    elif ai_backend == "mock":
        st.warning(
            "🤖 AI Backend: **MOCK** — deterministic hash-based stand-ins, "
            "not real spoof/speaker models. See `app/services/mock_ai_service.py`."
        )
    else:
        # Backend unreachable, or an older backend without the ai_backend
        # field — don't guess, since claiming either MOCK or REAL here
        # without confirmation could be wrong.
        st.caption("🤖 AI Backend: unknown (backend unreachable or not reporting status)")

st.title("🛡️ Voice Integrity Security Layer")
st.caption("Detects possible AI voice impersonation during high-risk calls or transactions.")

tab_enroll, tab_analyze = st.tabs(["🎙️ Speaker Enrollment", "📞 Call Simulation & Analysis"])

# ---------------------------------------------------------------------------
# SCREEN 1 — Speaker Enrollment
# ---------------------------------------------------------------------------
with tab_enroll:
    st.subheader("Enroll a New Speaker")
    st.caption("Register a reference voice sample so future calls can be checked against it.")

    left, right = st.columns([1.1, 1])

    with left:
        with st.form("enroll_form", clear_on_submit=False):
            name = st.text_input("Name", placeholder="e.g. Alice Sharma")
            role = st.text_input("Role", placeholder="e.g. customer, employee, executive")

            st.markdown("**Reference voice sample**")
            enroll_audio = audio_input_widget("enroll")

            submitted = st.form_submit_button("➕ Enroll Speaker", use_container_width=True)

        if submitted:
            if not name.strip():
                st.error("Please enter a name.")
            elif not role.strip():
                st.error("Please enter a role.")
            elif enroll_audio is None:
                st.error("Please upload or record a voice sample.")
            else:
                audio_bytes, filename, content_type = enroll_audio
                with st.spinner("Enrolling speaker..."):
                    result, error = enroll_speaker(
                        name.strip(), role.strip(), audio_bytes, filename, content_type
                    )
                if error:
                    st.error(f"Enrollment failed: {error}")
                else:
                    st.success(
                        f"✅ Speaker enrolled successfully! Assigned **User ID: {result['user_id']}**"
                    )
                    st.balloons()

    with right:
        st.markdown("**Currently Enrolled Speakers**")
        if st.button("🔄 Refresh list", key="refresh_users_enroll_tab"):
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
# SCREEN 2 — Call Simulation / Analysis
# ---------------------------------------------------------------------------
with tab_analyze:
    st.subheader("Simulate & Analyze a Call")
    st.caption("Run a call/transaction through the spoof detection, identity, and context risk pipeline.")

    users, users_error = fetch_users()
    if users_error:
        st.warning(users_error)
        users = []

    UNKNOWN_OPTION = "🔍 Unknown / No Claimed Identity"
    user_options = [UNKNOWN_OPTION] + [
        f"{u['user_id']} — {u['name']} ({u['role']})" for u in (users or [])
    ]

    with st.form("analyze_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            claimed_choice = st.selectbox("Claimed Caller Identity", user_options)
            transaction_value = st.number_input(
                "Transaction Value", min_value=0.0, step=100.0, format="%.2f"
            )

        with col2:
            urgency = st.selectbox("Urgency", ["low", "medium", "high"], index=1)
            caller_known_choice = st.radio(
                "Is the caller a known contact?", ["Yes", "No"], horizontal=True
            )

        st.markdown("**Call audio sample**")
        analyze_audio = audio_input_widget("analyze")

        analyze_submitted = st.form_submit_button("🔎 Analyze Call", use_container_width=True)

    if analyze_submitted:
        if analyze_audio is None:
            st.error("Please upload or record the call's audio sample.")
        else:
            audio_bytes, filename, content_type = analyze_audio
            claimed_user_id = None
            if claimed_choice != UNKNOWN_OPTION:
                claimed_user_id = int(claimed_choice.split(" — ")[0])

            with st.spinner("Analyzing call..."):
                result, error = analyze_call(
                    audio_bytes=audio_bytes,
                    filename=filename,
                    content_type=content_type,
                    claimed_user_id=claimed_user_id,
                    transaction_value=transaction_value,
                    urgency=urgency,
                    caller_known=(caller_known_choice == "Yes"),
                )

            if error:
                st.error(f"Analysis failed: {error}")
            else:
                st.session_state["last_analysis_result"] = result

    # --- Results view (persists across reruns, e.g. sidebar edits) ---
    result = st.session_state.get("last_analysis_result")
    if result:
        st.divider()
        st.markdown("## Results")

        spoof_score = result["spoof_score"]
        speaker_similarity = result["speaker_similarity"]
        context_risk = result["context_risk"]
        impersonation_risk = result["impersonation_risk"]
        backend_verdict = result["verdict"]

        tier_name, tier_color, tier_bg = get_risk_tier(impersonation_risk)

        # 5. Verdict — obvious from a distance
        render_verdict_banner(tier_name, tier_color, tier_bg, backend_verdict, impersonation_risk)

        # Latency: "audio in -> risk score out", measured server-side and
        # sent via response header (see app/routers/analyze.py). Relevant
        # for the "real-time" claim judges tend to ask about.
        processing_ms = result.get("_processing_time_ms")
        if processing_ms:
            st.caption(f"⏱️ Analyzed in {float(processing_ms):.0f} ms (server-side processing time)")

        # 1-4. Component score meters
        m1, m2 = st.columns(2)
        with m1:
            spoof_tier_name, spoof_color, _ = get_risk_tier(spoof_score)
            render_meter(
                "1. Spoof Score (AI-generated voice likelihood)",
                spoof_score,
                spoof_color,
                "Higher = more likely synthetic/spoofed audio.",
            )

            if speaker_similarity is None:
                render_meter(
                    "2. Speaker Similarity",
                    0.0,
                    "#9ca3af",
                    "No claimed identity provided — similarity could not be checked. "
                    "Identity mismatch risk is treated as maximum (1.0) per the risk formula.",
                )
            else:
                identity_mismatch = 1 - speaker_similarity
                _, mismatch_color, _ = get_risk_tier(identity_mismatch)
                render_meter(
                    "2. Speaker Similarity (match to claimed identity)",
                    speaker_similarity,
                    mismatch_color,
                    "Higher = more likely the claimed speaker.",
                )

        with m2:
            ctx_tier_name, ctx_color, _ = get_risk_tier(context_risk)
            render_meter(
                "3. Context Risk",
                context_risk,
                ctx_color,
                "Based on caller familiarity, transaction value, urgency, and call time.",
            )

            render_meter(
                "4. Overall Impersonation Risk",
                impersonation_risk,
                tier_color,
                "0.5 × Spoof Risk + 0.3 × Identity Mismatch Risk + 0.2 × Context Risk",
            )

        # 6. Recommended action
        render_recommended_action(tier_name, tier_color, tier_bg)

        with st.expander("Raw API response (for debugging / demo transparency)"):
            # Exclude the client-side-only latency field so this accurately
            # reflects the actual /analyze JSON contract (Section 3) — the
            # latency is shown separately above, sourced from a response
            # header, not the response body.
            api_only = {k: v for k, v in result.items() if k != "_processing_time_ms"}
            st.json(api_only)
