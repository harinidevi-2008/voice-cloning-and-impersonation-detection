# Voice Integrity Security Layer — Member 2 Backend

FastAPI backend for the Voice Integrity Security Layer hackathon project.
Implements `/enroll`, `/users`, and `/analyze`, plus the context and risk
fusion engines. AI model calls (spoof detection, speaker verification) are
**mocked** in `app/services/mock_ai_service.py` until Member 1's real models
are ready — see that file's docstring for the exact swap-in instructions.

## 1. Install

Requires Python 3.9+.

```bash
cd member2_backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -r dashboard/requirements.txt   # only needed if you'll run the dashboard too
```

## 2. Run everything (quickest option)

```bash
python run_dev.py
```

Starts the FastAPI backend (`http://127.0.0.1:8000`) and the Streamlit
dashboard (`http://127.0.0.1:8501`) together, and stops both cleanly on
Ctrl+C. This is a convenience wrapper only — see below to run either piece
on its own (e.g. if you only need the API, or want them on different
machines).

## 3. Run the API on its own

From the `member2_backend/` directory:

```bash
uvicorn app.main:app --reload --port 8000
```

- API base URL: `http://127.0.0.1:8000`
- Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`
- A SQLite DB is auto-created at `data/voice_integrity.db` on first run.
- Uploaded audio is saved under `data/audio_uploads/`.

## 4. Run the tests

```bash
pytest -v
```

Covers: risk engine math, context engine math (including the midnight-wrap
"unusual time" check), and end-to-end API behavior (enroll → list users,
analyze with/without a claimed identity, validation errors).

## 5. Testing the endpoints manually

The easiest option is the Swagger UI at `/docs` — it lets you upload files
through a form in the browser. Below are `curl` and PowerShell equivalents.

You'll need a real (or dummy) audio file for the upload. Any `.wav`/`.mp3`
file works for the mock stage. **Naming tip:** the mock AI service reads
filename hints — files named like `genuine_*.wav`, `clone_*.wav`, or
`impersonator_*.wav` (matching Member 3's demo audio convention) will
produce realistic-looking scores; anything else gets a neutral score.

### POST /enroll

**curl**
```bash
curl -X POST "http://127.0.0.1:8000/enroll" \
  -F "name=Alice Sharma" \
  -F "role=customer" \
  -F "audio_file=@genuine_alice.wav"
```

**PowerShell**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/enroll" `
  -F "name=Alice Sharma" `
  -F "role=customer" `
  -F "audio_file=@genuine_alice.wav"
```

**Expected output**
```json
{
  "user_id": 1
}
```

---

### GET /users

**curl**
```bash
curl "http://127.0.0.1:8000/users"
```

**PowerShell**
```powershell
curl.exe "http://127.0.0.1:8000/users"
```

**Expected output**
```json
[
  {
    "user_id": 1,
    "name": "Alice Sharma",
    "role": "customer",
    "enrolled_at": "2026-09-02T10:15:30.123456+00:00"
  }
]
```

---

### POST /analyze — with a claimed identity

**curl**
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "audio_file=@clone_alice.wav" \
  -F "claimed_user_id=1" \
  -F "transaction_value=250000" \
  -F "urgency=high" \
  -F "caller_known=false"
```

**PowerShell**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/analyze" `
  -F "audio_file=@clone_alice.wav" `
  -F "claimed_user_id=1" `
  -F "transaction_value=250000" `
  -F "urgency=high" `
  -F "caller_known=false"
```

**Expected output** (numbers vary by filename hash, shape is fixed)
```json
{
  "spoof_score": 0.891,
  "speaker_similarity": 0.284,
  "context_risk": 0.85,
  "impersonation_risk": 0.836,
  "verdict": "HIGH_RISK_LIKELY_IMPERSONATION"
}
```

---

### POST /analyze — no claimed identity (`claimed_user_id` omitted)

**curl**
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "audio_file=@genuine_unknown_caller.wav" \
  -F "transaction_value=500" \
  -F "urgency=low" \
  -F "caller_known=true"
```

**Expected output**
```json
{
  "spoof_score": 0.041,
  "speaker_similarity": null,
  "context_risk": 0.05,
  "impersonation_risk": 0.331,
  "verdict": "LOW_RISK_LIKELY_GENUINE"
}
```
Note: `speaker_similarity: null` forces `IdentityMismatchRisk = 1.0` by
design (see `risk_engine.compute_identity_mismatch_risk` docstring) — an
unverifiable identity is treated as a risk factor, not ignored.

## 6. Real AI models (Member 1's integration)

Member 1's real models are integrated and available behind a switch —
**both backends coexist**, selected by an environment variable:

```bash
# Default: fast, deterministic mock (what the automated tests use)
uvicorn app.main:app --reload --port 8000

# Real models: XLS-R+AASIST spoof detection, ECAPA-TDNN speaker verification
export VISL_AI_BACKEND=real          # Windows PowerShell: $env:VISL_AI_BACKEND="real"
uvicorn app.main:app --reload --port 8000
```

### One-time setup for the real backend

```bash
pip install -r requirements-real-ai.txt
```

This installs torch, torchaudio, librosa, speechbrain, scikit-learn, and
related packages. The ECAPA-TDNN model itself downloads automatically from
Hugging Face on first use (needs internet the first time; cached after
that under `app/services/ai_models/checkpoints/ecapa/`). AASIST's model
code and pretrained weights are already included in this repo under
`app/services/ai_models/checkpoints/aasist/` (sourced from the official
[clovaai/aasist](https://github.com/clovaai/aasist), MIT licensed — the
zip you shared didn't include them, only the training/eval scaffolding).

### Validate it before trusting it

Run this once on a machine with `requirements-real-ai.txt` installed,
**before** relying on the API/dashboard with real models:

```bash
python check_real_ai_integration.py path/to/a/real_voice_sample.wav
```

It enrolls a test speaker, confirms `get_similarity` finds the right
embedding back, and runs real spoof detection — exercising the exact code
path the API uses. I verified the integration logic (path resolution, the
enroll/verify round-trip, control flow) using dependency-stubbed tests in
my own environment, but the actual PyTorch/SpeechBrain model math has not
been run end-to-end yet — that's what this script is for.

### How the code is organized

```
app/services/
├── mock_ai_service.py      # default backend — unchanged, still what tests use
├── real_ai_service.py      # real backend — wraps ai_models/ under the same 3-function contract
├── ai_service.py           # picks mock vs real based on AI_BACKEND, for get_spoof_score/get_similarity
└── ai_models/               # Member 1's models, relocated + path-fixed
    ├── spoof_detector.py     # AASIST wrapper (was models/spoof_detector.py)
    ├── speaker_verifier.py   # ECAPA-TDNN wrapper (was models/speaker_verifier.py)
    ├── preprocess.py         # audio loading (was utils/preprocess.py, unchanged)
    ├── embedding_store.py    # speaker embeddings DB (was utils/database.py)
    └── checkpoints/           # AASIST model + weights, ECAPA cache dir
```

### The one integration bug this required fixing

Member 1's `enroll_speaker()` returns an ID from its **own** embeddings
database, and `get_similarity()` looks up the stored embedding by that same
ID. This backend's main database (`app/db/database.py`) was independently
generating its own IDs — so the two would silently drift apart the moment
enrollment counts diverged, and `get_similarity` would check the wrong
person's voiceprint (or find nothing).

**Fix:** `app/routers/enroll.py` now creates the user in the main database
*first*, then calls `real_ai_service.enroll_speaker_at(user_id, ...)` to
pin the embedding to that exact ID — added specifically for this purpose,
alongside the original `enroll_speaker()` (kept for interface completeness).
This only affects the real backend; the mock backend's behavior is
unchanged.

A second, harmless issue in the original repo: `utils/config.py` imports a
constant from itself (`from utils.config import SPOOF_THRESHOLD`, inside
`config.py`) and doesn't define it — this file isn't imported by anything
in the actual pipeline, so it doesn't affect the integration, but it's
worth flagging back to Member 1.

## 7. Hardening fixes (Windows compatibility, correctness, security)

A hardening pass found and fixed 7 real issues, verified with 17 new
automated tests (38 total, up from 21). Full before/after detail and
verification method for each is in the code comments at the cited
locations — this is a summary.

1. **Windows ECAPA/SpeechBrain crash (WinError 1314).**
   `speaker_verifier.py` had a relative `savedir` path (resolved against
   whatever directory the process launched from, not this project) and
   used SpeechBrain's default `LocalStrategy.SYMLINK`, which needs
   Administrator/Developer Mode on Windows. Fixed: absolute path via
   `__file__`, and `local_strategy=LocalStrategy.COPY_SKIP_CACHE` —
   SpeechBrain's own supported no-symlink strategy, confirmed by reading
   `speechbrain/utils/fetching.py` directly.

2. **AASIST spoof score was reading the wrong class and using the wrong
   math.** The official `clovaai/aasist` training/eval code
   (`checkpoints/aasist/`) confirms column 0 = spoof, column 1 = bonafide,
   and that the model is trained with `nn.CrossEntropyLoss` (a joint
   2-class softmax), not two independent logits. The original code read
   column 1 (bonafide, the wrong class) and applied `sigmoid` to it alone
   (mathematically invalid for a jointly-trained output). Fixed with a
   proper softmax over both columns, extracted into a stdlib-only pure
   function (`app/services/ai_models/aasist_scoring.py`) so it's unit
   tested without needing torch.

3. **"Ghost users" on enrollment failure.** If voiceprint registration
   failed after the user row was already created, the row stayed in the
   database with no embedding. `app/routers/enroll.py` now rolls back
   (deletes the user row and the uploaded file) on any failure.

4. **Corrupt audio caused a 500, not a clean error.** Extension checks
   only look at the filename. A new `AudioDecodeError` (stdlib-only, so
   mock mode never imports librosa) is raised by `preprocess.py` on decode
   failure and caught by both routers as a 400.

5. **Path traversal via uploaded filenames.** `upload_file.filename` was
   used directly in constructing the saved path. Now sanitized via
   `basename` + a character allowlist + a `commonpath` defense-in-depth
   check, confirmed live with an actual `../../../etc/passwd.wav` request.

6. **Dashboard always claimed "mock" regardless of the real setting.**
   `GET /` now reports the actual `AI_BACKEND` value (not part of the
   frozen `/analyze`/`/enroll`/`/users` contract), and the dashboard
   displays whichever is actually true.

7. **Unknown-identity risk (`speaker_similarity=None` → risk=1.0).**
   Reviewed and kept as-is — this is intentional: an unverifiable identity
   is itself a risk signal for a fraud-detection system, not neutral
   information, and treating it as anything less than maximal would create
   an incentive to simply not claim an identity. Documentation strengthened
   in `risk_engine.py`; existing test coverage confirmed unchanged.

## 8. Real-time fraud detection refactor (dashboard + audio pipeline)

The dashboard and audio pipeline were substantially reworked for a live
demo: manual entry of transaction amount, urgency, and known-contact
status has been **removed from the UI** — these are now auto-derived from
the call audio itself.

### What changed

1. **Universal audio input.** `/enroll` and `/analyze` now accept WAV, MP3,
   M4A, AAC, FLAC, OGG, and MP4 (previously WAV-only). Every upload is
   converted server-side to mono 16kHz WAV via `ffmpeg`
   (`app/services/audio_conversion.py`) before any model sees it —
   verified with real stereo/44.1kHz WAV, MP3, and M4A inputs.

2. **Real microphone recording** via `streamlit-mic-recorder` (both
   screens), replacing the old placeholder. Recording auto-submits for
   analysis/enrollment as soon as it stops — no separate click needed.

3. **Auto-detected transaction amount** (`app/services/entity_extraction.py`):
   parses the call transcript for amounts, supporting digits, ₹/Rs
   markers, Indian numbering (lakh/crore), and spelled-out numbers
   ("fifty thousand" → 50000). Pure regex — no ML model needed for this
   part.

4. **Auto-detected urgency** (`app/services/urgency_detector.py`):
   keyword-based classifier (LOW/MEDIUM/HIGH) using the keyword lists
   from the spec, configurable in `app/config.py`.

5. **Auto-detected known-contact status**: derived from speaker similarity
   vs. `KNOWN_CONTACT_SIMILARITY_THRESHOLD` (default 0.75) when an identity
   is claimed; defaults to `False` (not known) otherwise — consistent
   with this project's existing "unverifiable is a risk signal, not a
   safe default" principle.

6. **Transcription** (`app/services/transcription_service.py`): real
   backend uses `faster-whisper` (CTranslate2, not full PyTorch — kept
   independent of the AASIST/ECAPA torch stack); mock backend produces a
   deterministic transcript keyed off the *original* filename, so the
   whole pipeline stays testable without network access or a model
   download.

7. **`analysis.db`**: every analyzed call is now logged (call_id,
   timestamp, transcript, spoof_score, similarity, amount, urgency, risk)
   — a second, separate SQLite database from the user/embedding stores,
   since it has a different shape (one row per call) and lifecycle
   (append-only).

8. **Dark cybersecurity theme**, live transcript panel, and 3-card risk
   dashboard (AI Spoof Probability / Speaker Match / Fraud Risk) in the
   results view.

### The `/analyze` contract, and how backward compatibility was preserved

`transaction_value`, `urgency`, and `caller_known` are now **optional**
form fields (previously required) — if omitted, they're auto-derived; if
explicitly sent, they're still honored exactly as before. This is how the
dashboard could drop manual entry entirely without breaking the endpoint's
existing shape or any caller that still sends all three explicitly.

The response gained 5 new **optional, additive** fields (`transcript`,
`detected_amount`, `detected_urgency`, `known_contact`, `call_id`) —
adding fields to a JSON response is backward compatible; the original 5
contracted fields (`spoof_score`, `speaker_similarity`, `context_risk`,
`impersonation_risk`, `verdict`) are unchanged, and a test
(`test_analyze_returns_latency_header`) explicitly asserts they're still
present.

### A bug found and fixed during this refactor

The first version of the converted-audio filename discarded the original
upload's filename entirely (replaced with a bare UUID). This silently
broke the mock backend's core demo mechanism: both `mock_ai_service.py`
and the mock transcription service key their deterministic behavior off
filename hints (`genuine_`/`clone_`/`impersonator_`, or a digit sequence
for the mock amount) — with the original name gone, none of that worked
anymore. Fixed by preserving the original filename in the converted
output, and additionally by threading the *true original* filename
through to the mock transcription service explicitly (`filename_hint`
parameter) rather than parsing it back out of an increasingly complex
generated path. Covered by `tests/test_transcription_service.py`.

### Setup

```bash
pip install -r requirements.txt              # now includes soundfile (tests)
pip install -r dashboard/requirements.txt     # now includes streamlit-mic-recorder
pip install -r requirements-real-ai.txt       # now includes faster-whisper

# ffmpeg is a SYSTEM dependency (not pip) — see requirements.txt for
# install commands per OS.
```

### What I could not verify in this environment

Real Whisper transcription (model download + inference) was **not**
tested end-to-end here — this sandbox's network doesn't reach
huggingface.co. The integration code was checked against the actual
installed `faster-whisper` library's real method signatures, and the
mock backend + full pipeline (transcribe → extract → detect → risk) was
verified live end-to-end. Confirm real transcription works on a machine
with normal internet access before a demo.

## 9. Second hardening pass (Speak Now bug, urgency explainability, call history)

A follow-up pass on the refactor above found and fixed one genuinely
critical bug, plus extended a few features per more precise spec:

### Critical bug: real microphone recordings were being rejected

`.webm` was never added to `ALLOWED_AUDIO_EXTENSIONS` (`app/config.py`).
Browsers' native `MediaRecorder` API — used by `streamlit-mic-recorder`,
the actual "Speak Now" implementation — produces webm/Opus, not WAV,
directly. Every real recording would have hit HTTP 400 ("Unsupported
audio file type '.webm'") before ever reaching ffmpeg conversion. This
had never been exercised end-to-end before, because automated testing
can't drive real browser microphone JS — confirmed live with a real
webm/Opus fixture through the actual `/analyze` endpoint. Fixed by adding
`.webm` to the allowlist and confirming ffmpeg decodes it correctly
(already true — ffmpeg's Opus support was already present).

### Other changes

- **PCM 16-bit made explicit**: `audio_conversion.py` now passes
  `-acodec pcm_s16le` explicitly rather than relying on ffmpeg's default
  codec selection, which could vary across builds.
- **Recording filenames**: "Speak Now" recordings are now named
  `recording_<timestamp>.<ext>` before upload (extension matches the
  actual browser-produced encoding — see the bug above for why lying
  about it in the filename wasn't the right fix).

  **Known mock-mode limitation, confirmed live:** since these filenames
  carry no meaningful content hint (unlike Member 3's `genuine_`/`clone_`
  upload convention), the *mock* backend's transcript for a live
  recording will be generic — mock mode never actually listens to audio,
  by design, so it has nothing else to key off. A real webm recording
  sent through `/analyze` still produces a valid response, just with a
  generic mock transcript. Only `AI_BACKEND=real` (actual faster-whisper)
  genuinely transcribes what was said — if you want "Speak Now" to
  visibly react to actual speech content in a demo, that requires the
  real backend, not mock.
- **Urgency explainability**: `detect_urgency_detailed()`
  (`app/services/urgency_detector.py`) now also returns a confidence
  score and the matched keywords, surfaced in the API response
  (`urgency_confidence`, `urgency_keywords` — both additive/optional) and
  the dashboard's urgency badge.
- **`call_logs` table** (renamed from `analysis`, with a one-time
  migration for any existing rows): added a `speaker_name` column,
  populated by looking up the claimed user's enrolled name.
- **New endpoint** `GET /analysis/recent` (not part of the original
  Section 3 contract — additive) powers the dashboard's new **Recent
  Analyses** section, keeping the dashboard a pure HTTP client rather
  than reading the SQLite file directly.
- **Metric card theming**: `st.metric` widgets weren't picking up the
  card styling by default; added explicit CSS targeting Streamlit's
  `stMetric` test IDs.

## 10. Streamlit Dashboard

The dashboard is a thin HTTP client over this backend — it never imports
backend code directly, only calls `/enroll`, `/users`, and `/analyze` over
HTTP. This means backend and dashboard can be run on different machines/ports
and demoed independently.

### Install dashboard dependencies

```bash
pip install -r dashboard/requirements.txt
```

### Run (with the backend already running separately)

```bash
# Terminal 1
uvicorn app.main:app --reload --port 8000

# Terminal 2
streamlit run dashboard/streamlit_app.py
```

Opens at `http://localhost:8501`. The sidebar shows live backend connectivity
status and lets you change the API URL if the backend isn't on the default
`http://127.0.0.1:8000`.

### What's in it

**Tab 1 — Speaker Enrollment**
Name, role, and an audio sample (upload a file, or record live via
`st.audio_input` if your Streamlit version supports it — falls back to
upload-only otherwise). Submits to `/enroll`, shows the generated `user_id`,
and lists all currently enrolled speakers (refreshed from `/users`).

**Tab 2 — Call Simulation & Analysis**
A dropdown of enrolled speakers (from `/users`, plus an explicit
"Unknown / No Claimed Identity" option), transaction value, urgency,
caller-known toggle, and an audio sample. Submits to `/analyze` and renders:
- Colored meters for Spoof Score, Speaker Similarity, Context Risk, and
  Overall Impersonation Risk.
- A large, color-coded verdict banner (green/amber/orange/red) that's
  readable from across a room for a demo.
- A recommended action tied to the risk level.
- The raw JSON response in a collapsed expander, for judges who want to see
  the underlying numbers.

**Note on risk tiers:** the backend's own `verdict` field has three buckets
(LOW/MEDIUM/HIGH — see `app/config.py`). The dashboard adds a fourth
*display-only* tier, CRITICAL, by splitting HIGH at an impersonation_risk of
0.85 (configurable at the top of `dashboard/streamlit_app.py`). This is a
presentation-layer decision only — it does not change the backend's contract
or verdict logic — and the backend's raw verdict string is always shown
alongside it for transparency.

### Testing the dashboard

1. Start the backend, then the dashboard (see above).
2. Go to **Speaker Enrollment**, fill in a name/role, upload a short audio
   file (any `.wav`/`.mp3` works with the mock stage), and click
   **Enroll Speaker**. Confirm the success message and user ID appear, and
   the enrolled-speakers table updates.
3. Go to **Call Simulation & Analysis**, pick that speaker from the dropdown,
   fill in a transaction value/urgency/caller-known, upload an audio file,
   and click **Analyze Call**. Confirm all four meters, the verdict banner,
   and the recommended action render.
4. Try it again with **Unknown / No Claimed Identity** selected — confirm
   the Speaker Similarity meter shows the "not verified" state instead of a
   percentage.
5. Try an audio filename containing `clone_` or `impersonator_` with a high
   transaction value and `urgency=high` — the mock AI service is
   filename-aware, so this reliably produces a HIGH/CRITICAL result for
   demo purposes.

## 11. Latency measurement

The team's work-division doc calls for measuring end-to-end latency
("audio in → risk score out") for the "real-time" claim judges tend to ask
about. The `/analyze` response body's shape is frozen by the interface
contract (exactly 5 fields), so timing is **not** added as a JSON field.
Instead:

- The backend returns an `X-Processing-Time-Ms` response header on every
  `/analyze` call, and logs a matching line to the console:
  ```
  2026-09-03 18:17:18 INFO visl.analyze: analyze: backend=mock claimed_user_id=1 ai_ms=0.0 total_ms=0.4 verdict=LOW_RISK_LIKELY_GENUINE
  ```
- The dashboard reads that header and shows it under the verdict banner
  (e.g. "⏱️ Analyzed in 340 ms"), separately from the raw API response
  shown in the debug expander — so the JSON contract stays visibly
  untouched even in the UI.
- Covered by `tests/test_api.py::test_analyze_returns_latency_header`,
  which also asserts the response body is still exactly the 5 contracted
  fields (nothing leaked in).

With the mock backend this will always read sub-millisecond — the
meaningful number to capture for the PPT is with `AI_BACKEND=real`, where
`ai_ms` in the log line isolates just the model inference time from the
rest of the request (file save, DB lookups, risk fusion).

## 12. Project structure

```
member2_backend/
├── app/
│   ├── main.py                  # FastAPI app, CORS, router registration
│   ├── config.py                # all weights/thresholds/paths + AI_BACKEND toggle
│   ├── schemas.py                # Pydantic response models
│   ├── routers/
│   │   ├── enroll.py              # POST /enroll (mock/real ID-sync, audio conversion)
│   │   ├── users.py               # GET /users
│   │   ├── analyze.py             # POST /analyze (auto-detect amount/urgency/known)
│   │   └── analysis.py            # GET /analysis/recent (Recent Analyses section)
│   ├── services/
│   │   ├── mock_ai_service.py     # default backend — fast, deterministic
│   │   ├── real_ai_service.py     # real backend — wraps ai_models/
│   │   ├── ai_service.py          # picks mock vs real, per AI_BACKEND
│   │   ├── audio_conversion.py    # ffmpeg -> mono 16kHz WAV for any input format
│   │   ├── transcription_service.py  # faster-whisper (real) / mock transcript
│   │   ├── entity_extraction.py   # amount extraction from transcript
│   │   ├── urgency_detector.py    # keyword-based urgency classifier
│   │   ├── ai_models/              # Member 1's models, relocated + path-fixed
│   │   │   ├── spoof_detector.py
│   │   │   ├── speaker_verifier.py
│   │   │   ├── preprocess.py
│   │   │   ├── embedding_store.py
│   │   │   ├── exceptions.py        # AudioDecodeError (stdlib-only)
│   │   │   ├── aasist_scoring.py     # pure softmax logic, testable w/o torch
│   │   │   └── checkpoints/         # AASIST weights + ECAPA cache dir
│   │   ├── context_engine.py      # context_risk computation
│   │   └── risk_engine.py         # fusion formula + verdict
│   ├── db/
│   │   ├── database.py            # SQLite: enrolled users
│   │   ├── analysis_db.py         # SQLite: per-call analysis log
│   │   └── crud.py                # user CRUD (+ delete for rollback)
│   └── storage/
│       └── audio_store.py         # upload validation, saving, filename sanitization
├── dashboard/
│   ├── streamlit_app.py           # 2-tab dashboard, dark theme, mic recording
│   └── requirements.txt           # streamlit, requests, streamlit-mic-recorder
├── tests/                         # 58 tests (+1 skipped without speechbrain)
├── check_real_ai_integration.py    # standalone script to validate AI_BACKEND=real
├── data/                          # created at runtime (dbs + uploads)
├── run_dev.py                     # convenience: starts backend + dashboard together
├── pytest.ini
├── .gitignore
├── requirements.txt                # backend deps (mock backend only)
├── requirements-real-ai.txt         # extra deps needed for AI_BACKEND=real (+ faster-whisper)
└── README.md
```

## 13. Project Status (Member 2 scope)

| Item | Status |
|---|---|
| FastAPI backend (`/enroll`, `/users`, `/analyze`, `/analysis/recent`) | ✅ Done, 64 automated tests passing (1 skipped without speechbrain) |
| Context engine + risk fusion engine | ✅ Done, unit-tested, matches Section 3 formula exactly |
| SQLite persistence (users, embeddings, per-call analysis log) | ✅ Done |
| Streamlit dashboard: dark theme, real mic recording, live transcript, risk cards | ✅ Done, verified against live backend |
| Universal audio input (WAV/MP3/M4A/AAC/FLAC/OGG/MP4 → normalized WAV) | ✅ Done, verified with real multi-format audio |
| Auto-detected transaction amount, urgency, known-contact (no manual entry) | ✅ Done, verified end-to-end incl. exact spec examples |
| One-command dev launcher (`run_dev.py`) | ✅ Done, verified start + graceful shutdown |
| Integration with Member 1's real spoof/speaker models | ✅ Code complete — mock/real toggle, ID-sync bug fixed, missing AASIST files sourced. ⚠️ Not yet run with the real PyTorch/SpeechBrain stack end-to-end — run `check_real_ai_integration.py` to confirm before demo day |
| Real transcription (faster-whisper) | ✅ Integration code verified against the real library's API. ⚠️ Model download/inference NOT verified end-to-end — this sandbox can't reach huggingface.co. Confirm on a machine with internet before a demo |
| End-to-end latency measurement | ✅ Done — `X-Processing-Time-Ms` header + server log line + dashboard display, verified live |
| Verdict visibility ("unmistakable from across a room") | ✅ Done — large bordered color banner, verified via automated rendering test |
| Demo audio, PPT, documentation, rehearsal | Out of scope for Member 2 — see Members 3 & 4 |
