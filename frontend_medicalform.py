import streamlit as st
import requests
import json
import io
import base64
from PIL import Image
from datetime import datetime, date
import pandas as pd
from collections import Counter

# optional graph render fallback
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    NX_AVAILABLE = True
except Exception:
    NX_AVAILABLE = False

# Hardcoded medical map from backend for condition inference
MEDICAL_MAP = {
    "chest pain": "Possible cardiac issue",
    "mild chest pain": "Possible cardiac issue",
    "shortness of breath": "Respiratory distress",
    "dyspnea": "Respiratory distress",
    "fever": "Possible infection",
    "cough": "Respiratory infection",
    "persistent cough": "Possible bronchitis",
    "sore throat": "Upper respiratory infection",
    "headache": "Neurological symptom",
}

# ------------------------------
# PAGE CONFIG + GLOBAL STYLE
# ------------------------------
st.set_page_config(
    page_title="MedForm AI — Staff Console",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://x.ai',
        'Report a bug': None,
        'About': "MedForm AI: Intelligent Medical Data Extraction & Graph Management"
    }
)

# ---------- GLOBAL STYLE (Enhanced for modern aesthetics) ----------
st.markdown("""
<style>
.stApp { 
    background: linear-gradient(to bottom, #f5f7fa, #e8effa); 
    font-family: 'Inter', sans-serif; 
}

/* Typography improvements */
h1, h2, h3, h4 { 
    font-family: 'Inter', sans-serif; 
    color: #1e3a8a; 
    font-weight: 600; 
}

/* Buttons with hover effects and gradients */
div.stButton > button, .stDownloadButton > button {
    padding: 10px 22px;
    font-size: 15px !important;
    border-radius: 8px !important;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
div.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

/* Primary gradient */
.big-primary {
    background: linear-gradient(90deg, #2f6fed, #3aa0ff) !important;
    color: white !important;
}

/* Better cards with subtle shadows and hover */
.card {
    background: white;
    padding: 18px 20px 16px 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
    transition: all 0.3s ease;
}
.card:hover {
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    transform: translateY(-2px);
}

/* Sidebar refinement: Modern, clean, with subtle gradient and animations */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #eef2f7, #d9e4f0);
    padding: 20px 10px;
    border-right: 1px solid #d1d5db;
}
[data-testid="stSidebar"] > div > div > div > div > div {
    transition: all 0.3s ease;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    margin-bottom: 10px;
    border-radius: 8px;
    background: white;
    color: #2f6fed;
    font-weight: 500;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #2f6fed;
    color: white;
}
[data-testid="stSidebar"] .stTextInput > div > div > input {
    border-radius: 8px;
    border: 1px solid #d1d5db;
    padding: 10px;
    transition: border 0.3s ease;
}
[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: #2f6fed;
    box-shadow: 0 0 0 2px rgba(47, 111, 237, 0.2);
}

/* Metrics styling: More modern with colors */
div[data-testid="metric-container"] {
    background: linear-gradient(to bottom, #ffffff, #f9fafb);
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* Charts responsive with better styling */
div[data-testid="stChart"] {
    width: 100%;
    border-radius: 8px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# SIDEBAR — Redesigned for dynamism and engagement
# ------------------------------
with st.sidebar:
    # Logo/Title with icon for visual interest
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <span style="font-size: 28px; margin-right: 10px;">🩺</span>
        <h2 style="margin: 0; font-weight: 600;">MedForm AI</h2>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Internal Staff Console v2.1")

    # Structured sections with expanders for reduced clutter
    with st.expander("⚙️ Configuration", expanded=True):
        API_URL = st.text_input("API Endpoint", value="http://localhost:8000", help="Backend server URL")
        operator = st.text_input(
            "Operator ID",
            placeholder="e.g. nurse_01",
            help="Used for audit logging on approved records."
        )

    with st.expander("🚀 Quick Actions", expanded=True):
        if st.button("🔄 Refresh App", help="Reload the entire application", key="refresh_app"):
            st.rerun()
        if st.button("🧹 Reset Session", help="Clear all session data", key="reset_session"):
            for k in list(st.session_state.keys()):
                if k not in ("_widgets",):
                    st.session_state.pop(k, None)
            st.success("Session cleared.")
            st.rerun()

    # Footer with subtle styling
    st.markdown("---")
    st.caption(f"© MedForm AI {datetime.now().year} — Internal use only")
    st.caption("Built with ❤️ by Suraj Varma")

# ------------------------------
# BACKEND HELPERS (Unchanged)
# ------------------------------
def safe_json(r):
    try: return r.json()
    except: return {"raw": getattr(r, "text", str(r))}

def call_extract_text(text):
    try: return requests.post(f"{API_URL}/extract/text", data={"text": text}, timeout=60)
    except Exception as e: return e

def call_extract_image(bytes_data, name, type_):
    try: return requests.post(
        f"{API_URL}/extract/image",
        files={"file": (name, bytes_data, type_)},
        timeout=120
    )
    except Exception as e: return e

def call_extract_audio(bytes_data, name, type_):
    try: return requests.post(
        f"{API_URL}/extract/audio",
        files={"file": (name, bytes_data, type_)},
        timeout=180
    )
    except Exception as e: return e

def post_graph_insert(obj):
    if operator:
        obj["_approved_by"] = operator
    try: return requests.post(f"{API_URL}/graph/insert", json=obj)
    except Exception as e: return e

def list_patients():
    try: return requests.get(f"{API_URL}/patients/list")
    except Exception as e: return e

def get_patient_graph(pid):
    try: return requests.get(f"{API_URL}/graph/patient/{pid}")
    except Exception as e: return e

def get_patient_graph_image(pid):
    try: return requests.get(f"{API_URL}/graph/patient/{pid}/image")
    except Exception as e: return e

def load_all_records():
    try: return requests.get(f"{API_URL}/records/all")
    except Exception as e: return e

def search_records(name):
    try: return requests.get(f"{API_URL}/records/search", params={"name": name})
    except Exception as e: return e

# ------------------------------
# KPI COMPUTATION HELPER (Unchanged)
# ------------------------------
@st.cache_data(ttl=60)  # Cache for 1 minute for pseudo-real-time
def compute_kpis():
    resp = load_all_records()
    if not hasattr(resp, "ok") or not resp.ok:
        return None
    
    records = safe_json(resp).get("records", [])
    if not records:
        return None
    
    today = date.today()
    total_patients = len(records)
    patients_today = sum(1 for r in records if r.get("ingested_at", "")[:10] == today.isoformat())
    
    all_symptoms = []
    ages = []
    for rec in records:
        for detail in rec.get("details", []):
            if detail["type"] == "Symptom":
                all_symptoms.append(detail["props"].get("name", "").lower())
        age = rec.get("age")
        if age and age.isdigit():
            ages.append(int(age))
    
    symptom_counts = Counter(all_symptoms)
    top_symptoms = symptom_counts.most_common(5)
    
    condition_counts = Counter()
    for sym in all_symptoms:
        cond = MEDICAL_MAP.get(sym)
        if cond:
            condition_counts[cond] += 1
    
    average_age = sum(ages) / len(ages) if ages else 0
    
    # Age histogram
    if ages:
        age_series = pd.Series(ages)
        age_hist = age_series.value_counts(bins=10, sort=False)
    else:
        age_hist = pd.Series()
    
    return {
        "total_patients": total_patients,
        "patients_today": patients_today,
        "average_age": round(average_age, 1),
        "top_symptoms": top_symptoms,
        "condition_counts": dict(condition_counts),
        "age_hist": age_hist,
        "last_updated": datetime.utcnow().isoformat()
    }

# ------------------------------
# SESSION STATE INIT (Unchanged)
# ------------------------------
st.session_state.setdefault("last_parsed", None)
st.session_state.setdefault("recent_patients", [])
st.session_state.setdefault("records_df", None)

# ------------------------------
# MAIN LAYOUT TABS — ADDED DASHBOARD TAB (Unchanged)
# ------------------------------
tab_dashboard, tab_text, tab_image, tab_audio, tab_summary, tab_graph, tab_records = st.tabs([
    "🏠 Dashboard",
    "✏️ Text Extract",
    "🖼️ Image Extract",
    "🎙️ Audio Extract",
    "👤 Review & Approve",
    "📊 Graph",
    "📚 Records"
])

# ===============================================================
# TAB 0 — DASHBOARD WITH ADVANCED KPIS & VIZ (Minor styling tweaks)
# ===============================================================
with tab_dashboard:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Operational Dashboard")

    if st.button("🔄 Refresh Data", help="Fetch latest data from backend", use_container_width=True):
        compute_kpis.clear()  # Clear cache for fresh data
        st.rerun()
    
    with st.spinner("Loading KPIs and Visualizations..."):
        kpis = compute_kpis()
    
    if not kpis:
        st.info("No data available. Run extractions and approvals to populate records.")
    else:
        # KPI Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Patients", kpis["total_patients"])
        col2.metric("Patients Today", kpis["patients_today"])
        col3.metric("Average Age", kpis["average_age"])
        
        st.markdown("---")
        
        # Advanced Visualizations
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.subheader("Top Symptoms")
            if kpis["top_symptoms"]:
                sym_df = pd.DataFrame(kpis["top_symptoms"], columns=["Symptom", "Count"]).set_index("Symptom")
                st.bar_chart(sym_df, use_container_width=True)
            else:
                st.info("No symptoms data.")
            
            st.subheader("Inferred Conditions")
            if kpis["condition_counts"]:
                cond_df = pd.DataFrame.from_dict(kpis["condition_counts"], orient="index", columns=["Count"]).sort_values("Count", ascending=False)
                st.bar_chart(cond_df, use_container_width=True)
            else:
                st.info("No conditions inferred.")

        with col_v2:
            st.subheader("Age Distribution")
            if not kpis["age_hist"].empty:
                st.bar_chart(kpis["age_hist"], use_container_width=True)
            else:
                st.info("No age data.")
        
        st.caption(f"Last updated: {kpis['last_updated']} (UTC). Use 'Refresh Data' for real-time updates.")

    # Quick recent patients
    st.markdown("---")
    st.subheader("Recent Patients")
    patient_list = st.session_state.get("recent_patients", [])
    resp = list_patients()
    if hasattr(resp, "ok") and resp.ok:
        srv_list = safe_json(resp).get("patients", [])
        for p in srv_list:
            if p not in patient_list:
                patient_list.append(p)
    patient_list = patient_list[:10]  # Limit to recent 10
    
    if patient_list:
        st.dataframe(pd.DataFrame({"Patient ID": patient_list}), use_container_width=True)
    else:
        st.info("No recent patients.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================================================
# TAB 1 — TEXT EXTRACTION (Unchanged)
# ===============================================================
with tab_text:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Text Extraction")
    text_input = st.text_area(
        "Clinical Notes",
        placeholder="Paste unstructured intake notes here (German/English)...",
        height=150,
        help="Enter patient details like symptoms, vitals, etc."
    )

    if st.button("Run Extraction (Text)", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Enter some text first.")
        else:
            with st.spinner("Extracting …"):
                resp = call_extract_text(text_input)
                if isinstance(resp, Exception):
                    st.error(f"Error: {resp}")
                elif not resp.ok:
                    st.error(safe_json(resp))
                else:
                    data = safe_json(resp)
                    parsed = data.get("parsed")
                    if isinstance(parsed, dict):
                        st.session_state["last_parsed"] = parsed
                        pid = parsed.get("_patient_id")
                        if pid and pid not in st.session_state["recent_patients"]:
                            st.session_state["recent_patients"].insert(0, pid)
                        st.success("Extraction complete. Proceed to Review & Approve tab.")
                        with st.expander("View Raw Response", expanded=False):
                            st.json(data)
                    else:
                        st.warning("No structured data detected.")
                        st.json(data)

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================================================
# TAB 2 — IMAGE EXTRACTION (Unchanged)
# ===============================================================
with tab_image:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Image Extraction")

    uploaded_image = st.file_uploader("Upload Scanned Form (PNG/JPG)", type=["png","jpg","jpeg"], help="Upload images of patient forms for OCR and extraction.")
    if uploaded_image:
        st.image(uploaded_image, caption="Preview", use_container_width=True)

    if st.button("Run Extraction (Image)", type="primary", use_container_width=True):
        if not uploaded_image:
            st.warning("Upload an image first.")
        else:
            with st.spinner("Processing image..."):
                resp = call_extract_image(
                    uploaded_image.getvalue(),  # Use getvalue() for fresh read
                    uploaded_image.name,
                    uploaded_image.type
                )
                if isinstance(resp, Exception):
                    st.error(f"Error: {resp}")
                elif not resp.ok:
                    st.error(safe_json(resp))
                else:
                    data = safe_json(resp)
                    parsed = data.get("parsed")
                    if isinstance(parsed, dict):
                        st.session_state["last_parsed"] = parsed
                        pid = parsed.get("_patient_id")
                        if pid and pid not in st.session_state["recent_patients"]:
                            st.session_state["recent_patients"].insert(0, pid)
                        st.success("Image extraction complete. Proceed to Review & Approve tab.")
                        with st.expander("View Raw Response (incl. OCR)", expanded=False):
                            st.json(data)
                    else:
                        st.warning("Unstructured response:")
                        st.json(data)

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================================================
# TAB 3 — AUDIO EXTRACTION (Unchanged)
# ===============================================================
with tab_audio:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Audio Extraction")

    uploaded_audio = st.file_uploader("Upload Voice Note (WAV/MP3)", type=["mp3","wav"], help="Upload audio recordings for transcription and extraction.")
    if uploaded_audio:
        st.audio(uploaded_audio)

    if st.button("Run Extraction (Audio)", type="primary", use_container_width=True):
        if not uploaded_audio:
            st.warning("Upload audio first.")
        else:
            with st.spinner("Transcribing & extracting..."):
                resp = call_extract_audio(
                    uploaded_audio.getvalue(),
                    uploaded_audio.name,
                    uploaded_audio.type
                )
                if isinstance(resp, Exception):
                    st.error(f"Error: {resp}")
                elif not resp.ok:
                    st.error(safe_json(resp))
                else:
                    data = safe_json(resp)
                    parsed = data.get("parsed")
                    if isinstance(parsed, dict):
                        st.session_state["last_parsed"] = parsed
                        pid = parsed.get("_patient_id")
                        if pid and pid not in st.session_state["recent_patients"]:
                            st.session_state["recent_patients"].insert(0, pid)
                        st.success("Audio extraction complete. Proceed to Review & Approve tab.")
                        with st.expander("View Raw Response (incl. Transcription)", expanded=False):
                            st.json(data)
                    else:
                        st.warning("Unstructured response:")
                        st.json(data)
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================================================
# TAB 4 — SUMMARY REVIEW & APPROVAL (Unchanged)
# ===============================================================
with tab_summary:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Review & Approval")

    parsed = st.session_state.get("last_parsed")
    if parsed is None:
        st.info("Run an extraction first from Text/Image/Audio tabs to load data here.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Form for editing
        with st.form(key="review_form"):
            st.markdown("### Edit Extracted Data")
            parsed = parsed.copy()

            # Input for patient details
            col1, col2, col3 = st.columns([2, 1, 1])
            parsed["patient_name"] = col1.text_input("Patient Name", parsed.get("patient_name", ""), help="Full name of the patient")
            parsed["age"] = col2.text_input("Age", parsed.get("age", ""), help="Age in years")
            parsed["duration"] = col3.text_input("Duration", parsed.get("duration", ""), help="Duration of symptoms")

            # Symptoms input
            parsed["symptoms"] = st.text_area("Symptoms (comma-separated)", parsed.get("symptoms", ""), height=80, help="List symptoms separated by commas")

            # Vital signs input (ensure we handle empty vital_keys correctly)
            vit = parsed.get("vital_signs", {})
            if isinstance(vit, dict):
                st.markdown("#### Vital Signs")
                vital_keys = list(vit.keys())

                # Handle the case when vital_keys is empty
                num_columns = max(1, min(3, len(vital_keys)))  # Ensure at least one column
                cols = st.columns(num_columns)

                # Loop through vital keys and create text inputs
                for i, k in enumerate(vital_keys):
                    vit[k] = cols[i % num_columns].text_input(k.capitalize(), vit.get(k, ""))

                # Update parsed data with the modified vital signs
                parsed["vital_signs"] = vit

            # Recommendations input
            parsed["recommendations"] = st.text_area("Recommendations", parsed.get("recommendations", ""), height=90, help="Any recommendations or notes")


            st.markdown("---")
            submit = st.form_submit_button("✅ Approve & Insert to Graph", type="primary", use_container_width=True)
            if submit:
                obj = parsed.copy()
                obj["_ingested_at"] = datetime.utcnow().isoformat() + "Z"
                with st.spinner("Writing to graph..."):
                    resp = post_graph_insert(obj)
                    if isinstance(resp, Exception):
                        st.error(f"Error: {resp}")
                    elif not resp.ok:
                        st.error(safe_json(resp))
                    else:
                        out = safe_json(resp)
                        st.success("Successfully inserted into graph.")
                        with st.expander("Insertion Details"):
                            st.json(out)
                        st.session_state["last_parsed"] = None  # Clear after approval
                        st.rerun()  # Refresh to update recent patients

        colA, colB = st.columns(2)
        if colA.button("💾 Save Edits Locally", use_container_width=True):
            st.session_state["last_parsed"] = parsed
            st.success("Edits saved to session.")

        if colB.button("🗑️ Discard Changes", use_container_width=True):
            st.session_state["last_parsed"] = None
            st.success("Session cleared.")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ===============================================================
# TAB 5 — GRAPH VIEWER (Unchanged)
# ===============================================================
with tab_graph:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Patient Knowledge Graph")

    patient_list = st.session_state.get("recent_patients", [])

    resp = list_patients()
    if hasattr(resp, "ok") and resp.ok:
        srv_list = safe_json(resp).get("patients", [])
        patient_list = list(set(patient_list + srv_list))[:50]

    if not patient_list:
        st.info("No patient data available.")
    else:
        pid = st.selectbox("Select Patient ID", [""] + patient_list, help="Choose a patient to view their graph")
        if pid:
            with st.spinner("Loading graph..."):
                gjson = safe_json(get_patient_graph(pid))
                with st.expander("Graph JSON Data", expanded=False):
                    st.json(gjson)

                gimg = get_patient_graph_image(pid)
                if hasattr(gimg, "ok") and gimg.ok:
                    b64 = safe_json(gimg).get("image_base64")
                    if b64:
                        st.image(f"data:image/png;base64,{b64}", caption="Visual Graph Representation", use_container_width=True)
                else:
                    st.warning("Graph image unavailable from backend.")

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================================================
# TAB 6 — RECORDS (Improved: full-columns, Name/ID search, duplicate handling)
# ===============================================================
with tab_records:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Historical Patient Records")

    # --- Helper (local to the tab) ------------------------------------------------
    import re

    def is_probable_id(q: str) -> bool:
        q = (q or "").strip()
        if not q:
            return False
        # heuristic: long hex/uuid-like strings or all digits or contains ':' etc
        if re.match(r'^[0-9a-fA-F\-]{8,}$', q):  # UUID-ish or long hex
            return True
        if q.isdigit() and len(q) >= 4:
            return True
        return False

    def normalize_records(arr):
        """
        Flatten a list of record dicts into a DataFrame suitable for display.
        Ensures consistent columns, converts nested lists/dicts to JSON strings,
        builds small preview fields (symptoms_preview, patient id normalized, ingested_at normalized).
        """
        rows = []
        for r in arr:
            row = dict(r)  # shallow copy to avoid mutation
            # normalize patient id
            pid = r.get("_patient_id") or r.get("patient_id") or r.get("id") or r.get("patientId")
            row["_patient_id_normalized"] = pid
            # normalize ingested_at
            ing = r.get("_ingested_at") or r.get("ingested_at") or r.get("ingestedAt") or r.get("ingested")
            row["ingested_at_normalized"] = ing

            # extract symptom preview from details list if present
            details = r.get("details") or []
            if isinstance(details, list):
                symptoms = []
                for d in details:
                    # flexible extraction of symptom name
                    props = d.get("props", {}) if isinstance(d, dict) else {}
                    name = (props.get("name") or props.get("symptom") or "").strip()
                    if not name:
                        # if detail itself contains name key
                        name = d.get("name") if isinstance(d, dict) else ""
                    if name:
                        symptoms.append(name)
                row["symptoms_preview"] = ", ".join(symptoms[:3])
                row["symptoms_full"] = ", ".join(symptoms)
            else:
                row["symptoms_preview"] = ""
                row["symptoms_full"] = ""

            # ensure there is a top-level display name
            row["patient_name_normalized"] = r.get("patient_name") or r.get("name") or r.get("full_name") or ""

            rows.append(row)

        df = pd.json_normalize(rows)

        # convert nested lists/dicts into compact JSON strings so they show as table cells
        for col in df.columns:
            sample = df[col].dropna()
            if not sample.empty:
                s0 = sample.iloc[0]
                if isinstance(s0, (list, dict)):
                    # Apply JSON.dumps to lists and dicts individually
                    df[col] = df[col].apply(lambda x: json.dumps(x, default=str) if isinstance(x, (list, dict)) else str(x))
                else:
                    # Non-list or non-dict values (like strings or numbers)
                    df[col] = df[col].apply(lambda x: json.dumps(x, default=str) if pd.notnull(x) else "")

        # sort columns with normalized/previews near the left for readability
        preferred_cols = [
            "patient_name_normalized", "_patient_id_normalized",
            "ingested_at_normalized", "age", "symptoms_preview", "symptoms_full"
        ]
        cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
        df = df[cols]
        return df

    # --- Search UI --------------------------------------------------------------
    left, mid, right = st.columns([3,1,1])
    query = left.text_input("Search (name or ID)", placeholder="Type patient name (partial ok) or exact ID...", help="Auto-detects ID vs name if set to Auto. Use 'ID' mode for exact id searches.")
    mode = mid.selectbox("Mode", options=["Auto-detect", "Name", "ID"], index=0)
    if right.button("🔍 Search", use_container_width=True, key="records_search"):
        if not query or not query.strip():
            st.warning("Enter a name or ID to search.")
        else:
            q = query.strip()
            with st.spinner("Searching records..."):
                matches = []
                search_success = True
                # If mode set to ID or auto and likely id -> filter by id across fields
                try:
                    if mode == "ID" or (mode == "Auto-detect" and is_probable_id(q)):
                        # load all and filter locally by patient id fields (safer when API lacks id endpoint)
                        resp = load_all_records()
                        if not (hasattr(resp, "ok") and resp.ok):
                            st.error(f"Error fetching records: {safe_json(resp)}")
                            search_success = False
                        else:
                            arr = safe_json(resp).get("records", [])
                            for r in arr:
                                pid = r.get("_patient_id") or r.get("patient_id") or r.get("id") or r.get("patientId")
                                if pid and str(pid).strip() == q:
                                    matches.append(r)
                    else:
                        # Name mode or auto-detect name: use search_records endpoint (partial match)
                        resp = search_records(q)
                        if not (hasattr(resp, "ok") and resp.ok):
                            st.error(f"Error searching records: {safe_json(resp)}")
                            search_success = False
                        else:
                            # backend may return results under different keys; accommodate both
                            data = safe_json(resp)
                            arr = data.get("results") or data.get("records") or data.get("matches") or []
                            matches = arr
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    search_success = False

                if search_success:
                    if not matches:
                        st.info("No matches found.")
                    else:
                        df = normalize_records(matches)
                        # Provide compact preview table
                        st.markdown("**Search results preview**")
                        st.dataframe(df, use_container_width=True)

                        # If duplicate names exist, group and let user pick a specific record
                        name_col = "patient_name_normalized" if "patient_name_normalized" in df.columns else None
                        if name_col:
                            groups = df.groupby(name_col)
                            dup_names = [name for name, g in groups if len(g) > 1 and name]
                        else:
                            dup_names = []

                        if dup_names:
                            st.info(f"Found {len(dup_names)} name(s) with multiple records. Expand to pick a specific record.")
                            for name in dup_names:
                                g = groups.get_group(name)
                                with st.expander(f"Matches for '{name}' ({len(g)} records)", expanded=False):
                                    # build labels that clearly differentiate records
                                    labels = []
                                    for i, row in g.reset_index(drop=True).iterrows():
                                        pid = row.get("_patient_id_normalized") or ""
                                        ing = row.get("ingested_at_normalized") or ""
                                        age = row.get("age") or ""
                                        preview = row.get("symptoms_preview") or ""
                                        label = f"{i+1}. {name} — ID: {pid} — Date: {ing} — Age: {age} — {preview}"
                                        labels.append(label)
                                    choice = st.radio(f"Select the exact record for '{name}'", options=labels, key=f"dup_select_{name}")
                                    sel_idx = labels.index(choice)
                                    sel_row = g.reset_index(drop=True).iloc[sel_idx]
                                    # Show actions for selected row
                                    col1, col2 = st.columns([3,1])
                                    if col1.button("📄 Show Full Record (JSON)", key=f"show_json_{name}"):
                                        st.json(json.loads(sel_row.to_json(orient="index")))
                                    if col2.download_button("⬇️ Download JSON", sel_row.to_json(orient="index").encode('utf-8'),
                                                            file_name=f"record_{sel_row.get('_patient_id_normalized') or name}_{sel_idx+1}.json",
                                                            mime="application/json"):
                                        pass

                        # Also provide single-selection list for all returned records (useful when no duplicates)
                        st.markdown("---")
                        st.markdown("### Select a record to view or download")
                        selection_labels = []
                        for i, row in df.reset_index(drop=True).iterrows():
                            name = row.get("patient_name_normalized") or "<no name>"
                            pid = row.get("_patient_id_normalized") or ""
                            ing = row.get("ingested_at_normalized") or ""
                            preview = row.get("symptoms_preview") or ""
                            label = f"{i+1}. {name} — {pid} — {ing} — {preview}"
                            selection_labels.append((label, i))  # keep index for lookup

                        selected_label = st.selectbox("Pick record", options=[l for l, _ in selection_labels], index=0)
                        sel_index = [idx for lab, idx in selection_labels if lab == selected_label][0]
                        selected_row = df.reset_index(drop=True).iloc[sel_index]

                        with st.expander("Selected record — full view", expanded=True):
                            # display nicely: pretty json, raw table for that row, download buttons
                            # Reconstruct original JSON if possible (we show normalized row)
                            st.json(json.loads(selected_row.to_json(orient="index")))
                            st.markdown("**Row preview (table)**")
                            st.table(selected_row.to_frame().T)

                            dl_col1, dl_col2 = st.columns(2)
                            # CSV download of single row
                            csv_bytes = selected_row.to_frame().T.to_csv(index=False).encode('utf-8')
                            dl_col1.download_button("⬇️ Download CSV", csv_bytes,
                                                    file_name=f"record_row_{sel_index+1}.csv",
                                                    mime="text/csv")
                            # JSON download
                            dl_col2.download_button("⬇️ Download JSON", selected_row.to_json(orient="index").encode('utf-8'),
                                                    file_name=f"record_{selected_row.get('_patient_id_normalized') or sel_index+1}.json",
                                                    mime="application/json")

    st.markdown("---")

    # --- Load All Records button (flat, full-columns view) -----------------------
    if st.button("📥 Load All Records", use_container_width=True, key="load_all_records"):
        with st.spinner("Loading all records..."):
            resp = load_all_records()
            if not (hasattr(resp, "ok") and resp.ok):
                st.error(f"Error fetching records: {safe_json(resp)}")
            else:
                arr = safe_json(resp).get("records", [])
                if not arr:
                    st.info("No records found.")
                else:
                    df_all = normalize_records(arr)
                    # store in session for reuse
                    st.session_state["records_df"] = df_all

                    st.markdown("**All records (flattened)**")
                    st.dataframe(df_all, use_container_width=True)

                    # download full CSV
                    csv = df_all.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download All Records (CSV)",
                        csv,
                        file_name=f"medform_records_{datetime.utcnow().date()}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    # If there's a cached dataframe, show it and let user filter/sort a bit
    if st.session_state.get("records_df") is not None:
        st.markdown("---")
        st.markdown("**Cached records (from last load)**")
        df_cache = st.session_state["records_df"]
        st.dataframe(df_cache, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
