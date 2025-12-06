# MedForm AI 🩺

**Full‑Stack Medical Data Extraction & Knowledge Graph (Streamlit + FastAPI + Neo4j)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x‑Bolt‑URI-green)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

MedForm AI is a **comprehensive platform for structured medical data extraction**. It ingests clinical notes from **text, scanned forms, or voice recordings**, uses an LLM-powered extraction pipeline, optionally enhanced with OCR or speech-to-text models, and persists structured data in **Neo4j as a semantic knowledge graph**. A staff-facing **Streamlit console** allows review, approval, visualization, and management of patient records.

* **Frontend Dashboard:**
  ![Frontend Dashboard](https://github.com/suraj5424/Medicalform-multimodal-agent-fastapi-neo4j-streamlit/blob/main/Dashboard.png)

---

## 1. Project Overview

* Accepts multiple input modalities: **text**, **images**, or **audio recordings**.
* Extracts structured fields via LLMs: patient name, age, symptoms, vitals, recommendations, and other medical data.
* Normalizes and stores extracted data as **graph nodes and relationships** in Neo4j.
* Provides API endpoints for querying data and rendering interactive **graph visualizations**.
* Staff console for **reviewing, editing, approving, and managing** historical records.

---

## 2. Architecture

**High-level workflow:**

1. Staff / Clinician / Data Entry Staff interact with the **Streamlit Frontend**:

   * Upload text → `/extract/text`
   * Upload scanned images → `/extract/image` (OCR)
   * Upload voice notes → `/extract/audio` (transcription)
   * Review & Approve extracted data → `/graph/insert`
   * Visualize patient graph → `/graph/patient/{pid}` & `/graph/patient/{pid}/image`
   * Access dashboard and historical records → `/records/*` or `/patients/list`

2. **FastAPI backend** processes extraction, normalization, and graph insertion.

3. **Neo4j database** stores patient records as nodes and relationships for semantic querying.

**Tech stack:** Python 3.10+, FastAPI, Streamlit, Neo4j 5.x, Tesseract OCR, Whisper (optional), matplotlib & NetworkX for graph rendering, requests, python-dotenv.

---

## 3. Key Features ✨

**Multi-modal extraction**

* Free-text clinical notes
* Scanned forms processed via OCR
* Voice notes transcribed via Whisper or alternative STT models

**Structured JSON extraction via LLM**

* Schema enforced: `patient_name`, `age`, `duration`, `symptoms`, `vital_signs`, `recommendations`
* Backend normalization: `_patient_id`, `_ingested_at` (timestamp), `_symptom_list`

**Graph-based knowledge storage**

* Nodes: Patient, Symptom, Vital, Condition, Duration, Recommendation
* Relationships: HAS_SYMPTOM, HAS_VITAL, HAS_RECOMMENDATION, HAS_DURATION, INDICATES
* Idempotent inserts: repeated submissions update existing nodes rather than creating duplicates

**Staff console**

* Dashboard with KPIs, symptom distribution, and age statistics
* Extraction preview, editing, and approval workflow
* Graph viewer with JSON and PNG visualization
* Export data in CSV or JSON format

**Extensible & maintainable architecture**

* Modular backend for extraction, normalization, graph storage, and frontend
* Clear separation of concerns enables easy extension for additional medical mappings or new input modalities

---

## 4. Prerequisites & Setup

**System requirements:**

* Python 3.10+
* Tesseract OCR with required language packs (`eng` recommended, `deu` optional)
* `ffmpeg` for audio transcription
* Running Neo4j instance accessible from backend

**Environment variables (`.env`)**

* `OPENROUTER_API_KEY` = your OpenRouter API key
* `NEO4J_URI` = bolt://localhost:7687 (or your Neo4j URI)
* `NEO4J_USER` = neo4j (or your username)
* `NEO4J_PASS` = your password
* `WHISPER_MODEL` = base (optional for audio transcription)

> Keep `.env` out of version control to secure API keys and credentials.

---

## 5. Running Locally (Development)

1. Clone the repository.
2. Install dependencies for backend and frontend from the code files.
3. Ensure Neo4j, Tesseract and ffmpeg are installed and properly configured.
4. Start the backend: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`.
5. Start the frontend: navigate to `frontend` folder → `streamlit run app.py`.
6. Set **API URL** in the Streamlit sidebar (default: `http://localhost:8501`).

You should now have access to the full extraction, review, graph visualization, and record management functionality.

---

## 6. API Endpoints

| Path                         | Method | Description                                            |
| ---------------------------- | ------ | ------------------------------------------------------ |
| `/extract/text`              | POST   | Parse free-text clinical notes                         |
| `/extract/image`             | POST   | Upload image → OCR + parsing                           |
| `/extract/audio`             | POST   | Upload audio → transcription + parsing                 |
| `/graph/insert`              | POST   | Insert or update approved structured data in Neo4j     |
| `/patients/list`             | GET    | Retrieve list of patient IDs (recent first)            |
| `/graph/patient/{pid}`       | GET    | Get JSON representation of patient graph               |
| `/graph/patient/{pid}/image` | GET    | Get base64 PNG rendering of patient graph              |
| `/records/all`               | GET    | Retrieve flattened historical records for all patients |
| `/records/search?name=…`     | GET    | Search patients by name (partial or case-insensitive)  |

---

## 7. Data Model & Normalization

* **Extraction schema from LLM:** `patient_name`, `age`, `duration`, `symptoms`, `vital_signs`, `recommendations`
* **Normalized fields added by backend:** `_patient_id`, `_ingested_at`, `_symptom_list`
* **Graph nodes:** Patient, Symptom, Vital, Condition, Duration, Recommendation
* **Relationships:** HAS_SYMPTOM, HAS_VITAL, HAS_RECOMMENDATION, HAS_DURATION, INDICATES (symptom → condition)
* Idempotent merge ensures repeated submissions update existing data instead of creating duplicates

---

## 8. Security & Privacy 🔐

* Keep `.env` and API keys secret.
* Use HTTPS in production and implement authentication and role-based access control.
* For real patient data, comply with GDPR/HIPAA: consider encryption, pseudonymization, audit logging, and data retention policies.

---

## 9. Troubleshooting & Common Issues

| Issue                        | Solution                                                                           |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| Missing `OPENROUTER_API_KEY` | Ensure `.env` exists and contains a valid API key                                  |
| OCR errors                   | Verify Tesseract is installed and language packs match (`eng`, `deu`)              |
| Audio extraction fails       | Ensure `ffmpeg` is installed, WHISPER_MODEL configured, and model loaded correctly |
| Neo4j connection fails       | Check `NEO4J_URI`, credentials, and network connectivity                           |
| Invalid JSON from LLM        | Inspect raw response; adjust prompt or fallback to alternative model               |

---

## 10. Images & Screenshots 🖼️
* **Patient Knowledge Graph (matplotlib/NetworkX):**
![Patient Knowledge Graph ](https://github.com/suraj5424/Medicalform-multimodal-agent-fastapi-neo4j-streamlit/blob/main/image.png)

Tip: decode base64 output from `/graph/patient/{pid}/image` and save as PNG for visualization.

---

## 11. Contributing & Next Steps 🚀

* Add **unit tests** for parsing, normalization, and graph insertion.
* Improve LLM prompts or expand **medical mapping rules** to cover more symptoms and conditions.
* Harden production deployment with authentication, HTTPS, and audit logging.
* Add a **CI pipeline**: linting, type checking, tests.
* Extend schema for additional medical metadata, structured vitals, demographics, or event history.

---

## 12. License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
