import os
import io
import uuid
import json
import base64
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Optional but recommended
import pytesseract
from PIL import Image
import whisper

load_dotenv()

# =========================
# Config
# =========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY missing in .env")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Load Whisper once (optional – falls back to text-only if not available)
whisper_model = None
try:
    whisper_model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))
except Exception as e:
    print(f"[WARN] Whisper not available: {e}")

# =========================
# FastAPI App
# =========================
app = FastAPI(title="MedForm AI Backend", version="2.0")

# =========================
# Medical condition mapping (extend as needed)
# =========================
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

# =========================
# LLM Call via OpenRouter (with vision support)
# =========================
import requests

MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "mistralai/pixtral-12b:free",
    "anthropic/claude-3-haiku",
    "x-ai/grok-4.1-fast:free",
]

def call_llm(text: str, image_b64: str | None = None) -> Dict[str, Any]:
    prompt = f"""You are an expert medical data extractor for German and English patient records.
Extract EXACTLY these fields as JSON only (no explanations, no markdown unless it's inside the JSON):

{{
  "patient_name": str or null,
  "age": str or null,
  "duration": str or null,
  "symptoms": str or null,
  "vital_signs": {{}} or null,
  "recommendations": str or null
}}

Use null if information is missing. Be very precise. Respond with valid JSON only.

Text/Form content:
{text}
"""

    content: list = [{"type": "text", "text": prompt}]
    if image_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
        })

    for model in MODELS:
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                },
                timeout=90,
            )
            if resp.status_code == 200:
                data = resp.json()
                raw = data["choices"][0]["message"]["content"]
                return {"model": model, "raw": raw.strip()}
        except:
            continue
    raise HTTPException(502, "All LLM providers failed")

# =========================
# Parsing & Normalizing
# =========================
def parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
    return {"error": "Invalid JSON from LLM", "raw": raw}

def normalize_extraction(data: dict) -> dict:
    if "error" in data:
        base = {k: None for k in ["patient_name","age","duration","symptoms","vital_signs","recommendations"]}
        base["_raw"] = data.get("raw")
    else:
        base = {
            "patient_name": data.get("patient_name"),
            "age": data.get("age"),
            "duration": data.get("duration"),
            "symptoms": data.get("symptoms"),
            "vital_signs": data.get("vital_signs") or {},
            "recommendations": data.get("recommendations"),
        }

    # Generate patient ID if missing
    base["_patient_id"] = data.get("_patient_id") or f"pat-{uuid.uuid4().hex[:10]}"
    base["_ingested_at"] = datetime.utcnow().isoformat() + "Z"
    
    # Symptom list
    sym = base["symptoms"]
    if isinstance(sym, str):
        sym = [s.strip() for s in sym.replace(";", ",").split(",") if s.strip()]
    elif not isinstance(sym, list):
        sym = []
    base["_symptom_list"] = sym

    return base

# =========================
# Graph Storage
# =========================
def insert_into_graph(data: dict):
    pid = data["_patient_id"]
    name = data["patient_name"] or "Unbekannt"

    with driver.session() as session:
        # Patient node
        session.run("""
            MERGE (p:Patient {id: $pid})
            SET p.name = $name,
                p.age = $age,
                p.ingested_at = $ingested_at
        """, pid=pid, name=name, age=data["age"], ingested_at=data["_ingested_at"])

        # Duration
        if data["duration"]:
            session.run("""
                MERGE (d:Duration {value: $val})
                WITH d MATCH (p:Patient {id: $pid})
                MERGE (p)-[:HAS_DURATION]->(d)
            """, val=data["duration"], pid=pid)

        # Vital signs
        if data["vital_signs"] and isinstance(data["vital_signs"], dict):
            for k, v in data["vital_signs"].items():
                if v is not None:
                    session.run("""
                        MERGE (v:Vital {name: $name})
                        ON CREATE SET v.value = $value
                        ON MATCH SET v.value = $value
                        WITH v MATCH (p:Patient {id: $pid})
                        MERGE (p)-[:HAS_VITAL]->(v)
                    """, name=k.title(), value=str(v), pid=pid)

        # Recommendations
        if data["recommendations"]:
            session.run("""
                MERGE (r:Recommendation {text: $text})
                WITH r MATCH (p:Patient {id: $pid})
                MERGE (p)-[:HAS_RECOMMENDATION]->(r)
            """, text=data["recommendations"], pid=pid)

        # Symptoms + auto-condition mapping
        for symptom in data["_symptom_list"]:
            norm = symptom.strip().lower()
            session.run("""
                MERGE (s:Symptom {name: $name})
                WITH s MATCH (p:Patient {id: $pid})
                MERGE (p)-[:HAS_SYMPTOM]->(s)
            """, name=norm, pid=pid)

            condition = MEDICAL_MAP.get(norm)
            if condition:
                session.run("""
                    MERGE (c:Condition {name: $cname})
                    WITH c
                    MATCH (s:Symptom {name: $sname})
                    MERGE (s)-[:INDICATES]->(c)
                """, cname=condition, sname=norm)

    return {"patient_id": pid, "status": "inserted"}

# =========================
# Endpoints
# =========================

@app.post("/extract/text")
async def extract_text(text: str = Form(...)):
    llm = call_llm(text)
    parsed = parse_json_response(llm["raw"])
    normalized = normalize_extraction(parsed)
    graph = insert_into_graph(normalized)
    return {"parsed": normalized, "model": llm["model"], "graph": graph}

@app.post("/extract/image")
async def extract_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    ocr_text = pytesseract.image_to_string(img, lang="deu+eng")
    b64 = base64.b64encode(contents).decode()

    llm = call_llm(ocr_text or " ", image_b64=b64)
    parsed = parse_json_response(llm["raw"])
    normalized = normalize_extraction(parsed)
    graph = insert_into_graph(normalized)

    return {"parsed": normalized, "ocr_text": ocr_text, "model": llm["model"], "graph": graph}

@app.post("/extract/audio")
async def extract_audio(file: UploadFile = File(...)):
    if not whisper_model:
        raise HTTPException(503, "Whisper model not loaded")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(contents)
        f.close()
        result = whisper_model.transcribe(f.name, language="de")
        os.unlink(f.name)

    transcription = result["text"]
    llm = call_llm(transcription)
    parsed = parse_json_response(llm["raw"])
    normalized = normalize_extraction(parsed)
    graph = insert_into_graph(normalized)

    return {"parsed": normalized, "transcription": transcription, "model": llm["model"], "graph": graph}

@app.post("/graph/insert")
async def graph_insert(data: dict):
    normalized = normalize_extraction(data)
    graph = insert_into_graph(normalized)
    return {"parsed": normalized, "graph": graph}

@app.get("/patients/list")
def list_patients():
    with driver.session() as session:
        res = session.run("MATCH (p:Patient) RETURN p.id AS id ORDER BY p.ingested_at DESC")
        return {"patients": [r["id"] for r in res]}

@app.get("/graph/patient/{pid}")
def get_patient_graph(pid: str):
    with driver.session() as session:
        res = session.run("""
            MATCH (p:Patient {id: $pid})-[r]->(n)
            RETURN type(r) AS rel, labels(n)[0] AS type, properties(n) AS props
        """, pid=pid)
        nodes = [dict(record) for record in res]
        if not nodes:
            raise HTTPException(404, "Patient not found")
        return {"patient_id": pid, "graph": nodes}

@app.get("/graph/patient/{pid}/image")
def get_patient_graph_image(pid: str):
    import matplotlib.pyplot as plt
    import networkx as nx
    import textwrap

    with driver.session() as session:
        result = session.run("""
            MATCH (p:Patient {id: $pid})-[r]->(n)
            RETURN p.name AS name, type(r) AS rel, labels(n)[0] AS label, properties(n) AS props
        """, pid=pid)

        records = list(result)
        if not records:
            raise HTTPException(404, "Patient not found")

        patient_name = (records[0]["name"] or pid)[:40]

        # --- Graph ---
        G = nx.DiGraph()

        # Define a readable label for the patient
        G.add_node(
            pid,
            label=f"Patient\n{patient_name}",
            ntype="Patient",
        )

        # Color palette
        color_map = {
            "Patient":       "#b3e6ff",
            "Symptom":       "#ff9999",
            "Vital":         "#ffd480",
            "Condition":     "#cc99ff",
            "Duration":      "#99ffcc",
            "Recommendation":"#ffff99",
        }

        # Helper: safe text wrapping
        def wrap(text, width=20):
            if not text:
                return "?"
            return "\n".join(textwrap.wrap(str(text), width=width))

        # Create child nodes
        for r in records:
            kind = r["label"]
            props = r["props"]
            rel = r["rel"]

            # Determine node id + text
            if kind == "Vital":
                key = props.get("name", "")
                val = props.get("value", "")
                node_id = f"Vital:{key}"
                label = f"{key}:\n{val}"
            elif kind == "Symptom":
                name = props.get("name", "")
                node_id = f"Symptom:{name}"
                label = name
            elif kind == "Condition":
                name = props.get("name", "")
                node_id = f"Condition:{name}"
                label = name
            elif kind == "Duration":
                val = props.get("value", "")
                node_id = f"Duration:{val}"
                label = val
            elif kind == "Recommendation":
                text = props.get("text", "")
                node_id = f"Rec:{text[:30]}"
                label = text
            else:
                node_id = f"{kind}:{str(props)[:20]}"
                label = str(props)

            G.add_node(
                node_id,
                label=wrap(label),
                ntype=kind,
            )
            G.add_edge(pid, node_id, label=rel)

        # --- Draw graph ---
        plt.figure(figsize=(14, 10))
        
        # Try GraphViz layout if available; fallback to spring layout.
        try:
            pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        except:
            pos = nx.spring_layout(G, k=1.2, iterations=50)

        # Extract color list
        node_colors = [
            color_map.get(G.nodes[n].get("ntype", "Patient"), "#e6f7ff")
            for n in G.nodes()
        ]

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=3000,
            edgecolors="black"
        )

        # Draw labels
        labels = {n: G.nodes[n]["label"] for n in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels, font_size=9, font_weight="bold"
        )

        # Draw edges
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", width=1.5)

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            label_pos=0.5
        )

        # Export image
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=200)
        plt.close()

        return {
            "image_base64": base64.b64encode(buf.getvalue()).decode()
        }



@app.get("/records/all")
def all_records():
    with driver.session() as session:
        res = session.run("""
            MATCH (p:Patient)
            OPTIONAL MATCH (p)-[r]->(n)
            RETURN p.id AS patient_id, p.name AS name, type(r) AS rel, n, p.ingested_at as ingested_at
        """)
        data = {}
        for r in res:
            pid = r["patient_id"]
            if pid not in data:
                data[pid] = {
                    "patient_id": pid,
                    "name": r["name"],
                    "ingested_at": r["ingested_at"],
                    "records": []
                }

            if r["n"] is not None:
                data[pid]["records"].append({
                    "rel": r["rel"],
                    "node": dict(r["n"])
                })

        # Sort by ingestion time
        sorted_records = sorted(
            data.values(),
            key=lambda p: p.get("ingested_at") or "",
            reverse=True
        )
        return {"records": sorted_records}

@app.get("/records/search")
def search_records(name: str | None = None):
    if not name:
        raise HTTPException(400, "Parameter 'name' required")
    with driver.session() as session:
        res = session.run("""
            MATCH (p:Patient)
            WHERE toLower(p.name) CONTAINS toLower($name)
            RETURN p.id, p.name
            LIMIT 50
        """, name=name)
        return {"results": [dict(r) for r in res]}

@app.get("/")
def root():
    return {"status": "MedForm AI Backend v2 – Ready", "date": datetime.now(timezone.utc).isoformat()}


