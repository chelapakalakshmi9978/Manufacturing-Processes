"""
RAG (Retrieval-Augmented Generation) Module
Process Data Retrieval — provides context documents to IBM Granite agents.
Simulates a vector store of manufacturing knowledge documents.
"""

import json
import math
from datetime import datetime
from granite_client import GraniteClient


# ---------------------------------------------------------------------------
# Manufacturing Knowledge Base (simulated document corpus)
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    {
        "id": "KB-001",
        "title": "CNC Machining Temperature Standards",
        "category": "process_standards",
        "content": (
            "CNC machining processes require maintaining workpiece temperature between 180–220°C for optimal "
            "material removal rates. Temperatures above 220°C cause thermal expansion, leading to dimensional "
            "inaccuracies of ±0.05 mm or more. Coolant flow must be maintained at 8–12 L/min. "
            "Thermal runaway is the leading cause of scrap parts in precision machining."
        ),
        "tags": ["temperature", "cnc", "dimensional accuracy"],
    },
    {
        "id": "KB-002",
        "title": "Vibration Analysis in Rotating Equipment",
        "category": "maintenance",
        "content": (
            "Acceptable vibration levels for CNC spindles and lathes are below 2.5 mm/s RMS. "
            "Vibration above 3.0 mm/s indicates bearing wear, tool imbalance, or fixture looseness. "
            "High vibration directly causes surface roughness defects (Ra > 1.6 μm) and reduces tool life by up to 40%. "
            "ISO 10816 defines vibration severity zones for industrial machinery."
        ),
        "tags": ["vibration", "bearings", "surface roughness", "maintenance"],
    },
    {
        "id": "KB-003",
        "title": "Hydraulic Press Pressure Control",
        "category": "process_standards",
        "content": (
            "Press forming requires consistent hydraulic pressure between 95–105 bar for structural integrity. "
            "Pressure fluctuations > ±5 bar cause material deformation and micro-cracks in formed parts. "
            "Pressure must be monitored at 10 Hz minimum. Pressure relief valve settings should be verified every 8 hours. "
            "Hydraulic oil temperature must remain below 60°C to maintain viscosity."
        ),
        "tags": ["pressure", "press", "structural weakness", "hydraulic"],
    },
    {
        "id": "KB-004",
        "title": "Surface Finish Quality Standards",
        "category": "quality_standards",
        "content": (
            "Surface roughness (Ra) must not exceed 1.6 μm for precision components per ISO 1302. "
            "Factors affecting surface finish: cutting speed, feed rate, tool geometry, and vibration. "
            "Ra values above 2.0 μm indicate worn tooling or incorrect cutting parameters. "
            "Regular profilometer measurements every 50 parts are recommended for process control."
        ),
        "tags": ["surface roughness", "quality", "tooling", "ISO 1302"],
    },
    {
        "id": "KB-005",
        "title": "Material Properties: Steel A36",
        "category": "materials",
        "content": (
            "ASTM A36 structural steel: Tensile strength 400–550 MPa, Yield strength ≥ 250 MPa. "
            "Optimal machining temperature range: 15–35°C ambient. "
            "Susceptible to work hardening if feed rates are too low. "
            "Requires carbide tooling at speeds above 150 m/min. "
            "Dimensional tolerance achievable: ±0.025 mm with proper fixturing."
        ),
        "tags": ["steel", "material", "A36", "machining"],
    },
    {
        "id": "KB-006",
        "title": "Humidity Effects on Manufacturing",
        "category": "environmental",
        "content": (
            "Relative humidity above 60% causes corrosion on ferrous materials and measurement errors in CMM inspection. "
            "Humidity below 30% causes static discharge risks and material brittleness in polymers. "
            "Optimal shop floor humidity: 40–55% RH. "
            "Hygroscopic materials like nylon and ABS must be dried before machining."
        ),
        "tags": ["humidity", "environment", "corrosion", "measurement"],
    },
    {
        "id": "KB-007",
        "title": "Defect Classification and Root Cause Analysis",
        "category": "quality_standards",
        "content": (
            "Manufacturing defect taxonomy: "
            "(1) Dimensional defects — caused by thermal expansion, tool wear, fixture play; "
            "(2) Surface defects — caused by vibration, worn tools, incorrect cutting fluid; "
            "(3) Structural defects — caused by material inconsistency, incorrect heat treatment, overloading; "
            "(4) Cosmetic defects — surface marks, burrs, scratches from handling. "
            "Root cause analysis should follow the 5-Why methodology."
        ),
        "tags": ["defects", "root cause", "quality", "classification"],
    },
    {
        "id": "KB-008",
        "title": "Predictive Maintenance Intervals",
        "category": "maintenance",
        "content": (
            "Recommended maintenance intervals for precision machining equipment: "
            "Daily — check coolant level, spindle runout, fixture torque; "
            "Weekly — replace cutting tools, check belt tension, calibrate sensors; "
            "Monthly — full spindle inspection, hydraulic fluid analysis, CMM calibration; "
            "Annually — full machine rebuild and alignment verification per manufacturer spec."
        ),
        "tags": ["maintenance", "schedule", "spindle", "calibration"],
    },
    {
        "id": "KB-009",
        "title": "Speed and Torque Optimization",
        "category": "process_standards",
        "content": (
            "Optimal spindle speed for steel: 800–1200 RPM for rough cutting, 1500–2500 RPM for finishing. "
            "Torque limits for CNC spindles: 45–75 Nm. Exceeding torque limit by > 20% risks spindle motor damage. "
            "Speed deviations > 10% from setpoint indicate drive belt slippage or encoder fault. "
            "Adaptive feed control systems reduce torque spikes by up to 30%."
        ),
        "tags": ["speed", "torque", "spindle", "cnc", "optimization"],
    },
    {
        "id": "KB-010",
        "title": "Statistical Process Control (SPC) Guidelines",
        "category": "quality_standards",
        "content": (
            "SPC control chart rules (Western Electric): A process is out of control when: "
            "(1) One point beyond 3σ; (2) Two of three consecutive points beyond 2σ; "
            "(3) Four of five consecutive points beyond 1σ; (4) Eight consecutive points on one side of center. "
            "Process Capability Index (Cpk) must be ≥ 1.33 for production release. "
            "Gauge R&R studies must show measurement system variation < 10% of tolerance."
        ),
        "tags": ["SPC", "control charts", "Cpk", "statistics", "quality"],
    },
]


class RAGRetriever:
    """
    Retrieval-Augmented Generation system for manufacturing knowledge.
    Uses TF-IDF-like keyword scoring for document retrieval.
    """

    def __init__(self):
        self.documents = KNOWLEDGE_BASE
        self.granite = GraniteClient()
        self._build_index()
        print(f"[RAGRetriever] Loaded {len(self.documents)} knowledge documents.")

    def _build_index(self):
        """Build a simple inverted index for fast keyword retrieval."""
        self.index = {}
        for doc in self.documents:
            words = set(
                (doc["title"] + " " + doc["content"] + " " + " ".join(doc["tags"])).lower().split()
            )
            for word in words:
                self.index.setdefault(word, []).append(doc["id"])

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Retrieve top-k relevant documents for a given query."""
        query_words = query.lower().split()
        scores = {}
        for word in query_words:
            for doc_id in self.index.get(word, []):
                scores[doc_id] = scores.get(doc_id, 0) + 1

        # Sort by relevance score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        doc_map = {d["id"]: d for d in self.documents}
        results = [doc_map[doc_id] for doc_id, _ in ranked[:top_k] if doc_id in doc_map]
        return results

    def retrieve_for_reading(self, reading: dict) -> list:
        """Auto-retrieve relevant documents for a manufacturing sensor reading."""
        out_of_spec = reading.get("out_of_spec", [])
        machine = reading.get("machine_id", "")
        material = reading.get("material", "")
        query = " ".join(out_of_spec + [machine, material])
        return self.retrieve(query, top_k=3)

    def generate_rag_report(self, reading: dict, analysis: dict, prediction: dict) -> str:
        """Use RAG + IBM Granite to generate a knowledge-augmented quality report."""
        docs = self.retrieve_for_reading(reading)
        context = "\n\n".join(
            f"[{d['title']}]\n{d['content']}" for d in docs
        )

        prompt = f"""You are a Manufacturing Quality Expert AI. Use the following knowledge base excerpts to provide an expert analysis.

KNOWLEDGE BASE:
{context}

CURRENT PRODUCTION STATUS:
- Machine: {reading.get('machine_id')} | Batch: {reading.get('batch_id')}
- Out-of-spec parameters: {', '.join(reading.get('out_of_spec', ['None']))}
- Quality severity: {analysis.get('severity', 0):.1%}
- Top predicted defect: {prediction.get('top_defect', 'None')} ({prediction.get('top_defect_probability', 0):.1%} probability)
- Process forecast: {prediction.get('forecast', {}).get('direction', 'unknown')}

Based on the knowledge base and current data, provide a comprehensive quality control report (5–6 sentences) covering:
1. Root cause analysis referencing relevant standards
2. Defect risk assessment
3. Specific corrective actions with technical details
4. Compliance and quality standard considerations"""

        return self.granite.generate(prompt, max_tokens=700)

    def get_all_documents(self) -> list:
        return self.documents
