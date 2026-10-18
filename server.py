"""
Flask API Server — serves the dashboard and exposes agent data as JSON endpoints.
"""

import threading
import time
from flask import Flask, jsonify, render_template_string, send_from_directory
from flask_cors import CORS

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, MONITORING_INTERVAL
from agents.monitoring_agent import ProcessMonitoringAgent
from agents.analysis_agent import QualityAnalysisAgent
from agents.prediction_agent import DefectPredictionAgent
from agents.optimization_agent import ProcessOptimizationAgent
from rag_retriever import RAGRetriever

app = Flask(__name__, static_folder="static")
CORS(app)

# ---------------------------------------------------------------------------
# Agent instances (shared state)
# ---------------------------------------------------------------------------
monitor_agent = ProcessMonitoringAgent()
analysis_agent = QualityAnalysisAgent()
prediction_agent = DefectPredictionAgent()
optimization_agent = ProcessOptimizationAgent()
rag_retriever = RAGRetriever()

# Shared state updated by background thread
_latest_state = {}
_state_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Background monitoring loop
# ---------------------------------------------------------------------------

def _ai_pipeline(reading, history):
    """Run the AI agents (may block on Granite API) in its own thread."""
    try:
        analysis = analysis_agent.analyze(reading, history)
        prediction = prediction_agent.predict(reading, history, analysis)
        optimization = optimization_agent.optimize(reading, analysis, prediction)
        with _state_lock:
            _latest_state.update({
                "analysis": analysis,
                "prediction": prediction,
                "optimization": optimization,
            })
    except Exception as exc:
        print(f"[AIPipeline] Error: {exc}")


def _run_monitoring_loop():
    """
    Background thread: collects sensor data every MONITORING_INTERVAL seconds.
    The AI pipeline (Granite calls) runs in a separate thread so slow API
    responses never block sensor collection or dashboard updates.
    """
    print("[Server] Background monitoring loop started.")
    while True:
        try:
            reading = monitor_agent.collect_data()
            history = monitor_agent.get_history()

            # Always update reading + history immediately (no Granite wait)
            with _state_lock:
                _latest_state["reading"] = reading
                _latest_state["history"] = history[-30:]
                # Seed analysis/prediction/optimization with defaults on first run
                if "analysis" not in _latest_state:
                    _latest_state["analysis"] = {
                        "risk_level": "LOW", "severity": 0.0, "anomalies": [],
                        "out_of_spec_count": len(reading.get("out_of_spec", [])),
                        "narrative": "IBM Granite analysis pending...", "trend_analysis": {},
                    }
                if "prediction" not in _latest_state:
                    _latest_state["prediction"] = {
                        "overall_risk": 0.0, "risk_category": "LOW", "top_defect": "None",
                        "top_defect_probability": 0.0, "defect_scores": {},
                        "forecast": {"direction": "stable", "confidence": 0.7, "horizon": "next 5 readings"},
                        "narrative": "IBM Granite prediction pending...",
                    }
                if "optimization" not in _latest_state:
                    _latest_state["optimization"] = {
                        "priority": "LOW", "immediate_actions": [], "parameter_adjustments": {},
                        "preventive_maintenance": [], "narrative": "IBM Granite optimization pending...",
                    }

            # Fire AI pipeline in background — doesn't block sensor collection
            ai_thread = threading.Thread(target=_ai_pipeline, args=(reading, list(history)), daemon=True)
            ai_thread.start()

        except Exception as exc:
            print(f"[MonitoringLoop] Error: {exc}")
        time.sleep(MONITORING_INTERVAL)


# ---------------------------------------------------------------------------
# REST API endpoints
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    with _state_lock:
        state = dict(_latest_state)
    if not state or "reading" not in state:
        return jsonify({"status": "initializing"}), 202
    return jsonify({
        "status": "ok",
        "tick": state["reading"].get("tick"),
        "timestamp": state["reading"].get("timestamp"),
        "machine_id": state["reading"].get("machine_id"),
        "batch_id": state["reading"].get("batch_id"),
        "risk_level": state["analysis"].get("risk_level"),
        "severity": state["analysis"].get("severity"),
        "out_of_spec_count": state["analysis"].get("out_of_spec_count"),
        "top_defect": state["prediction"].get("top_defect"),
        "top_defect_probability": state["prediction"].get("top_defect_probability"),
        "overall_risk": state["prediction"].get("overall_risk"),
        "optimization_priority": state["optimization"].get("priority"),
        "production_rate": state["reading"].get("production_rate"),
    })


@app.route("/api/reading")
def api_reading():
    with _state_lock:
        return jsonify(_latest_state.get("reading", {}))


@app.route("/api/analysis")
def api_analysis():
    with _state_lock:
        return jsonify(_latest_state.get("analysis", {}))


@app.route("/api/prediction")
def api_prediction():
    with _state_lock:
        return jsonify(_latest_state.get("prediction", {}))


@app.route("/api/optimization")
def api_optimization():
    with _state_lock:
        return jsonify(_latest_state.get("optimization", {}))


@app.route("/api/history")
def api_history():
    with _state_lock:
        history = _latest_state.get("history", [])
    # Return lightweight history for charting
    chart_data = []
    for h in history:
        params = h.get("parameters", {})
        chart_data.append({
            "tick": h.get("tick"),
            "timestamp": h.get("timestamp"),
            "temperature": params.get("temperature"),
            "pressure": params.get("pressure"),
            "vibration": params.get("vibration"),
            "speed": params.get("speed"),
            "torque": params.get("torque"),
            "surface_roughness": params.get("surface_roughness"),
            "thickness": params.get("thickness"),
            "out_of_spec_count": len(h.get("out_of_spec", [])),
        })
    return jsonify(chart_data)


@app.route("/api/analysis/history")
def api_analysis_history():
    data = analysis_agent.get_history()
    return jsonify([{
        "timestamp": d.get("timestamp"),
        "risk_level": d.get("risk_level"),
        "severity": d.get("severity"),
        "out_of_spec_count": d.get("out_of_spec_count"),
        "anomaly_count": len(d.get("anomalies", [])),
    } for d in data])


@app.route("/api/prediction/history")
def api_prediction_history():
    data = prediction_agent.get_history()
    return jsonify([{
        "timestamp": d.get("timestamp"),
        "overall_risk": d.get("overall_risk"),
        "risk_category": d.get("risk_category"),
        "top_defect": d.get("top_defect"),
        "defect_scores": d.get("defect_scores", {}),
    } for d in data])


@app.route("/api/rag/report")
def api_rag_report():
    with _state_lock:
        state = dict(_latest_state)
    if not state:
        return jsonify({"report": "System initializing…"}), 202
    report = rag_retriever.generate_rag_report(
        state["reading"], state["analysis"], state["prediction"]
    )
    docs = rag_retriever.retrieve_for_reading(state["reading"])
    return jsonify({
        "report": report,
        "source_documents": [{"id": d["id"], "title": d["title"], "category": d["category"]} for d in docs],
    })


@app.route("/api/knowledge")
def api_knowledge():
    docs = rag_retriever.get_all_documents()
    return jsonify([{"id": d["id"], "title": d["title"], "category": d["category"], "tags": d["tags"]} for d in docs])


@app.route("/api/machines")
def api_machines():
    return jsonify(monitor_agent.get_machine_status())


@app.route("/")
def index():
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------

def start_server():
    # Start background monitoring thread
    t = threading.Thread(target=_run_monitoring_loop, daemon=True)
    t.start()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG, use_reloader=False)


if __name__ == "__main__":
    start_server()
