"""
Manufacturing Process Quality Control Agent
Main entry point — starts the multi-agent system and web server.

IBM Granite AI · IBM Watsonx · Project ID: 3cb648e7-2430-4e25-86ab-bc51d756ef68
"""

import os
import sys
import time
from colorama import init, Fore, Style

init(autoreset=True)

BANNER = f"""
{Fore.CYAN}+==================================================================+
|   Manufacturing Process Quality Control System                   |
|   Powered by IBM Granite AI - IBM Watsonx                        |
|   Project: 3cb648e7-2430-4e25-86ab-bc51d756ef68                  |
+==================================================================+{Style.RESET_ALL}
"""

AGENT_INFO = [
    ("Process Monitoring Agent",     "Collects real-time sensor & machine data"),
    ("Quality Analysis Agent",       "Detects anomalies with IBM Granite insights"),
    ("Defect Prediction Agent",      "Predicts defects using ML + Granite reasoning"),
    ("Process Optimization Agent",   "Suggests corrective actions via Granite"),
    ("RAG Retriever",                "Retrieves manufacturing knowledge for reports"),
]


def print_startup():
    print(BANNER)
    print(f"{Fore.YELLOW}  Initializing agents:{Style.RESET_ALL}")
    for name, desc in AGENT_INFO:
        print(f"  {Fore.GREEN}OK{Style.RESET_ALL}  {name:<35} {Fore.CYAN}{desc}{Style.RESET_ALL}")
    print()


def run_cli_demo():
    """Run a quick CLI demo (no web server) to verify agents work."""
    from agents.monitoring_agent import ProcessMonitoringAgent
    from agents.analysis_agent import QualityAnalysisAgent
    from agents.prediction_agent import DefectPredictionAgent
    from agents.optimization_agent import ProcessOptimizationAgent
    from rag_retriever import RAGRetriever

    print(f"\n{Fore.CYAN}━━━ CLI Demo Mode (5 readings) ━━━{Style.RESET_ALL}\n")

    monitor    = ProcessMonitoringAgent()
    analysis   = QualityAnalysisAgent()
    prediction = DefectPredictionAgent()
    optimizer  = ProcessOptimizationAgent()
    rag        = RAGRetriever()

    for i in range(5):
        print(f"\n{Fore.YELLOW}── Reading {i+1}/5 ──────────────────────────────────{Style.RESET_ALL}")
        reading = monitor.collect_data()
        history = monitor.get_history()

        print(monitor.status_summary())

        analysis_result = analysis.analyze(reading, history)
        print(f"\n  Risk Level  : {Fore.RED if analysis_result['risk_level'] in ('HIGH','CRITICAL') else Fore.GREEN}{analysis_result['risk_level']}{Style.RESET_ALL}")
        print(f"  Severity    : {analysis_result['severity']:.3f}")

        pred_result = prediction.predict(reading, history, analysis_result)
        print(f"  Top Defect  : {pred_result['top_defect']} ({pred_result['top_defect_probability']:.1%})")

        opt_result = optimizer.optimize(reading, analysis_result, pred_result)
        print(f"  Opt Priority: {opt_result['priority']}")

        if reading.get("out_of_spec"):
            print(f"\n  {Fore.CYAN}Granite Analysis:{Style.RESET_ALL}")
            print(f"  {analysis_result['narrative'][:300]}…")

        time.sleep(0.5)

    print(f"\n{Fore.GREEN}[DONE] CLI demo complete.{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}To start the full web dashboard, run:{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}python main.py --server{Style.RESET_ALL}")


def run_server():
    """Start the Flask web server with background monitoring."""
    from config import FLASK_PORT
    print(f"\n{Fore.CYAN}Starting web server…{Style.RESET_ALL}")
    print(f"  Dashboard URL : {Fore.GREEN}http://localhost:{FLASK_PORT}/{Style.RESET_ALL}")
    print(f"  API Base URL  : {Fore.GREEN}http://localhost:{FLASK_PORT}/api/{Style.RESET_ALL}")
    print(f"\n  {Fore.YELLOW}Press Ctrl+C to stop.{Style.RESET_ALL}\n")
    from server import start_server
    start_server()


if __name__ == "__main__":
    print_startup()

    mode = sys.argv[1] if len(sys.argv) > 1 else "--server"

    if mode == "--demo":
        run_cli_demo()
    elif mode == "--server":
        run_server()
    else:
        print(f"Usage: python main.py [--server | --demo]")
        print(f"  --server  Start the Flask web dashboard (default)")
        print(f"  --demo    Run a quick 5-reading CLI demo")
