import sys
sys.path.insert(0, '.')

from config import QUALITY_THRESHOLDS, RISK_LEVELS, AGENT_NAMES
print('config OK — thresholds:', list(QUALITY_THRESHOLDS.keys()))

from agents.monitoring_agent import ProcessMonitoringAgent
m = ProcessMonitoringAgent()
r = m.collect_data()
print('monitoring_agent OK — tick:', r['tick'], '| out_of_spec:', r['out_of_spec'])

from agents.analysis_agent import QualityAnalysisAgent
a = QualityAnalysisAgent()

# Run analysis (Granite may fallback if API unreachable)
history = m.get_history()
result = a.analyze(r, history)
print('analysis_agent OK — risk_level:', result['risk_level'], '| severity:', result['severity'])

from agents.prediction_agent import DefectPredictionAgent
p = DefectPredictionAgent()
pred = p.predict(r, history, result)
print('prediction_agent OK — top_defect:', pred['top_defect'], '| risk:', pred['overall_risk'])

from agents.optimization_agent import ProcessOptimizationAgent
o = ProcessOptimizationAgent()
opt = o.optimize(r, result, pred)
print('optimization_agent OK — priority:', opt['priority'])

from rag_retriever import RAGRetriever
rag = RAGRetriever()
docs = rag.retrieve('temperature cnc')
ids = [d['id'] for d in docs]
print('rag_retriever OK — retrieved:', ids)

print()
print('ALL MODULES VALIDATED SUCCESSFULLY.')
