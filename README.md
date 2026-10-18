# Manufacturing Process Quality Control System
## IBM Granite AI · Multi-Agent Architecture

---

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure IBM credentials (create a .env file with):
# IBM_API_KEY=bpSKgmuggXW8pMu1rc0G0XL22LglM8dkx2kYkQxoxjtZ
# IBM_PROJECT_ID=3cb648e7-2430-4e25-86ab-bc51d756ef68
# IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com

# 3a. Start the full web dashboard (recommended)
python main.py --server

# 3b. Run a quick CLI demo (no browser needed)
python main.py --demo
```

Then open **http://localhost:5000** in your browser.

---

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  IBM Granite AI (Watsonx)                     │
│              granite-3-8b-instruct model                      │
└─────────────────────────┬────────────────────────────────────┘
                          │ Natural Language Generation
         ┌────────────────┼────────────────┐
         │                │                │
┌────────▼──────┐ ┌───────▼──────┐ ┌──────▼──────────┐
│  Quality      │ │  Defect      │ │  Process        │
│  Analysis     │ │  Prediction  │ │  Optimization   │
│  Agent        │ │  Agent       │ │  Agent          │
└───────────────┘ └──────────────┘ └─────────────────┘
         ▲                │
         │                │
┌────────┴──────────────────────────┐
│       Process Monitoring Agent     │
│  (Simulated SCADA/PLC sensors)     │
└───────────────────────────────────┘
         │
┌────────▼──────────────────────────┐
│       RAG Retriever                │
│  (Manufacturing Knowledge Base)    │
│  10 expert documents indexed       │
└───────────────────────────────────┘
         │
┌────────▼──────────────────────────┐
│       Flask API Server             │
│  REST endpoints + Dashboard        │
└───────────────────────────────────┘
```

---

### Project Structure

```
manufacturing process/
├── main.py                    # Entry point
├── server.py                  # Flask API + background loop
├── config.py                  # Thresholds & configuration
├── granite_client.py          # IBM Granite model wrapper
├── rag_retriever.py           # RAG system (10 KB documents)
├── dashboard.html             # Interactive web dashboard
├── requirements.txt
├── agents/
│   ├── monitoring_agent.py    # Agent 1: Sensor data collection
│   ├── analysis_agent.py      # Agent 2: Anomaly detection
│   ├── prediction_agent.py    # Agent 3: Defect prediction
│   └── optimization_agent.py  # Agent 4: Process optimization
```

---

### API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Interactive dashboard |
| `GET /api/status` | Current system KPIs |
| `GET /api/reading` | Latest sensor reading |
| `GET /api/analysis` | Latest quality analysis |
| `GET /api/prediction` | Latest defect predictions |
| `GET /api/optimization` | Latest optimization recommendations |
| `GET /api/history` | Last 30 sensor readings (chart data) |
| `GET /api/rag/report` | RAG-enhanced Granite report |
| `GET /api/machines` | Machine status map |
| `GET /api/knowledge` | Knowledge base document list |

---

### Monitored Parameters

| Parameter | Min | Max | Unit |
|---|---|---|---|
| Temperature | 180 | 220 | °C |
| Pressure | 95 | 105 | bar |
| Vibration | 0 | 2.5 | mm/s |
| Humidity | 30 | 60 | % |
| Speed | 950 | 1050 | RPM |
| Torque | 45 | 75 | Nm |
| Thickness | 9.8 | 10.2 | mm |
| Surface Roughness | 0 | 1.6 | μm |

---

### Agents

| Agent | Role | AI Method |
|---|---|---|
| Process Monitoring | Collects & simulates sensor data | Rule-based simulation |
| Quality Analysis | Detects threshold + Z-score anomalies | Statistical + IBM Granite |
| Defect Prediction | Scores 5 defect types | Weighted ML + IBM Granite |
| Process Optimization | Generates corrective actions | Rule-based + IBM Granite |
| RAG Retriever | Knowledge retrieval | TF-IDF + IBM Granite |
