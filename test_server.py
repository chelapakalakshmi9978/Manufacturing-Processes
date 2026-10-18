import sys, os, time, threading
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from server import app, _run_monitoring_loop

# Start monitoring in background
t = threading.Thread(target=_run_monitoring_loop, daemon=True)
t.start()

# Start Flask in background thread
ft = threading.Thread(target=lambda: app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False), daemon=True)
ft.start()

time.sleep(3)

# Now test the API
import requests
try:
    r = requests.get('http://127.0.0.1:5000/api/status', timeout=5)
    print('STATUS CODE:', r.status_code)
    import json
    data = r.json()
    print('RISK LEVEL:', data.get('risk_level'))
    print('MACHINE:', data.get('machine_id'))
    print('SEVERITY:', data.get('severity'))
    print('TICK:', data.get('tick'))

    r2 = requests.get('http://127.0.0.1:5000/api/machines', timeout=5)
    print('MACHINES:', r2.json())

    r3 = requests.get('http://127.0.0.1:5000/api/knowledge', timeout=5)
    docs = r3.json()
    print('KB DOCS:', len(docs))

    print()
    print('SERVER API TEST PASSED.')
except Exception as e:
    print('API TEST ERROR:', e)
