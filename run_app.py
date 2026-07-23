# ============================================================
# run_app.py — Launch the Streamlit Application
# CIS6005 Student Health Risk Prediction
# ============================================================
# HOW TO RUN:
#   Open Anaconda Prompt
#   cd D:\Student_Health_Risk_Prediction
#   python run_app.py
# ============================================================

import subprocess
import sys
from pathlib import Path

app_path = Path(__file__).parent / 'streamlit_app' / 'app.py'

print("=" * 55)
print("  CIS6005 — Student Health Risk Predictor")
print("  Launching Streamlit Application...")
print("=" * 55)
print(f"  App: {app_path}")
print("  URL: http://localhost:8501")
print("=" * 55)

subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    str(app_path),
    "--server.port=8501",
    "--browser.gatherUsageStats=false"
])
