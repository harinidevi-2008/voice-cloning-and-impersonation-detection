"""Run the repository-level real-AI integration check from ``scripts/``."""

import os
import runpy
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
runpy.run_path(os.path.join(PROJECT_ROOT, "check_real_ai_integration.py"), run_name="__main__")
