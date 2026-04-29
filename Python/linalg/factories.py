"""
factories.py — фабричные функции для DSP/Python/linalg/
"""
import sys
import os

# Ensure project root is on Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
