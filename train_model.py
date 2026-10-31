"""
Root execution entrypoint for Image Blur Classifier training pipeline.
Runs memory-safe end-to-end training and evaluation.
"""

import sys
import os

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train_model import run_pipeline

if __name__ == "__main__":
    run_pipeline()
