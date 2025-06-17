# ==============================================================================
# FILE 1: project_config.py (NO CHANGE)
# You can manage all project settings from this file.
# ==============================================================================
import os

# --- Hugging Face Credentials ---
HF_TOKEN = ""
MODEL_NAME = "pyannote/embedding"

# --- Directory Paths ---
DATA_DIR = "Your Input Data Path"
OUTPUT_DIR = "Your Output Data Path"

# --- Speaker Count Estimation Settings ---
ESTIMATOR_MIN_SPEAKERS = 2
ESTIMATOR_MAX_SPEAKERS = 30
ESTIMATOR_STEP = 1
ESTIMATOR_GRAPH_PATH = os.path.join(OUTPUT_DIR, "1_speaker_estimation_graph.png")

# --- Main Clustering Settings ---
MINIMUM_SAMPLES = 400
CLUSTERING_GRAPH_PATH = os.path.join(OUTPUT_DIR, "2_clustering_result_graph.png")