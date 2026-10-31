import os

# Configuration settings for RAM-safe, CPU-efficient blur classifier

BATCH_SIZE = 32
N_JOBS = 4  # Conservative worker count to prevent RAM spikes

# Image Processing Settings
FFT_SIZE = 512
CACHE_FEATURES = True

# Directory Paths
DATA_DIR = "data"
RAW_DATA_DIR = "blur_dataset"  # Kwentar Blur Dataset (https://www.kaggle.com/datasets/kwentar/blur-dataset)
MANIFEST_DIR = os.path.join(DATA_DIR, "manifests")
FEATURE_DIR = os.path.join(DATA_DIR, "features")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "blur_classifier.pkl")
LOG_DIR = "logs"
FAILED_LOG_PATH = os.path.join(LOG_DIR, "failed_images.txt")

# Exact Feature Order (Must be preserved identically during training and inference)
FEATURE_NAMES = [
    "laplacian_variance",
    "edge_strength",
    "noise",
    "brenner_sharpness",
    "canny_edge_density",
    "fft_high_frequency_ratio",
    "blur_effect"
]

# Binary Class Mapping
# 0 = BLURRY, 1 = NOT_BLURRY
CLASS_MAPPING = {
    0: "BLURRY",
    1: "NOT_BLURRY"
}

# Label Mapping from Raw Dataset Categories
LABEL_MAPPING = {
    "sharp": 1,
    "defocused_blurred": 0,
    "defocus": 0,
    "motion_blurred": 0,
    "motion": 0
}

# Random Forest Hyperparameters
RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 6,
    "min_samples_leaf": 2,
    "random_state": 42,
    "class_weight": "balanced",
    "n_jobs": N_JOBS
}

RANDOM_STATE = 42
