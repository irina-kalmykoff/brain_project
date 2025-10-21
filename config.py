# config.py
import os
import sys

# Detect which folder we're running from
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_FOLDER_NAME = os.path.basename(CURRENT_DIR)

# Check if we're in a "clean" repo or the original
if "step2" in CURRENT_FOLDER_NAME or "clean" in CURRENT_FOLDER_NAME or not os.path.exists("features"):
    # We're in a clean repo, point to original data
    DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "SingleWordProductionDutch")
    print(f"Running from clean repo, using data from: {DATA_DIR}")
else:
    # We're in the original repo with data
    DATA_DIR = CURRENT_DIR
    print(f"Running from original repo with local data")

# Dataset paths - both are Dutch datasets
DUTCH_10_PATH = os.path.join(DATA_DIR, "Dutch_10patients")  # Renamed from SingleWordProductionDutch-iBIDS
DUTCH_30_PATH = os.path.join(DATA_DIR, "Dutch_30patients")  # New 30-patient dataset

# Backward compatibility - point to 10-patient dataset by default
BIDS_PATH = DUTCH_10_PATH  

# Output paths can be shared or separate
OUTPUT_PATH = os.path.join(DATA_DIR, "features")
RESULTS_PATH = os.path.join(DATA_DIR, "results")

# Separate output directories for each dataset
FEATURES_10 = os.path.join(OUTPUT_PATH, "dutch10")
FEATURES_30 = os.path.join(OUTPUT_PATH, "dutch30")
RESULTS_10 = os.path.join(RESULTS_PATH, "dutch10")
RESULTS_30 = os.path.join(RESULTS_PATH, "dutch30")

# Legacy aliases for backward compatibility
FEATURES_DIR = OUTPUT_PATH
RESULTS_DIR = RESULTS_PATH

# Helper function to get paths for specific dataset
def get_dataset_paths(dataset='dutch10'):
    """
    Get paths for specific dataset
    
    Args:
        dataset: 'dutch10' or 'dutch30'
    
    Returns:
        dict with data_path, features_path, results_path
    """
    if dataset == 'dutch10':
        return {
            'data_path': DUTCH_10_PATH,
            'features_path': FEATURES_10,
            'results_path': RESULTS_10,
            'format': 'nwb'
        }
    elif dataset == 'dutch30':
        return {
            'data_path': DUTCH_30_PATH,
            'features_path': FEATURES_30,
            'results_path': RESULTS_30,
            'format': 'numpy'
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

# Auto-create directories if they don't exist
def ensure_directories():
    """Create all necessary directories"""
    dirs = [
        DUTCH_10_PATH, DUTCH_30_PATH,
        FEATURES_10, FEATURES_30,
        RESULTS_10, RESULTS_30,
        os.path.join(DUTCH_30_PATH, "raw"),
        os.path.join(DUTCH_30_PATH, "preprocessed")
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)