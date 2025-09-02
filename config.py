# config.py
import os

# Path to your data directory
DATA_DIR = r"D:\Documents\UM DACS\bachelor\UM DACS\bachelor\mozg\code\SingleWordProductionDutch"

# Or use relative path if the folders are siblings
# DATA_DIR = "../SingleWordProductionDutch"

# Define paths to specific data folders
FEATURES_DIR = os.path.join(DATA_DIR, "features")
IBIDS_DIR = os.path.join(DATA_DIR, "SingleWordProductionDutch-iBIDS")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
AUDIO_DIR = os.path.join(DATA_DIR, "original_audio")