# Converted from Untitled2.ipynb

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Audio, display
from pynwb import NWBHDF5IO

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, IntSlider, FloatSlider, Dropdown, Checkbox

from brain_audio_decoder import BrainAudioDecoder
from custom_decoder import CustomBrainAudioDecoder
from brain_audio_decoder_viz import BrainAudioDecoderViz
from acoustic_change_detector import AcousticChangeDetector
from phoneme_validator import PhonemeValidator
from phonetic_dictionary import PhoneticDictionary
from feature_vizualizer import PhonemeFeatureVisualizer
from pipeline import UnifiedPhonemePipeline

# Define paths
path_bids = './SingleWordProductionDutch-iBIDS'  # Path to the BIDS dataset
path_output = './features'  # Path to save extracted features
path_results = './results'  # Path to save results

visualizer = PhonemeFeatureVisualizer(output_dir='./phoneme_visualizations')

# Pipeline creation with PCA components management
use_augmentation = True
feature_extraction_method = 'high_gamma' #'high_gamma'  #
optimal_pca_components = 50  # Use your optimal value determined earlier

# Try to load existing pipeline, otherwise create new one
try:
    # Try loading existing pipeline
    pipeline = UnifiedPhonemePipeline.load_saved(path_results, method=feature_extraction_method)
    print(f"Loaded existing {feature_extraction_method} pipeline")
    
    # Check and update PCA components if needed
    current_pca = getattr(pipeline, 'pca_components', None)
    if current_pca != optimal_pca_components:
        print(f"Updating PCA components from {current_pca} to {optimal_pca_components}")
        pipeline.set_pca_components(optimal_pca_components)
        
        # Re-run data steps with new PCA components
        print("Re-processing data with updated PCA components...")
        pipeline.step4_initialize_detector()    
        pipeline.step5_accumulate_data()
        pipeline.step6_resolve_unknowns()
        pipeline.step7_filter_unknowns()
        
        # Save the updated pipeline
        pipeline.save()
        print(f"Updated and saved {feature_extraction_method} pipeline with {optimal_pca_components} PCA components")
    
except (FileNotFoundError, AttributeError, TypeError) as e:
    # No existing pipeline found, create new one
    print(f"No existing {feature_extraction_method} pipeline found. Creating new one...")
    
    pipeline = UnifiedPhonemePipeline(
        path_bids=path_bids,
        path_output=path_output,
        path_results=path_results,
        feature_extraction_method=feature_extraction_method,
        unknown_keep_ratio=0.1,
        channel_correlation_threshold=0.3,  # ADD THIS
        prioritize_regions=True,  # ADD THIS
        channel_selection='best_correlation',
        pca_components=optimal_pca_components,  # Set optimal PCA components
        debug_mode=True
    )
    
    # Run all steps
    print("Running pipeline steps...")
    pipeline.step1_initialize_decoder()
    pipeline.step2_stratify_participants()    
    pipeline.step3_create_split()
    pipeline.step4_initialize_detector()    
    pipeline.step5_accumulate_data()
    pipeline.step6_resolve_unknowns()
    pipeline.step7_filter_unknowns()
    
    # Save the pipeline
    pipeline.save()
    print(f"Created and saved new {feature_extraction_method} pipeline with {optimal_pca_components} PCA components")

# The pipeline is now ready to use with optimal PCA components
print(f"Pipeline ready with {feature_extraction_method} features and {optimal_pca_components} PCA components")

phonetic_dict = PhoneticDictionary()
# Get FILTERED data (this is what you feed to the Markov model):
train_data = pipeline.get_training_data(filtered=True)  # or pipeline.train_filtered

# visualizer.process_batches(train_data, method='high_gamma', band=(70, 150))

# # You could also visualize other frequency bands
# visualizer.process_batches(train_data, method='theta', band=(4, 8))
# visualizer.process_batches(train_data, method='alpha', band=(8, 13))
# visualizer.process_batches(train_data, method='beta', band=(13, 30))
