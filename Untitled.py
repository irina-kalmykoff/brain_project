# Converted from Untitled.ipynb

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

# Define paths
path_bids = './SingleWordProductionDutch-iBIDS'
path_output = './features'
path_results = './results'

# Create directories if they don't exist
os.makedirs(path_output, exist_ok=True)
os.makedirs(path_results, exist_ok=True)

# Initialize decoder
decoder = BrainAudioDecoder(
    path_bids=path_bids,
    path_output=path_output,
    path_results=path_results,
    win_length=0.05,
    frameshift=0.01,
    model_order=4,
    step_size=5,
    n_components=50
)

results = decoder.extract_features_all_participants()

results

features, spectrogram, words, feature_names = decoder.load_features('sub-01')  # Change to any participant ID

participant_ids = [f'sub-{i:02d}' for i in range(1, 11)]
        
all_data = {}
    
# Process each participant
for participant_id in participant_ids:
    try:
        print(f"Loading data for {participant_id}...")
        features, spectrogram, words, feature_names = decoder.load_features(participant_id)
            
        # Print basic stats
        print(f"  Features shape: {features.shape if hasattr(features, 'shape') else 'unknown'}")
        print(f"  Spectrogram shape: {spectrogram.shape if hasattr(spectrogram, 'shape') else 'unknown'}")
        print(f"  Words count: {len(words)}")
            
        # Store data
        all_data[participant_id] = {
                'features': features,
                'spectrogram': spectrogram,
                'words': words,
                'feature_names': feature_names
            }
            
        print(f"  Successfully loaded data for {participant_id}")
            
    except Exception as e:
        print(f"Error loading data for {participant_id}: {str(e)}")
    
# Print summary
successful = [p_id for p_id, data in all_data.items() if 'features' in data]
print(f"\nSuccessfully loaded data for {len(successful)}/{len(participant_ids)} participants")

phonetic_dict = PhoneticDictionary()
phonetic_dict.add_phoneme_groups()

all_phoneme_features = []
all_phoneme_labels = []
all_phoneme_groups = []
all_phoneme_participant_ids = []
all_phoneme_words = []

# Collect features from all participants into a single list
all_features = []
all_labels = []
all_participant_ids = []

# Iterate through participants and extract features
for participant_id, data in all_data.items():
    features = data['features']
    feature_names = data['feature_names'] #channel name
    words = data['words']
    
    # Add each feature with its corresponding word and participant ID
    for i, (feature, word) in enumerate(zip(features, words)):
        all_features.append(feature)
        all_labels.append(word)
        all_participant_ids.append(participant_id)

print(f"Collected {len(all_features)} features from {len(all_data)} participants")
# phoneme_spectrograms = all_data['spectrograms']
# phoneme_labels = all_data['phoneme_labels']
# phoneme_words = all_data['phoneme_words']

phonetic_dict = PhoneticDictionary()
phonetic_dict.add_phoneme_groups()

# Collect phoneme information
all_phoneme_features = []
all_phoneme_labels = []
all_phoneme_groups = []
all_phoneme_participant_ids = []
all_phoneme_words = []

# Process each participant
for participant_id, data in all_data.items():
    features = data['features']
    words = data['words']
    
    print(f"Extracting phonemes for {participant_id}...")
    
    # Process each word
    phoneme_count = 0
    for i, word in enumerate(words):
        if not isinstance(word, str) or not word:
            continue
            
        if word in phonetic_dict:
            # Extract phonemes for this word
            phonemes = phonetic_dict.extract_phonemes(word)
            
            # Get the feature for this word
            feature = features[i] if i < len(features) else None
            
            if feature is not None:
                # Store each phoneme with its corresponding feature
                for phoneme in phonemes:
                    all_phoneme_features.append(feature)
                    all_phoneme_labels.append(phoneme)
                    group = phonetic_dict.get_phoneme_group(phoneme) or 'unknown'
                    all_phoneme_groups.append(group)
                    all_phoneme_participant_ids.append(participant_id)
                    all_phoneme_words.append(word)
                    
                    phoneme_count += 1
    
    print(f"  Extracted {phoneme_count} phonemes from {participant_id}")

# Get counts of all phonemes
phoneme_counts = Counter(all_phoneme_labels)

phoneme_counts

# # Keep only phonemes with at least 5 instances
# min_occurrences = 5
# valid_phonemes = [p for p, c in phoneme_counts.items() if c >= min_occurrences]
# valid_indices = [i for i, p in enumerate(phoneme_labels) if p in valid_phonemes]

# print(f"Keeping {len(valid_indices)} samples with {len(valid_phonemes)} phonemes")
# print(f"Filtered out {len(phoneme_features) - len(valid_indices)} rare phoneme instances")

# # Extract valid samples
# valid_features = [phoneme_features[i] for i in valid_indices]
# valid_spectrograms = [phoneme_spectrograms[i] for i in valid_indices] if phoneme_spectrograms is not None else None
# valid_labels = [phoneme_labels[i] for i in valid_indices]
# valid_words = [phoneme_words[i] for i in valid_indices]

from sklearn.preprocessing import StandardScaler

# First, check the dimensions of all features
feature_dims = [feat.shape[0] if hasattr(feat, 'shape') else len(feat) for feat in all_phoneme_features]
dim_counts = Counter(feature_dims)

print("Feature dimensions:")
for dim, count in dim_counts.most_common():
    print(f"  Dimension {dim}: {count} features ({count/len(feature_dims):.1%})")

# Find the most common dimension
most_common_dim = dim_counts.most_common(1)[0][0]
print(f"Most common dimension: {most_common_dim}")

# Prepare features for clustering - use only features with the most common dimension
valid_indices = []
flattened_features = []

for i, feature in enumerate(all_phoneme_features):
    feat_dim = feature.shape[0] if hasattr(feature, 'shape') else len(feature)
    
    if feat_dim == most_common_dim:
        # Keep only features with the most common dimension
        valid_indices.append(i)
        
        # Process the feature
        try:
            if hasattr(feature, 'shape'):
                if len(feature.shape) == 1:
                    # Already 1D
                    flat_feat = feature
                else:
                    # Take mean across time (assumed to be first dimension)
                    flat_feat = np.mean(feature, axis=0)
            else:
                # Convert to numpy array if not already
                flat_feat = np.array(feature)
                
            flattened_features.append(flat_feat)
        except Exception as e:
            print(f"Error processing feature {i}: {e}")
            valid_indices.pop()  # Remove the index we just added

print(f"Keeping {len(valid_indices)} features with dimension {most_common_dim}")

# Also filter the labels and other data to match
valid_labels = [all_phoneme_labels[i] for i in valid_indices]
valid_groups = [all_phoneme_groups[i] for i in valid_indices]
valid_participant_ids = [all_phoneme_participant_ids[i] for i in valid_indices]
valid_words = [all_phoneme_words[i] for i in valid_indices]

# Check the shape of one feature to determine if we need additional processing
if len(flattened_features) > 0:
    first_feature = flattened_features[0]
    if hasattr(first_feature, 'shape') and len(first_feature.shape) > 0 and first_feature.shape[0] > 1:
        # Features are already vectors, stack them directly
        flattened_features_array = np.vstack(flattened_features)
    else:
        # Features might be scalars, reshape to 2D
        flattened_features_array = np.array(flattened_features).reshape(-1, 1)

    print(f"Flattened features shape: {flattened_features_array.shape}")

    # Now standardize
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(flattened_features_array)

    print(f"Scaled features shape: {scaled_features.shape}")
    
    # Count the distribution of phonemes in the valid set
    valid_phoneme_counts = Counter(valid_labels)
    print("\nTop phonemes in valid set:")
    for phoneme, count in valid_phoneme_counts.most_common(20):
        print(f"  {phoneme}: {count}")
else:
    print("No valid features found!")


trainData=np.dot(trainData, pca.components_[:numComps,:].T)
            testData = np.dot(testData, pca.components_[:numComps,:].T)

# # Perform dimensionality reduction for visualization
# try:
#     import umap
#     reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
#     embedding = reducer.fit_transform(scaled_features)
#     dim_reduction = "UMAP"
# except ImportError:
#     print("UMAP not installed, falling back to PCA")
#     from sklearn.decomposition import PCA
#     reducer = PCA(n_components=2)
#     embedding = reducer.fit_transform(scaled_features)
#     dim_reduction = "PCA"

# # Visualize phoneme distribution
# plt.figure(figsize=(12, 10))
# unique_labels = np.unique(valid_labels)
# colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

# for i, label in enumerate(unique_labels):
#     mask = np.array(valid_labels) == label
#     plt.scatter(embedding[mask, 0], embedding[mask, 1], 
#                 color=colors[i], label=label, alpha=0.7)

# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title(f'{dim_reduction} projection of phoneme segments with original labels')
# plt.tight_layout()
# plt.savefig(os.path.join(path_results, f'phoneme_{dim_reduction.lower()}_original.png'), dpi=300)
# plt.show()

# # Apply clustering
# n_clusters = len(phonetic_dict.phoneme_groups)  # Use number of phoneme groups as cluster count
# print(f"Clustering into {n_clusters} groups based on phoneme groups")

# clustering = AgglomerativeClustering(n_clusters=n_clusters)
# cluster_labels = clustering.fit_predict(scaled_features)

# # Visualize clusters
# plt.figure(figsize=(12, 10))
# for i in range(n_clusters):
#     mask = cluster_labels == i
#     plt.scatter(embedding[mask, 0], embedding[mask, 1], 
#                 label=f'Cluster {i}', alpha=0.7)

# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title(f'{dim_reduction} projection with {n_clusters} discovered clusters')
# plt.tight_layout()
# plt.savefig(os.path.join(path_results, f'phoneme_{dim_reduction.lower()}_clusters.png'), dpi=300)
# plt.show()

# # Analyze cluster composition
# cluster_composition = {}
# for i in range(n_clusters):
#     mask = cluster_labels == i
#     labels_in_cluster = [valid_labels[j] for j in range(len(valid_labels)) if mask[j]]
#     label_counts = Counter(labels_in_cluster)
#     cluster_composition[i] = label_counts

# print("Cluster composition:")
# for cluster, composition in cluster_composition.items():
#     print(f"Cluster {cluster}:")
#     for label, count in composition.most_common(5):  # Show top 5
#         print(f"  {label}: {count}")

# Map phonemes to phoneme groups
phoneme_groups = [phonetic_dict.get_phoneme_group(p) or 'unknown' for p in valid_labels]
unique_groups = np.unique(phoneme_groups)

# Count phoneme group frequencies
group_counts = Counter(phoneme_groups)
print("Phoneme group distribution:")
for group, count in group_counts.most_common():
    print(f"  {group}: {count}")

# Visualize by phoneme group
plt.figure(figsize=(12, 10))
group_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))

for i, group in enumerate(unique_groups):
    mask = np.array(phoneme_groups) == group
    plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                color=group_colors[i], label=group, alpha=0.7)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title(f'{dim_reduction} projection of phoneme segments by phoneme group')
plt.tight_layout()
plt.savefig(os.path.join(path_results, f'phoneme_{dim_reduction.lower()}_by_group.png'), dpi=300)
plt.show()

# Compare cluster assignments with phoneme groups
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

ari = adjusted_rand_score(phoneme_groups, cluster_labels)
ami = adjusted_mutual_info_score(phoneme_groups, cluster_labels)

print(f"Agreement between clusters and phoneme groups:")
print(f"  Adjusted Rand Index: {ari:.3f}")
print(f"  Adjusted Mutual Information: {ami:.3f}")

# Map phonemes to phoneme groups
phoneme_groups = [phonetic_dict.get_phoneme_group(p) or 'unknown' for p in valid_labels]
unique_groups = np.unique(phoneme_groups)

# Count phoneme group frequencies
group_counts = Counter(phoneme_groups)
print("Phoneme group distribution:")
for group, count in group_counts.most_common():
    print(f"  {group}: {count}")

# Visualize by phoneme group
plt.figure(figsize=(12, 10))
group_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))

for i, group in enumerate(unique_groups):
    mask = np.array(phoneme_groups) == group
    plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                color=group_colors[i], label=group, alpha=0.7)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title(f'{dim_reduction} projection of phoneme segments by phoneme group')
plt.tight_layout()
plt.savefig(os.path.join(path_results, f'phoneme_{dim_reduction.lower()}_by_group.png'), dpi=300)
plt.show()

# Compare cluster assignments with phoneme groups
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

ari = adjusted_rand_score(phoneme_groups, cluster_labels)
ami = adjusted_mutual_info_score(phoneme_groups, cluster_labels)

print(f"Agreement between clusters and phoneme groups:")
print(f"  Adjusted Rand Index: {ari:.3f}")
print(f"  Adjusted Mutual Information: {ami:.3f}")

# Map phonemes to phoneme groups
phoneme_groups = [phonetic_dict.get_phoneme_group(p) or 'unknown' for p in valid_labels]
unique_groups = np.unique(phoneme_groups)

# Count phoneme group frequencies
group_counts = Counter(phoneme_groups)
print("Phoneme group distribution:")
for group, count in group_counts.most_common():
    print(f"  {group}: {count}")

# Visualize by phoneme group
plt.figure(figsize=(12, 10))
group_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))

for i, group in enumerate(unique_groups):
    mask = np.array(phoneme_groups) == group
    plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                color=group_colors[i], label=group, alpha=0.7)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title(f'{dim_reduction} projection of phoneme segments by phoneme group')
plt.tight_layout()
plt.savefig(os.path.join(path_results, f'phoneme_{dim_reduction.lower()}_by_group.png'), dpi=300)
plt.show()

# Compare cluster assignments with phoneme groups
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

ari = adjusted_rand_score(phoneme_groups, cluster_labels)
ami = adjusted_mutual_info_score(phoneme_groups, cluster_labels)

print(f"Agreement between clusters and phoneme groups:")
print(f"  Adjusted Rand Index: {ari:.3f}")
print(f"  Adjusted Mutual Information: {ami:.3f}")

from sklearn.metrics import silhouette_score, silhouette_samples

# Calculate silhouette scores
silhouette_avg = silhouette_score(scaled_features, cluster_labels)
print(f"Overall silhouette score: {silhouette_avg:.3f}")

# Get silhouette scores for each sample
sample_silhouette = silhouette_samples(scaled_features, cluster_labels)

# Identify high-confidence samples
confidence_threshold = 0.3  # Adjust based on your data
high_confidence = sample_silhouette > confidence_threshold
high_conf_indices = np.where(high_confidence)[0]

print(f"Found {len(high_conf_indices)} high-confidence samples ({len(high_conf_indices)/len(sample_silhouette):.1%})")

# Visualize high vs. low confidence samples
plt.figure(figsize=(12, 10))
plt.scatter(embedding[~high_confidence, 0], embedding[~high_confidence, 1], 
            color='lightgray', label='Low confidence', alpha=0.5)
plt.scatter(embedding[high_confidence, 0], embedding[high_confidence, 1], 
            color='blue', label='High confidence', alpha=0.7)

plt.legend()
plt.title(f'Sample confidence (threshold={confidence_threshold})')
plt.tight_layout()
plt.savefig(os.path.join(path_results, 'phoneme_confidence.png'), dpi=300)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Use high-confidence samples for training
X_train = np.array(flattened_features)[high_confidence]
y_train = cluster_labels[high_confidence]

# Use low-confidence samples for testing
X_test = np.array(flattened_features)[~high_confidence]
y_test = cluster_labels[~high_confidence]

# Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Semi-supervised model accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))

# Apply model to all samples to get refined labels
refined_labels = clf.predict(flattened_features)

# Visualize refined labels
plt.figure(figsize=(12, 10))
for i in range(n_clusters):
    mask = refined_labels == i
    plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                label=f'Refined Cluster {i}', alpha=0.7)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Phoneme segments with refined labels')
plt.tight_layout()
plt.savefig(os.path.join(path_results, 'phoneme_refined_clusters.png'), dpi=300)
plt.show()

# Analyze which features are most important for clustering
if hasattr(clf, 'feature_importances_'):
    plt.figure(figsize=(12, 6))
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.title('Feature importances')
    plt.bar(range(min(20, len(importances))), importances[indices[:20]])
    plt.xticks(range(min(20, len(importances))), indices[:20], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(path_results, 'feature_importance.png'), dpi=300)
    plt.show()

# Extract acoustic features from spectrograms if available
if valid_spectrograms is not None:
    # Process spectrograms to extract features
    acoustic_features = []
    for spec in valid_spectrograms:
        # Simple features: mean and std per frequency band
        mean_spec = np.mean(spec, axis=0)
        std_spec = np.std(spec, axis=0)
        combined = np.concatenate([mean_spec, std_spec])
        acoustic_features.append(combined)
    
    # Standardize
    scaler_acoustic = StandardScaler()
    scaled_acoustic = scaler_acoustic.fit_transform(acoustic_features)
    
    # Cluster acoustic features
    acoustic_clusters = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(scaled_acoustic)
    
    # Compare with EEG-based clusters
    acoustic_ari = adjusted_rand_score(cluster_labels, acoustic_clusters)
    acoustic_ami = adjusted_mutual_info_score(cluster_labels, acoustic_clusters)
    
    print(f"Agreement between EEG and acoustic clustering:")
    print(f"  Adjusted Rand Index: {acoustic_ari:.3f}")
    print(f"  Adjusted Mutual Information: {acoustic_ami:.3f}")
    
    # Visualize acoustic clusters
    try:
        acoustic_embedding = reducer.fit_transform(scaled_acoustic)
        
        plt.figure(figsize=(12, 10))
        for i in range(n_clusters):
            mask = acoustic_clusters == i
            plt.scatter(acoustic_embedding[mask, 0], acoustic_embedding[mask, 1], 
                        label=f'Acoustic Cluster {i}', alpha=0.7)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Clustering based on acoustic features')
        plt.tight_layout()
        plt.savefig(os.path.join(path_results, 'acoustic_clusters.png'), dpi=300)
        plt.show()
    except:
        print("Could not visualize acoustic clusters - reducer might be UMAP-specific")
