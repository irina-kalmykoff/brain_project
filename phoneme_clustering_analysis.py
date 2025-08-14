import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict, Counter
import pandas as pd

from debugger import DebugMixin

class PhonemeClusteringAnalysis(DebugMixin):
    """
    Analyze whether your phoneme grouping is optimal for EEG-based recognition.
    """
    
    def __init__(self, phonetic_dict, debug_mode=False):
        
        super().init(class_name="PhonemeClusteringAnalysis", debug_mode=debug_mode)
            
        if debug_mode is not None:
            self.DEBUG_MODE = debug_mode
        self.log(f"Initialized with DEBUG_MODE={self.DEBUG_MODE}")
        
        self.phonetic_dict = phonetic_dict
        self.debug_mode = debug_mode
        
        # Define alternative grouping strategies
        self.grouping_strategies = {
            'articulation_place': {
                'front_vowels': ['i', 'ɪ', 'e', 'ɛ', 'ɛi', 'y', 'ʏ'],
                'back_vowels': ['u', 'o', 'ɔ', 'a', 'ɑ', 'ɑu', 'œy', 'ə'],
                'labial': ['p', 'b', 'f', 'v', 'm', 'w'],
                'alveolar': ['t', 'd', 's', 'z', 'n', 'l', 'r'],
                'palatal': ['ʃ', 'ʒ', 'j', 'ɲ', 'c', 'ɟ'],
                'velar': ['k', 'g', 'x', 'ɣ', 'ŋ', 'χ', 'ʁ'],
                'glottal': ['h', 'ɦ', 'ʔ']
            },
            'manner': {
                'vowels': ['i', 'ɪ', 'e', 'ɛ', 'y', 'ʏ', 'ə', 'u', 'o', 'ɔ', 'a', 'ɑ'],
                'diphthongs': ['ɛi', 'œy', 'ɑu'],
                'plosives': ['p', 'b', 't', 'd', 'k', 'g', 'ʔ'],
                'fricatives': ['f', 'v', 's', 'z', 'ʃ', 'ʒ', 'x', 'ɣ', 'h', 'ɦ', 'χ', 'ʁ'],
                'nasals': ['m', 'n', 'ŋ', 'ɲ'],
                'liquids': ['l', 'r'],
                'glides': ['j', 'w']
            },
            'voicing': {
                'voiced_obstruents': ['b', 'd', 'g', 'v', 'z', 'ʒ', 'ɣ'],
                'voiceless_obstruents': ['p', 't', 'k', 'f', 's', 'ʃ', 'x', 'h', 'χ'],
                'sonorants': ['m', 'n', 'ŋ', 'l', 'r', 'j', 'w'],
                'vowels': ['i', 'ɪ', 'e', 'ɛ', 'y', 'ʏ', 'ə', 'u', 'o', 'ɔ', 'a', 'ɑ', 'ɛi', 'œy', 'ɑu']
            },
            'complexity': {
                'simple_vowels': ['i', 'e', 'a', 'o', 'u', 'ə'],
                'complex_vowels': ['ɪ', 'ɛ', 'y', 'ʏ', 'ɔ', 'ɑ', 'ɛi', 'œy', 'ɑu'],
                'simple_consonants': ['p', 't', 'k', 'm', 'n', 's', 'l', 'r'],
                'complex_consonants': ['b', 'd', 'g', 'f', 'v', 'z', 'ʃ', 'ʒ', 'x', 'ɣ', 'ŋ', 'j', 'w', 'h']
            }
        }
    
    def analyze_confusion_patterns(self, true_labels, predicted_labels, current_grouping='articulation_place'):
        """
        Analyze which phoneme groups are being confused with each other.
        This helps identify if the grouping makes sense for your EEG data.
        """
        print("="*60)
        print("CONFUSION PATTERN ANALYSIS")
        print("="*60)
        
        # Get current grouping
        grouping = self.grouping_strategies[current_grouping]
        
        # Map phonemes to groups
        phoneme_to_group = {}
        for group, phonemes in grouping.items():
            for phoneme in phonemes:
                phoneme_to_group[phoneme] = group
        
        # Map labels to groups
        true_groups = [phoneme_to_group.get(l, 'unknown') for l in true_labels]
        pred_groups = [phoneme_to_group.get(l, 'unknown') for l in predicted_labels]
        
        # Calculate confusion matrix
        groups = sorted(set(true_groups + pred_groups))
        cm = confusion_matrix(true_groups, pred_groups, labels=groups)
        
        # Normalize by row (true class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', 
                    xticklabels=groups, yticklabels=groups, cmap='YlOrRd')
        plt.title(f'Confusion Matrix - {current_grouping}')
        plt.ylabel('True Group')
        plt.xlabel('Predicted Group')
        plt.tight_layout()
        plt.show()
        
        # Analyze confusion patterns
        print(f"\nGrouping strategy: {current_grouping}")
        print("-"*40)
        
        # Find most confused pairs
        confusion_pairs = []
        for i, true_group in enumerate(groups):
            for j, pred_group in enumerate(groups):
                if i != j and cm_normalized[i, j] > 0.1:  # >10% confusion
                    confusion_pairs.append((true_group, pred_group, cm_normalized[i, j]))
        
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print("\nMost confused group pairs (>10% confusion rate):")
        for true_g, pred_g, rate in confusion_pairs[:10]:
            print(f"  {true_g} → {pred_g}: {rate:.2%}")
        
        # Calculate per-group accuracy
        print("\nPer-group accuracy:")
        group_accuracies = {}
        for i, group in enumerate(groups):
            if cm[i].sum() > 0:
                accuracy = cm[i, i] / cm[i].sum()
                group_accuracies[group] = accuracy
                print(f"  {group}: {accuracy:.2%} ({cm[i].sum()} samples)")
        
        return cm_normalized, group_accuracies
    
    def test_alternative_groupings(self, features, labels):
        """
        Test different grouping strategies to see which works best with your EEG features.
        """
        print("="*60)
        print("TESTING ALTERNATIVE GROUPINGS")
        print("="*60)
        
        results = {}
        
        for strategy_name, grouping in self.grouping_strategies.items():
            print(f"\nTesting: {strategy_name}")
            print("-"*40)
            
            # Map phonemes to groups
            phoneme_to_group = {}
            for group, phonemes in grouping.items():
                for phoneme in phonemes:
                    phoneme_to_group[phoneme] = group
            
            # Map labels to groups
            groups = [phoneme_to_group.get(l, 'unknown') for l in labels]
            
            # Prepare features (average pooling for variable length)
            X = []
            for feat in features:
                if feat.ndim > 1:
                    X.append(np.mean(feat, axis=0))
                else:
                    X.append(feat)
            X = np.array(X)
            
            # Calculate silhouette score (how well-separated the groups are)
            if len(set(groups)) > 1:
                try:
                    score = silhouette_score(X, groups, sample_size=min(1000, len(X)))
                    print(f"  Silhouette score: {score:.3f} (higher is better)")
                except:
                    score = -1
                    print("  Silhouette score: Could not compute")
            else:
                score = -1
            
            # Count group sizes
            group_counts = Counter(groups)
            print(f"  Number of groups: {len(group_counts)}")
            print(f"  Group balance (std/mean): {np.std(list(group_counts.values()))/np.mean(list(group_counts.values())):.2f}")
            
            results[strategy_name] = {
                'silhouette': score,
                'n_groups': len(group_counts),
                'balance': np.std(list(group_counts.values()))/np.mean(list(group_counts.values())),
                'group_counts': group_counts
            }
        
        # Rank strategies
        print("\n" + "="*60)
        print("RANKING OF GROUPING STRATEGIES")
        print("="*60)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['silhouette'], reverse=True)
        for i, (strategy, metrics) in enumerate(sorted_results, 1):
            print(f"{i}. {strategy}:")
            print(f"   Silhouette: {metrics['silhouette']:.3f}")
            print(f"   Groups: {metrics['n_groups']}")
            print(f"   Balance: {metrics['balance']:.2f}")
        
        return results
    
    def visualize_phoneme_space(self, features, labels, method='pca', grouping='articulation_place'):
        """
        Visualize how phonemes cluster in your EEG feature space.
        This shows if your grouping aligns with the actual EEG patterns.
        """
        print(f"\nVisualizing phoneme space with {method.upper()}...")
        
        # Prepare features
        X = []
        valid_labels = []
        for feat, label in zip(features, labels):
            if feat.ndim > 1:
                X.append(np.mean(feat, axis=0))
            else:
                X.append(feat)
            valid_labels.append(label)
        X = np.array(X)
        
        # Reduce dimensionality
        if method == 'pca':
            reducer = PCA(n_components=2)
        else:  # tsne
            reducer = TSNE(n_components=2, random_state=42)
        
        X_reduced = reducer.fit_transform(X)
        
        # Get grouping
        grouping = self.grouping_strategies[grouping]
        phoneme_to_group = {}
        for group, phonemes in grouping.items():
            for phoneme in phonemes:
                phoneme_to_group[phoneme] = group
        
        # Map labels to groups
        groups = [phoneme_to_group.get(l, 'unknown') for l in valid_labels]
        unique_groups = sorted(set(groups))
        
        # Create color map
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_groups)))
        group_colors = {g: colors[i] for i, g in enumerate(unique_groups)}
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Individual phonemes
        unique_phonemes = sorted(set(valid_labels))
        phoneme_colors = plt.cm.hsv(np.linspace(0, 1, len(unique_phonemes)))
        
        for i, phoneme in enumerate(unique_phonemes):
            mask = np.array(valid_labels) == phoneme
            ax1.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                       c=[phoneme_colors[i]], label=phoneme, alpha=0.6, s=30)
        
        ax1.set_title(f'Individual Phonemes ({method.upper()})')
        ax1.set_xlabel('Component 1')
        ax1.set_ylabel('Component 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        
        # Right plot: Grouped phonemes
        for group in unique_groups:
            mask = np.array(groups) == group
            ax2.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                       c=[group_colors[group]], label=group, alpha=0.6, s=50)
        
        ax2.set_title(f'Grouped by {grouping} ({method.upper()})')
        ax2.set_xlabel('Component 1')
        ax2.set_ylabel('Component 2')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        return X_reduced, groups
    
    def hierarchical_clustering_analysis(self, features, labels, n_clusters_to_try=[3, 5, 7, 10]):
        """
        Use hierarchical clustering to discover natural groupings in your EEG data.
        Compare these with your predefined groupings.
        """
        print("="*60)
        print("HIERARCHICAL CLUSTERING ANALYSIS")
        print("="*60)
        
        # Prepare features - get average per phoneme type
        phoneme_features = defaultdict(list)
        for feat, label in zip(features, labels):
            if feat.ndim > 1:
                phoneme_features[label].append(np.mean(feat, axis=0))
            else:
                phoneme_features[label].append(feat)
        
        # Average features for each phoneme
        phoneme_names = []
        X_phonemes = []
        for phoneme, feats in phoneme_features.items():
            phoneme_names.append(phoneme)
            X_phonemes.append(np.mean(feats, axis=0))
        
        X_phonemes = np.array(X_phonemes)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(X_phonemes, method='ward')
        
        # Plot dendrogram
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix, labels=phoneme_names, orientation='top')
        plt.title('Hierarchical Clustering of Phonemes (based on EEG features)')
        plt.xlabel('Phoneme')
        plt.ylabel('Distance')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        
        # Try different numbers of clusters
        print("\nTesting different cluster numbers:")
        best_clustering = None
        best_score = -1
        
        for n_clusters in n_clusters_to_try:
            # Get clusters
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            cluster_labels = clustering.fit_predict(X_phonemes)
            
            # Create cluster dictionary
            clusters = defaultdict(list)
            for phoneme, cluster in zip(phoneme_names, cluster_labels):
                clusters[f'cluster_{cluster}'].append(phoneme)
            
            # Evaluate clustering
            if len(features) < 10000:  # Only if not too many samples
                # Map all samples to clusters
                sample_clusters = []
                for feat, label in zip(features, labels):
                    phoneme_idx = phoneme_names.index(label) if label in phoneme_names else -1
                    if phoneme_idx >= 0:
                        sample_clusters.append(cluster_labels[phoneme_idx])
                    else:
                        sample_clusters.append(-1)
                
                # Calculate silhouette score
                X_samples = []
                valid_clusters = []
                for feat, cluster in zip(features, sample_clusters):
                    if cluster >= 0:
                        if feat.ndim > 1:
                            X_samples.append(np.mean(feat, axis=0))
                        else:
                            X_samples.append(feat)
                        valid_clusters.append(cluster)
                
                if len(set(valid_clusters)) > 1:
                    score = silhouette_score(X_samples, valid_clusters, 
                                           sample_size=min(1000, len(X_samples)))
                else:
                    score = -1
            else:
                score = -1
            
            print(f"\n{n_clusters} clusters - Silhouette: {score:.3f}")
            for cluster_name, phonemes in clusters.items():
                print(f"  {cluster_name}: {', '.join(phonemes)}")
            
            if score > best_score:
                best_score = score
                best_clustering = clusters
        
        return best_clustering
    
    def compare_with_predictions(self, true_labels, predicted_labels, features):
        """
        Analyze if confusion patterns suggest a different grouping would work better.
        """
        print("="*60)
        print("CONFUSION-BASED GROUPING ANALYSIS")
        print("="*60)
        
        # Build confusion matrix for individual phonemes
        unique_phonemes = sorted(set(true_labels + predicted_labels))
        cm = confusion_matrix(true_labels, predicted_labels, labels=unique_phonemes)
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        # Find which phonemes are most often confused with each other
        confusion_strength = cm_norm + cm_norm.T  # Symmetrize
        
        # Perform clustering on confusion matrix
        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
        
        # Use confusion as similarity (inverse for distance)
        distance_matrix = 1 - confusion_strength
        np.fill_diagonal(distance_matrix, 0)
        
        # Hierarchical clustering on confusion patterns
        linkage_matrix = linkage(distance_matrix, method='average')
        
        # Plot dendrogram
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix, labels=unique_phonemes, orientation='top')
        plt.title('Phoneme Clustering Based on Confusion Patterns')
        plt.xlabel('Phoneme')
        plt.ylabel('Distance (1 - confusion rate)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        
        # Get suggested clusters based on confusion
        suggested_clusters = {}
        for n_clusters in [4, 6, 8]:
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            cluster_dict = defaultdict(list)
            for phoneme, cluster in zip(unique_phonemes, clusters):
                cluster_dict[f'group_{cluster}'].append(phoneme)
            suggested_clusters[n_clusters] = cluster_dict
            
            print(f"\nSuggested grouping with {n_clusters} groups (based on confusion):")
            for group, phonemes in cluster_dict.items():
                print(f"  {group}: {', '.join(phonemes)}")
        
        return suggested_clusters
