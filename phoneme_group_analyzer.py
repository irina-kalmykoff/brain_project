"""
Phoneme Group Analyzer

Analyzes how well different phoneme groupings can be distinguished
from iEEG features. Tests vowel and consonant subgroups to find
the best classification targets for a specific patient.

Usage:
    from phoneme_group_analyzer import analyze_phoneme_groups
    results = analyze_phoneme_groups(pipeline, patient_id='P06')
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kurtosis, skew
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import os


# Phoneme group definitions based on Dutch phonology
PHONEME_GROUPS = {
    # Vowel groups by position and length
    'front_short_vowels': {'i', 'ɪ', 'e', 'ɛ', 'y', 'ʏ', 'ø'},
    'front_long_vowels': {'iː', 'i:', 'eː', 'e:', 'yː', 'y:', 'øː', 'ø:'},
    'back_short_vowels': {'ɔ', 'o', 'u', 'ɑ', 'ə'},
    'back_long_vowels': {'oː', 'o:', 'uː', 'u:', 'aː', 'a:', 'ɑː', 'ɑ:'},
    
    # Consonant groups by manner of articulation
    'plosives': {'p', 'b', 't', 'd', 'k', 'g', 'c'},
    'fricatives': {'f', 'v', 's', 'z', 'x', 'ɣ', 'h', 'ʃ', 'ʒ'},
    'nasals': {'m', 'n', 'ŋ'},
    'liquids_glides': {'l', 'r', 'ʋ', 'j', 'w'},
    
    # Consonant groups by voicing
    'voiced_consonants': {'b', 'd', 'g', 'v', 'z', 'ɣ', 'm', 'n', 'ŋ', 'l', 'r', 'ʋ', 'j', 'w'},
    'voiceless_consonants': {'p', 't', 'k', 'f', 's', 'x', 'h', 'ʃ', 'c'},
    
    # Consonant groups by place of articulation
    'labial_consonants': {'p', 'b', 'f', 'v', 'm', 'ʋ', 'w'},
    'alveolar_consonants': {'t', 'd', 's', 'z', 'n', 'l', 'r'},
    'velar_consonants': {'k', 'g', 'x', 'ɣ', 'ŋ'},
    
    # Combined vowel groups
    'all_front_vowels': {'i', 'ɪ', 'e', 'ɛ', 'y', 'ʏ', 'ø', 'iː', 'i:', 'eː', 'e:', 'yː', 'y:', 'øː', 'ø:'},
    'all_back_vowels': {'ɔ', 'o', 'u', 'ɑ', 'ə', 'oː', 'o:', 'uː', 'u:', 'aː', 'a:', 'ɑː', 'ɑ:'},
    'all_short_vowels': {'i', 'ɪ', 'e', 'ɛ', 'y', 'ʏ', 'ø', 'ɔ', 'o', 'u', 'ɑ', 'ə'},
    'all_long_vowels': {'iː', 'i:', 'eː', 'e:', 'yː', 'y:', 'øː', 'ø:', 'oː', 'o:', 'uː', 'u:', 'aː', 'a:', 'ɑː', 'ɑ:'},
    
    # Diphthongs
    'diphthongs': {'ɛi', 'œy', 'ɑu', 'ʌu', 'ei', 'au', 'ou', 'ui'},
}

# Classification comparisons to test
COMPARISONS = [
    # Vowel comparisons
    ('all_front_vowels', 'all_back_vowels', 'Front vs Back Vowels'),
    ('all_short_vowels', 'all_long_vowels', 'Short vs Long Vowels'),
    ('front_short_vowels', 'back_short_vowels', 'Front vs Back (Short)'),
    ('front_long_vowels', 'back_long_vowels', 'Front vs Back (Long)'),
    
    # Consonant comparisons
    ('plosives', 'fricatives', 'Plosives vs Fricatives'),
    ('plosives', 'nasals', 'Plosives vs Nasals'),
    ('fricatives', 'nasals', 'Fricatives vs Nasals'),
    ('voiced_consonants', 'voiceless_consonants', 'Voiced vs Voiceless'),
    ('labial_consonants', 'alveolar_consonants', 'Labial vs Alveolar'),
    ('labial_consonants', 'velar_consonants', 'Labial vs Velar'),
    ('alveolar_consonants', 'velar_consonants', 'Alveolar vs Velar'),
    
    # Mixed comparisons
    ('nasals', 'liquids_glides', 'Nasals vs Liquids/Glides'),
    ('plosives', 'liquids_glides', 'Plosives vs Liquids/Glides'),
]


def extract_summary_features(feat):
    """
    Extract summary statistics from pipeline feature matrix.
    
    Args:
        feat: numpy array of shape (n_frames, n_channels) or (1, n_features)
    
    Returns:
        dict: Summary features
    """
    if feat is None or feat.size == 0:
        return None
    
    if feat.ndim == 1:
        feat = feat.reshape(1, -1)
    
    n_frames, n_features = feat.shape
    summary = {}
    
    if n_frames == 1:
        signal = feat.flatten()
        
        summary['mean'] = np.mean(signal)
        summary['std'] = np.std(signal)
        summary['max'] = np.max(signal)
        summary['min'] = np.min(signal)
        summary['range'] = summary['max'] - summary['min']
        summary['total_energy'] = np.sum(signal ** 2)
        summary['mean_energy'] = np.mean(signal ** 2)
        summary['n_frames'] = n_frames
        
        if len(signal) > 3:
            kurt_val = kurtosis(signal, nan_policy='omit')
            summary['kurtosis'] = kurt_val if np.isfinite(kurt_val) else 0
            skew_val = skew(signal, nan_policy='omit')
            summary['skewness'] = skew_val if np.isfinite(skew_val) else 0
        else:
            summary['kurtosis'] = 0
            summary['skewness'] = 0
        
        rms = np.sqrt(np.mean(signal ** 2))
        summary['peak_to_rms'] = np.max(np.abs(signal)) / (rms + 1e-10)
        summary['crest_factor'] = np.max(np.abs(signal)) / (np.std(signal) + 1e-10)
        summary['line_length'] = np.sum(np.abs(np.diff(signal))) / (len(signal) + 1e-10)
        centered = signal - np.mean(signal)
        summary['zero_crossing'] = np.sum(np.abs(np.diff(np.sign(centered))) > 0) / (len(centered) + 1e-10)
        
    else:
        summary['mean'] = np.mean(feat)
        summary['std'] = np.std(feat)
        summary['max'] = np.max(feat)
        summary['min'] = np.min(feat)
        summary['range'] = summary['max'] - summary['min']
        
        frame_means = np.mean(feat, axis=1)
        summary['temporal_mean'] = np.mean(frame_means)
        summary['temporal_std'] = np.std(frame_means)
        summary['temporal_range'] = np.max(frame_means) - np.min(frame_means)
        
        summary['total_energy'] = np.sum(feat ** 2)
        summary['mean_energy'] = np.mean(feat ** 2)
        summary['n_frames'] = n_frames
        
        flat = feat.flatten()
        if len(flat) > 3:
            kurt_val = kurtosis(flat, nan_policy='omit')
            summary['kurtosis'] = kurt_val if np.isfinite(kurt_val) else 0
            skew_val = skew(flat, nan_policy='omit')
            summary['skewness'] = skew_val if np.isfinite(skew_val) else 0
        else:
            summary['kurtosis'] = 0
            summary['skewness'] = 0
        
        rms = np.sqrt(np.mean(feat ** 2))
        summary['peak_to_rms'] = np.max(np.abs(feat)) / (rms + 1e-10)
        summary['crest_factor'] = np.max(np.abs(feat)) / (np.std(feat) + 1e-10)
        summary['line_length'] = np.sum(np.abs(np.diff(frame_means))) / (len(frame_means) + 1e-10)
        centered = frame_means - np.mean(frame_means)
        summary['zero_crossing'] = np.sum(np.abs(np.diff(np.sign(centered))) > 0) / (len(centered) + 1e-10)
    
    return summary


def cohens_d(g1, g2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return abs(np.mean(g1) - np.mean(g2)) / (pooled_std + 1e-10)


def analyze_phoneme_groups(pipeline, patient_id=None, save_path=None):
    """
    Analyze how well different phoneme groupings can be distinguished.
    
    Args:
        pipeline: Dutch30Pipeline with train data extracted
        patient_id: Optional patient filter
        save_path: Directory to save figures
    
    Returns:
        dict: Analysis results for all comparisons
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    if not hasattr(pipeline, 'train') or pipeline.train is None:
        raise ValueError("Pipeline has no training data. Run steps 1-6 first.")
    
    train_data = pipeline.train
    features_list = train_data['features']
    labels = train_data['phoneme_labels']
    participant_ids = train_data['phoneme_participant_ids']
    
    # Filter by patient if specified
    if patient_id is not None:
        indices = [i for i, pid in enumerate(participant_ids) if pid == patient_id]
        if not indices:
            print(f"No data found for patient {patient_id}")
            return None
        features_list = [features_list[i] for i in indices]
        labels = [labels[i] for i in indices]
        participant_ids = [participant_ids[i] for i in indices]
    
    print(f"Analyzing {len(features_list)} samples for {patient_id if patient_id else 'all patients'}")
    print(f"Unique phonemes: {len(set(labels))}")
    
    # Extract summary features
    print("Extracting summary features...")
    summaries = []
    valid_indices = []
    
    for i, feat in enumerate(features_list):
        s = extract_summary_features(feat)
        if s is not None:
            summaries.append(s)
            valid_indices.append(i)
    
    labels = [labels[i] for i in valid_indices]
    
    if len(summaries) == 0:
        print("No valid samples found!")
        return None
    
    feature_names = list(summaries[0].keys())
    X = np.array([[s[fn] for fn in feature_names] for s in summaries])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # Analyze each comparison
    results = {}
    comparison_summary = []
    
    print("\n" + "="*70)
    print("PHONEME GROUP COMPARISONS")
    print("="*70)
    
    for group1_name, group2_name, comparison_name in COMPARISONS:
        group1 = PHONEME_GROUPS[group1_name]
        group2 = PHONEME_GROUPS[group2_name]
        
        # Find samples belonging to each group
        idx_g1 = [i for i, label in enumerate(labels) if label in group1]
        idx_g2 = [i for i, label in enumerate(labels) if label in group2]
        
        n1, n2 = len(idx_g1), len(idx_g2)
        
        if n1 < 5 or n2 < 5:
            print(f"\n{comparison_name}: SKIPPED (n1={n1}, n2={n2}, need >= 5 each)")
            continue
        
        # Calculate effect sizes for each feature
        feature_results = {}
        for j, feat_name in enumerate(feature_names):
            g1_vals = X[idx_g1, j]
            g2_vals = X[idx_g2, j]
            
            d = cohens_d(g1_vals, g2_vals)
            try:
                _, pval = mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')
            except:
                pval = 1.0
            
            feature_results[feat_name] = {
                'd': d,
                'pval': pval,
                'g1_mean': np.mean(g1_vals),
                'g2_mean': np.mean(g2_vals)
            }
        
        # Find best feature
        best_feat = max(feature_results.keys(), key=lambda f: feature_results[f]['d'])
        best_d = feature_results[best_feat]['d']
        
        # Classification accuracy
        y_binary = np.array([0]*n1 + [1]*n2)
        X_binary = np.vstack([X[idx_g1], X[idx_g2]])
        
        baseline = max(n1, n2) / (n1 + n2)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_binary)
        
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        try:
            scores = cross_val_score(clf, X_scaled, y_binary, cv=cv, scoring='accuracy')
            accuracy = np.mean(scores)
            accuracy_std = np.std(scores)
        except:
            accuracy = baseline
            accuracy_std = 0
        
        lift = accuracy / baseline
        
        # Store results
        results[comparison_name] = {
            'group1': group1_name,
            'group2': group2_name,
            'n1': n1,
            'n2': n2,
            'baseline': baseline,
            'accuracy': accuracy,
            'accuracy_std': accuracy_std,
            'lift': lift,
            'best_feature': best_feat,
            'best_d': best_d,
            'feature_results': feature_results,
            'phonemes_g1': [l for l in labels if l in group1],
            'phonemes_g2': [l for l in labels if l in group2]
        }
        
        comparison_summary.append({
            'name': comparison_name,
            'n1': n1,
            'n2': n2,
            'best_d': best_d,
            'lift': lift,
            'accuracy': accuracy
        })
        
        # Print summary
        d_indicator = "***" if best_d > 0.5 else "**" if best_d > 0.3 else "*" if best_d > 0.2 else ""
        lift_indicator = "+++" if lift > 1.1 else "++" if lift > 1.05 else "+" if lift > 1.0 else "-"
        
        print(f"\n{comparison_name}:")
        print(f"  Samples: {n1} vs {n2} (baseline: {baseline:.2f})")
        print(f"  Best feature: {best_feat} (d={best_d:.3f}) {d_indicator}")
        print(f"  Accuracy: {accuracy:.3f} +/- {accuracy_std:.3f} (lift: {lift:.2f}x) {lift_indicator}")
    
    # Summary ranking
    print("\n" + "="*70)
    print("RANKING BY LIFT")
    print("="*70)
    
    ranked = sorted(comparison_summary, key=lambda x: x['lift'], reverse=True)
    
    for i, comp in enumerate(ranked):
        status = "PROMISING" if comp['lift'] > 1.05 and comp['best_d'] > 0.25 else ""
        print(f"{i+1:2d}. {comp['name']:30s} | lift={comp['lift']:.3f} | d={comp['best_d']:.3f} | n={comp['n1']}+{comp['n2']} {status}")
    
    print("\n" + "="*70)
    print("RANKING BY EFFECT SIZE (Cohen's d)")
    print("="*70)
    
    ranked_d = sorted(comparison_summary, key=lambda x: x['best_d'], reverse=True)
    
    for i, comp in enumerate(ranked_d):
        status = "PROMISING" if comp['lift'] > 1.05 and comp['best_d'] > 0.25 else ""
        print(f"{i+1:2d}. {comp['name']:30s} | d={comp['best_d']:.3f} | lift={comp['lift']:.3f} | n={comp['n1']}+{comp['n2']} {status}")
    
    # Create visualization
    if comparison_summary:
        _plot_comparison_results(comparison_summary, results, patient_id, save_path)
    
    return results


def _plot_comparison_results(comparison_summary, results, patient_id, save_path):
    """Create visualization of comparison results."""
    
    title_suffix = f" - {patient_id}" if patient_id else " - All Patients"
    
    # Figure 1: Overview bar chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Phoneme Group Classification Analysis{title_suffix}", fontsize=14, fontweight='bold')
    
    # Top left: Lift by comparison
    ax = axes[0, 0]
    names = [c['name'] for c in comparison_summary]
    lifts = [c['lift'] for c in comparison_summary]
    colors = ['green' if l > 1.05 else 'orange' if l > 1.0 else 'red' for l in lifts]
    
    y_pos = np.arange(len(names))
    ax.barh(y_pos, lifts, color=colors, alpha=0.7)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax.axvline(1.05, color='orange', linestyle=':', linewidth=1.5, label='1.05x')
    ax.axvline(1.1, color='green', linestyle=':', linewidth=1.5, label='1.10x')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Lift (accuracy / baseline)')
    ax.set_title('Classification Lift by Comparison')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Top right: Cohen's d by comparison
    ax = axes[0, 1]
    d_vals = [c['best_d'] for c in comparison_summary]
    colors_d = ['green' if d > 0.5 else 'orange' if d > 0.3 else 'gold' if d > 0.2 else 'red' for d in d_vals]
    
    ax.barh(y_pos, d_vals, color=colors_d, alpha=0.7)
    ax.axvline(0.2, color='gray', linestyle=':', label='Small (0.2)')
    ax.axvline(0.5, color='gray', linestyle='--', label='Medium (0.5)')
    ax.axvline(0.8, color='gray', linestyle='-', label='Large (0.8)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Best Feature Cohen's d")
    ax.set_title('Effect Size by Comparison')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Bottom left: Sample sizes
    ax = axes[1, 0]
    n1_vals = [c['n1'] for c in comparison_summary]
    n2_vals = [c['n2'] for c in comparison_summary]
    
    ax.barh(y_pos - 0.2, n1_vals, height=0.4, label='Group 1', color='steelblue', alpha=0.7)
    ax.barh(y_pos + 0.2, n2_vals, height=0.4, label='Group 2', color='coral', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Number of Samples')
    ax.set_title('Sample Sizes per Group')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Bottom right: Lift vs Effect Size scatter
    ax = axes[1, 1]
    
    for i, comp in enumerate(comparison_summary):
        color = 'green' if comp['lift'] > 1.05 and comp['best_d'] > 0.25 else 'gray'
        ax.scatter(comp['best_d'], comp['lift'], s=100, c=color, alpha=0.7)
        ax.annotate(comp['name'][:15], (comp['best_d'], comp['lift']), 
                   fontsize=7, ha='left', va='bottom')
    
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax.axvline(0.2, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(1.05, color='green', linestyle=':', alpha=0.5)
    ax.axvline(0.3, color='green', linestyle=':', alpha=0.5)
    
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel("Lift")
    ax.set_title("Lift vs Effect Size (green = promising)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "phoneme_groups_overview.png"), dpi=150, bbox_inches='tight')
        print(f"\nSaved: phoneme_groups_overview.png")
    
    plt.show()
    
    # Figure 2: Top 3 comparisons detailed view
    ranked = sorted(comparison_summary, key=lambda x: x['lift'], reverse=True)[:3]
    
    if len(ranked) >= 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Top 3 Comparisons - Feature Details{title_suffix}", fontsize=13, fontweight='bold')
        
        for idx, comp in enumerate(ranked):
            ax = axes[idx]
            comp_name = comp['name']
            
            if comp_name not in results:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            res = results[comp_name]
            feat_res = res['feature_results']
            
            # Sort features by effect size
            sorted_feats = sorted(feat_res.keys(), key=lambda f: feat_res[f]['d'], reverse=True)[:10]
            
            y_pos = np.arange(len(sorted_feats))
            d_vals = [feat_res[f]['d'] for f in sorted_feats]
            colors = ['green' if d > 0.3 else 'orange' if d > 0.2 else 'red' for d in d_vals]
            
            ax.barh(y_pos, d_vals, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_feats, fontsize=9)
            ax.set_xlabel("Cohen's d")
            ax.set_title(f"{comp_name}\nlift={res['lift']:.3f}, n={res['n1']}+{res['n2']}")
            ax.axvline(0.2, color='gray', linestyle=':')
            ax.axvline(0.3, color='gray', linestyle='--')
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, "phoneme_groups_top3_details.png"), dpi=150, bbox_inches='tight')
            print(f"Saved: phoneme_groups_top3_details.png")
        
        plt.show()
    
    # Figure 3: Phoneme distribution in top comparison
    if ranked:
        best_comp = ranked[0]['name']
        res = results[best_comp]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Best Comparison: {best_comp}{title_suffix}", fontsize=13, fontweight='bold')
        
        # Phoneme counts in each group
        from collections import Counter
        
        ax = axes[0]
        phonemes_g1 = Counter(res['phonemes_g1'])
        phonemes_g2 = Counter(res['phonemes_g2'])
        
        all_phonemes = sorted(set(phonemes_g1.keys()) | set(phonemes_g2.keys()))
        x = np.arange(len(all_phonemes))
        
        counts_g1 = [phonemes_g1.get(p, 0) for p in all_phonemes]
        counts_g2 = [phonemes_g2.get(p, 0) for p in all_phonemes]
        
        width = 0.35
        ax.bar(x - width/2, counts_g1, width, label=res['group1'], color='steelblue', alpha=0.7)
        ax.bar(x + width/2, counts_g2, width, label=res['group2'], color='coral', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(all_phonemes, fontsize=10)
        ax.set_xlabel('Phoneme')
        ax.set_ylabel('Count')
        ax.set_title('Phoneme Distribution by Group')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Best feature distribution
        ax = axes[1]
        best_feat = res['best_feature']
        feat_res = res['feature_results'][best_feat]
        
        ax.text(0.5, 0.9, f"Best Feature: {best_feat}", transform=ax.transAxes, 
               fontsize=12, ha='center', fontweight='bold')
        ax.text(0.5, 0.75, f"Cohen's d = {feat_res['d']:.3f}", transform=ax.transAxes, 
               fontsize=11, ha='center')
        ax.text(0.5, 0.6, f"Group 1 mean: {feat_res['g1_mean']:.4f}", transform=ax.transAxes, 
               fontsize=10, ha='center', color='steelblue')
        ax.text(0.5, 0.5, f"Group 2 mean: {feat_res['g2_mean']:.4f}", transform=ax.transAxes, 
               fontsize=10, ha='center', color='coral')
        ax.text(0.5, 0.35, f"\nClassification:", transform=ax.transAxes, fontsize=11, ha='center')
        ax.text(0.5, 0.25, f"Accuracy: {res['accuracy']:.3f} (+/- {res['accuracy_std']:.3f})", 
               transform=ax.transAxes, fontsize=10, ha='center')
        ax.text(0.5, 0.15, f"Baseline: {res['baseline']:.3f}", transform=ax.transAxes, 
               fontsize=10, ha='center')
        ax.text(0.5, 0.05, f"Lift: {res['lift']:.3f}x", transform=ax.transAxes, 
               fontsize=11, ha='center', fontweight='bold',
               color='green' if res['lift'] > 1.05 else 'red')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, "phoneme_groups_best_detail.png"), dpi=150, bbox_inches='tight')
            print(f"Saved: phoneme_groups_best_detail.png")
        
        plt.show()


def compare_patients(pipeline, patient_ids, save_path=None):
    """
    Compare phoneme group classification across multiple patients.
    
    Args:
        pipeline: Dutch30Pipeline with train data for multiple patients
        patient_ids: List of patient IDs to compare
        save_path: Directory to save figures
    
    Returns:
        dict: Comparison results
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    all_results = {}
    
    for pid in patient_ids:
        print(f"\n{'='*70}")
        print(f"ANALYZING {pid}")
        print('='*70)
        
        results = analyze_phoneme_groups(pipeline, patient_id=pid, save_path=None)
        if results:
            all_results[pid] = results
    
    # Summary comparison
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("CROSS-PATIENT COMPARISON")
        print("="*70)
        
        # Find best comparison for each patient
        for pid, results in all_results.items():
            best_comp = max(results.keys(), key=lambda k: results[k]['lift'])
            best_lift = results[best_comp]['lift']
            best_d = results[best_comp]['best_d']
            print(f"{pid}: Best = {best_comp} (lift={best_lift:.3f}, d={best_d:.3f})")
    
    return all_results


# Main execution example
if __name__ == "__main__":
    print("Phoneme Group Analyzer")
    print("Usage:")
    print("  from phoneme_group_analyzer import analyze_phoneme_groups")
    print("  results = analyze_phoneme_groups(pipeline, patient_id='P06')")