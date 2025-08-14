import numpy as np
import pickle
import os
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict, Counter
from scipy.stats import gamma as gamma_dist
from markov_phoneme_model import MarkovPhonemeModel
from debugger import DebugMixin

class GMMHMMModel(MarkovPhonemeModel):
    """
    GMM-HMM model that inherits from MarkovPhonemeModel.
    Overrides only the acoustic/emission model to use GMMs instead of Random Forest.
    """
    
    def __init__(self, phonetic_dict=None, order=2, n_mix=5, 
                 output_dir='./models/gmm_hmm', debug_mode=False):
        """
        Initialize GMM-HMM model.
        
        Parameters:
        -----------
        n_mix : int
            Number of Gaussian mixtures per phoneme group
        """
        # Initialize parent class
        
        super().__init__(phonetic_dict=phonetic_dict, order=order, 
                        output_dir=output_dir, debug_mode=debug_mode)
        
        self.n_mix = n_mix
        self.gmms = {}  # Will store GMM for each phoneme group
        self.class_name = "GMMHMMModel" 
        self.log(f"Initialized GMM-HMM with {n_mix} mixtures per state")
    
    def _build_acoustic_model(self, features, group_labels):
        """
        Override parent's acoustic model to use GMMs instead of Random Forest.
        """
        self.log("Building GMM acoustic model...")
        
        # Flatten features to fixed size (use mean pooling)
        flattened_features = []
        for feat in features:
            if feat.ndim > 1:
                pooled = np.mean(feat, axis=0)
            else:
                pooled = feat
            flattened_features.append(pooled)
        
        X = np.array(flattened_features)
        
        # Filter out samples with NaN or Inf
        valid_indices = []
        for i, x in enumerate(X):
            if not np.any(np.isnan(x)) and not np.any(np.isinf(x)):
                valid_indices.append(i)
        
        X = X[valid_indices]
        y = [group_labels[i] for i in valid_indices]
        
        # Scale features
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Group features by label
        grouped_features = defaultdict(list)
        for feat, label in zip(X_scaled, y):
            grouped_features[label].append(feat)
        
        # Train GMM for each group
        for group, features_list in grouped_features.items():
            features_array = np.array(features_list)
            
            # Adjust number of components based on available data
            n_components = min(self.n_mix, len(features_list) // 2)
            n_components = max(1, n_components)
            
            if len(features_list) >= n_components:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='diag',
                    max_iter=100,
                    random_state=42
                )
                gmm.fit(features_array)
                self.gmms[group] = gmm
                self.log(f"Trained GMM for '{group}' with {n_components} components on {len(features_list)} samples")
        
        # Store the GMMs as our acoustic classifier for compatibility
        self.acoustic_classifier = self.gmms
        
        self.log(f"Trained GMM acoustic model on {len(X)} samples")
    
    def predict(self, features, use_viterbi=True):
        """
        Override predict to use GMM scoring.
        """
        if not self.gmms:
            self.log("Error: Model must be trained before prediction")
            return None, None
        
        # Process features
        flattened_features = []
        for feat in features:
            if isinstance(feat, np.ndarray):
                if feat.ndim > 1:
                    pooled = np.mean(feat, axis=0)
                else:
                    pooled = feat
            else:
                pooled = np.array(feat)
            flattened_features.append(pooled)
        
        X = np.array(flattened_features)
        X_scaled = self.feature_scaler.transform(X)
        
        # Get acoustic probabilities from GMMs
        acoustic_probs = self._compute_gmm_probabilities(X_scaled)
        
        if use_viterbi and len(features) > 1:
            # Use parent's Viterbi with our GMM probabilities
            predicted_sequence = self._viterbi_decode_gmm(acoustic_probs)
            predicted_groups = [self.group_encoder.classes_[i] for i in predicted_sequence]
            probabilities = acoustic_probs
        else:
            # Simple independent prediction
            predicted_indices = np.argmax(acoustic_probs, axis=1)
            predicted_groups = [self.group_encoder.classes_[i] for i in predicted_indices]
            probabilities = acoustic_probs
        
        return predicted_groups, probabilities
    
    def _compute_gmm_probabilities(self, X_scaled):
        """
        Compute probability matrix using GMMs.
        """
        n_samples = X_scaled.shape[0]
        n_classes = len(self.group_encoder.classes_)
        probs = np.zeros((n_samples, n_classes))
        
        for group_idx, group in enumerate(self.group_encoder.classes_):
            if group in self.gmms:
                # Get log probabilities from GMM
                log_probs = self.gmms[group].score_samples(X_scaled)
                probs[:, group_idx] = np.exp(log_probs)
            else:
                # Small probability for unseen groups
                probs[:, group_idx] = 1e-10
        
        # Normalize to get probabilities
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        return probs
    
    def _viterbi_decode_gmm(self, emission_probs):
        """
        Simplified Viterbi for GMM (reuses parent's structure).
        """
        n_samples = emission_probs.shape[0]
        n_states = emission_probs.shape[1]
        
        # Initialize
        viterbi = np.zeros((n_samples, n_states))
        backpointer = np.zeros((n_samples, n_states), dtype=int)
        
        # Initial probabilities
        states = self.group_encoder.classes_
        for i, state in enumerate(states):
            viterbi[0, i] = self.initial_probs.get(state, 1e-10) * emission_probs[0, i]
        
        # Forward pass
        for t in range(1, n_samples):
            for j, curr_state in enumerate(states):
                max_prob = 0
                best_prev = 0
                
                for i, prev_state in enumerate(states):
                    # Use first-order transitions for simplicity
                    context = (prev_state,)
                    if context in self.transition_probs:
                        trans_prob = self.transition_probs[context].get(curr_state, 1e-10)
                    else:
                        trans_prob = 1.0 / n_states
                    
                    prob = viterbi[t-1, i] * trans_prob * emission_probs[t, j]
                    
                    if prob > max_prob:
                        max_prob = prob
                        best_prev = i
                
                viterbi[t, j] = max_prob
                backpointer[t, j] = best_prev
        
        # Backtrack
        path = np.zeros(n_samples, dtype=int)
        path[-1] = np.argmax(viterbi[-1, :])
        
        for t in range(n_samples - 2, -1, -1):
            path[t] = backpointer[t + 1, path[t + 1]]
        
        return path
    
    def save_model(self, path=None):
        """
        Save the GMM-HMM model.
        """
        if path is None:
            path = os.path.join(self.output_dir, 'gmm_hmm_model.pkl')
        
        model_data = {
            'transition_probs': self.transition_probs,
            'initial_probs': self.initial_probs,
            'gmms': self.gmms,
            'feature_scaler': self.feature_scaler,
            'group_encoder': self.group_encoder,
            'order': self.order,
            'n_mix': self.n_mix
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.log(f"GMM-HMM model saved to {path}")
    
    def load_model(self, path=None):
        """
        Load a saved GMM-HMM model.
        """
        if path is None:
            path = os.path.join(self.output_dir, 'gmm_hmm_model.pkl')
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.transition_probs = model_data['transition_probs']
        self.initial_probs = model_data['initial_probs']
        self.gmms = model_data['gmms']
        self.acoustic_classifier = self.gmms  # For compatibility
        self.feature_scaler = model_data['feature_scaler']
        self.group_encoder = model_data['group_encoder']
        self.order = model_data['order']
        self.n_mix = model_data['n_mix']
        
        self.log(f"GMM-HMM model loaded from {path}")


class HSMMModel(MarkovPhonemeModel):
    """
    HSMM model that inherits from MarkovPhonemeModel.
    Adds explicit duration modeling on top of the base HMM.
    """
    
    def __init__(self, phonetic_dict=None, order=2, max_duration=50,
                 output_dir='./models/hsmm', debug_mode=False):
        """
        Initialize HSMM model.
        
        Parameters:
        -----------
        max_duration : int
            Maximum duration for any phoneme (in frames)
        """
        # Initialize parent class
        super().__init__(phonetic_dict=phonetic_dict, order=order,
                        output_dir=output_dir, debug_mode=debug_mode)
        
        self.max_duration = max_duration
        self.duration_models = {}  # Duration distribution for each phoneme group
        self.class_name = "HSMMModel"
        self.log(f"Initialized HSMM with max duration {max_duration}")
    
    def train(self, features, phoneme_labels, words=None, participant_ids=None):
        """
        Train the HSMM model (extends parent's train method).
        """
        # First, call parent's train method to build transitions and emissions
        parent_results = super().train(features, phoneme_labels, words, participant_ids)
        
        # Then add duration modeling
        self.log("Adding duration models to HSMM...")
        
        # Map phonemes to groups for duration modeling
        group_labels = []
        for phoneme in phoneme_labels:
            if phoneme == '?':
                group_labels.append('unknown')
            elif phoneme in self.phoneme_to_group:
                group_labels.append(self.phoneme_to_group[phoneme])
            else:
                group_labels.append('unknown')
        
        # Extract duration statistics
        self._build_duration_models(features, group_labels)
        
        return {
            **parent_results,
            'duration_models': len(self.duration_models)
        }
    
    def _build_duration_models(self, features, group_labels):
        """
        Build duration models for each phoneme group.
        This is specific to HSMM.
        """
        durations = defaultdict(list)
        
        # Collect durations for each group
        for feat, label in zip(features, group_labels):
            if isinstance(feat, np.ndarray) and feat.ndim > 1:
                duration = feat.shape[0]  # Time dimension
                durations[label].append(duration)
        
        # Fit duration models (using Gamma distribution for flexibility)
        for group, dur_list in durations.items():
            if len(dur_list) > 1:
                mean_dur = np.mean(dur_list)
                var_dur = np.var(dur_list)
                
                # Gamma parameters (method of moments)
                if var_dur > 0:
                    shape = (mean_dur ** 2) / var_dur
                    scale = var_dur / mean_dur
                else:
                    shape = mean_dur
                    scale = 1.0
                
                self.duration_models[group] = {
                    'type': 'gamma',
                    'shape': max(0.1, shape),  # Ensure positive
                    'scale': max(0.1, scale),  # Ensure positive
                    'mean': mean_dur,
                    'std': np.std(dur_list),
                    'min': min(dur_list),
                    'max': min(max(dur_list), self.max_duration)
                }
                
                self.log(f"Duration model for '{group}': mean={mean_dur:.1f}, std={np.std(dur_list):.1f}")
    
    def predict(self, features, use_viterbi=True):
        """
        Predict using HSMM with duration constraints.
        """
        if self.acoustic_classifier is None:
            self.log("Error: Model must be trained before prediction")
            return None, None
        
        # Get base predictions from parent
        base_predictions, base_probs = super().predict(features, use_viterbi=False)
        
        if not self.duration_models:
            # If no duration models, fall back to parent's prediction
            return base_predictions, base_probs
        
        # Enhanced prediction with duration constraints
        if use_viterbi and len(features) > 1:
            # Use duration-aware Viterbi
            predicted_groups = self._duration_aware_prediction(features, base_probs)
            return predicted_groups, base_probs
        else:
            # For single predictions, incorporate duration score
            predicted_groups = []
            
            for i, feat in enumerate(features):
                if isinstance(feat, np.ndarray) and feat.ndim > 1:
                    duration = feat.shape[0]
                    
                    # Get base prediction
                    base_pred = base_predictions[i] if i < len(base_predictions) else 'unknown'
                    
                    # Check if duration is reasonable for this phoneme
                    if base_pred in self.duration_models:
                        dur_model = self.duration_models[base_pred]
                        
                        # If duration is very unlikely, consider alternatives
                        if duration < dur_model['min'] * 0.5 or duration > dur_model['max'] * 1.5:
                            # Find better matching phoneme based on duration
                            best_group = base_pred
                            best_score = self._duration_score(duration, dur_model)
                            
                            for group, model in self.duration_models.items():
                                score = self._duration_score(duration, model)
                                if score > best_score:
                                    best_score = score
                                    best_group = group
                            
                            predicted_groups.append(best_group)
                        else:
                            predicted_groups.append(base_pred)
                    else:
                        predicted_groups.append(base_pred)
                else:
                    predicted_groups.append(base_predictions[i] if i < len(base_predictions) else 'unknown')
            
            return predicted_groups, base_probs
    
    def _duration_score(self, duration, dur_model):
        """
        Calculate duration probability score.
        """
        if dur_model['type'] == 'gamma':
            # Use gamma distribution
            prob = gamma_dist.pdf(duration, dur_model['shape'], scale=dur_model['scale'])
            return np.log(prob + 1e-10)
        else:
            # Gaussian approximation
            mean = dur_model['mean']
            std = max(1.0, dur_model['std'])  # Avoid division by zero
            prob = np.exp(-0.5 * ((duration - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            return np.log(prob + 1e-10)
    
    def _duration_aware_prediction(self, features, base_probs):
        """
        Simplified duration-aware segmentation.
        """
        predicted_groups = []
        
        # For each feature, consider duration constraints
        for i, feat in enumerate(features):
            if isinstance(feat, np.ndarray) and feat.ndim > 1:
                duration = feat.shape[0]
                
                # Get probability distribution for this frame
                if i < base_probs.shape[0]:
                    probs = base_probs[i]
                    
                    # Weight probabilities by duration likelihood
                    weighted_probs = probs.copy()
                    
                    for j, group in enumerate(self.group_encoder.classes_):
                        if group in self.duration_models:
                            dur_score = self._duration_score(duration, self.duration_models[group])
                            # Convert log probability to weight
                            weight = np.exp(dur_score / 10)  # Scale factor to avoid extreme weights
                            weighted_probs[j] *= weight
                    
                    # Normalize
                    weighted_probs = weighted_probs / weighted_probs.sum()
                    
                    # Select best
                    best_idx = np.argmax(weighted_probs)
                    predicted_groups.append(self.group_encoder.classes_[best_idx])
                else:
                    predicted_groups.append('unknown')
            else:
                predicted_groups.append('unknown')
        
        return predicted_groups
    
    def save_model(self, path=None):
        """
        Save the HSMM model.
        """
        if path is None:
            path = os.path.join(self.output_dir, 'hsmm_model.pkl')
        
        # Get parent's model data
        model_data = {
            'transition_probs': self.transition_probs,
            'initial_probs': self.initial_probs,
            'acoustic_classifier': self.acoustic_classifier,
            'feature_scaler': self.feature_scaler,
            'group_encoder': self.group_encoder,
            'order': self.order,
            # HSMM-specific
            'duration_models': self.duration_models,
            'max_duration': self.max_duration
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.log(f"HSMM model saved to {path}")
    
    def load_model(self, path=None):
        """
        Load a saved HSMM model.
        """
        if path is None:
            path = os.path.join(self.output_dir, 'hsmm_model.pkl')
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load parent's components
        self.transition_probs = model_data['transition_probs']
        self.initial_probs = model_data['initial_probs']
        self.acoustic_classifier = model_data['acoustic_classifier']
        self.feature_scaler = model_data['feature_scaler']
        self.group_encoder = model_data['group_encoder']
        self.order = model_data['order']
        
        # Load HSMM-specific components
        self.duration_models = model_data['duration_models']
        self.max_duration = model_data['max_duration']
        
        self.log(f"HSMM model loaded from {path}")

