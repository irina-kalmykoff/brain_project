import numpy as np
import os
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


import matplotlib.pyplot as plt
import pickle
from debugger import DebugMixin

class MarkovPhonemeModel(DebugMixin):
    """
    A Markov chain-based model for phoneme sequence prediction.
    This model combines local acoustic features with transition probabilities
    between phoneme groups to improve prediction accuracy.
    """
    
    def __init__(self, phonetic_dict=None, order=2, output_dir='./models/markov_phoneme', 
                 debug_mode=False, use_groups=True):
        """
        Initialize the Markov chain phoneme model.
        
        Parameters:
        -----------
        phonetic_dict : PhoneticDictionary or None
            Phonetic dictionary for phoneme grouping
        order : int
            Order of the Markov chain (1 = bigram, 2 = trigram, etc.)
        output_dir : str
            Directory to save model outputs
        debug_mode : bool
            Whether to enable debug mode
        """
        super().__init__(class_name="MarkovPhonemeModel", debug_mode=debug_mode)
        if debug_mode is not None:
            self.DEBUG_MODE = debug_mode
        self.log(f"Initialized with DEBUG_MODE={self.DEBUG_MODE}")
        
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.use_groups = use_groups
        
        # Set up phonetic dictionary
        if phonetic_dict is None:
            from phonetic_dictionary import PhoneticDictionary
            self.phonetic_dict = PhoneticDictionary()
        else:
            self.phonetic_dict = phonetic_dict
        
        # Ensure phoneme groups are available
        if not hasattr(self.phonetic_dict, 'phoneme_groups'):
            self.phonetic_dict.add_phoneme_groups()
        
        self.phoneme_groups = self.phonetic_dict.phoneme_groups
        self.phoneme_to_group = self.phonetic_dict.phoneme_to_group
        
        # Markov chain parameters
        self.order = order
        self.transition_probs = {}
        self.emission_probs = {}
        self.initial_probs = {}
        
        # Simple acoustic classifier (will be trained on features)
        self.acoustic_classifier = None
        self.feature_scaler = None
        self.trained_classes = None
        self.class_to_index = None
        self.index_to_class = None
        
        # Label encoder for phoneme groups (including 'unknown')
        # Needed to maintain a consistent mapping of all possible phoneme groups, 
        # which is different from the trained_classes
        self.group_encoder = LabelEncoder()
        all_groups = list(self.phoneme_groups.keys()) + ['unknown']
        self.group_encoder.fit(all_groups)
        
        self.log(f"Initialized MarkovPhonemeModel with order={order}")
    
    def train(self, features, phoneme_labels, words=None, participant_ids=None):
        """
        Train the Markov chain model.
        
        Parameters:
        -----------
        features : list
            List of feature arrays for each phoneme
        phoneme_labels : list
            List of phoneme labels
        words : list or None
            List of words each phoneme belongs to
        participant_ids : list or None
            List of participant IDs
            
        Returns:
        --------
        dict
            Training results
        """
        self.log("Training Markov chain model...")
        
        # Filter out samples with NaN or Inf features
        valid_indices = []
        for i, feat in enumerate(features):
            if isinstance(feat, np.ndarray):
                if feat.ndim > 1:
                    feat_flat = np.mean(feat, axis=0)
                else:
                    feat_flat = feat
                
                if not np.any(np.isnan(feat_flat)) and not np.any(np.isinf(feat_flat)):
                    valid_indices.append(i)
        
        # Filter data
        features = [features[i] for i in valid_indices]
        phoneme_labels = [phoneme_labels[i] for i in valid_indices]
        if words is not None:
            words = [words[i] for i in valid_indices]
            
        if self.use_groups:
            # Detect if labels are already groups
            known_groups = set(self.phoneme_groups.keys())
            known_groups.add('unknown')
            
            labels_are_groups = sum(1 for label in phoneme_labels[:20] if label in known_groups) > 10
            
            if labels_are_groups:
                self.log("Detected that labels are already phoneme groups")
                group_labels = phoneme_labels
            else:
                self.log("Mapping phonemes to groups...")
                group_labels = []
                for phoneme in phoneme_labels:
                    if phoneme == '?':
                        group_labels.append('unknown')
                    elif phoneme in self.phoneme_to_group:
                        group_labels.append(self.phoneme_to_group[phoneme])
                    else:
                        group_labels.append('unknown')
            
            training_labels = group_labels
        else:
            # Use raw phonemes without conversion
            self.log("Using raw phonemes (no group conversion)")
            training_labels = phoneme_labels
        
        # store unique classes on which the model is trained
        self.trained_classes = sorted(list(set(training_labels)))
        self.class_to_index = {cls: i for i, cls in enumerate(self.trained_classes)}
        self.index_to_class = {i: cls for cls, i in self.class_to_index.items()}
        
        self.log(f"Training on {len(self.trained_classes)} classes: {self.trained_classes}")
        
        # Log the distribution
        label_counter = Counter(training_labels)
        self.log(f"Group distribution: {dict(label_counter)}")
        
        # Rest of the training continues as normal
        self._build_transition_model(training_labels, words)
        self._build_acoustic_model(features, training_labels)
        self._build_initial_probs(training_labels, words)
        
        #self.save_model()
        
        return {
            'transition_matrix_size': len(self.transition_probs),
            'num_states': len(self.phoneme_groups),
            'training_samples': len(features),
            'group_distribution': dict(label_counter)
        }

    
    def _build_transition_model(self, group_labels, words=None):
        """
        Build transition probability matrix from training sequences.
        """
        self.log("Building transition model...")
        
        # Count transitions
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        if words is not None:
            # Group by words to get proper sequences
            word_sequences = defaultdict(list)
            for label, word in zip(group_labels, words):
                word_sequences[word].append(label)
            
            # Count transitions within each word
            for word, sequence in word_sequences.items():
                for i in range(len(sequence) - self.order):
                    # Create context (previous states)
                    context = tuple(sequence[i:i+self.order])
                    next_state = sequence[i+self.order]
                    transition_counts[context][next_state] += 1
        else:
            # Treat entire sequence as continuous
            for i in range(len(group_labels) - self.order):
                context = tuple(group_labels[i:i+self.order])
                next_state = group_labels[i+self.order]
                transition_counts[context][next_state] += 1
        
        # Convert counts to probabilities with smoothing
        self.transition_probs = {}
        alpha = 0.1  # Smoothing parameter
        
        # Get all possible states (including 'unknown')
        all_states = list(self.group_encoder.classes_)
        num_states = len(all_states)
        
        for context, next_states in transition_counts.items():
            total = sum(next_states.values())
            
            # Apply Laplace smoothing
            self.transition_probs[context] = {}
            for state in all_states:
                count = next_states.get(state, 0)
                self.transition_probs[context][state] = (count + alpha) / (total + alpha * num_states)
        
        # Add default transitions for unseen contexts
        self.default_transition = {state: 1.0 / len(self.trained_classes) 
                                  for state in self.trained_classes}
                                  
        self.log(f"Built transition model with {len(self.transition_probs)} contexts")
    
    def _build_acoustic_model(self, features, group_labels):
        """
        Build a simple acoustic model using averaged features per group.
        """
        self.log("Building acoustic model...")
    
        # Flatten features
        flattened_features = []
        for feat in features:
            if feat.ndim > 1:
                pooled = np.mean(feat, axis=0)
            else:
                pooled = feat
            flattened_features.append(pooled)
        
        X = np.array(flattened_features)
        
        # Filter out invalid samples
        valid_indices = []
        for i, x in enumerate(X):
            if not np.any(np.isnan(x)) and not np.any(np.isinf(x)):
                valid_indices.append(i)
        
        X = X[valid_indices]
        y = [group_labels[i] for i in valid_indices]
        
        
        # IMPORTANT: Track which groups we're training on
        self.trained_groups = sorted(list(set(y)))
        self.log(f"Training on groups: {self.trained_groups}")
        
        # Create a mapping from group names to indices for the classifier
        self.group_to_classifier_idx = {group: i for i, group in enumerate(self.trained_classes)}
        y_encoded = np.array([self.group_to_classifier_idx[label] for label in y])
        
        # Scale features
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        
        X_scaled += np.random.normal(0, 0.01, X_scaled.shape)
        # Train classifier with balanced class weights
        self.acoustic_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,  # to prevent overfitting
            min_samples_leaf=5,   #  to prevent overfitting
            class_weight='balanced',  # This helps with imbalanced classes
            random_state=42
        )
        self.acoustic_classifier.fit(X_scaled, y_encoded)
        
        self.log(f"Trained acoustic model on {len(X)} samples with {len(self.trained_groups)} groups")
        
    def _build_initial_probs(self, group_labels, words=None):
        """
        Build initial state probabilities.
        """
        self.log("Building initial state probabilities...")
        
        if words is not None:
            # Count first phoneme of each word
            word_first_phonemes = {}
            for label, word in zip(group_labels, words):
                if word not in word_first_phonemes:
                    word_first_phonemes[word] = label
            
            # Count occurrences
            first_phoneme_counts = Counter(word_first_phonemes.values())
        else:
            # Use overall distribution
            first_phoneme_counts = Counter(group_labels)
        
        # Convert to probabilities
        total = sum(first_phoneme_counts.values())
        alpha = 0.1  # Smoothing
        
        self.initial_probs = {}
        for state in self.trained_classes:
            count = first_phoneme_counts.get(state, 0)
            self.initial_probs[state] = (count + alpha) / (total + alpha * len(self.trained_classes))
        
        self.log(f"Built initial probabilities for {len(self.initial_probs)} states")
    
    def predict(self, features, use_viterbi=True):
        """
        Predict phoneme groups for a sequence of features.
        
        Parameters:
        -----------
        features : list or numpy.ndarray
            Feature arrays to predict
        use_viterbi : bool
            Whether to use Viterbi decoding for sequence prediction
            
        Returns:
        --------
        tuple
            (predicted_groups, probabilities)
        """
        if self.acoustic_classifier is None:
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
        
        # Get predictions from classifier
        classifier_probs = self.acoustic_classifier.predict_proba(X_scaled)
        classifier_preds = self.acoustic_classifier.predict(X_scaled)
        
        # Debug first prediction
        if len(classifier_preds) > 0:
            self.debug(f"First classifier prediction: {classifier_preds[0]}")
            if hasattr(self, 'classifier_group_mapping'):
                self.debug(f"Mapped to: {self.classifier_group_mapping.get(classifier_preds[0], 'UNMAPPED')}")
        
        # Map to group names
        predicted_labels = []
        for pred_idx in classifier_preds:
            if pred_idx in self.index_to_class:
                predicted_labels.append(self.index_to_class[pred_idx])
            else:
                self.log(f"Warning: Prediction index {pred_idx} not in mapping!")
                predicted_groups.append('unknown')
        
        # For compatibility, also return probabilities (simplified)
        n_all_groups = len(self.group_encoder.classes_)
        probabilities = np.zeros((len(predicted_labels), n_all_groups))
        
        for i, probs in enumerate(classifier_probs):
            for j, class_idx in enumerate(range(len(probs))):
                if class_idx in self.index_to_class:
                    group = self.index_to_class[class_idx]
                    if group in self.group_encoder.classes_:
                        group_idx = np.where(self.group_encoder.classes_ == group)[0][0]
                        probabilities[i, group_idx] = probs[j]
        
        # Apply Viterbi smoothing if requested
        if use_viterbi and len(features) > 1:
            predicted_groups = self._apply_viterbi_smoothing(predicted_labels, probabilities)
        
        return predicted_labels, probabilities 
    
    def _apply_viterbi_smoothing(self, predictions, probabilities):
        """
        Apply simple Viterbi smoothing to predictions.
        """
        # Simple smoothing - could be enhanced with full Viterbi
        smoothed = [predictions[0]]
        
        for i in range(1, len(predictions)):
            # Check transition probability
            prev = smoothed[-1]
            curr = predictions[i]
            
            context = (prev,)
            if context in self.transition_probs:
                trans_prob = self.transition_probs[context].get(curr, 0.1)
                
                # If transition is very unlikely, consider alternatives
                if trans_prob < 0.01:
                    # Find more likely transition
                    best_next = curr
                    best_prob = trans_prob
                    
                    for alt_state in self.trained_classes:
                        alt_prob = self.transition_probs[context].get(alt_state, 0.1)
                        if alt_prob > best_prob * 2:  # Significantly better
                            # Check if acoustic score is reasonable
                            if alt_state in self.group_encoder.classes_:
                                alt_idx = np.where(self.group_encoder.classes_ == alt_state)[0][0]
                                if probabilities[i, alt_idx] > 0.1:  # Reasonable acoustic score
                                    best_next = alt_state
                                    best_prob = alt_prob
                    
                    smoothed.append(best_next)
                else:
                    smoothed.append(curr)
            else:
                smoothed.append(curr)
        
        return smoothed
        
    def _viterbi_decode(self, emission_probs):
        """
        Viterbi algorithm for finding most likely state sequence.
        Simplified version for proof of concept.
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
                        trans_prob = 1.0 / n_states  # Uniform if unknown
                    
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
    
    def evaluate(self, features, true_labels, use_viterbi=True):
        """
        Evaluate the model on test data.
        """
        # Map true labels to groups
        true_groups = []
        for phoneme in true_labels:
            # Check if it's already a group (from pipeline.get_test_data with use_groups=True)
            if phoneme in self.phoneme_groups.keys():
                # It's already a group name
                true_groups.append(phoneme)
            elif phoneme == '?' or phoneme == 'unknown':
                true_groups.append('unknown')
            elif phoneme in self.phoneme_to_group:
                # It's a phoneme, map to group
                true_groups.append(self.phoneme_to_group[phoneme])
            else:
                # Unknown phoneme
                true_groups.append('unknown')
        
        # Get predictions
        predicted_groups, _ = self.predict(features, use_viterbi=use_viterbi)
        
        # Debug: Print first few comparisons
        if self.DEBUG_MODE:
            self.log("First 10 comparisons:")
            for i in range(min(10, len(true_groups))):
                self.log(f"  {i}: True='{true_groups[i]}' vs Pred='{predicted_groups[i]}' -> {true_groups[i] == predicted_groups[i]}")
        
        # Calculate accuracy
        accuracy = accuracy_score(true_groups, predicted_groups)
        
        # Calculate confusion matrix
        all_groups = list(self.phoneme_groups.keys()) + ['unknown']
        conf_matrix = confusion_matrix(true_groups, predicted_groups, labels=all_groups)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'true_groups': true_groups,
            'predicted_groups': predicted_groups
        }
    
    def save_model(self, path=None):
        """Save the model to disk."""
        if path is None:
            path = os.path.join(self.output_dir, 'markov_model.pkl')
        
        model_data = {
            'transition_probs': self.transition_probs,
            'initial_probs': self.initial_probs,
            'acoustic_classifier': self.acoustic_classifier,
            'feature_scaler': self.feature_scaler,
            'group_encoder': self.group_encoder,
            'order': self.order
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.log(f"Model saved to {path}")
    
    def load_model(self, path=None):
        """Load a saved model."""
        if path is None:
            path = os.path.join(self.output_dir, 'markov_model.pkl')
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.transition_probs = model_data['transition_probs']
        self.initial_probs = model_data['initial_probs']
        self.acoustic_classifier = model_data['acoustic_classifier']
        self.feature_scaler = model_data['feature_scaler']
        self.group_encoder = model_data['group_encoder']
        self.order = model_data['order']
        
        self.log(f"Model loaded from {path}")
    
    def visualize_transitions(self, save_path=None):
        """
        Visualize the transition probability matrix.
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'transition_matrix.png')
        
        # Create a simplified transition matrix for visualization
        # Use the same state list as the encoder
        states = list(self.group_encoder.classes_)
        n_states = len(states)
        
        # First-order transitions only for visualization
        trans_matrix = np.zeros((n_states, n_states))
        
        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                context = (from_state,)
                if context in self.transition_probs:
                    trans_matrix[i, j] = self.transition_probs[context].get(to_state, 0)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.imshow(trans_matrix, cmap='Blues', aspect='auto')
        plt.colorbar(label='Transition Probability')
        
        plt.xticks(range(n_states), states, rotation=45)
        plt.yticks(range(n_states), states)
        plt.xlabel('To State')
        plt.ylabel('From State')
        plt.title('Phoneme Group Transition Probabilities')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"Transition matrix visualization saved to {save_path}")
        
        return trans_matrix