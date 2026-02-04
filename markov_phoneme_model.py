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
    
    """A Markov chain-based model for phoneme sequence prediction.
    
       Combines a neural signal classifier (trained on EEG features) with
       phonotactic transition probabilities (derived from the phonetic
       dictionary corpus) to improve prediction accuracy. The transition
       model captures which phonemes are likely to follow each other in
       Dutch, while the neural signal classifier provides per-phoneme
       evidence from intracranial EEG recordings.
    """
    
    def __init__(self, phonetic_dict=None, order=2, output_dir='./models/markov_phoneme', 
             debug_mode=False, use_groups=False, class_weight='balanced', classifier_type='random_forest'):
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
        use_groups : bool
            Whether to use phoneme groups instead of raw phonemes
        class_weight : str or dict or None
            Class weight for classifiers that support it
        classifier_type : str
            Type of classifier to use. Options:
            - 'random_forest': RandomForestClassifier (default)
            - 'extra_trees': ExtraTreesClassifier
            - 'gradient_boosting': GradientBoostingClassifier
            - 'logistic_regression': LogisticRegression
            - 'knn': KNeighborsClassifier
            - 'gaussian_nb': GaussianNB
        """
        super().__init__(class_name="MarkovPhonemeModel", debug_mode=debug_mode)
        if debug_mode is not None:
            self.DEBUG_MODE = debug_mode
        self.log(f"Initialized with DEBUG_MODE={self.DEBUG_MODE}")
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.use_groups = use_groups
        self.class_weight = class_weight
        self.classifier_type = classifier_type
        
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
        
        # Neural signal classifier (trained on EEG features)
        self.neural_classifier = None
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
    
    def train(self, features: list, phoneme_labels: list, words=None, participant_ids=None) -> dict:
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
        """
        self.log("Training Markov chain model")
        
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
        self._build_corpus_transition_model()
        self._build_neural_classifier(features, training_labels)
        #self._build_neural_model(features, training_labels)
        #self._build_initial_probs(training_labels, words)
        
        #self.save_model()
        
        return {
            'transition_matrix_size': len(self.transition_probs),
            'num_states': len(self.phoneme_groups),
            'training_samples': len(features),
            'group_distribution': dict(label_counter)
        }

    '''
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
    '''
    
    def _build_neural_model(self, features, group_labels):
        """
         Trains the neural signal classifier on EEG features.

       Fits a RandomForest classifier on pooled EEG feature vectors
       to produce per-phoneme (or per-group) emission probabilities.
       Features are flattened by averaging across the temporal axis,
       then scaled before training.

       Args:
           features: List of EEG feature arrays, one per phoneme
               segment. Each array has shape (n_frames, n_channels)
               or (n_channels,) if already pooled.
           group_labels: List of target labels corresponding to each
               feature array. These are either raw phoneme strings or
               group names depending on self.use_groups.
        """
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.preprocessing import StandardScaler
        
        self.log(f"Building neural model with classifier: {self.classifier_type}")

        # Flatten features
        X = self._flatten_features(features)
        
        # Filter out invalid samples
        valid_indices = []
        for i, x in enumerate(X):
            if not np.any(np.isnan(x)) and not np.any(np.isinf(x)):
                valid_indices.append(i)
        
        X = X[valid_indices]
        y = [group_labels[i] for i in valid_indices]
        
        # Track which groups we're training on
        self.trained_groups = sorted(list(set(y)))
        self.log(f"Training on groups: {self.trained_groups}")
        
        # Create mapping from group names to indices
        self.group_to_classifier_idx = {group: i for i, group in enumerate(self.trained_classes)}
        y_encoded = np.array([self.group_to_classifier_idx[label] for label in y])
        
        # Scale features
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Add small noise for regularization
        rng = np.random.default_rng(42)
        X_scaled += rng.normal(0, 0.01, X_scaled.shape)
        
        # Select classifier based on type
        if self.classifier_type == 'extra_trees':
            self.neural_classifier = ExtraTreesClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=2,
                class_weight=self.class_weight,
                random_state=42,
                n_jobs=-1
            )
        elif self.classifier_type == 'gradient_boosting':
            # Note: GradientBoosting doesn't support class_weight directly
            self.neural_classifier = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.classifier_type == 'logistic_regression':
            self.neural_classifier = LogisticRegression(
                max_iter=1000,
                class_weight=self.class_weight,
                random_state=42,
                n_jobs=-1
            )
        elif self.classifier_type == 'knn':
            # Note: KNN doesn't support class_weight
            self.neural_classifier = KNeighborsClassifier(
                n_neighbors=30,
                n_jobs=-1
            )
        elif self.classifier_type == 'gaussian_nb':
            # Note: GaussianNB doesn't support class_weight
            self.neural_classifier = GaussianNB()
        else:  # Default: random_forest
            self.neural_classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=2,
                class_weight=self.class_weight,
                random_state=42,
                n_jobs=-1
            )
        
        self.neural_classifier.fit(X_scaled, y_encoded)
        
        self.log(f"Trained neural signal classifier on {len(X)} samples with {len(self.trained_groups)} classes")
        
    def _build_initial_probs(self, group_labels, words=None):
        """
        Build initial state probabilities.
        """
        self.log("Building initial state probabilities")
        
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
        if self.neural_classifier is None:
            self.log("Error: Model must be trained before prediction")
            return None, None
    
        # Process features
        X = self._flatten_features(features)
        X_scaled = self.feature_scaler.transform(X)
        
        # Get predictions from classifier
        classifier_probs = self.neural_classifier.predict_proba(X_scaled)
        classifier_preds = self.neural_classifier.predict(X_scaled)
        
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
        
        Args:
            features: List of feature arrays
            true_labels: List of true phoneme labels
            use_viterbi: Whether to use Viterbi decoding
            
        Returns:
            Dict with accuracy, confusion_matrix, and label comparisons
        """
        # Get predictions (already in correct format based on self.use_groups)
        predicted, _ = self.predict(features, use_viterbi=use_viterbi)
        
        # Convert true labels to match prediction format
        if self.use_groups:
            # Convert true phonemes to groups
            true_converted = []
            for phoneme in true_labels:
                # Already a group name
                if phoneme in self.phoneme_groups:
                    true_converted.append(phoneme)
                # Unknown markers
                elif phoneme in ('?', 'unknown'):
                    true_converted.append('unknown')
                # Known phoneme - map to group
                elif phoneme in self.phoneme_to_group:
                    true_converted.append(self.phoneme_to_group[phoneme])
                # Unknown phoneme
                else:
                    true_converted.append('unknown')
            
            all_labels = list(self.phoneme_groups.keys()) + ['unknown']
        else:
            # Raw phoneme mode - no conversion needed
            true_converted = list(true_labels)
            
            # Filter to only labels that exist in both sets for confusion matrix
            all_labels = sorted(set(true_converted) | set(predicted))
        
        # Handle unseen labels in test set (not in training)
        # This prevents errors when test has phonemes not seen in training
        train_labels_set = set(self.group_labels) if hasattr(self, 'group_labels') else set()
        
        # Debug output
        if self.DEBUG_MODE:
            self.log(f"Evaluation mode: {'groups' if self.use_groups else 'raw phonemes'}")
            self.log(f"Unique true labels: {len(set(true_converted))}")
            self.log(f"Unique predictions: {len(set(predicted))}")
            self.log("First 10 comparisons:")
            for i in range(min(10, len(true_converted))):
                match = "OK" if true_converted[i] == predicted[i] else "X"
                self.log(f"  {i}: True='{true_converted[i]}' vs Pred='{predicted[i]}' [{match}]")
        
        # Calculate accuracy
        correct = sum(1 for t, p in zip(true_converted, predicted) if t == p)
        accuracy = correct / len(true_converted) if true_converted else 0.0
        
        # Calculate confusion matrix
        try:
            conf_matrix = confusion_matrix(true_converted, predicted, labels=all_labels)
        except Exception as e:
            if self.DEBUG_MODE:
                self.log(f"Confusion matrix error: {e}")
            conf_matrix = None
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'true_labels': true_converted,
            'predicted_labels': predicted,
            'n_correct': correct,
            'n_total': len(true_converted)
        }
    
    def save_model(self, path=None):
        """Save the model to disk."""
        if path is None:
            path = os.path.join(self.output_dir, 'markov_model.pkl')
        
        model_data = {
            'transition_probs': self.transition_probs,
            'initial_probs': self.initial_probs,
            'neural_classifier': self.neural_classifier,
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
        self.neural_classifier = model_data['neural_classifier']
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
        
    def _flatten_features(self, features):
        """
        Convert variable-dimension feature arrays to 1D vectors for classification.
        
        For 2D features (frames, channels), applies statistical pooling by
        concatenating mean and standard deviation across the time axis.
        This preserves more temporal information than simple averaging.
        
        Args:
            features: List of feature arrays. Can be 1D (n_channels,) or 
                     2D (n_frames, n_channels).
        
        Returns:
            numpy.ndarray: 2D array of shape (n_samples, n_features) where
                          n_features is n_channels for 1D input or 
                          2 * n_channels for 2D input.
        """
        flattened_features = []
        for feat in features:
            if feat.ndim > 1:
                pooled = np.concatenate([
                    np.mean(feat, axis=0),
                    np.std(feat, axis=0)
                ])
            else:
                # 1D: pad with zeros to match 2D output (mean + std)
                pooled = np.concatenate([feat, np.zeros_like(feat)])
            flattened_features.append(pooled)
        
        return np.array(flattened_features)
        
        
    def _build_corpus_transition_model(self, smoothing_alpha=None):
        """Builds transition and initial probabilities from the phonetic dictionary corpus.

        Iterates over single-word entries in the phonetic dictionary,
        extracts their phoneme sequences, and counts transitions using a
        sliding window of size ``self.order``. Sentence-level entries
        (keys containing spaces) are skipped to avoid counting cross-word
        transitions that would not occur within the word-level decoding
        context of this pipeline.

        When ``self.use_groups`` is True, phonemes are mapped to their
        articulatory groups before counting. Phonemes that do not map to
        any group are skipped rather than counted as 'unknown', since
        unknown mappings in the corpus indicate a gap in the group
        definitions rather than genuine signal.

        Also populates ``self.initial_probs`` from the first phoneme (or
        group) of each word, replacing the need for a separate call to
        ``_build_initial_probs``.

        Args:
            smoothing_alpha: Laplace smoothing constant for probability
                estimation. If None, falls back to
                ``self.config.laplace_smoothing_alpha`` when a config
                object is available, otherwise defaults to 0.1.

        Raises:
            ValueError: If no valid phoneme sequences are found in the
                corpus, which would indicate a broken phonetic dictionary.
        """
        from collections import defaultdict, Counter

        self.log("Building transition model from phonetic dictionary corpus...")

        # Resolve smoothing alpha from config or fallback
        if smoothing_alpha is None:
            if hasattr(self, 'config') and hasattr(self.config, 'laplace_smoothing_alpha'):
                smoothing_alpha = self.config.laplace_smoothing_alpha
            else:
                smoothing_alpha = 0.1

        transition_counts = defaultdict(lambda: defaultdict(int))
        initial_counts = Counter()
        total_words = 0
        total_sequences_skipped = 0

        for word, transcription in self.phonetic_dict.dictionary.items():
            # Skip sentence-level entries to keep transitions within-word only
            if ' ' in word:
                continue

            phonemes = self.phonetic_dict.extract_phonemes(word)

            if not phonemes:
                total_sequences_skipped += 1
                continue

            # Map to groups if configured
            if self.use_groups:
                mapped = []
                for phoneme in phonemes:
                    if phoneme in self.phoneme_to_group:
                        mapped.append(self.phoneme_to_group[phoneme])
                    # Skip phonemes without a group mapping entirely;
                    # they represent gaps in group definitions, not real unknowns
                sequence = mapped
            else:
                sequence = phonemes

            if len(sequence) < 1:
                total_sequences_skipped += 1
                continue

            total_words += 1

            # Count initial state
            initial_counts[sequence[0]] += 1

            # Count transitions with sliding window of self.order
            for i in range(len(sequence) - self.order):
                context = tuple(sequence[i:i + self.order])
                next_state = sequence[i + self.order]
                transition_counts[context][next_state] += 1

        if total_words == 0:
            raise ValueError(
                "No valid phoneme sequences found in corpus. "
                "Check that phonetic_dict.dictionary contains single-word entries "
                "with valid transcriptions."
            )

        self.log(f"  Corpus words processed: {total_words}")
        self.log(f"  Sequences skipped (empty after mapping): {total_sequences_skipped}")
        self.log(f"  Unique transition contexts: {len(transition_counts)}")

        # Determine state space for normalization
        if self.use_groups:
            all_states = list(self.group_encoder.classes_)
        else:
            # Collect all phonemes that appeared in any sequence
            corpus_phonemes = set()
            for word, transcription in self.phonetic_dict.dictionary.items():
                if ' ' in word:
                    continue
                corpus_phonemes.update(self.phonetic_dict.extract_phonemes(word))
            all_states = sorted(corpus_phonemes)

        num_states = len(all_states)

        # Normalize transition counts to probabilities with Laplace smoothing
        self.transition_probs = {}
        for context, next_state_counts in transition_counts.items():
            total = sum(next_state_counts.values())
            self.transition_probs[context] = {}
            for state in all_states:
                count = next_state_counts.get(state, 0)
                self.transition_probs[context][state] = (
                    (count + smoothing_alpha) / (total + smoothing_alpha * num_states)
                )

        # Default uniform transitions for contexts not seen in corpus
        self.default_transition = {
            state: 1.0 / num_states for state in all_states
        }

        # Normalize initial counts to probabilities with Laplace smoothing
        total_initial = sum(initial_counts.values())
        self.initial_probs = {}
        for state in all_states:
            count = initial_counts.get(state, 0)
            self.initial_probs[state] = (
                (count + smoothing_alpha) / (total_initial + smoothing_alpha * num_states)
            )

        self.log(f"  Transition contexts: {len(self.transition_probs)}")
        self.log(f"  Initial prob states: {len(self.initial_probs)}")
        self.log(f"  Smoothing alpha: {smoothing_alpha}")