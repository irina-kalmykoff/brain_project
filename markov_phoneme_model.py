import numpy as np
import os
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from debugger import DebugMixin


class SnakeActivation(nn.Module):
    """Snake activation: x + (1/alpha) * sin(alpha * x)^2."""
    def __init__(self, dim, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim) * alpha)

    def forward(self, x):
        return x + (1.0 / (self.alpha + 1e-8)) * torch.sin(self.alpha * x) ** 2


def build_nn_model(n_features, n_classes, activation='relu'):
    """Build a three-layer MLP with the specified activation function."""
    if activation == 'relu':
        act1, act2 = nn.ReLU(), nn.ReLU()
    elif activation == 'snake':
        act1, act2 = SnakeActivation(256), SnakeActivation(128)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    return nn.Sequential(
        nn.Linear(n_features, 256),
        act1,
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        act2,
        nn.Dropout(0.3),
        nn.Linear(128, n_classes),
    )


class TorchClassifierWrapper:
    """Wraps a PyTorch model with sklearn-compatible predict/predict_proba."""

    def __init__(self, n_features, n_classes, activation='relu',
                 class_weight='balanced', random_state=42,
                 epochs=300, batch_size=64, lr=0.001, weight_decay=1e-4):
        self.n_features = n_features
        self.n_classes = n_classes
        self.activation = activation
        self.class_weight = class_weight
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        self.model = build_nn_model(self.n_features, self.n_classes, self.activation)

        # Compute class weights
        counts = Counter(int(v) for v in y)
        total = sum(counts.values())
        weights = torch.FloatTensor([
            total / (self.n_classes * counts.get(i, 1))
            for i in range(self.n_classes)
        ])

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=weights)

        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)

        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X_t))
            for i in range(0, len(X_t), self.batch_size):
                idx = perm[i:i + self.batch_size]
                out = self.model(X_t[idx])
                loss = criterion(out, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X))
            return logits.argmax(dim=1).numpy()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X))
            return torch.softmax(logits, dim=1).numpy()

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
             debug_mode=False, use_groups=False, class_weight='balanced', classifier_type='random_forest', scaler_type='standard', feature_pooling_method='flatten',  random_state=37):
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
        self.random_state = random_state
        self.scaler_type = scaler_type
        self.feature_pooling_method = feature_pooling_method
        
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

        
        return {
            'transition_matrix_size': len(self.transition_probs),
            'num_states': len(self.phoneme_groups),
            'training_samples': len(features),
            'group_distribution': dict(label_counter)
        }

    
    def _build_neural_classifier(self, features, group_labels):
        """Trains the neural signal classifier on EEG features.

        Args:
            features: List of EEG feature arrays, one per phoneme segment.
                Each array has shape (n_frames, n_channels).
            group_labels: List of target labels corresponding to each
                feature array.
        """
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        self.log(f"Building neural classifier: {self.classifier_type}")
        
        lengths = [feat.shape[0] for feat in features]
        print(f"feature lengths unique values: {set(lengths)}")
        print(f"feature shape example: {features[0].shape}")
        
        # Track which classes we're training on
        self.trained_groups = sorted(list(set(group_labels)))
        self.log(f"Training on groups: {self.trained_groups}")
        
        # Create mapping from group names to indices
        self.group_to_classifier_idx = {group: i for i, group in enumerate(self.trained_classes)}
        y = np.array([self.group_to_classifier_idx[label] for label in group_labels])
        
        if self.classifier_type == 'dtw_knn':
            # DTW classifier works directly on variable-length sequences
            # Filter invalid samples first
            valid_features = []
            valid_labels = []
            
            for i, feat in enumerate(features):
                if feat.ndim == 1:
                    feat = feat.reshape(-1, 1)
                if not np.any(np.isnan(feat)) and not np.any(np.isinf(feat)):
                    valid_features.append(feat)
                    valid_labels.append(y[i])
            
            self.neural_classifier = DTWKNNClassifier(k=5)
            self.neural_classifier.fit(valid_features, valid_labels)
            self.feature_scaler = None  # Not needed for DTW
            self._use_dtw = True
            
            self.log(f"Trained DTW-KNN classifier on {len(valid_features)} samples")
            
        else:
            # Pool features to fixed size for sklearn classifiers
            X = self._pool_features(features, method=self.feature_pooling_method)
            
            # Filter invalid samples
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
            X = X[valid_mask]
            y = y[valid_mask]
            
            # Scale features
            if self.scaler_type == 'robust':
                from sklearn.preprocessing import RobustScaler
                self.feature_scaler = RobustScaler()
            elif self.scaler_type == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                self.feature_scaler = MinMaxScaler()
            elif self.scaler_type == 'none':
                self.feature_scaler = None
            else:
                self.feature_scaler = StandardScaler()

            if self.feature_scaler is not None:
                X_scaled = self.feature_scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Select classifier
            if self.classifier_type == 'logistic_regression':
                self.neural_classifier = LogisticRegression(
                    C=0.1,
                    solver='lbfgs',
                    penalty='l2',
                    max_iter=1000,
                    tol=1e-3,
                    class_weight=self.class_weight,
                    random_state=37,
                    n_jobs=-1
                )
            elif self.classifier_type == 'extra_trees':
                self.neural_classifier = ExtraTreesClassifier(
                    n_estimators=1000,
                    max_depth=None,
                    min_samples_leaf=1,
                    class_weight=self.class_weight,
                    random_state=37,
                    n_jobs=-1
                )
            elif self.classifier_type in ('nn_relu', 'nn_snake'):
                activation = 'snake' if self.classifier_type == 'nn_snake' else 'relu'
                self.neural_classifier = TorchClassifierWrapper(
                    n_features=X_scaled.shape[1],
                    n_classes=len(self.trained_classes),
                    activation=activation,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                )
            else:  # random_forest
                self.neural_classifier = RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=1,
                    class_weight=self.class_weight,
                    random_state=37,
                    n_jobs=-1
                )
            
            self.neural_classifier.fit(X_scaled, y)
            self._use_dtw = False
            
            self.log(f"Trained {self.classifier_type} on {len(X)} samples, {X.shape[1]} features")
        
    def predict(self, features, use_viterbi=True):
        """Predict phoneme labels for a sequence of neural signal features.

        Args:
            features: List of feature arrays to predict.
            use_viterbi: Whether to use Viterbi decoding for sequence
                prediction.

        Returns:
            Tuple of (predicted_labels, probabilities).
        """
        if self.neural_classifier is None:
            self.log("Error: Model must be trained before prediction")
            return None, None

        if getattr(self, '_use_dtw', False):
            classifier_preds = self.neural_classifier.predict(features)
            classifier_probs = self.neural_classifier.predict_proba(features)
        else:
            X = self._pool_features(features, method=self.feature_pooling_method)
            if self.feature_scaler is not None:
                X_scaled = self.feature_scaler.transform(X)
            else:
                X_scaled = X

            classifier_probs = self.neural_classifier.predict_proba(X_scaled)
            classifier_preds = self.neural_classifier.predict(X_scaled)

        # Map classifier indices to label names
        predicted_labels = []
        for pred_idx in classifier_preds:
            if pred_idx in self.index_to_class:
                predicted_labels.append(self.index_to_class[pred_idx])
            else:
                self.log(f"Warning: Prediction index {pred_idx} not in mapping!")
                predicted_labels.append('unknown')

        # Build emission probability matrix aligned to trained_classes
        n_classes = len(self.trained_classes)
        probabilities = np.zeros((len(predicted_labels), n_classes))

        for i, probs in enumerate(classifier_probs):
            for j in range(len(probs)):
                if j < n_classes:
                    probabilities[i, j] = probs[j]

        # Apply Viterbi decoding if requested
        if use_viterbi and len(features) > 1:
            path = self._viterbi_decode(probabilities)
            predicted_labels = [self.trained_classes[idx] for idx in path]

        return predicted_labels, probabilities 
    
        
    def _viterbi_decode(self, emission_probs):
        """Find the most likely state sequence using the Viterbi algorithm.

        Operates in log-space to avoid numerical underflow on long
        sequences. Uses trained_classes as the state space so it works
        in both raw phoneme and group modes.

        Args:
            emission_probs: Array of shape (n_samples, n_trained_classes)
                with classifier output probabilities.

        Returns:
            Array of integer indices into self.trained_classes.
        """
        n_samples = emission_probs.shape[0]
        n_states = len(self.trained_classes)

        viterbi = np.full((n_samples, n_states), -np.inf)
        backpointer = np.zeros((n_samples, n_states), dtype=int)

        # Initial step
        for i, state in enumerate(self.trained_classes):
            init_p = self.initial_probs.get(state, 1e-10)
            emit_p = emission_probs[0, i]
            if init_p > 0 and emit_p > 0:
                viterbi[0, i] = np.log(init_p) + np.log(emit_p)

        # Forward pass
        for t in range(1, n_samples):
            for j, curr_state in enumerate(self.trained_classes):
                emit_p = emission_probs[t, j]
                log_emit = np.log(emit_p) if emit_p > 0 else -np.inf

                for i, prev_state in enumerate(self.trained_classes):
                    context = (prev_state,)
                    if context in self.transition_probs:
                        trans_p = self.transition_probs[context].get(curr_state, 1e-10)
                    else:
                        trans_p = self.default_transition.get(curr_state, 1.0 / n_states)

                    log_trans = np.log(trans_p) if trans_p > 0 else -np.inf
                    score = viterbi[t-1, i] + log_trans + log_emit

                    if score > viterbi[t, j]:
                        viterbi[t, j] = score
                        backpointer[t, j] = i

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
    
    def _pool_features(self, features, method='auto'):
        """Convert features to fixed-size vector.
        
        Args:
            features: List of arrays with shape (n_frames_i, n_channels).
            method: 'flatten', 'pool', or 'auto' (detect based on length consistency)
            
        Returns:
            2D numpy array of shape (n_samples, n_features).
        """
        # Determine method
        if method == 'auto':
            lengths = [feat.shape[0] for feat in features]
            method = 'flatten' if len(set(lengths)) == 1 else 'pool'
        
        pooled = []
        
        for feat in features:
            if feat.ndim == 1:
                feat = feat.reshape(-1, 1)
            
            if method == 'flatten':
                pooled.append(feat.flatten())
            else:
                mean = feat.mean(axis=0)
                std = feat.std(axis=0)
                min_val = feat.min(axis=0)
                max_val = feat.max(axis=0)
                pooled.append(np.concatenate([mean, std, min_val, max_val]))
        
        return np.array(pooled) 
        
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

       # self.log(f"  Corpus words processed: {total_words}")
       # self.log(f"  Sequences skipped (empty after mapping): {total_sequences_skipped}")
       # self.log(f"  Unique transition contexts: {len(transition_counts)}")

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

        #self.log(f"  Transition contexts: {len(self.transition_probs)}")
        #self.log(f"  Initial prob states: {len(self.initial_probs)}")
        #self.log(f"  Smoothing alpha: {smoothing_alpha}")
        
        
        
class DTWKNNClassifier:
    """K-Nearest Neighbors classifier using Dynamic Time Warping distance.
    
    Handles variable-length sequences without resampling or padding.
    """
    
    def __init__(self, k=5, n_jobs=-1):
        """Initialize DTW-KNN classifier.
        
        Args:
            k: Number of nearest neighbors for voting.
            n_jobs: Parallel jobs for distance computation. -1 uses all cores.
        """
        self.k = k
        self.n_jobs = n_jobs
        self.train_features = None
        self.train_labels = None
        self.classes_ = None
    
    def fit(self, features, labels):
        """Store training data.
        
        Args:
            features: List of arrays with shape (n_frames_i, n_channels).
            labels: Array of integer labels.
        """
        self.train_features = features
        self.train_labels = np.array(labels)
        self.classes_ = np.unique(labels)
        return self
    
    def _dtw_distance(self, a, b):
        """Compute DTW distance between two feature sequences.
        
        Args:
            a: Array of shape (n_frames_a, n_channels).
            b: Array of shape (n_frames_b, n_channels).
            
        Returns:
            Scalar DTW distance.
        """
        n, m = len(a), len(b)
        
        # Cost matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(a[i-1] - b[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1]
                )
        
        return dtw_matrix[n, m]
    
    def predict(self, features):
        """Predict labels using DTW distance and KNN voting.
        
        Args:
            features: List of arrays with shape (n_frames_i, n_channels).
            
        Returns:
            Array of predicted integer labels.
        """
        predictions = []
        
        for test_feat in features:
            distances = []
            
            for train_feat in self.train_features:
                dist = self._dtw_distance(test_feat, train_feat)
                distances.append(dist)
            
            distances = np.array(distances)
            nearest_idx = np.argsort(distances)[:self.k]
            nearest_labels = self.train_labels[nearest_idx]
            
            # Majority vote
            counts = np.bincount(nearest_labels, minlength=len(self.classes_))
            predictions.append(np.argmax(counts))
        
        return np.array(predictions)
    
    def predict_proba(self, features):
        """Predict class probabilities based on neighbor distances.
        
        Args:
            features: List of arrays with shape (n_frames_i, n_channels).
            
        Returns:
            Array of shape (n_samples, n_classes) with probabilities.
        """
        probas = []
        
        for test_feat in features:
            distances = []
            
            for train_feat in self.train_features:
                dist = self._dtw_distance(test_feat, train_feat)
                distances.append(dist)
            
            distances = np.array(distances)
            nearest_idx = np.argsort(distances)[:self.k]
            nearest_labels = self.train_labels[nearest_idx]
            nearest_dists = distances[nearest_idx]
            
            # Convert distances to weights (inverse distance)
            weights = 1.0 / (nearest_dists + 1e-10)
            weights = weights / weights.sum()
            
            # Weighted vote
            proba = np.zeros(len(self.classes_))
            for label, weight in zip(nearest_labels, weights):
                proba[label] += weight
            
            probas.append(proba)
        
        return np.array(probas)