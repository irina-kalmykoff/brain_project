import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from debugger import DebugMixin

class HybridRNNHMMDecoder(DebugMixin):
    """
    Hybrid RNN-HMM model for phoneme decoding from brain signals.
    
    This class implements a hybrid approach that combines:
    1. A neural network (RNN/LSTM) to extract features from EEG signals
    2. A Hidden Markov Model (HMM) to model phoneme transitions and sequences
    
    The neural network handles the complex feature extraction from brain signals,
    while the HMM captures the temporal and sequential structure of phonemes.
    """
    
    def __init__(self, output_dir='./results/hybrid_model', debug_mode=False, 
                 phonetic_dict=None, hmm_states_per_phoneme=3):
        """
        Initialize the hybrid RNN-HMM decoder.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save model files and results
        debug_mode : bool
            Whether to enable debug mode
        phoneme_dict : dict or None
            Dictionary mapping words to phoneme sequences
        hmm_states_per_phoneme : int
            Number of HMM states to use per phoneme (typically 3-5 for speech)
        """
        # Initialize the DebugMixin
        super().__init__(class_name="HybridRNNHMMDecoder", debug_mode=debug_mode)
        
        # Store parameters
        self.output_dir = output_dir
        self.phonetic_dict = phonetic_dict
        self.hmm_states_per_phoneme = hmm_states_per_phoneme
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models
        self.neural_model = None
        self.hmm_models = {}  # One HMM per phoneme
        self.label_encoder = None  # Will be set during training
        self.phoneme_list = []  # List of all phonemes
        
        # Parameters for the neural network
        self.nn_input_shape = None
        self.nn_output_dim = None
        self.feature_scaler = StandardScaler()
        
        # Hyperparameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.dropout_rate = 0.3
        
        # Phoneme transition probabilities (will be learned from data)
        self.transition_probs = None
        
        self.log(f"Initialized HybridRNNHMMDecoder with hmm_states_per_phoneme={hmm_states_per_phoneme}")
    
    def build_neural_network(self, input_shape, output_dim, model_type='bilstm'):
        """
        Build the neural network component of the hybrid model.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input features (sequence_length, feature_dim)
        output_dim : int
            Dimension of output (number of phoneme classes)
        model_type : str
            Type of neural network to use ('lstm', 'bilstm', 'stacked')
            
        Returns:
        --------
        tensorflow.keras.Model
            The neural network model
        """
        self.nn_input_shape = input_shape
        self.nn_output_dim = output_dim
        
        self.log(f"Building neural network with input_shape={input_shape}, output_dim={output_dim}")
        
        if model_type == 'lstm':
            model = Sequential([
                Input(shape=input_shape),
                LSTM(128, return_sequences=True),
                Dropout(self.dropout_rate),
                LSTM(64),
                Dropout(self.dropout_rate),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(self.dropout_rate),
                Dense(output_dim, activation='softmax')
            ])
        
        elif model_type == 'bilstm':
            model = Sequential([
                Input(shape=input_shape),
                Bidirectional(LSTM(128, return_sequences=True)),
                Dropout(self.dropout_rate),
                Bidirectional(LSTM(64)),
                Dropout(self.dropout_rate),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(self.dropout_rate),
                Dense(output_dim, activation='softmax')
            ])
        
        elif model_type == 'stacked':
            model = Sequential([
                Input(shape=input_shape),
                Bidirectional(LSTM(128, return_sequences=True)),
                Dropout(self.dropout_rate),
                Bidirectional(LSTM(128, return_sequences=True)),
                Dropout(self.dropout_rate),
                Bidirectional(LSTM(64)),
                Dropout(self.dropout_rate),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(self.dropout_rate),
                Dense(output_dim, activation='softmax')
            ])
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        self.neural_model = model
        return model
    
    def train_neural_network(self, features, labels, validation_split=0.2, epochs=50, patience=10):
        """
        Train the neural network component on EEG features.
        
        Parameters:
        -----------
        features : list
            List of feature arrays for each phoneme segment
        labels : list
            List of phoneme labels corresponding to each feature array
        validation_split : float
            Proportion of data to use for validation
        epochs : int
            Maximum number of training epochs
        patience : int
            Patience for early stopping
            
        Returns:
        --------
        history : tensorflow.keras.callbacks.History
            Training history
        """
        self.log(f"Training neural network with {len(features)} samples")
        
        # Convert list of features to uniform arrays
        # First, find the maximum sequence length
        max_length = max(f.shape[0] for f in features)
        self.log(f"Maximum sequence length: {max_length}")
        
        # Standardize lengths by padding/truncating
        standardized_features = self._standardize_length(features, max_length)
        
        # Get unique phoneme labels and encode them
        unique_labels = sorted(list(set(labels)))
        self.phoneme_list = unique_labels
        self.log(f"Found {len(unique_labels)} unique phonemes: {unique_labels}")
        
        # Create a simple label encoder
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.label_encoder = label_to_idx
        
        # Encode labels
        encoded_labels = [label_to_idx[label] for label in labels]
        one_hot_labels = to_categorical(encoded_labels, num_classes=len(unique_labels))
        
        # Build neural network if not already built
        if self.neural_model is None:
            input_shape = (max_length, standardized_features.shape[2])
            self.build_neural_network(input_shape, len(unique_labels))
        
        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                os.path.join(self.output_dir, 'best_nn_model.keras'),
                monitor='val_loss',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        history = self.neural_model.fit(
            standardized_features, one_hot_labels,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the model
        self.neural_model.save(os.path.join(self.output_dir, 'neural_model.keras'))
        
        # Plot training history
        self._plot_training_history(history)
        
        return history
    
    def _standardize_length(self, features, target_length):
        """
        Standardize the length of feature arrays.
        
        Parameters:
        -----------
        features : list
            List of feature arrays
        target_length : int
            Target sequence length
            
        Returns:
        --------
        numpy.ndarray
            Standardized features array with shape (n_samples, target_length, feature_dim)
        """
        n_samples = len(features)
        
        # Get feature dimension from first non-empty feature
        feature_dim = None
        for feat in features:
            if feat is not None and feat.shape[0] > 0:
                feature_dim = feat.shape[1] if feat.ndim > 1 else 1
                break
        
        if feature_dim is None:
            self.log("Error: Could not determine feature dimension")
            return None
        
        # Initialize output array
        standardized = np.zeros((n_samples, target_length, feature_dim))
        
        for i, feat in enumerate(features):
            if feat is None or feat.shape[0] == 0:
                continue
                
            if feat.ndim == 1:
                feat = feat.reshape(-1, 1)
                
            length = min(feat.shape[0], target_length)
            
            # Copy the data
            standardized[i, :length, :] = feat[:length, :]
        
        return standardized
    
    def _plot_training_history(self, history):
        """
        Plot training history.
        
        Parameters:
        -----------
        history : tensorflow.keras.callbacks.History
            Training history
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot accuracy
        axs[0].plot(history.history['accuracy'])
        axs[0].plot(history.history['val_accuracy'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].legend(['Train', 'Validation'], loc='upper left')
        
        # Plot loss
        axs[1].plot(history.history['loss'])
        axs[1].plot(history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
        plt.close()
    
    def train_hmm_models(self, features, labels, n_iter=100):
        """
        Train HMM models for each phoneme.
        
        Parameters:
        -----------
        features : list
            List of feature arrays for each phoneme segment
        labels : list
            List of phoneme labels corresponding to each feature array
        n_iter : int
            Number of iterations for HMM training
            
        Returns:
        --------
        dict
            Dictionary of trained HMM models, one per phoneme
        """
        self.debug(f"Training HMM models for {len(set(labels))} phonemes")
        
        # Group features by phoneme
        phoneme_features = {}
        for feat, label in zip(features, labels):
            if label not in phoneme_features:
                phoneme_features[label] = []
            phoneme_features[label].append(feat)
        
        # Train one HMM per phoneme
        hmm_models = {}
        
        for phoneme, feats in phoneme_features.items():
            self.log(f"Training HMM for phoneme '{phoneme}' with {len(feats)} samples")
            
            # Skip phonemes with too few examples
            if len(feats) < 2:
                self.log(f"Warning: Skipping phoneme '{phoneme}' with only {len(feats)} samples")
                continue
            
            # Combine all features for this phoneme
            X = np.vstack(feats)
            
            # Initialize and train HMM
            # GaussianHMM works well for continuous observations like EEG features
            model = hmm.GaussianHMM(
                n_components=self.hmm_states_per_phoneme,
                covariance_type="diag",
                n_iter=n_iter,
                random_state=42
            )
            
            try:
                model.fit(X)
                hmm_models[phoneme] = model
                self.log(f"Successfully trained HMM for phoneme '{phoneme}'")
            except Exception as e:
                self.log(f"Error training HMM for phoneme '{phoneme}': {e}")
        
        self.hmm_models = hmm_models
        
        # Save the models
        with open(os.path.join(self.output_dir, 'hmm_models.pkl'), 'wb') as f:
            pickle.dump(hmm_models, f)
        
        return hmm_models
    
    def extract_nn_features(self, eeg_features):
        """
        Extract neural network features from EEG data.
        
        Parameters:
        -----------
        eeg_features : numpy.ndarray
            EEG features
            
        Returns:
        --------
        numpy.ndarray
            Neural network features
        """
        if self.neural_model is None:
            self.debug("Error: Neural model not trained")
            return None
        
        # Use the trained model to get predictions (activations from the last layer) 
        self.log("Extracting features using trained neural model")
        
        try:
            # Get neural network predictions as features
            # These are softmax probabilities that can be used as features for the HMM
            nn_features = self.neural_model.predict(eeg_features)
            return nn_features
        except Exception as e:
            self.debug(f"Error extracting features: {e}")
            
            # Fallback: if we can't get predictions, return a simple feature representation
            # For example, just use standardized EEG features
            self.log("Using fallback feature extraction")
            
            # Flatten each time step's features into a single vector
            if eeg_features.ndim > 2:
                n_samples = eeg_features.shape[0]
                flattened = np.reshape(eeg_features, (n_samples, -1))
                return flattened
            else:
                return eeg_features
        
        return nn_features
    
    def learn_phoneme_transitions(self, word_phonemes):
        """
        Learn phoneme transition probabilities from word phoneme sequences.
        
        Parameters:
        -----------
        word_phonemes : list
            List of phoneme sequences for words
            
        Returns:
        --------
        numpy.ndarray
            Transition probability matrix
        """
        self.log("Learning phoneme transition probabilities")
        
        # Count transitions
        transition_counts = np.zeros((len(self.phoneme_list), len(self.phoneme_list)))
        
        for phoneme_seq in word_phonemes:
            for i in range(len(phoneme_seq) - 1):
                # Get indices of current and next phoneme
                current = self.label_encoder.get(phoneme_seq[i])
                next_phoneme = self.label_encoder.get(phoneme_seq[i + 1])
                
                # Skip if either phoneme is not in our vocabulary
                if current is None or next_phoneme is None:
                    continue
                
                transition_counts[current, next_phoneme] += 1
        
        # Convert counts to probabilities
        transition_probs = np.zeros_like(transition_counts, dtype=float)
        
        for i in range(transition_counts.shape[0]):
            row_sum = np.sum(transition_counts[i, :])
            if row_sum > 0:
                transition_probs[i, :] = transition_counts[i, :] / row_sum
            else:
                # If no transitions from this phoneme, use uniform distribution
                transition_probs[i, :] = 1.0 / transition_counts.shape[1]
        
        self.transition_probs = transition_probs
        
        # Save transition probabilities
        np.save(os.path.join(self.output_dir, 'transition_probs.npy'), transition_probs)
        
        return transition_probs
    
    def decode_sequence(self, eeg_features, use_transitions=True):
        """
        Decode a sequence of phonemes from EEG features.
        
        Parameters:
        -----------
        eeg_features : numpy.ndarray
            EEG features
        use_transitions : bool
            Whether to use phoneme transition probabilities
            
        Returns:
        --------
        list
            Decoded phoneme sequence
        """
        if self.neural_model is None or not self.hmm_models:
            self.log("Error: Models not trained")
            return None
        
        # Get neural network predictions
        nn_probs = self.neural_model.predict(eeg_features)
        
        # If not using transitions, just return the most likely phonemes
        if not use_transitions or self.transition_probs is None:
            # Get most likely phoneme for each time step
            predicted_indices = np.argmax(nn_probs, axis=1)
            predicted_phonemes = [self.phoneme_list[idx] for idx in predicted_indices]
            return predicted_phonemes
        
        # Viterbi decoding with transitions
        # Initialize log probabilities
        log_probs = np.log(nn_probs + 1e-10)  # Add small constant to avoid log(0)
        log_transitions = np.log(self.transition_probs + 1e-10)
        
        n_frames = log_probs.shape[0]
        n_phonemes = log_probs.shape[1]
        
        # Initialize Viterbi variables
        viterbi = np.zeros((n_frames, n_phonemes))
        backpointers = np.zeros((n_frames, n_phonemes), dtype=int)
        
        # Initialize first frame
        viterbi[0, :] = log_probs[0, :]
        
        # Forward pass
        for t in range(1, n_frames):
            for j in range(n_phonemes):
                # Find the most likely previous phoneme
                best_prob = -np.inf
                best_prev = -1
                
                for i in range(n_phonemes):
                    prob = viterbi[t-1, i] + log_transitions[i, j] + log_probs[t, j]
                    if prob > best_prob:
                        best_prob = prob
                        best_prev = i
                
                viterbi[t, j] = best_prob
                backpointers[t, j] = best_prev
        
        # Backtracking
        path = np.zeros(n_frames, dtype=int)
        path[-1] = np.argmax(viterbi[-1, :])
        
        for t in range(n_frames - 2, -1, -1):
            path[t] = backpointers[t+1, path[t+1]]
        
        # Convert indices to phoneme labels
        predicted_phonemes = [self.phoneme_list[idx] for idx in path]
        
        return predicted_phonemes
    
    def train_hybrid_model(self, eeg_features, phoneme_labels, word_phonemes=None, 
                          validation_split=0.2, nn_epochs=50, hmm_iter=100, patience=10):
        """
        Train the complete hybrid model.
        
        Parameters:
        -----------
        eeg_features : list
            List of EEG feature arrays
        phoneme_labels : list
            List of phoneme labels
        word_phonemes : list or None
            List of phoneme sequences for words (for transition learning)
        validation_split : float
            Proportion of data to use for validation
        nn_epochs : int
            Maximum number of epochs for neural network training
        hmm_iter : int
            Number of iterations for HMM training
        patience : int
            Patience for early stopping
            
        Returns:
        --------
        dict
            Training results
        """
        self.log("Training hybrid RNN-HMM model")
        
        # 1. Train the neural network
        nn_history = self.train_neural_network(
            eeg_features, 
            phoneme_labels, 
            validation_split=validation_split,
            epochs=nn_epochs,
            patience=patience
        )
        
        # 2. Extract neural network features
        # Standardize EEG features first
        max_length = max(f.shape[0] for f in eeg_features)
        standardized_eeg = self._standardize_length(eeg_features, max_length)
        
        # Extract features from the penultimate layer
        nn_features = self.extract_nn_features(standardized_eeg)
        self.log(f"Extracted neural features with shape {nn_features.shape}")
        
        # 3. Train HMM models
        hmm_models = self.train_hmm_models(
            nn_features, 
            phoneme_labels,
            n_iter=hmm_iter
        )
        
        # 4. Learn phoneme transitions if word phonemes are provided
        if word_phonemes:
            transition_probs = self.learn_phoneme_transitions(word_phonemes)
        
        # Save model metadata
        metadata = {
            'phoneme_list': self.phoneme_list,
            'label_encoder': self.label_encoder,
            'hmm_states_per_phoneme': self.hmm_states_per_phoneme
        }
        
        with open(os.path.join(self.output_dir, 'model_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        # Return training results
        return {
            'nn_history': nn_history.history,
            'hmm_models': len(hmm_models),
            'phoneme_list': self.phoneme_list,
            'has_transitions': word_phonemes is not None
        }
    
    def evaluate(self, eeg_features, true_phonemes, use_transitions=True):
        """
        Evaluate the hybrid model on test data.
        
        Parameters:
        -----------
        eeg_features : list
            List of EEG feature arrays
        true_phonemes : list
            List of true phoneme sequences
        use_transitions : bool
            Whether to use phoneme transition probabilities
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        self.log("Evaluating hybrid model")
        
        # Standardize EEG features
        max_length = max(f.shape[0] for f in eeg_features)
        standardized_eeg = self._standardize_length(eeg_features, max_length)
        
        # Get neural network predictions
        nn_probs = self.neural_model.predict(standardized_eeg)
        nn_predictions = np.argmax(nn_probs, axis=1)
        
        # Convert to phoneme labels
        nn_phonemes = [self.phoneme_list[idx] for idx in nn_predictions]
        
        # Decode with HMM if using transitions
        if use_transitions and self.transition_probs is not None:
            hybrid_phonemes = self.decode_sequence(standardized_eeg, use_transitions=True)
        else:
            hybrid_phonemes = nn_phonemes
        
        # Calculate accuracy
        nn_accuracy = np.mean([p == t for p, t in zip(nn_phonemes, true_phonemes)])
        hybrid_accuracy = np.mean([p == t for p, t in zip(hybrid_phonemes, true_phonemes)])
        
        # Create confusion matrix and classification report
        # Encode true phonemes
        true_indices = [self.label_encoder.get(p, -1) for p in true_phonemes]
        # Filter out phonemes not in our vocabulary
        valid_indices = [i for i, idx in enumerate(true_indices) if idx >= 0]
        
        if valid_indices:
            true_filtered = [true_indices[i] for i in valid_indices]
            nn_filtered = [nn_predictions[i] for i in valid_indices]
            
            conf_matrix = confusion_matrix(
                true_filtered, 
                nn_filtered,
                labels=range(len(self.phoneme_list))
            )
            
            class_report = classification_report(
                true_filtered,
                nn_filtered,
                labels=range(len(self.phoneme_list)),
                target_names=self.phoneme_list,
                output_dict=True
            )
        else:
            conf_matrix = None
            class_report = None
        
        # Plot confusion matrix
        if conf_matrix is not None:
            self._plot_confusion_matrix(conf_matrix)
        
        # Return evaluation results
        results = {
            'nn_accuracy': nn_accuracy,
            'hybrid_accuracy': hybrid_accuracy,
            'nn_phonemes': nn_phonemes,
            'hybrid_phonemes': hybrid_phonemes,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
        
        self.log(f"Neural network accuracy: {nn_accuracy:.4f}")
        self.log(f"Hybrid model accuracy: {hybrid_accuracy:.4f}")
        
        return results
    
    def _plot_confusion_matrix(self, conf_matrix):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        conf_matrix : numpy.ndarray
            Confusion matrix
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        tick_marks = np.arange(len(self.phoneme_list))
        plt.xticks(tick_marks, self.phoneme_list, rotation=45)
        plt.yticks(tick_marks, self.phoneme_list)
        
        # Add values
        thresh = conf_matrix.max() / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Phoneme')
        plt.xlabel('Predicted Phoneme')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()
    
    def save_model(self):
        """
        Save the complete hybrid model.
        """
        self.log("Saving hybrid model")
        
        # Neural network is already saved during training
        
        # Save HMM models
        with open(os.path.join(self.output_dir, 'hmm_models.pkl'), 'wb') as f:
            pickle.dump(self.hmm_models, f)
        
        # Save transition probabilities
        if self.transition_probs is not None:
            np.save(os.path.join(self.output_dir, 'transition_probs.npy'), self.transition_probs)
        
        # Save metadata
        metadata = {
            'phoneme_list': self.phoneme_list,
            'label_encoder': self.label_encoder,
            'hmm_states_per_phoneme': self.hmm_states_per_phoneme,
            'nn_input_shape': self.nn_input_shape,
            'nn_output_dim': self.nn_output_dim
        }
        
        with open(os.path.join(self.output_dir, 'model_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        self.log("Model saved successfully")
    
    def load_model(self, model_dir=None):
        """
        Load a previously saved hybrid model.
        
        Parameters:
        -----------
        model_dir : str or None
            Directory containing the saved model. If None, uses self.output_dir.
            
        Returns:
        --------
        bool
            True if loaded successfully, False otherwise
        """
        if model_dir is None:
            model_dir = self.output_dir
        
        self.log(f"Loading hybrid model from {model_dir}")
        
        try:
            # Load metadata
            with open(os.path.join(model_dir, 'model_metadata.pkl'), 'rb') as f:
                metadata = pickle.load(f)
            
            self.phoneme_list = metadata['phoneme_list']
            self.label_encoder = metadata['label_encoder']
            self.hmm_states_per_phoneme = metadata['hmm_states_per_phoneme']
            self.nn_input_shape = metadata['nn_input_shape']
            self.nn_output_dim = metadata['nn_output_dim']
            
            # Load neural network
            self.neural_model = load_model(os.path.join(model_dir, 'neural_model.keras'))
            
            # Load HMM models
            with open(os.path.join(model_dir, 'hmm_models.pkl'), 'rb') as f:
                self.hmm_models = pickle.load(f)
            
            # Load transition probabilities if available
            transition_path = os.path.join(model_dir, 'transition_probs.npy')
            if os.path.exists(transition_path):
                self.transition_probs = np.load(transition_path)
            
            self.log("Model loaded successfully")
            return True
            
        except Exception as e:
            self.log(f"Error loading model: {e}")
            return False
    
    def prepare_word_phonemes(self, words):
        """
        Prepare phoneme sequences for words using the phonetic dictionary.
        
        Parameters:
        -----------
        words : list
            List of words
            
        Returns:
        --------
        list
            List of phoneme sequences
        """
        if self.phonetic_dict is None:
            self.log("Warning: No phonetic dictionary available")
            return None
        
        phoneme_sequences = []
        
        for word in words:
            if word in self.phonetic_dict:
                # Get phonetic transcription
                transcription = self.phonetic_dict[word]
                
                # Clean transcription
                cleaned = transcription.replace('ˈ', '').replace('(', '').replace(')', '').replace("'", '')
                
                # Extract phonemes
                phonemes = []
                i = 0
                while i < len(cleaned):
                    # Check for complex phonemes
                    complex_found = False
                    for cp in ['ɛi', 'œy', 'ɑu', 'ɵ:', 'ɛ:', 'a:', 'o:', 'e:', 'øk', 'ɔf', 'ts', 'ŋk', 'sx', 'ɔx', 'ɪx']:
                        if i + len(cp) <= len(cleaned) and cleaned[i:i+len(cp)] == cp:
                            phonemes.append(cp)
                            i += len(cp)
                            complex_found = True
                            break
                    
                    if not complex_found:
                        phonemes.append(cleaned[i])
                        i += 1
                
                phoneme_sequences.append(phonemes)
            else:
                # Word not in dictionary
                phoneme_sequences.append([])
        
        return phoneme_sequences
    
    def process_accumulated_data(self, accumulated_data):
        """
        Process accumulated phoneme data for model training.
        
        Parameters:
        -----------
        accumulated_data : dict
            Output from detector.accumulate_phoneme_data
            
        Returns:
        --------
        dict
            Processed data ready for training
        """
        self.log("Processing accumulated phoneme data")
        
        # Extract relevant data
        eeg_features = accumulated_data['features']
        phoneme_labels = accumulated_data['phoneme_labels']
        words = accumulated_data['phoneme_words']
        
        # Remove unknown phonemes ('?')
        valid_indices = [i for i, p in enumerate(phoneme_labels) if p != '?']
        
        if len(valid_indices) < len(phoneme_labels):
            self.log(f"Removing {len(phoneme_labels) - len(valid_indices)} unknown phonemes")
            
            eeg_features = [eeg_features[i] for i in valid_indices]
            phoneme_labels = [phoneme_labels[i] for i in valid_indices]
            words = [words[i] for i in valid_indices]
        
        # Prepare word phoneme sequences for transition learning
        unique_words = list(set(words))
        word_phonemes = self.prepare_word_phonemes(unique_words)
        
        return {
            'eeg_features': eeg_features,
            'phoneme_labels': phoneme_labels,
            'words': words,
            'word_phonemes': word_phonemes
        }
    
    def train_with_accumulated_data(self, train_data, test_data=None, 
                                  validation_split=0.2, nn_epochs=50, 
                                  hmm_iter=100, patience=10):
        """
        Train the hybrid model using accumulated data.
        
        Parameters:
        -----------
        train_data : dict
            Training data from accumulate_phoneme_data
        test_data : dict or None
            Test data from accumulate_phoneme_data
        validation_split : float
            Proportion of training data to use for validation
        nn_epochs : int
            Maximum number of epochs for neural network training
        hmm_iter : int
            Number of iterations for HMM training
        patience : int
            Patience for early stopping
            
        Returns:
        --------
        dict
            Training and evaluation results
        """
        # Process training data
        processed_train = self.process_accumulated_data(train_data)
        
        # Train the hybrid model
        train_results = self.train_hybrid_model(
            processed_train['eeg_features'],
            processed_train['phoneme_labels'],
            processed_train['word_phonemes'],
            validation_split=validation_split,
            nn_epochs=nn_epochs,
            hmm_iter=hmm_iter,
            patience=patience
        )
        
        # Evaluate on test data if provided
        eval_results = None
        if test_data is not None:
            processed_test = self.process_accumulated_data(test_data)
            
            eval_results = self.evaluate(
                processed_test['eeg_features'],
                processed_test['phoneme_labels'],
                use_transitions=True
            )
        
        # Save the model
        self.save_model()
        
        # Return results
        return {
            'train_results': train_results,
            'eval_results': eval_results
        }