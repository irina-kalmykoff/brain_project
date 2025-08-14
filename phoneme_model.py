import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, 
    Input, TimeDistributed, Attention, Concatenate, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Bidirectional, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from debugger import DebugMixin

class PhonemeDecoderModel(DebugMixin):
    """
    A class to handle phoneme prediction from brain signals.
    This model is designed to work with the phoneme-level data processed
    by AcousticChangeDetector and CustomBrainAudioDecoder.
    """
    
    def __init__(self, model_type='lstm_cnn', output_dir='./models', decoder = None, debug_mode=False, **kwargs):
        """
        Initialize the phoneme decoder model.
        
        Parameters:
        -----------
        model_type : str
            Type of model architecture to use. Options:
            - 'lstm_cnn': LSTM with CNN feature extraction
            - 'transformer': Transformer-based model
            - 'attention': CNN with attention mechanism
            - 'ensemble': Ensemble of multiple models
        output_dir : str
            Directory to save model files and results
        debug_mode : bool
            Whether to enable debug mode
        """
        # Initialize the DebugMixin
        super().__init__(class_name="PhonemeDecoderModel", debug_mode=debug_mode)
        if debug_mode is not None:
            self.DEBUG_MODE = debug_mode
        self.log(f"Initialized with DEBUG_MODE={self.DEBUG_MODE}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model attributes
        self.model_type = model_type
        self.kwargs = kwargs
        self.decoder = decoder
        self.model = None
        self.label_encoder = LabelEncoder()
        
        self.log(f"Initialized PhonemeDecoderModel with model_type={model_type}")
    
    def prepare_data(self, features, labels, validation_split=0.2, test_split=0.1, max_length=None):
        """
        Prepare data for training and evaluation.
        """
        
        self.log(f"Preparing data with {len(features)} samples")
        
        # Check for empty inputs
        if not features or not labels:
            self.log("Error: Empty features or labels provided")
            return None
        
        # Fit label encoder
        self.label_encoder.fit(labels)
        self.phoneme_classes = self.label_encoder.classes_
        self.num_classes = len(self.phoneme_classes)
        
        self.log(f"Found {self.num_classes} unique phoneme classes")
        self.debug(f"Phoneme classes: {self.phoneme_classes}")
        
        # Encode labels
        encoded_labels = self.label_encoder.transform(labels)
        
        # Convert to one-hot encoding
        onehot_labels = to_categorical(encoded_labels, self.num_classes)
        
        # Split data into train, validation, and test sets
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, onehot_labels, test_size=test_split, random_state=42, stratify=encoded_labels
        )
        
        train_features, val_features, train_labels, val_labels = train_test_split(
            train_features, train_labels, test_size=validation_split/(1-test_split), 
            random_state=42, stratify=np.argmax(train_labels, axis=1)
        )
        
        self.log(f"Data split: {len(train_features)} train, {len(val_features)} validation, {len(test_features)} test")
        
        # Standardize sequence lengths
        if max_length is None:
            max_length = max(f.shape[0] for f in features)
        
        self.max_sequence_length = max_length
        self.debug(f"Using max sequence length: {self.max_sequence_length}")
        
        # Get feature dimension from first feature
        if features and features[0].ndim > 1:
            self.feature_dim = features[0].shape[1]
        else:
            self.feature_dim = 1
        
        self.input_shape = (self.max_sequence_length, self.feature_dim)
        self.debug(f"Input shape: {self.input_shape}")
        
        # Standardize lengths
        train_features_std = self._standardize_length(train_features, self.max_sequence_length)
        val_features_std = self._standardize_length(val_features, self.max_sequence_length)
        test_features_std = self._standardize_length(test_features, self.max_sequence_length)
        
        # Return prepared data
        return {
            'train_features': train_features_std,
            'train_labels': train_labels,
            'val_features': val_features_std,
            'val_labels': val_labels,
            'test_features': test_features_std,
            'test_labels': test_labels,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }
    
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
        standardized = []
        
        for feat in features:
            if feat.shape[0] > target_length:
                # Truncate
                standardized.append(feat[:target_length])
            elif feat.shape[0] < target_length:
                # Pad with zeros
                if feat.ndim > 1:
                    padding = np.zeros((target_length - feat.shape[0], feat.shape[1]))
                    standardized.append(np.vstack([feat, padding]))
                else:
                    # Handle 1D arrays
                    padding = np.zeros(target_length - feat.shape[0])
                    standardized.append(np.concatenate([feat, padding]))
            else:
                standardized.append(feat)
        
        # Convert to numpy array
        return np.array(standardized)
    
    def build_model(self):
        """
        Build the model architecture based on the specified model_type.
        
        Returns:
        --------
        tensorflow.keras.Model
            The constructed model
        """
        if self.input_shape is None or self.num_classes is None:
            self.log("Error: Data must be prepared before building the model")
            return None
        
        if self.model_type == 'lstm_cnn':
            self.model = self._build_lstm_cnn_model()
        elif self.model_type == 'transformer':
            self.model = self._build_transformer_model()
        elif self.model_type == 'attention':
            self.model = self._build_attention_model()
        elif self.model_type == 'ensemble':
            self.model = self._build_ensemble_model()
        else:
            self.log(f"Unknown model type: {self.model_type}. Using default LSTM-CNN model.")
            self.model = self._build_lstm_cnn_model()
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        self.model.summary()
        
        return self.model
    
    def _build_lstm_cnn_model(self):
        """
        Build an LSTM model with CNN feature extraction.
        
        Returns:
        --------
        tensorflow.keras.Model
            LSTM-CNN model
        """
        self.log("Building LSTM-CNN model")
        
        model = Sequential([
            # Convolutional layers for feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
                   input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            # Bidirectional LSTM layers for sequence modeling
            Bidirectional(LSTM(100, return_sequences=True)),
            Dropout(0.3),
            
            Bidirectional(LSTM(50)),
            Dropout(0.3),
            
            # Output layers
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _build_transformer_model(self):
        """
        Build a Transformer-based model.
        
        Returns:
        --------
        tensorflow.keras.Model
            Transformer model
        """
        self.log("Building Transformer model")
        
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Initial feature extraction
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        
        # Transformer blocks (simplified)
        for i in range(2):  # 2 transformer blocks
            # Multi-head self-attention
            attention_output = MultiHeadAttention(
                num_heads=8, key_dim=64
            )(x, x)
            
            # Add & normalize
            x = LayerNormalization()(x + attention_output)
            
            # Feed forward network
            ffn_output = Dense(128, activation='relu')(x)
            ffn_output = Dense(64)(ffn_output)
            
            # Add & normalize
            x = LayerNormalization()(x + ffn_output)
        
        # Final classification layers
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def _build_attention_model(self):
        """
        Build a CNN model with attention mechanism.
        
        Returns:
        --------
        tensorflow.keras.Model
            CNN with attention model
        """
        self.log("Building CNN with attention model")
        
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # CNN feature extraction (time-distributed)
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x)
        
        # Self-attention mechanism
        attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        
        # Combine attention with original features
        x = Concatenate()([x, attention_output])
        
        # LSTM layer to process the sequence
        x = Bidirectional(LSTM(64))(x)
        
        # Final dense layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def _build_ensemble_model(self):
        """
        Build an ensemble of models.
        Note: This implementation uses a soft voting ensemble within a single model.
        
        Returns:
        --------
        tensorflow.keras.Model
            Ensemble model
        """
        self.log("Building ensemble model")
        
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Branch 1: CNN
        cnn = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        cnn = MaxPooling1D(2)(cnn)
        cnn = Conv1D(128, 3, activation='relu', padding='same')(cnn)
        cnn = Flatten()(cnn)
        cnn = Dense(128, activation='relu')(cnn)
        
        # Branch 2: LSTM
        lstm = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        lstm = Bidirectional(LSTM(32))(lstm)
        lstm = Dense(128, activation='relu')(lstm)
        
        # Branch 3: Attention
        att = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        att_output = MultiHeadAttention(num_heads=4, key_dim=32)(att, att)
        att = Concatenate()([att, att_output])
        att = GlobalAveragePooling1D()(att)
        att = Dense(128, activation='relu')(att)
        
        # Combine branches
        combined = Concatenate()([cnn, lstm, att])
        combined = Dense(256, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        outputs = Dense(self.num_classes, activation='softmax')(combined)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def train(self, prepared_data, epochs=50, batch_size=32, patience=10):
        """
        Train the model using prepared data.
        
        Parameters:
        -----------
        prepared_data : dict
            Dictionary containing prepared data splits from prepare_data method
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Batch size for training
        patience : int
            Patience for early stopping
            
        Returns:
        --------
        tensorflow.keras.callbacks.History
            Training history
        """
        if self.model is None:
            self.log("Building model before training")
            self.build_model()
        
        # Extract data
        train_features = prepared_data['train_features']
        train_labels = prepared_data['train_labels']
        val_features = prepared_data['val_features']
        val_labels = prepared_data['val_labels']
        
        self.log(f"Training model with {len(train_features)} samples for up to {epochs} epochs")
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                os.path.join(self.output_dir, 'best_model.keras'),
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
        self.history = self.model.fit(
            train_features, train_labels,
            validation_data=(val_features, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the model and metadata
        self.save_model()
        
        # Plot training history
        self.plot_training_history()
        
        return self.history
    
    def evaluate(self, prepared_data=None, test_features=None, test_labels=None):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        prepared_data : dict or None
            Dictionary containing prepared data splits from prepare_data method
        test_features : numpy.ndarray or None
            Test features if not using prepared_data
        test_labels : numpy.ndarray or None
            Test labels if not using prepared_data
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            self.log("Error: Model must be trained or loaded before evaluation")
            return None
        
        # Extract test data
        if prepared_data is not None:
            test_features = prepared_data['test_features']
            test_labels = prepared_data['test_labels']
        elif test_features is None or test_labels is None:
            self.log("Error: Either prepared_data or test_features/test_labels must be provided")
            return None
        
        self.log(f"Evaluating model on {len(test_features)} test samples")
        
        # Evaluate the model
        metrics = self.model.evaluate(test_features, test_labels, verbose=1)
        
        # Get predictions
        predictions = self.model.predict(test_features)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(true_classes, predicted_classes)
        
        # Generate classification report
        class_report = classification_report(
            true_classes, predicted_classes,
            target_names=self.phoneme_classes,
            output_dict=True
        )
        
        # Log results
        self.log(f"Test loss: {metrics[0]:.4f}, Test accuracy: {metrics[1]:.4f}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(conf_matrix)
        
        # Return evaluation results
        return {
            'loss': metrics[0],
            'accuracy': metrics[1],
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': true_classes
        }
    
    def predict(self, features):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        features : list or numpy.ndarray
            Feature arrays to predict on
            
        Returns:
        --------
        tuple
            (predicted_phonemes, probabilities)
        """
        if self.model is None:
            self.log("Error: Model must be trained or loaded before prediction")
            return None, None
        
        # Standardize input length
        if isinstance(features, list):
            features = self._standardize_length(features, self.max_sequence_length)
        
        # Make predictions
        probabilities = self.model.predict(features)
        
        # Convert to phoneme classes
        predicted_indices = np.argmax(probabilities, axis=1)
        predicted_phonemes = self.label_encoder.inverse_transform(predicted_indices)
        
        return predicted_phonemes, probabilities
    
    def save_model(self, model_path=None):
        """
        Save the model and associated metadata.
        
        Parameters:
        -----------
        model_path : str or None
            Path to save the model. If None, uses the default path.
        """
        if self.model is None:
            self.log("Error: No model to save")
            return
        
        if model_path is None:
            model_path = os.path.join(self.output_dir, 'phoneme_decoder_model.keras')
        
        # Save the model
        self.model.save(model_path)
        self.log(f"Model saved to {model_path}")
        
        # Save label encoder classes
        classes_path = os.path.join(self.output_dir, 'phoneme_classes.npy')
        np.save(classes_path, self.phoneme_classes)
        
        # Save model metadata
        metadata_path = os.path.join(self.output_dir, 'model_metadata.npz')
        np.savez(
            metadata_path,
            model_type=self.model_type,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            max_sequence_length=self.max_sequence_length,
            feature_dim=self.feature_dim
        )
        
        self.log(f"Model metadata saved to {metadata_path}")
    
    def load_model(self, model_path=None, metadata_path=None, classes_path=None):
        """
        Load a saved model and associated metadata.
        
        Parameters:
        -----------
        model_path : str or None
            Path to the saved model. If None, uses the default path.
        metadata_path : str or None
            Path to the saved metadata. If None, uses the default path.
        classes_path : str or None
            Path to the saved phoneme classes. If None, uses the default path.
            
        Returns:
        --------
        bool
            True if loaded successfully, False otherwise
        """
        # Set default paths if not provided
        if model_path is None:
            model_path = os.path.join(self.output_dir, 'phoneme_decoder_model.keras')
        if metadata_path is None:
            metadata_path = os.path.join(self.output_dir, 'model_metadata.npz')
        if classes_path is None:
            classes_path = os.path.join(self.output_dir, 'phoneme_classes.npy')
        
        # Check if files exist
        if not os.path.exists(model_path):
            self.log(f"Error: Model file not found at {model_path}")
            return False
        
        # Load the model
        try:
            self.model = load_model(model_path)
            self.log(f"Model loaded from {model_path}")
        except Exception as e:
            self.log(f"Error loading model: {e}")
            return False
        
        # Load metadata if available
        if os.path.exists(metadata_path):
            try:
                metadata = np.load(metadata_path, allow_pickle=True)
                self.model_type = str(metadata['model_type'])
                self.input_shape = tuple(metadata['input_shape'])
                self.num_classes = int(metadata['num_classes'])
                self.max_sequence_length = int(metadata['max_sequence_length'])
                self.feature_dim = int(metadata['feature_dim'])
                self.log(f"Model metadata loaded from {metadata_path}")
            except Exception as e:
                self.log(f"Error loading metadata: {e}. Using default values.")
                # Try to infer from model
                if self.model.input_shape is not None:
                    self.input_shape = self.model.input_shape[1:]
                    self.max_sequence_length = self.input_shape[0]
                    self.feature_dim = self.input_shape[1]
                if self.model.output_shape is not None:
                    self.num_classes = self.model.output_shape[1]
        
        # Load phoneme classes if available
        if os.path.exists(classes_path):
            try:
                self.phoneme_classes = np.load(classes_path, allow_pickle=True)
                self.label_encoder.classes_ = self.phoneme_classes
                self.log(f"Phoneme classes loaded from {classes_path}")
            except Exception as e:
                self.log(f"Error loading phoneme classes: {e}")
        
        return True
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Parameters:
        -----------
        save_path : str or None
            Path to save the plot. If None, uses the default path.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.history is None:
            self.log("Error: No training history to plot")
            return None
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'training_history.png')
        
        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot accuracy
        axs[0].plot(self.history.history['accuracy'])
        axs[0].plot(self.history.history['val_accuracy'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].legend(['Train', 'Validation'], loc='upper left')
        
        # Plot loss
        axs[1].plot(self.history.history['loss'])
        axs[1].plot(self.history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.log(f"Training history plot saved to {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self, conf_matrix, save_path=None):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        conf_matrix : numpy.ndarray
            Confusion matrix to plot
        save_path : str or None
            Path to save the plot. If None, uses the default path.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if self.phoneme_classes is None:
            self.log("Error: Phoneme classes not available")
            return None
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        
        # Create figure
        plt.figure(figsize=(12, 10))
        plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        tick_marks = np.arange(len(self.phoneme_classes))
        plt.xticks(tick_marks, self.phoneme_classes, rotation=45)
        plt.yticks(tick_marks, self.phoneme_classes)
        
        # Add text
        thresh = conf_matrix.max() / 2
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Phoneme')
        plt.xlabel('Predicted Phoneme')
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.log(f"Confusion matrix plot saved to {save_path}")
        
        return plt.gcf()
    
    def compare_with_baseline(self, baseline_results, phoneme_results, save_path=None):
        """
        Compare phoneme model results with baseline model results.
        
        Parameters:
        -----------
        baseline_results : dict
            Results from the baseline model
        phoneme_results : dict
            Results from the phoneme model
        save_path : str or None
            Path to save the comparison plot. If None, uses the default path.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'model_comparison.png')
        
        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot accuracy comparison
        models = ['Baseline', 'Phoneme Model']
        accuracy = [
            baseline_results.get('accuracy', baseline_results.get('correlation', 0)),
            phoneme_results.get('accuracy', 0)
        ]
        
        axs[0].bar(models, accuracy)
        axs[0].set_title('Model Accuracy Comparison')
        axs[0].set_ylabel('Accuracy / Correlation')
        axs[0].set_ylim(0, 1)
        
        for i, v in enumerate(accuracy):
            axs[0].text(i, v + 0.05, f"{v:.4f}", ha='center')
        
        # Plot additional metrics if available
        if 'classification_report' in phoneme_results:
            report = phoneme_results['classification_report']
            metrics = ['precision', 'recall', 'f1-score']
            values = [report['macro avg'][m] for m in metrics]
            
            axs[1].bar(metrics, values)
            axs[1].set_title('Phoneme Model Metrics')
            axs[1].set_ylabel('Score')
            axs[1].set_ylim(0, 1)
            
            for i, v in enumerate(values):
                axs[1].text(i, v + 0.05, f"{v:.4f}", ha='center')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.log(f"Model comparison plot saved to {save_path}")
        
        return fig
    
    def train_with_accumulated_data(self, train_accumulated_data, test_accumulated_data=None, 
                             validation_split=0.2, epochs=50, batch_size=32, patience=10,
                             handle_unseen_phonemes='filter', resolve_unknown=True):
        """
        Train the model using data accumulated with detector.accumulate_phoneme_data.
        
        Parameters:
        -----------
        train_accumulated_data : dict
            Output from detector.accumulate_phoneme_data for training data
        test_accumulated_data : dict or None
            Output from detector.accumulate_phoneme_data for test data
        validation_split : float
            Proportion of training data to use for validation
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Batch size for training
        patience : int
            Patience for early stopping
        handle_unseen_phonemes : str
            How to handle phonemes in test set not seen in training:
            - 'filter': Remove instances with unseen phonemes
            - 'add': Add unseen phonemes to the label encoder (increases model complexity)
            
        Returns:
        --------
        dict
            Training results including history and evaluation metrics
        """
        self.log("Training model with accumulated phoneme data")
        
        train_phoneme_batch = {
                'phoneme_labels': train_accumulated_data['phoneme_labels'],
                'phoneme_spectrogram_segments': train_accumulated_data['spectrograms'] if 'spectrograms' in train_accumulated_data else [],
                'phoneme_words': train_accumulated_data['phoneme_words'],
                'phoneme_positions': train_accumulated_data.get('phoneme_positions', [0] * len(train_accumulated_data['phoneme_labels'])),
                'phoneme_participant_ids': train_accumulated_data.get('phoneme_participant_ids', ['unknown'] * len(train_accumulated_data['phoneme_labels']))
            }
        # Count unknown phonemes in training data
        unknown_train = train_phoneme_batch['phoneme_labels'].count('?')
        if unknown_train > 0:
            self.log(f"Training data contains {unknown_train} unknown phonemes ('?')")
            
            # Resolve unknown phonemes if requested
            if resolve_unknown and hasattr(self, 'resolve_unknown_phonemes'):
                self.log("Attempting to resolve unknown phonemes in training data...")
                resolved_train_batch = self.resolve_unknown_phonemes(train_phoneme_batch)
                
                # Update the training data with resolved phonemes
                train_accumulated_data['phoneme_labels'] = resolved_train_batch['phoneme_labels']
                
                
                # Count how many were resolved
                remaining_unknown = train_accumulated_data['phoneme_labels'].count('?')
                self.log(f"Resolved {unknown_train - remaining_unknown} of {unknown_train} unknown phonemes in training data")
        
        # For test data (if provided)
        if test_accumulated_data is not None:
            test_phoneme_batch = {
                'phoneme_labels': test_accumulated_data['phoneme_labels'],
                'phoneme_spectrogram_segments': test_accumulated_data['spectrograms'] if 'spectrograms' in test_accumulated_data else [],
                'phoneme_words': test_accumulated_data['phoneme_words'],
                'phoneme_positions': test_accumulated_data.get('phoneme_positions', [0] * len(test_accumulated_data['phoneme_labels'])),
                'phoneme_participant_ids': test_accumulated_data.get('phoneme_participant_ids', ['unknown'] * len(test_accumulated_data['phoneme_labels']))
            }
        
        # Count unknown phonemes in test data
        unknown_test = test_phoneme_batch['phoneme_labels'].count('?')
        if unknown_test > 0:
            self.log(f"Test data contains {unknown_test} unknown phonemes ('?')")
            
            # Resolve unknown phonemes if requested
            if resolve_unknown and hasattr(self, 'resolve_unknown_phonemes'):
                self.log("Attempting to resolve unknown phonemes in test data...")
                resolved_test_batch = self.resolve_unknown_phonemes(test_phoneme_batch)
                
                # Update the test data with resolved phonemes
                test_accumulated_data['phoneme_labels'] = resolved_test_batch['phoneme_labels']
                
                # Count how many were resolved
                remaining_unknown = test_accumulated_data['phoneme_labels'].count('?')
                self.log(f"Resolved {unknown_test - remaining_unknown} of {unknown_test} unknown phonemes in test data")



        # Extract features and labels from accumulated data
        train_features = train_accumulated_data['features']
        train_labels = train_accumulated_data['phoneme_labels']
        
        if not train_features or not train_labels:
            self.log("Error: Empty training features or labels")
            return None
        
        self.log(f"Training data: {len(train_features)} samples, {len(set(train_labels))} unique phonemes")
        
        # Process test data early to potentially update label set
        test_features = None
        test_labels = None
        
        if test_accumulated_data is not None:
            test_features = test_accumulated_data['features']
            test_labels = test_accumulated_data['phoneme_labels']
            
            if test_features and test_labels:
                self.log(f"Found {len(test_features)} test samples with {len(set(test_labels))} unique phonemes")
                
                # Identify phonemes in test set not in train set
                train_phonemes = set(train_labels)
                test_phonemes = set(test_labels)
                unseen_phonemes = test_phonemes - train_phonemes
                
                if unseen_phonemes:
                    self.log(f"Warning: Test set contains {len(unseen_phonemes)} phonemes not in training set: {sorted(unseen_phonemes)}")
                    
                    if handle_unseen_phonemes == 'add':
                        self.log("Adding unseen phonemes to label set")
                        # Add test phonemes to training set (just for label encoder)
                        # This doesn't add any training examples, just expands the label space
                        all_phonemes = list(train_labels) + list(unseen_phonemes)
                        # Fit label encoder on combined set
                        self.label_encoder.fit(all_phonemes)
                        self.phoneme_classes = self.label_encoder.classes_
                        self.num_classes = len(self.phoneme_classes)
                        self.log(f"Updated phoneme classes to {self.num_classes} total classes")
                    elif handle_unseen_phonemes == 'filter':
                        self.log("Filtering test samples with unseen phonemes")
                        # Filter out test samples with unseen phonemes
                        valid_indices = [i for i, label in enumerate(test_labels) if label in train_phonemes]
                        if valid_indices:
                            test_features = [test_features[i] for i in valid_indices]
                            test_labels = [test_labels[i] for i in valid_indices]
                            self.log(f"Filtered test set now has {len(test_features)} samples")
                        else:
                            self.log("Warning: No valid test samples remain after filtering!")
                            test_features = None
                            test_labels = None
        
        # If we haven't already expanded the label set with test phonemes:
        if handle_unseen_phonemes != 'add' or test_accumulated_data is None:
            # Fit label encoder on training data only
            self.label_encoder.fit(train_labels)
            self.phoneme_classes = self.label_encoder.classes_
            self.num_classes = len(self.phoneme_classes)
        
        self.log(f"Using {self.num_classes} unique phoneme classes")
        self.debug(f"Phoneme classes: {self.phoneme_classes}")
        
        # Diagnose class distribution
        encoded_labels = self.label_encoder.transform(train_labels)
        class_counts = np.bincount(encoded_labels)
        rare_classes = np.where(class_counts == 1)[0]
        if len(rare_classes) > 0:
            self.log(f"Warning: Found {len(rare_classes)} phoneme classes with only 1 sample:")
            for rare_idx in rare_classes:
                rare_phoneme = self.phoneme_classes[rare_idx]
                self.log(f"  - Phoneme '{rare_phoneme}' has only 1 sample")
        
        # Encode training labels
        onehot_labels = to_categorical(encoded_labels, self.num_classes)
        
        # Get feature dimension from first feature
        if train_features and train_features[0].ndim > 1:
            self.feature_dim = train_features[0].shape[1]
        else:
            self.feature_dim = 1
        
        # Determine max sequence length
        max_length = max(f.shape[0] for f in train_features)
        self.max_sequence_length = max_length
        self.input_shape = (self.max_sequence_length, self.feature_dim)
        
        # Standardize feature lengths
        train_features_std = self._standardize_length(train_features, self.max_sequence_length)
        
        # Split into train and validation
        if validation_split > 0:
            try:
                # Try to create validation split with stratification
                if len(rare_classes) > 0:
                    self.log("Warning: Some phoneme classes have only 1 sample. Disabling stratification.")
                    # Create validation split without stratification
                    train_idx, val_idx = train_test_split(
                        np.arange(len(train_features_std)),
                        test_size=validation_split,
                        random_state=42,
                        stratify=None  # Disable stratification
                    )
                else:
                    # Create validation split with stratification
                    train_idx, val_idx = train_test_split(
                        np.arange(len(train_features_std)),
                        test_size=validation_split,
                        random_state=42,
                        stratify=encoded_labels
                    )
                
                X_train = train_features_std[train_idx]
                y_train = onehot_labels[train_idx]
                X_val = train_features_std[val_idx]
                y_val = onehot_labels[val_idx]
                
                self.log(f"Training data split: {len(X_train)} train, {len(X_val)} validation samples")
            except ValueError as e:
                self.log(f"Error in train/validation split: {e}. Using all data for training.")
                X_train = train_features_std
                y_train = onehot_labels
                X_val = None
                y_val = None
        else:
            # No validation split
            X_train = train_features_std
            y_train = onehot_labels
            X_val = None
            y_val = None
            self.log("No validation split requested. Using all data for training.")
        
        # Process test data
        X_test = None
        y_test = None
        if test_features and test_labels:
            self.log(f"Processing test data: {len(test_features)} samples")
            
            # Standardize test features
            X_test = self._standardize_length(test_features, self.max_sequence_length)
            
            try:
                # Transform using the label encoder
                test_encoded_labels = self.label_encoder.transform(test_labels)
                y_test = to_categorical(test_encoded_labels, self.num_classes)
                self.log(f"Processed test data: {len(X_test)} samples")
            except ValueError as e:
                self.log(f"Error encoding test labels: {e}. Test evaluation will be skipped.")
                X_test = None
                y_test = None
        
        # Build the model if not already built
        if self.model is None:
            self.build_model()
        
        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                os.path.join(self.output_dir, 'best_model.keras'),
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        if X_val is not None:
            self.log(f"Training model with validation data for up to {epochs} epochs")
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.log(f"Training model without validation data for up to {epochs} epochs")
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        # Evaluate on test data if available
        evaluation_results = {}
        if X_test is not None and y_test is not None:
            self.log("Evaluating model on test data")
            metrics = self.model.evaluate(X_test, y_test, verbose=1)
            
            # Get predictions
            predictions = self.model.predict(X_test)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_test, axis=1)
            
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(true_classes, predicted_classes)
            
            unique_test_labels = np.unique(true_classes)
            report_target_names = [self.phoneme_classes[i] for i in unique_test_labels]

            # Generate classification report
            class_report = classification_report(
                true_classes, predicted_classes,
                labels=unique_test_labels,  # Only use labels present in test data
                target_names=report_target_names,
                output_dict=True
            )
            
            evaluation_results = {
                'loss': metrics[0],
                'accuracy': metrics[1],
                'confusion_matrix': conf_matrix,
                'classification_report': class_report,
                'predictions': predictions,
                'predicted_classes': predicted_classes,
                'true_classes': true_classes
            }
            
            self.log(f"Test loss: {metrics[0]:.4f}, Test accuracy: {metrics[1]:.4f}")
        else:
            # Evaluate on validation data if no test data
            self.log("No valid test data available. Using validation metrics.")
            if X_val is not None:
                evaluation_results = {
                    'accuracy': max(self.history.history['val_accuracy']),
                    'loss': min(self.history.history['val_loss'])
                }
            else:
                evaluation_results = {
                    'accuracy': max(self.history.history['accuracy']),
                    'loss': min(self.history.history['loss'])
                }
        
        # Save the model and metadata
        self.save_model()
        
        # Plot training history
        self.plot_training_history()
        
        return {
            'history': self.history,
            'evaluation': evaluation_results,
            'model': self.model
        }