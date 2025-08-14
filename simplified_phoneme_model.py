import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, 
    Input, Bidirectional, BatchNormalization, Concatenate
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from debugger import DebugMixin

class SimplifiedPhonemeModel(DebugMixin):
    """
    A simplified model for phoneme classification using grouped phoneme categories.
    Designed to work with limited EEG data from multiple participants.
    """
    
    def __init__(self, phonetic_dict=None, output_dir='./models/simplified_phoneme', debug_mode=False):
        # Initialize the DebugMixin
        super().__init__(class_name="SimplifiedPhonemeModel", debug_mode=debug_mode)
        if debug_mode is not None:
            self.DEBUG_MODE = debug_mode
        self.log(f"Initialized with DEBUG_MODE={self.DEBUG_MODE}")
        
        # Store parameters
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up phonetic dictionary
        if phonetic_dict is None:
            from phonetic_dictionary import PhoneticDictionary
            self.phonetic_dict = PhoneticDictionary()
        else:
            self.phonetic_dict = phonetic_dict
        
        # Ensure phoneme groups are available
        if not hasattr(self.phonetic_dict, 'phoneme_groups'):
            self.phonetic_dict.add_phoneme_groups()
        
        # Get phoneme groups from the dictionary
        self.phoneme_groups = self.phonetic_dict.phoneme_groups
        self.phoneme_to_group = self.phonetic_dict.phoneme_to_group
        
        # Initialize model attributes
        self.model = None
        self.group_encoder = LabelEncoder()
        self.history = None
        self.input_shape = None
        self.num_groups = len(self.phoneme_groups)
        self.phoneme_group_names = list(self.phoneme_groups.keys())
        
        # Fit the group encoder
        self.group_encoder.fit(self.phoneme_group_names)
        
        self.log(f"Initialized SimplifiedPhonemeModel with {self.num_groups} phoneme groups")
        self.debug(f"Phoneme groups: {self.phoneme_groups}")
    

    
    def prepare_data(self, features, phoneme_labels, participant_ids=None, 
               include_participant_info=True, validation_split=0.2, test_split=0.1, 
               max_length=None, balance_classes=True, padding_method='interpolate'):
        """
        Prepare data for training and evaluation, mapping phonemes to groups.
        
        Parameters:
        -----------
        features : list
            List of feature arrays for each phoneme segment
        phoneme_labels : list
            List of phoneme labels corresponding to each feature array
        participant_ids : list or None
            List of participant IDs for each segment (optional)
        include_participant_info : bool
            Whether to include participant ID as a feature
        validation_split : float
            Proportion of data to use for validation
        test_split : float
            Proportion of data to use for testing
        max_length : int or None
            Maximum sequence length for standardization
        balance_classes : bool
            Whether to balance class representation in training set
        padding_method : str
            Method for standardizing sequence lengths:
              'truncate_pad': Truncate longer sequences and pad shorter ones (original)
              'zero_pad': Only pad shorter sequences but keep longer ones intact
              'interpolate': Resample all sequences to target_length
            
        Returns:
        --------
        dict
            Dictionary containing processed data splits
        """
        # [... rest of the method unchanged until standardization part ...]
        
        # Standardize sequence lengths
        if max_length is None:
            if padding_method == 'zero_pad':
                # For zero_pad, we might want the longest sequence
                max_length = max(f.shape[0] for f in features)
            else:
                # For other methods, we might want a more typical length
                lengths = [f.shape[0] for f in features]
                max_length = int(np.percentile(lengths, 90))  # Use 90th percentile to avoid outliers
        
        self.max_sequence_length = max_length
        self.debug(f"Using {padding_method} with max sequence length: {self.max_sequence_length}")
        
        # Get feature dimension from first feature
        if features and features[0].ndim > 1:
            self.feature_dim = features[0].shape[1]
        else:
            self.feature_dim = 1
        
        self.input_shape = (self.max_sequence_length, self.feature_dim)
        self.debug(f"Input shape: {self.input_shape}")
        
        # Standardize using the selected method
        train_features_std = self._standardize_length(train_features, self.max_sequence_length, method=padding_method)
        val_features_std = self._standardize_length(val_features, self.max_sequence_length, method=padding_method)
        test_features_std = self._standardize_length(test_features, self.max_sequence_length, method=padding_method)
        
        # Return prepared data
        return {
            'train_features': train_features_std,
            'train_labels': train_groups,
            'train_participants': train_participants,
            'val_features': val_features_std,
            'val_labels': val_groups,
            'val_participants': val_participants,
            'test_features': test_features_std,
            'test_labels': test_groups,
            'test_participants': test_participants,
            'input_shape': self.input_shape,
            'num_groups': self.num_groups,
            'participant_dim': participant_features.shape[1] if participant_features is not None else None
        }
    
    def _standardize_length(self, features, target_length, method='truncate_pad'):
        """
        Standardize the length of feature arrays using various methods.
        
        Parameters:
        -----------
        features : list
            List of feature arrays
        target_length : int
            Target sequence length
        method : str
            Method to use: 
              'truncate_pad': Truncate longer sequences and pad shorter ones (original)
              'zero_pad': Only pad shorter sequences but keep longer ones intact
              'interpolate': Resample all sequences to target_length
                
        Returns:
        --------
        numpy.ndarray
            Standardized features array with shape (n_samples, target_length, feature_dim)
        """
        standardized = []
        
        if method == 'interpolate':
            from scipy.interpolate import interp1d
            
            for feat in features:
                # Get original time points
                orig_time = np.arange(feat.shape[0])
                # Generate new time points
                new_time = np.linspace(0, feat.shape[0]-1, target_length)
                
                # Create interpolated feature
                if feat.ndim > 1:
                    # For 2D arrays (time x features)
                    resampled = np.zeros((target_length, feat.shape[1]))
                    for j in range(feat.shape[1]):
                        if len(orig_time) > 1:  # Need at least 2 points for interpolation
                            interp_func = interp1d(
                                orig_time, feat[:, j], 
                                kind='linear', 
                                bounds_error=False, 
                                fill_value='extrapolate'
                            )
                            resampled[:, j] = interp_func(new_time)
                        else:
                            # If only one time point, duplicate it
                            resampled[:, j] = feat[0, j]
                else:
                    # For 1D arrays
                    if len(orig_time) > 1:
                        interp_func = interp1d(
                            orig_time, feat, 
                            kind='linear', 
                            bounds_error=False, 
                            fill_value='extrapolate'
                        )
                        resampled = interp_func(new_time)
                    else:
                        resampled = np.full(target_length, feat[0])
                
                standardized.append(resampled)
        
        elif method == 'zero_pad':
            for feat in features:
                # Keep original length if longer than target
                if feat.shape[0] > target_length:
                    standardized.append(feat)
                elif feat.shape[0] < target_length:
                    # Zero padding for shorter sequences
                    if feat.ndim > 1:
                        padding = np.zeros((target_length - feat.shape[0], feat.shape[1]))
                        standardized.append(np.vstack([feat, padding]))
                    else:
                        padding = np.zeros(target_length - feat.shape[0])
                        standardized.append(np.concatenate([feat, padding]))
                else:
                    standardized.append(feat)
        
        else:  # Default: 'truncate_pad'
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
                        padding = np.zeros(target_length - feat.shape[0])
                        standardized.append(np.concatenate([feat, padding]))
                else:
                    standardized.append(feat)
        
        # Convert to numpy array
        return np.array(standardized)
        
    
    
    def build_model(self, include_participant_info=False, participant_dim=None):
        """
        Build the simplified model architecture.
        
        Parameters:
        -----------
        include_participant_info : bool
            Whether to include participant ID as an input feature
        participant_dim : int or None
            Dimension of participant one-hot encoding
            
        Returns:
        --------
        tensorflow.keras.Model
            The constructed model
        """
        if self.input_shape is None or self.num_groups is None:
            self.log("Error: Data must be prepared before building the model", level="ERROR")
            return None
        
        self.log("Building simplified phoneme model")
        
        if include_participant_info and participant_dim is not None:
            # Model with participant information as additional input
            # Main input for EEG features
            eeg_input = Input(shape=self.input_shape, name='eeg_input')
            
            # Participant input
            participant_input = Input(shape=(participant_dim,), name='participant_input')
            
            # Process EEG with CNN
            x = Conv1D(32, 5, activation='relu', padding='same')(eeg_input)
            x = BatchNormalization()(x)
            x = MaxPooling1D(2)(x)
            
            x = Conv1D(64, 5, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(2)(x)
            
            # Process with LSTM
            x = Bidirectional(LSTM(64))(x)
            x = Dropout(0.5)(x)
            
            # Flatten and prepare for concatenation
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            
            # Process participant information
            p = Dense(16, activation='relu')(participant_input)
            p = BatchNormalization()(p)
            
            # Combine EEG features with participant info
            combined = Concatenate()([x, p])
            
            # Final classification layers
            combined = Dropout(0.5)(combined)
            output = Dense(self.num_groups, activation='softmax')(combined)
            
            # Create model with two inputs
            model = Model(inputs=[eeg_input, participant_input], outputs=output)
            
        else:
            # Simpler model without participant information
            model = Sequential([
                # Feature extraction with CNN
                Conv1D(32, 5, activation='relu', padding='same', input_shape=self.input_shape),
                BatchNormalization(),
                MaxPooling1D(2),
                
                Conv1D(64, 5, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling1D(2),
                
                # Sequence modeling with LSTM
                Bidirectional(LSTM(64)),
                Dropout(0.5),
                
                # Classification head
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(self.num_groups, activation='softmax')
            ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        self.model = model
        return model
    
    def train(self, prepared_data, epochs=50, batch_size=16, patience=10):
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
            # Check if we need to include participant info
            include_participant = (prepared_data.get('train_participants') is not None)
            participant_dim = prepared_data.get('participant_dim')
            
            self.log(f"Building model with participant info: {include_participant}")
            self.build_model(include_participant_info=include_participant, 
                          participant_dim=participant_dim)
        
        # Extract data
        train_features = prepared_data['train_features']
        train_labels = prepared_data['train_labels']
        val_features = prepared_data['val_features']
        val_labels = prepared_data['val_labels']
        
        # Extract participant information if available
        train_participants = prepared_data.get('train_participants')
        val_participants = prepared_data.get('val_participants')
        
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
        if train_participants is not None and val_participants is not None:
            # Train with participant information
            self.history = self.model.fit(
                [train_features, train_participants], train_labels,
                validation_data=([val_features, val_participants], val_labels),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Train without participant information
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
    
    def evaluate(self, prepared_data):
        """
        Evaluate the model on test data.
        """
        if self.model is None:
            self.log("Error: Model must be trained or loaded before evaluation", level="ERROR")
            return None
        
        # Extract test data
        test_features = prepared_data['test_features']
        test_labels = prepared_data['test_labels']
        test_participants = prepared_data.get('test_participants')
        
        self.log(f"Evaluating model on {len(test_features)} test samples")
        
        # Evaluate the model
        if test_participants is not None:
            metrics = self.model.evaluate([test_features, test_participants], test_labels, verbose=1)
            predictions = self.model.predict([test_features, test_participants])
        else:
            metrics = self.model.evaluate(test_features, test_labels, verbose=1)
            predictions = self.model.predict(test_features)
        
        # Get predicted groups
        predicted_groups = np.argmax(predictions, axis=1)
        true_groups = np.argmax(test_labels, axis=1)
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(true_groups, predicted_groups)
        
        # Get all unique classes used in the encoding
        all_classes = list(self.group_encoder.classes_)
        
        # Generate classification report
        class_report = classification_report(
            true_groups, predicted_groups,
            target_names=all_classes,  # Use the actual classes from the encoder
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
            'predicted_groups': predicted_groups,
            'true_groups': true_groups,
            'group_names': all_classes
        }
    
    def predict(self, features, participant_ids=None):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        features : list or numpy.ndarray
            Feature arrays to predict on
        participant_ids : list or None
            Participant IDs if the model was trained with participant information
            
        Returns:
        --------
        tuple
            (predicted_phoneme_groups, probabilities)
        """
        if self.model is None:
            self.log("Error: Model must be trained or loaded before prediction", level="ERROR")
            return None, None
        
        # Standardize input length
        if isinstance(features, list):
            features = self._standardize_length(features, self.max_sequence_length)
        
        # Make predictions
        if participant_ids is not None and hasattr(self.model, 'inputs') and len(self.model.inputs) > 1:
            # Encode participant IDs
            participant_encoder = LabelEncoder()
            participant_encoder.fit(participant_ids)
            encoded_participants = participant_encoder.transform(participant_ids)
            
            # Convert to one-hot
            participant_features = to_categorical(encoded_participants)
            
            # Predict with participant information
            probabilities = self.model.predict([features, participant_features])
        else:
            # Predict without participant information
            probabilities = self.model.predict(features)
        
        # Convert to phoneme group names
        predicted_indices = np.argmax(probabilities, axis=1)
        predicted_groups = self.group_encoder.inverse_transform(predicted_indices)
        
        return predicted_groups, probabilities
    
    def save_model(self, model_path=None):
        """
        Save the model and associated metadata.
        """
        if self.model is None:
            self.log("Error: No model to save", level="ERROR")
            return
        
        if model_path is None:
            model_path = os.path.join(self.output_dir, 'simplified_phoneme_model.keras')
        
        # Save the model
        self.model.save(model_path)
        self.log(f"Model saved to {model_path}")
        
        # Save phoneme groups
        groups_path = os.path.join(self.output_dir, 'phoneme_groups.npz')
        np.savez(
            groups_path,
            phoneme_groups=self.phoneme_groups,
            phoneme_group_names=self.phoneme_group_names,
            group_encoder_classes=self.group_encoder.classes_
        )
        
        # Save model metadata
        metadata_path = os.path.join(self.output_dir, 'model_metadata.npz')
        np.savez(
            metadata_path,
            input_shape=self.input_shape,
            max_sequence_length=self.max_sequence_length,
            feature_dim=self.feature_dim,
            num_groups=self.num_groups
        )
        
        self.log(f"Model metadata saved to {metadata_path}")
    
    def load_model(self, model_path=None, groups_path=None, metadata_path=None):
        """
        Load a saved model and associated metadata.
        """
        # Set default paths if not provided
        if model_path is None:
            model_path = os.path.join(self.output_dir, 'simplified_phoneme_model.keras')
        if groups_path is None:
            groups_path = os.path.join(self.output_dir, 'phoneme_groups.npz')
        if metadata_path is None:
            metadata_path = os.path.join(self.output_dir, 'model_metadata.npz')
        
        # Check if files exist
        if not os.path.exists(model_path):
            self.log(f"Error: Model file not found at {model_path}", level="ERROR")
            return False
        
        # Load the model
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.log(f"Model loaded from {model_path}")
        except Exception as e:
            self.log(f"Error loading model: {e}", level="ERROR")
            return False
        
        # Load phoneme groups if available
        if os.path.exists(groups_path):
            try:
                groups_data = np.load(groups_path, allow_pickle=True)
                
                if 'phoneme_groups' in groups_data:
                    self.phoneme_groups = groups_data['phoneme_groups'].item()
                
                if 'phoneme_group_names' in groups_data:
                    self.phoneme_group_names = groups_data['phoneme_group_names'].tolist()
                    self.num_groups = len(self.phoneme_group_names)
                
                if 'group_encoder_classes' in groups_data:
                    self.group_encoder.classes_ = groups_data['group_encoder_classes']
                
                # Recreate reverse mapping
                self.phoneme_to_group = {}
                for group, phonemes in self.phoneme_groups.items():
                    for phoneme in phonemes:
                        self.phoneme_to_group[phoneme] = group
                
                self.log(f"Phoneme groups loaded from {groups_path}")
            except Exception as e:
                self.log(f"Error loading phoneme groups: {e}", level="WARNING")
        
        # Load metadata if available
        if os.path.exists(metadata_path):
            try:
                metadata = np.load(metadata_path, allow_pickle=True)
                
                if 'input_shape' in metadata:
                    self.input_shape = tuple(metadata['input_shape'])
                
                if 'max_sequence_length' in metadata:
                    self.max_sequence_length = int(metadata['max_sequence_length'])
                
                if 'feature_dim' in metadata:
                    self.feature_dim = int(metadata['feature_dim'])
                
                if 'num_groups' in metadata:
                    self.num_groups = int(metadata['num_groups'])
                
                self.log(f"Model metadata loaded from {metadata_path}")
            except Exception as e:
                self.log(f"Error loading metadata: {e}", level="WARNING")
        
        return True
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if self.history is None:
            self.log("Error: No training history to plot", level="ERROR")
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
        """Plot confusion matrix"""
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        
        # Get all unique classes from the encoder
        all_classes = list(self.group_encoder.classes_)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        tick_marks = np.arange(len(all_classes))
        plt.xticks(tick_marks, all_classes, rotation=45)
        plt.yticks(tick_marks, all_classes)
        
        # Add text
        thresh = conf_matrix.max() / 2
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Group')
        plt.xlabel('Predicted Group')
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.log(f"Confusion matrix plot saved to {save_path}")
        
        return plt.gcf()
    

    def plot_detailed_confusion_matrix(self, conf_matrix, save_path=None):
        """
        Plot a detailed confusion matrix with percentages and counts.
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'detailed_confusion_matrix.png')
        
        # Get all unique classes from the encoder
        all_classes = list(self.group_encoder.classes_)
        
        # Create figure (make it larger for better readability)
        plt.figure(figsize=(12, 10))
        
        # Calculate percentages per row (true class)
        conf_pct = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_pct = np.nan_to_num(conf_pct)  # Replace NaN with 0
        
        # Plot with percentages
        plt.imshow(conf_pct, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        plt.title('Confusion Matrix (Normalized by True Class)', fontsize=14)
        plt.colorbar(label='Percentage')
        
        # Add labels
        tick_marks = np.arange(len(all_classes))
        plt.xticks(tick_marks, all_classes, rotation=45, fontsize=10)
        plt.yticks(tick_marks, all_classes, fontsize=10)
        
        # Add text showing both percentage and count
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                if conf_pct[i, j] > 0:
                    plt.text(j, i, f"{conf_pct[i, j]:.1%}\n({conf_matrix[i, j]})",
                            ha="center", va="center", 
                            color="white" if conf_pct[i, j] > 0.5 else "black",
                            fontsize=9)
        
        plt.tight_layout()
        plt.ylabel('True Phoneme Group', fontsize=12)
        plt.xlabel('Predicted Phoneme Group', fontsize=12)
        
        # Add a descriptive subtitle
        plt.figtext(0.5, 0.01, 
                    "Each cell shows: percentage of true class predicted as that class (count of examples)",
                    ha="center", fontsize=10)
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.log(f"Detailed confusion matrix saved to {save_path}")
        
        return plt.gcf()
    
    def analyze_participant_performance(self, features, phoneme_labels, participant_ids, save_path=None):
        """
        Analyze model performance per participant.
        
        Parameters:
        -----------
        features : list or numpy.ndarray
            Feature arrays
        phoneme_labels : list
            Phoneme labels
        participant_ids : list
            Participant IDs
        save_path : str or None
            Path to save the analysis
            
        Returns:
        --------
        dict
            Dictionary containing performance metrics per participant
        """
        if self.model is None:
            self.log("Error: Model must be trained or loaded before analysis", level="ERROR")
            return None
        
        # Map phonemes to groups
        group_labels = self.map_phonemes_to_groups(phoneme_labels)
        
        # Filter out unknown groups
        valid_indices = [i for i, group in enumerate(group_labels) if group != 'unknown']
        
        features = [features[i] for i in valid_indices]
        group_labels = [group_labels[i] for i in valid_indices]
        participant_ids = [participant_ids[i] for i in valid_indices]
        
        # Standardize features
        features_std = self._standardize_length(features, self.max_sequence_length)
        
        # Encode group labels
        encoded_groups = self.group_encoder.transform([g for g in group_labels if g in self.phoneme_group_names])
        onehot_groups = to_categorical(encoded_groups, self.num_groups)
        
        # Get unique participants
        unique_participants = sorted(set(participant_ids))
        
        # Initialize results
        participant_results = {}
        
        # Analyze each participant
        for participant in unique_participants:
            # Get indices for this participant
            participant_indices = [i for i, p_id in enumerate(participant_ids) if p_id == participant]
            
            if len(participant_indices) < 5:  # Skip if too few samples
                self.log(f"Skipping {participant}: only {len(participant_indices)} samples")
                continue
            
            # Get participant data
            participant_features = features_std[participant_indices]
            participant_groups = onehot_groups[participant_indices]
            
            # Predict
            if hasattr(self.model, 'inputs') and len(self.model.inputs) > 1:
                # For models with participant input, create one-hot
                participant_encoder = LabelEncoder()
                participant_encoder.fit(unique_participants)
                encoded_p = participant_encoder.transform([participant] * len(participant_indices))
                participant_onehot = to_categorical(encoded_p, len(unique_participants))
                
                # Predict with participant info
                predictions = self.model.predict([participant_features, participant_onehot])
            else:
                # Predict without participant info
                predictions = self.model.predict(participant_features)
            
            # Evaluate
            predicted_groups = np.argmax(predictions, axis=1)
            true_groups = np.argmax(participant_groups, axis=1)
            
            # Calculate accuracy
            accuracy = np.mean(predicted_groups == true_groups)
            
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(true_groups, predicted_groups, 
                                        labels=range(self.num_groups))
            
            # Store results
            participant_results[participant] = {
                'accuracy': accuracy,
                'sample_count': len(participant_indices),
                'confusion_matrix': conf_matrix,
                'predictions': predictions,
                'true_groups': true_groups
            }
            
            self.log(f"Participant {participant}: accuracy={accuracy:.4f}, samples={len(participant_indices)}")
        
        # Create summary plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'participant_performance.png')
        
        # Sort participants by accuracy
        sorted_participants = sorted(participant_results.items(), 
                                    key=lambda x: x[1]['accuracy'], 
                                    reverse=True)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Extract data for plotting
        p_ids = [p[0] for p in sorted_participants]
        accuracies = [p[1]['accuracy'] for p in sorted_participants]
        sample_counts = [p[1]['sample_count'] for p in sorted_participants]
        
        # Create bar plot
        bars = plt.bar(p_ids, accuracies)
        
        # Add sample count as text
        for i, (bar, count) in enumerate(zip(bars, sample_counts)):
            plt.text(bar.get_x() + bar.get_width()/2, 0.05,
                    f'n={count}', ha='center', va='bottom',
                    rotation=90, color='white', fontweight='bold')
        
        # Add labels and title
        plt.xlabel('Participant ID')
        plt.ylabel('Accuracy')
        plt.title('Model Performance by Participant')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return participant_results
        
    def analyze_class_performance(self, true_groups, predicted_groups, probabilities=None, save_path=None):
        """
        Analyze performance metrics for each phoneme group.
        
        Parameters:
        -----------
        true_groups : numpy.ndarray
            True group indices
        predicted_groups : numpy.ndarray
            Predicted group indices
        probabilities : numpy.ndarray or None
            Prediction probabilities (optional)
        save_path : str or None
            Path to save the plot
            
        Returns:
        --------
        dict
            Dictionary containing metrics for each class
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'class_performance.png')
        
        # Get all unique classes from the encoder
        all_classes = list(self.group_encoder.classes_)
        
        # Convert indices to class names
        true_class_names = [all_classes[idx] for idx in true_groups]
        predicted_class_names = [all_classes[idx] for idx in predicted_groups]
        
        # Calculate metrics per class
        precision, recall, f1, support = precision_recall_fscore_support(
            true_groups, predicted_groups, labels=range(len(all_classes))
        )
        
        # Calculate confusion confidence (average probability for each class)
        confidence = np.zeros(len(all_classes))
        if probabilities is not None:
            for i, class_idx in enumerate(range(len(all_classes))):
                # Get indices of samples predicted as this class
                pred_indices = np.where(predicted_groups == class_idx)[0]
                if len(pred_indices) > 0:
                    # Get the prediction probabilities for this class
                    class_probs = probabilities[pred_indices, class_idx]
                    confidence[i] = np.mean(class_probs)
        
        # Create a dataframe for easier analysis
        metrics_df = pd.DataFrame({
            'Phoneme Group': all_classes,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Support': support,
            'Confidence': confidence
        })
        
        # Sort by F1 score for better visualization
        metrics_df = metrics_df.sort_values('F1 Score', ascending=False)
        
        # Create a visualization
        fig, axs = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot precision, recall, and F1
        bar_width = 0.25
        indices = np.arange(len(metrics_df))
        
        axs[0].bar(indices - bar_width, metrics_df['Precision'], bar_width, 
                 label='Precision', color='#3498db')
        axs[0].bar(indices, metrics_df['Recall'], bar_width, 
                 label='Recall', color='#2ecc71')
        axs[0].bar(indices + bar_width, metrics_df['F1 Score'], bar_width, 
                 label='F1 Score', color='#e74c3c')
        
        axs[0].set_xlabel('Phoneme Group')
        axs[0].set_ylabel('Score')
        axs[0].set_title('Precision, Recall, and F1 Score by Phoneme Group')
        axs[0].set_xticks(indices)
        axs[0].set_xticklabels(metrics_df['Phoneme Group'], rotation=45)
        axs[0].legend()
        axs[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add text on top of bars for F1 score
        for i, v in enumerate(metrics_df['F1 Score']):
            axs[0].text(i + bar_width, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
        
        # Plot support (number of examples)
        bars = axs[1].bar(metrics_df['Phoneme Group'], metrics_df['Support'], color='#9b59b6')
        axs[1].set_xlabel('Phoneme Group')
        axs[1].set_ylabel('Number of Examples')
        axs[1].set_title('Number of Examples per Phoneme Group')
        axs[1].set_xticklabels(metrics_df['Phoneme Group'], rotation=45)
        axs[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add text on top of bars for support count
        for bar in bars:
            height = bar.get_height()
            axs[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f"{int(height)}", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.log(f"Class performance analysis saved to {save_path}")
        
        # Also save the metrics to CSV
        csv_path = os.path.join(self.output_dir, 'class_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        self.log(f"Class metrics saved to {csv_path}")
        
        return metrics_df.to_dict(orient='records')
    
    def train_with_grouped_data(self, train_features, train_group_labels, train_participant_ids=None,
                      test_features=None, test_group_labels=None, test_participant_ids=None,
                      include_participant_info=True, epochs=50, batch_size=16, patience=10,
                      min_occurrences=5):  # Add min_occurrences parameter
        """
        Train the model with pre-processed phoneme group data.
        This method works with your existing pipeline and just handles the final model training.
        
        Parameters:
        -----------
        train_features : list
            List of feature arrays for training
        train_group_labels : list
            List of phoneme group labels for training
        train_participant_ids : list or None
            List of participant IDs for training
        test_features : list or None
            List of feature arrays for testing
        test_group_labels : list or None
            List of phoneme group labels for testing
        test_participant_ids : list or None
            List of participant IDs for testing
        include_participant_info : bool
            Whether to include participant information
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        patience : int
            Patience for early stopping
        min_occurrences : int
            Minimum number of occurrences required for a phoneme group
            
        Returns:
        --------
        dict
            Training results
        """
        self.log(f"Training model with pre-processed phoneme group data")
        
        # 1. Count group occurrences for filtering
        from collections import Counter
        group_counts = Counter(train_group_labels)
        
        self.log("Phoneme group distribution before filtering:")
        for group, count in group_counts.most_common():
            self.log(f"  {group}: {count}")
        
        # Identify rare groups (fewer than min_occurrences)
        rare_groups = {group for group, count in group_counts.items() 
                      if count < min_occurrences}
        
        if rare_groups:
            self.log(f"Filtering out {len(rare_groups)} rare groups with fewer than {min_occurrences} occurrences: {rare_groups}")
        
        # 2. Get unique groups (excluding rare ones)
        unique_groups = [group for group in self.phoneme_groups.keys() 
                        if group not in rare_groups]
        
        if 'unknown' not in unique_groups and 'unknown' not in rare_groups:
            unique_groups.append('unknown')
        
        # Make sure group_encoder is properly fitted
        self.group_encoder.fit(unique_groups)
        self.num_groups = len(unique_groups)
        
        # 3. Filter for valid group labels (those that are in our known groups and not rare)
        valid_train_indices = [i for i, g in enumerate(train_group_labels) 
                             if g in unique_groups]
        
        self.log(f"Training with {len(valid_train_indices)} samples (filtered from {len(train_group_labels)})")
        if len(valid_train_indices) < len(train_group_labels):
            skipped = len(train_group_labels) - len(valid_train_indices)
            self.log(f"Skipped {skipped} samples with rare or unrecognized group labels")
        
        # Extract valid samples
        valid_train_features = [train_features[i] for i in valid_train_indices]
        valid_train_groups = [train_group_labels[i] for i in valid_train_indices]
        valid_train_participants = None
        if train_participant_ids is not None:
            valid_train_participants = [train_participant_ids[i] for i in valid_train_indices]
        
        # Check if we have enough data
        if not valid_train_features or not valid_train_groups:
            self.log("Error: No valid data after filtering")
            return None
        
        # 4. Show group distribution after filtering
        filtered_group_counts = Counter(valid_train_groups)
        self.log("Phoneme group distribution after filtering:")
        for group, count in filtered_group_counts.most_common():
            self.log(f"  {group}: {count}")
        
        # 5. Encode group labels
        train_encoded = self.group_encoder.transform(valid_train_groups)
        train_onehot = to_categorical(train_encoded, self.num_groups)
        
        # 6. Process test data if available
        test_onehot = None
        valid_test_features = None
        valid_test_participants = None
        
        if test_features is not None and test_group_labels is not None:
            valid_test_indices = [i for i, g in enumerate(test_group_labels) 
                                if g in unique_groups]
            
            self.log(f"Testing with {len(valid_test_indices)} samples (filtered from {len(test_group_labels)})")
            
            valid_test_features = [test_features[i] for i in valid_test_indices]
            valid_test_groups = [test_group_labels[i] for i in valid_test_indices]
            
            if test_participant_ids is not None:
                valid_test_participants = [test_participant_ids[i] for i in valid_test_indices]
            
            test_encoded = self.group_encoder.transform(valid_test_groups)
            test_onehot = to_categorical(test_encoded, self.num_groups)
        
        # 7. Standardize sequence lengths
        self.log("Standardizing feature lengths...")
        max_length = max(feat.shape[0] for feat in valid_train_features)
        self.max_sequence_length = max_length
        
        # Get feature dimension
        feature_dim = valid_train_features[0].shape[1]
        self.feature_dim = feature_dim
        
        self.input_shape = (self.max_sequence_length, self.feature_dim)
        self.log(f"Input shape: {self.input_shape}")
        
        # Standardize lengths
        train_features_std = self._standardize_length(valid_train_features, self.max_sequence_length)
        
        # 8. Prepare participant information if needed
        train_participants_encoded = None
        test_participants_encoded = None
        participant_dim = None
        
        if include_participant_info and valid_train_participants:
            from sklearn.preprocessing import LabelEncoder
            participant_encoder = LabelEncoder()
            participant_encoder.fit(valid_train_participants)
            
            train_participants_int = participant_encoder.transform(valid_train_participants)
            train_participants_encoded = to_categorical(train_participants_int)
            
            participant_dim = train_participants_encoded.shape[1]
        
        # 9. Split for validation if test data is not provided
        if test_features is None:
            # Split training data for validation
            
            train_idx, val_idx = train_test_split(
                np.arange(len(train_features_std)),
                test_size=0.2,
                random_state=42,
                stratify=train_encoded
            )
            
            prepared_data = {
                'train_features': train_features_std[train_idx],
                'train_labels': train_onehot[train_idx],
                'val_features': train_features_std[val_idx],
                'val_labels': train_onehot[val_idx],
                'input_shape': self.input_shape,
                'num_groups': self.num_groups,
            }
            
            if train_participants_encoded is not None:
                prepared_data['train_participants'] = train_participants_encoded[train_idx]
                prepared_data['val_participants'] = train_participants_encoded[val_idx]
                prepared_data['participant_dim'] = participant_dim
        else:
            # Standardize test features
            test_features_std = self._standardize_length(valid_test_features, self.max_sequence_length)
            
            # Process test participant info
            if valid_test_participants and include_participant_info and participant_dim is not None:
                test_participants_int = np.array([
                    np.where(participant_encoder.classes_ == p)[0][0] 
                    if p in participant_encoder.classes_ else 0
                    for p in valid_test_participants
                ])
                test_participants_encoded = to_categorical(test_participants_int, participant_dim)
            
            prepared_data = {
                'train_features': train_features_std,
                'train_labels': train_onehot,
                'val_features': test_features_std,
                'val_labels': test_onehot,
                'input_shape': self.input_shape,
                'num_groups': self.num_groups,
            }
            
            if train_participants_encoded is not None and test_participants_encoded is not None:
                prepared_data['train_participants'] = train_participants_encoded
                prepared_data['val_participants'] = test_participants_encoded
                prepared_data['participant_dim'] = participant_dim
        
        # 10. Train the model using the existing train method
        self.train(
            prepared_data=prepared_data,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        
        # 11. Evaluate if test data was provided separately from validation
        evaluation_results = None
        if test_features is not None and test_features is not prepared_data['val_features']:
            # This means we have separate test data not used for validation
            test_data = {
                'test_features': test_features_std,
                'test_labels': test_onehot,
            }
            
            if test_participants_encoded is not None:
                test_data['test_participants'] = test_participants_encoded
            
            evaluation_results = self.evaluate(test_data)
        else:
            # Use validation metrics from training
            evaluation_results = {
                'accuracy': max(self.history.history['val_accuracy']),
                'loss': min(self.history.history['val_loss'])
            }
        
        # 12. Store the last evaluation
        self.last_evaluation = evaluation_results
        
        return evaluation_results
        
    def analyze_examples(self, features, true_groups, predicted_groups, participant_ids=None, 
                   group_labels=None, max_examples=3, save_dir=None):
        """
        Analyze specific examples from each phoneme group, showing correct and incorrect predictions.
        
        Parameters:
        -----------
        features : numpy.ndarray
            Feature arrays
        true_groups : numpy.ndarray
            True group indices
        predicted_groups : numpy.ndarray
            Predicted group indices
        participant_ids : list or None
            List of participant IDs for each example
        group_labels : list or None
            List of group labels if not using indices
        max_examples : int
            Maximum number of examples to show per category
        save_dir : str or None
            Directory to save example visualizations
            
        Returns:
        --------
        dict
            Dictionary containing examples for each class
        """
        if save_dir is None:
            save_dir = os.path.join(self.output_dir, 'examples')
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Get all unique classes from the encoder
        all_classes = list(self.group_encoder.classes_)
        
        # Convert indices to class names if needed
        if group_labels is None:
            true_class_names = [all_classes[idx] for idx in true_groups]
            predicted_class_names = [all_classes[idx] for idx in predicted_groups]
        else:
            true_class_names = group_labels
            predicted_class_names = [all_classes[idx] for idx in predicted_groups]
        
        # Results dictionary
        examples_dict = {}
        
        # For each phoneme group
        for class_idx, class_name in enumerate(all_classes):
            # Get all examples for this true class
            true_class_indices = np.where(np.array(true_class_names) == class_name)[0]
            
            if len(true_class_indices) == 0:
                self.log(f"No examples found for phoneme group '{class_name}'")
                continue
            
            # Get correct predictions
            correct_indices = [i for i in true_class_indices if predicted_class_names[i] == class_name]
            
            # Get incorrect predictions
            incorrect_indices = [i for i in true_class_indices if predicted_class_names[i] != class_name]
            
            # Limit to max examples
            correct_examples = correct_indices[:max_examples] if len(correct_indices) > 0 else []
            incorrect_examples = incorrect_indices[:max_examples] if len(incorrect_indices) > 0 else []
            
            # Create a visualization
            n_correct = len(correct_examples)
            n_incorrect = len(incorrect_examples)
            n_rows = max(n_correct, 1) + max(n_incorrect, 1)
            
            if n_rows > 0:
                fig, axs = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows))
                
                # Handle case with only one subplot
                if n_rows == 1:
                    axs = [axs]
                
                # Plot correct examples
                row = 0
                if n_correct > 0:
                    for i, idx in enumerate(correct_examples):
                        feature = features[idx]
                        
                        # Create a heatmap of the feature
                        im = axs[row].imshow(feature.T, aspect='auto', origin='lower', cmap='viridis')
                        
                        # Add title with participant info if available
                        title = f"Correct: True={class_name}, Pred={predicted_class_names[idx]}"
                        if participant_ids is not None:
                            title += f" (Participant: {participant_ids[idx]})"
                        
                        axs[row].set_title(title)
                        axs[row].set_ylabel('Feature Dimension')
                        axs[row].set_xlabel('Time')
                        
                        # Add colorbar
                        plt.colorbar(im, ax=axs[row])
                        
                        row += 1
                else:
                    axs[row].text(0.5, 0.5, f"No correct examples for {class_name}",
                                ha='center', va='center', fontsize=12)
                    axs[row].axis('off')
                    row += 1
                
                # Plot incorrect examples
                if n_incorrect > 0:
                    for i, idx in enumerate(incorrect_examples):
                        feature = features[idx]
                        
                        # Create a heatmap of the feature
                        im = axs[row].imshow(feature.T, aspect='auto', origin='lower', cmap='viridis')
                        
                        # Add title with participant info if available
                        title = f"Incorrect: True={class_name}, Pred={predicted_class_names[idx]}"
                        if participant_ids is not None:
                            title += f" (Participant: {participant_ids[idx]})"
                        
                        axs[row].set_title(title)
                        axs[row].set_ylabel('Feature Dimension')
                        axs[row].set_xlabel('Time')
                        
                        # Add colorbar
                        plt.colorbar(im, ax=axs[row])
                        
                        row += 1
                else:
                    axs[row].text(0.5, 0.5, f"No incorrect examples for {class_name}",
                                ha='center', va='center', fontsize=12)
                    axs[row].axis('off')
                
                plt.tight_layout()
                
                # Save the figure
                save_path = os.path.join(save_dir, f"examples_{class_name}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                examples_dict[class_name] = {
                    'correct': [int(i) for i in correct_examples],
                    'incorrect': [int(i) for i in incorrect_examples],
                    'visualization_path': save_path
                }
                
                self.log(f"Examples for '{class_name}' saved to {save_path}")
        
        # Create a summary document with links to all visualizations
        summary_path = os.path.join(save_dir, 'examples_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("# Phoneme Group Examples Summary\n\n")
            for class_name in all_classes:
                if class_name in examples_dict:
                    n_correct = len(examples_dict[class_name]['correct'])
                    n_incorrect = len(examples_dict[class_name]['incorrect'])
                    f.write(f"## {class_name}\n")
                    f.write(f"- Correct examples: {n_correct}\n")
                    f.write(f"- Incorrect examples: {n_incorrect}\n")
                    f.write(f"- Visualization: {examples_dict[class_name]['visualization_path']}\n\n")
                else:
                    f.write(f"## {class_name}\n")
                    f.write("No examples found.\n\n")
        
        self.log(f"Examples summary saved to {summary_path}")
        
        return examples_dict
        

        
    