import numpy as np
import traceback

import tensorflow as tf
from tensorflow.keras import layers, models, Model  # Add Model here
from tensorflow.keras.callbacks import History
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder

import lightgbm as lgb
from typing import Tuple, List, Optional, Dict, Any

from debugger import DebugMixin


class SimplePhonemeModels(DebugMixin):
    """
    Minimalistic collection of diverse models for phoneme prediction.
    Designed to work with your existing pipeline.
    """
    
    def __init__(self, debug_mode=False):
        super().__init__(class_name="SimplePhonemeModels", debug_mode=debug_mode)
        if debug_mode is not None:
            self.DEBUG_MODE = debug_mode
        self.log(f"Initialized with DEBUG_MODE={self.DEBUG_MODE}")     
        
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
    def prepare_features_for_classical(self, features):
        """
        Convert variable-length sequences to fixed-size feature vectors for classical ML.
        Uses statistical summarization.
        """
        processed = []
        for feat in features:
            if feat.ndim > 1:
                # Statistical features: mean, std, min, max, percentiles
                summary = np.concatenate([
                    np.mean(feat, axis=0),
                    np.std(feat, axis=0),
                    np.min(feat, axis=0),
                    np.max(feat, axis=0),
                    np.percentile(feat, 25, axis=0),
                    np.percentile(feat, 75, axis=0)
                ])
            else:
                summary = feat
            processed.append(summary)
        return np.array(processed)
    
    # === CLASSICAL ML MODELS ===
    
    def train_naive_bayes(self, features, labels):
        """Gaussian Naive Bayes - Simple probabilistic classifier"""
        X = self.prepare_features_for_classical(features)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = GaussianNB()
        model.fit(X_scaled, labels)
        
        self.models['naive_bayes'] = model
        self.scalers['naive_bayes'] = scaler
        return model
    
    def train_logistic_regression(self, features, labels, C=1.0):
        """Logistic Regression with L2 regularization"""
        X = self.prepare_features_for_classical(features)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        model.fit(X_scaled, labels)
        
        self.models['logistic'] = model
        self.scalers['logistic'] = scaler
        return model
    
    def train_lda(self, features, labels):
        """Linear Discriminant Analysis - Finds linear combinations that maximize class separation"""
        X = self.prepare_features_for_classical(features)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LinearDiscriminantAnalysis()
        model.fit(X_scaled, labels)
        
        self.models['lda'] = model
        self.scalers['lda'] = scaler
        return model
    
    def train_knn(self, features, labels, n_neighbors=5):
        """K-Nearest Neighbors - Simple distance-based classifier"""
        X = self.prepare_features_for_classical(features)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_scaled, labels)
        
        self.models['knn'] = model
        self.scalers['knn'] = scaler
        return model
    
    def train_linear_svm(self, features, labels, C=1.0):
        """Linear Support Vector Machine - Fast and interpretable"""
        X = self.prepare_features_for_classical(features)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Increase max_iter to help with convergence
        model = LinearSVC(
            C=C, 
            random_state=42, 
            max_iter=5000,  # Increased from 2000
            tol=1e-4,       # Slightly relaxed tolerance
            dual=False      # Often faster for n_samples > n_features
        )
        model.fit(X_scaled, labels)
        
        self.models['linear_svm'] = model
        self.scalers['linear_svm'] = scaler
        return model
    
    def train_random_forest(self, features, labels, n_estimators=100):
        """Random Forest - Ensemble of decision trees"""
        X = self.prepare_features_for_classical(features)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_scaled, labels)
        
        self.models['random_forest'] = model
        self.scalers['random_forest'] = scaler
        return model
    
    def train_gradient_boosting(self, features, labels, n_estimators=100):
        """Gradient Boosting - Sequential ensemble"""
        X = self.prepare_features_for_classical(features)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_scaled, labels)
        
        self.models['gradient_boosting'] = model
        self.scalers['gradient_boosting'] = scaler
        return model
    
    def train_lightgbm(self, features, labels):
        """LightGBM - Fast gradient boosting"""
        X = self.prepare_features_for_classical(features)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        model.fit(X_scaled, labels)
        
        self.models['lightgbm'] = model
        self.scalers['lightgbm'] = scaler
        return model
    
    # === NEURAL NETWORK MODELS ===
    
    def build_simple_mlp(self, input_shape, num_classes):
        """Simple Multi-Layer Perceptron"""
        model = models.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def build_1d_cnn(self, input_shape, num_classes):
        """Simple 1D CNN for sequence data"""
        model = models.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def build_simple_rnn(self, input_shape, num_classes):
        """Simple RNN (using GRU for stability)"""
        model = models.Sequential([
            layers.GRU(64, input_shape=input_shape),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def build_tcn(self, input_shape, num_classes):
        """Temporal Convolutional Network - Alternative to RNNs"""
        inputs = layers.Input(shape=input_shape)
        
        # TCN block
        x = layers.Conv1D(32, 3, padding='causal', activation='relu')(inputs)
        x = layers.Conv1D(32, 3, padding='causal', activation='relu', dilation_rate=2)(x)
        x = layers.Conv1D(32, 3, padding='causal', activation='relu', dilation_rate=4)(x)
        
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        return models.Model(inputs, outputs)
        
    def train_neural_network(self, features, labels, model_type='mlp', epochs=30, batch_size=32):
        """Train a neural network model on the data."""
        
        # Convert labels to one-hot encoding
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)
        y_onehot = to_categorical(y_encoded, num_classes=num_classes)
        
        # Prepare features for neural networks (no balancing here - already done in pipeline)
        X_padded, input_shape = self._prepare_features_for_neural(features)
        
        # Initialize model variable to None
        model = None
        
        # Build the model
        try:
            if model_type == 'mlp':
                model = self.build_simple_mlp(input_shape, num_classes)
            elif model_type == 'cnn':
                model = self.build_1d_cnn(input_shape, num_classes)
            elif model_type == 'rnn':
                model = self.build_simple_rnn(input_shape, num_classes)
            elif model_type == 'tcn':
                model = self.build_tcn(input_shape, num_classes)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Check if model was created successfully
            if model is None:
                raise ValueError(f"Failed to create model of type: {model_type}")
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Early stopping to prevent overfitting
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Train the model (no sample weights - data already balanced)
            history = model.fit(
                X_padded, y_onehot,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1 if self.DEBUG_MODE else 0
            )
            
            # Store the model and metadata
            self.models[model_type] = model
            if not hasattr(self, 'label_encoders'):
                self.label_encoders = {}
            self.label_encoders[model_type] = label_encoder
            
            # Get training accuracy
            train_acc = history.history['accuracy'][-1]
            self.log(f"{model_type} training accuracy: {train_acc:.4f}")
            
            return model, history, label_encoder
            
        except Exception as e:
            self.log(f"Error building or training {model_type} model: {e}")
            import traceback
            traceback.print_exc()
            
            # Return None values to indicate failure
            return None, None, label_encoder

    def _prepare_features_for_neural(self, features: list) -> tuple:
        """
        Prepare features for neural network training by padding sequences
        """
        import numpy as np
        
        # Get maximum sequence length and feature dimension
        max_length = max(feat.shape[0] for feat in features)
        feature_dim = features[0].shape[1] if features[0].ndim > 1 else 1
        
        # Pad sequences to the same length
        padded_features = np.zeros((len(features), max_length, feature_dim))
        
        for i, feat in enumerate(features):
            if feat.ndim == 1:
                # Handle 1D feature vectors
                length = len(feat)
                padded_features[i, :length, 0] = feat
            else:
                # Handle 2D feature matrices
                length = feat.shape[0]
                padded_features[i, :length, :] = feat
        
        # Return input shape for the model
        input_shape = (max_length, feature_dim)
        
        return padded_features, input_shape

    def predict_with_neural(self, features, model_type='mlp'):
        """Make predictions with a neural network model."""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found")
        
        # Prepare features
        X_padded, input_shape = self._prepare_features_for_neural(features)
        
        # Check if the model expects a different input shape
        model = self.models[model_type]
        expected_shape = model.input_shape
        
        # Handle shape mismatch
        if expected_shape[1:] != X_padded.shape[1:]:
            self.log(f"Warning: Shape mismatch for {model_type}")
            self.log(f"Expected shape: {expected_shape}, Got shape: {X_padded.shape}")
            
            # Try to adjust the input shape
            if len(expected_shape) == 2:  # Flattened input expected
                # Model expects flattened input
                expected_features = expected_shape[1]
                current_features = X_padded.shape[1] * X_padded.shape[2] if len(X_padded.shape) > 2 else X_padded.shape[1]
                
                # Flatten the input
                X_padded_flat = X_padded.reshape(X_padded.shape[0], -1)
                
                if X_padded_flat.shape[1] < expected_features:
                    # Pad with zeros
                    padding = np.zeros((X_padded_flat.shape[0], expected_features - X_padded_flat.shape[1]))
                    X_padded = np.hstack([X_padded_flat, padding])
                elif X_padded_flat.shape[1] > expected_features:
                    # Truncate
                    X_padded = X_padded_flat[:, :expected_features]
                else:
                    X_padded = X_padded_flat
                    
                # Reshape back if needed
                if len(expected_shape) == 3:
                    X_padded = X_padded.reshape(X_padded.shape[0], expected_shape[1], expected_shape[2])
                    
            elif len(expected_shape) == 3:  # Sequential input expected
                expected_time = expected_shape[1]
                expected_features = expected_shape[2]
                
                # Adjust time dimension
                if X_padded.shape[1] < expected_time:
                    # Pad time dimension
                    padding = np.zeros((X_padded.shape[0], expected_time - X_padded.shape[1], X_padded.shape[2]))
                    X_padded = np.concatenate([X_padded, padding], axis=1)
                elif X_padded.shape[1] > expected_time:
                    # Truncate time dimension
                    X_padded = X_padded[:, :expected_time, :]
                
                # Adjust feature dimension
                if X_padded.shape[2] < expected_features:
                    # Pad feature dimension
                    padding = np.zeros((X_padded.shape[0], X_padded.shape[1], expected_features - X_padded.shape[2]))
                    X_padded = np.concatenate([X_padded, padding], axis=2)
                elif X_padded.shape[2] > expected_features:
                    # Truncate feature dimension
                    X_padded = X_padded[:, :, :expected_features]
        
        try:
            # Get predictions
            probabilities = self.models[model_type].predict(X_padded, verbose=0)
            predicted_indices = np.argmax(probabilities, axis=1)
            
            # Convert indices back to original labels
            if model_type in self.label_encoders:
                predictions = self.label_encoders[model_type].inverse_transform(predicted_indices)
            else:
                # If no label encoder, return indices as strings
                predictions = [str(i) for i in predicted_indices]
                self.log(f"Warning: No label encoder found for {model_type}")
            
            return predictions, probabilities
            
        except Exception as e:
            self.log(f"Error in prediction for {model_type}: {e}")
            # Return empty predictions
            return np.array(['unknown'] * len(features)), None
    
    # === TRAINING AND PREDICTION ===
    
    def train_all_classical(self, features: list, labels: list) -> tuple:
        """Train all classical ML models"""
        results = {}
        
        models_to_train = [
            ('naive_bayes', self.train_naive_bayes),
            ('logistic', self.train_logistic_regression),
            ('lda', self.train_lda),
            ('knn', self.train_knn),
            ('linear_svm', self.train_linear_svm),
            ('random_forest', self.train_random_forest),
            ('gradient_boosting', self.train_gradient_boosting),
            ('lightgbm', self.train_lightgbm)
        ]
        
        for name, train_func in models_to_train:
            try:
                self.log(f"Training {name}...")
                model = train_func(features, labels)
                
                # Calculate training accuracy
                X = self.prepare_features_for_classical(features)
                X_scaled = self.scalers[name].transform(X)
                train_acc = model.score(X_scaled, labels)
                
                results[name] = {'model': model, 'train_acc': train_acc}
                self.log(f"{name} training accuracy: {train_acc:.4f}")
                
            except Exception as e:
                self.log(f"Failed to train {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def predict(self, features: list, model_name: str) -> tuple:
        """Make predictions with a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        X = self.prepare_features_for_classical(features)
        X_scaled = self.scalers[model_name].transform(X)
        
        predictions = self.models[model_name].predict(X_scaled)
        
        if hasattr(self.models[model_name], 'predict_proba'):
            probabilities = self.models[model_name].predict_proba(X_scaled)
        else:
            probabilities = None
        
        return predictions, probabilities
        
    def train_all_neural(self, features: list, labels: list, epochs: int = 30, batch_size: int = 32):
        """
        Train all neural network models, returns dictionary with results for each model
        """
        results = {}
        
        models_to_train = ['mlp', 'cnn', 'rnn', 'tcn']
        
        for model_type in models_to_train:
            try:
                self.log(f"Training {model_type}...")
                model, history, _ = self.train_neural_network(
                    features, labels, 
                    model_type=model_type,
                    epochs=epochs,
                    batch_size=batch_size
                )
                
                if model is None or history is None:
                    self.log(f"Failed to train {model_type}")
                    results[model_type] = {'error': f'Training failed for {model_type}'}
                    continue
                
                # Get final training and validation accuracy
                train_acc = history.history['accuracy'][-1]
                val_acc = history.history.get('val_accuracy', [None])[-1]
                
                results[model_type] = {
                    'model': model,
                    'history': history,
                    'train_acc': train_acc,
                    'val_acc': val_acc
                }
                
                self.log(f"{model_type} training accuracy: {train_acc:.4f}")
                if val_acc:
                    self.log(f"{model_type} validation accuracy: {val_acc:.4f}")
                    
            except Exception as e:
                self.log(f"Failed to train {model_type}: {e}")
                traceback.print_exc()
                results[model_type] = {'error': str(e)}
        
        return results
        
    def train_all_models(self, features: list, labels: list, include_neural: bool = True,
                            epochs: int = 30, batch_size: int = 32) -> tuple:
        """
        Train all models (classical and neural if requested) and returns dictionary with results for each model
        """
        # Train classical models
        classical_results = self.train_all_classical(features, labels)        
        results = classical_results
        
        # Train neural models if requested
        if include_neural:
            try:
                neural_results = self.train_all_neural(features, labels, epochs=epochs, batch_size=batch_size)
                # Merge results
                results.update(neural_results)
            except Exception as e:
                self.log(f"Error training neural networks: {e}")

                traceback.print_exc()
        
        return results
        
    def evaluate_all_models(self, features, labels):
        """Evaluate all trained models on test data."""
        results = {}
        
        # Evaluate classical models
        for name in self.models:
            if name in ['mlp', 'cnn', 'rnn', 'tcn']:
                # Skip neural networks here - handle them separately
                continue
                
            try:
                # Prepare features
                X = self.prepare_features_for_classical(features)
                
                # Check if scaler exists for this model
                if name not in self.scalers:
                    self.log(f"Warning: No scaler found for {name}, skipping evaluation")
                    results[name] = None
                    continue
                    
                X_scaled = self.scalers[name].transform(X)
                
                # Evaluate
                accuracy = self.models[name].score(X_scaled, labels)
                results[name] = accuracy
                
                self.log(f"{name} test accuracy: {accuracy:.4f}")
            except Exception as e:
                self.log(f"Error evaluating {name}: {e}")
                results[name] = None
        
        # Evaluate neural networks
        for name in ['mlp', 'cnn', 'rnn', 'tcn']:
            if name not in self.models:
                continue
                
            try:
                # Use the predict_with_neural method with better error handling
                predictions, _ = self.predict_with_neural(features, name)
                
                if predictions is not None and len(predictions) > 0:
                    # Calculate accuracy
                    from sklearn.metrics import accuracy_score
                    accuracy = accuracy_score(labels, predictions)
                    results[name] = accuracy
                    
                    self.log(f"{name} test accuracy: {accuracy:.4f}")
                else:
                    self.log(f"No valid predictions from {name}")
                    results[name] = None
                    
            except Exception as e:
                self.log(f"Error evaluating {name}: {e}")
                import traceback
                traceback.print_exc()
                results[name] = None
        
        return results