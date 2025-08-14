import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
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
        
        model = LinearSVC(C=C, random_state=42, max_iter=2000)
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
    
    # === TRAINING AND PREDICTION ===
    
    def train_all_classical(self, features, labels):
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
    
    def predict(self, features, model_name):
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