import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
from debugger import DebugMixin
import os

class HybridPhonemeModels(DebugMixin):
    """
    Collection of alternative models suitable for small-sample phoneme recognition.
    """
    
    def __init__(self, phonetic_dict=None, output_dir='./models/hybrid', debug_mode=False):
        
        super().__init__(class_name="HybridPhonemeModels", debug_mode=debug_mode)
        if debug_mode is not None:
            self.DEBUG_MODE = debug_mode
        self.log(f"Initialized with DEBUG_MODE={self.DEBUG_MODE}")
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up phonetic dictionary
        if phonetic_dict is None:
            from phonetic_dictionary import PhoneticDictionary
            self.phonetic_dict = PhoneticDictionary()
        else:
            self.phonetic_dict = phonetic_dict
        
        if not hasattr(self.phonetic_dict, 'phoneme_groups'):
            self.phonetic_dict.add_phoneme_groups()
        
        self.models = {}
        self.results = {}
    
    def prepare_features(self, features, max_components=50):
        """
        Prepare features for traditional ML models.
        Uses temporal summarization instead of padding.
        """
        processed = []
        
        for feat in features:
            if feat.ndim > 1:
                # Extract statistical features from time series
                summary = []
                
                # Mean and std for each channel
                summary.extend(np.mean(feat, axis=0))
                summary.extend(np.std(feat, axis=0))
                
                # Additional temporal features
                summary.extend(np.percentile(feat, [25, 50, 75], axis=0).flatten())
                
                # Delta features (first difference)
                if feat.shape[0] > 1:
                    delta = np.diff(feat, axis=0)
                    summary.extend(np.mean(delta, axis=0))
                    summary.extend(np.std(delta, axis=0))
                
                processed.append(np.array(summary))
            else:
                processed.append(feat)
        
        X = np.array(processed)
        
        # Apply PCA if feature dimension is too high
        if X.shape[1] > max_components:
            self.pca = PCA(n_components=max_components)
            X = self.pca.fit_transform(X)
            self.log(f"Reduced features from {processed[0].shape[0]} to {max_components} dimensions")
        
        return X
    
    def train_gmm_model(self, features, labels, n_components=5):
        """
        Gaussian Mixture Model - good for modeling distributions with limited data.
        """
        self.log("Training GMM model...")
        
        X = self.prepare_features(features)
        
        # Map labels to groups
        groups = [self.phonetic_dict.phoneme_to_group.get(l, 'unknown') for l in labels]
        unique_groups = list(set(groups))
        
        # Train one GMM per phoneme group
        gmm_models = {}
        for group in unique_groups:
            group_indices = [i for i, g in enumerate(groups) if g == group]
            if len(group_indices) < n_components:
                n_comp = max(1, len(group_indices) // 2)
            else:
                n_comp = n_components
            
            gmm = GaussianMixture(n_components=n_comp, covariance_type='diag')
            gmm.fit(X[group_indices])
            gmm_models[group] = gmm
        
        self.models['gmm'] = {
            'models': gmm_models,
            'groups': unique_groups,
            'scaler': StandardScaler().fit(X)
        }
        
        # Evaluate
        predictions = self.predict_gmm(features)
        accuracy = accuracy_score(groups, predictions)
        
        self.results['gmm'] = {'accuracy': accuracy}
        self.log(f"GMM training complete. Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def predict_gmm(self, features):
        """Predict using GMM model."""
        if 'gmm' not in self.models:
            raise ValueError("GMM model not trained")
        
        X = self.prepare_features(features)
        X = self.models['gmm']['scaler'].transform(X)
        
        predictions = []
        for x in X:
            scores = {}
            for group, gmm in self.models['gmm']['models'].items():
                scores[group] = gmm.score_samples(x.reshape(1, -1))[0]
            
            predictions.append(max(scores, key=scores.get))
        
        return predictions
    
    def train_ensemble_model(self, features, labels, use_voting=True):
        """
        Ensemble of simpler models - often works well with small samples.
        """
        self.log("Training ensemble model...")
        
        X = self.prepare_features(features)
        y = [self.phonetic_dict.phoneme_to_group.get(l, 'unknown') for l in labels]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create ensemble
        models = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('lda', LinearDiscriminantAnalysis())
        ]
        
        trained_models = []
        for name, model in models:
            try:
                model.fit(X_scaled, y)
                trained_models.append((name, model))
                self.log(f"Trained {name} successfully")
            except Exception as e:
                self.log(f"Failed to train {name}: {e}")
        
        self.models['ensemble'] = {
            'models': trained_models,
            'scaler': scaler,
            'use_voting': use_voting
        }
        
        # Evaluate
        predictions = self.predict_ensemble(features)
        accuracy = accuracy_score(y, predictions)
        
        self.results['ensemble'] = {'accuracy': accuracy}
        self.log(f"Ensemble training complete. Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def predict_ensemble(self, features):
        """Predict using ensemble model."""
        if 'ensemble' not in self.models:
            raise ValueError("Ensemble model not trained")
        
        X = self.prepare_features(features)
        X_scaled = self.models['ensemble']['scaler'].transform(X)
        
        if self.models['ensemble']['use_voting']:
            # Voting ensemble
            all_predictions = []
            for name, model in self.models['ensemble']['models']:
                preds = model.predict(X_scaled)
                all_predictions.append(preds)
            
            # Majority voting
            all_predictions = np.array(all_predictions).T
            predictions = []
            for sample_preds in all_predictions:
                unique, counts = np.unique(sample_preds, return_counts=True)
                predictions.append(unique[np.argmax(counts)])
        else:
            # Average probabilities
            all_probs = []
            for name, model in self.models['ensemble']['models']:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_scaled)
                    all_probs.append(probs)
            
            if all_probs:
                avg_probs = np.mean(all_probs, axis=0)
                predictions = [self.models['ensemble']['models'][0][1].classes_[i] 
                              for i in np.argmax(avg_probs, axis=1)]
            else:
                predictions = self.models['ensemble']['models'][0][1].predict(X_scaled)
        
        return predictions
    
    def train_xgboost_model(self, features, labels):
        """
        XGBoost - often works well with small datasets.
        """
        self.log("Training XGBoost model...")
        
        # Check if we have enough data
        if len(features) < 10:
            self.log(f"Warning: Only {len(features)} samples. Need at least 10 for training.")
            return 0.0
            
        X = self.prepare_features(features)
        
        # Map labels to groups and then to integers
        groups = [self.phonetic_dict.phoneme_to_group.get(l, 'unknown') for l in labels]
        unique_groups = sorted(list(set(groups)))
        
        # Check if we have at least 2 classes
        if len(unique_groups) < 2:
            self.log(f"Error: Only {len(unique_groups)} unique group(s). Need at least 2 for classification.")
            return 0.0
        # Check minimum samples per class
        group_counts = Counter(groups)
        
        min_samples = min(group_counts.values())
        
        if min_samples < 2:
            self.log("Warning: Some groups have less than 2 samples.")
            for group, count in group_counts.items():
                if count < 2:
                    self.log(f"  {group}: {count} samples (may cause issues)")
        
        group_to_int = {g: i for i, g in enumerate(unique_groups)}
        y = np.array([group_to_int[g] for g in groups])
        
        # Check for NaN or Inf in features
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            self.log("Warning: Found NaN or Inf in features. Cleaning...")
            # Replace NaN with mean, Inf with max finite value
            X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(X.dtype).max, neginf=np.finfo(X.dtype).min)
        
        # Scale features
        scaler = StandardScaler()
        try:
            X_scaled = scaler.fit_transform(X)
        except Exception as e:
            self.log(f"Error scaling features: {e}")
            return 0.0
        
        # Adjust parameters based on data size
        n_samples = len(X_scaled)
        n_classes = len(unique_groups)
        
        # Dynamic parameter adjustment
        params = {
            'objective': 'multi:softprob',
            'num_class': n_classes,
            'max_depth': min(3, n_samples // 20),  # Shallower trees for small data
            'learning_rate': 0.1,
            'n_estimators': min(100, n_samples // 2),  # Fewer trees for small data
            'subsample': min(0.8, (n_samples - 10) / n_samples) if n_samples > 10 else 1.0,
            'colsample_bytree': 0.8,
            'min_child_weight': max(1, n_samples // 50),  # Prevent overfitting
            'random_state': 42
        }
        
        self.log(f"Training with {n_samples} samples, {n_classes} classes")
        self.log(f"Adjusted params: max_depth={params['max_depth']}, n_estimators={params['n_estimators']}")
        
        try:
            model = xgb.XGBClassifier(**params)
            model.fit(X_scaled, y)
        except Exception as e:
            self.log(f"Error training XGBoost: {e}")
            return 0.0
        
        self.models['xgboost'] = {
            'model': model,
            'scaler': scaler,
            'group_map': group_to_int,
            'int_to_group': {i: g for g, i in group_to_int.items()}
        }
        
        # Evaluate
        try:
            predictions = model.predict(X_scaled)
            accuracy = accuracy_score(y, predictions)
        except Exception as e:
            self.log(f"Error evaluating model: {e}")
            accuracy = 0.0
        
        self.results['xgboost'] = {'accuracy': accuracy}
        self.log(f"XGBoost training complete. Accuracy: {accuracy:.4f}")
        
        # Log feature importance (helpful for debugging)
        if hasattr(model, 'feature_importances_'):
            top_features = np.argsort(model.feature_importances_)[-5:]
            self.log(f"Top 5 important features: {top_features}")
        
        return accuracy
    
    def predict_xgboost(self, features):
        """Predict using XGBoost model."""
        if 'xgboost' not in self.models:
            raise ValueError("XGBoost model not trained")
        
        X = self.prepare_features(features)
        X_scaled = self.models['xgboost']['scaler'].transform(X)
        
        predictions_int = self.models['xgboost']['model'].predict(X_scaled)
        predictions = [self.models['xgboost']['int_to_group'][i] for i in predictions_int]
        
        return predictions
    
    def train_prototype_model(self, features, labels, n_prototypes=3):
        """
        Prototype-based learning - creates representative examples for each class.
        Good for interpretability and small samples.
        """
        self.log("Training prototype model...")
        
        X = self.prepare_features(features)
        groups = [self.phonetic_dict.phoneme_to_group.get(l, 'unknown') for l in labels]
        
        # Create prototypes for each group
        prototypes = {}
        for group in set(groups):
            group_indices = [i for i, g in enumerate(groups) if g == group]
            group_features = X[group_indices]
            
            if len(group_features) <= n_prototypes:
                # Use all samples as prototypes
                prototypes[group] = group_features
            else:
                # Use k-means to find prototypes
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_prototypes, random_state=42)
                kmeans.fit(group_features)
                prototypes[group] = kmeans.cluster_centers_
        
        self.models['prototype'] = {
            'prototypes': prototypes,
            'scaler': StandardScaler().fit(X)
        }
        
        # Evaluate
        predictions = self.predict_prototype(features)
        accuracy = accuracy_score(groups, predictions)
        
        self.results['prototype'] = {'accuracy': accuracy}
        self.log(f"Prototype model training complete. Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def predict_prototype(self, features):
        """Predict using prototype model."""
        if 'prototype' not in self.models:
            raise ValueError("Prototype model not trained")
        
        X = self.prepare_features(features)
        X = self.models['prototype']['scaler'].transform(X)
        
        predictions = []
        for x in X:
            min_dist = float('inf')
            best_group = None
            
            for group, protos in self.models['prototype']['prototypes'].items():
                for proto in protos:
                    dist = np.linalg.norm(x - proto)
                    if dist < min_dist:
                        min_dist = dist
                        best_group = group
            
            predictions.append(best_group)
        
        return predictions
    
    def compare_all_models(self, train_features, train_labels, test_features, test_labels):
        """
        Train and compare all models.
        """
        self.log("Comparing all models...")
        
        results = {}
        
        # Train all models
        models_to_test = [
            ('GMM', self.train_gmm_model),
            ('Ensemble', self.train_ensemble_model),
            ('XGBoost', self.train_xgboost_model),
            ('Prototype', self.train_prototype_model)
        ]
        
        for name, train_func in models_to_test:
            try:
                self.log(f"\nTraining {name}...")
                train_func(train_features, train_labels)
                
                # Test
                if name == 'GMM':
                    predictions = self.predict_gmm(test_features)
                elif name == 'Ensemble':
                    predictions = self.predict_ensemble(test_features)
                elif name == 'XGBoost':
                    predictions = self.predict_xgboost(test_features)
                elif name == 'Prototype':
                    predictions = self.predict_prototype(test_features)
                
                # Evaluate
                test_groups = [self.phonetic_dict.phoneme_to_group.get(l, 'unknown') 
                              for l in test_labels]
                accuracy = accuracy_score(test_groups, predictions)
                
                results[name] = {
                    'train_acc': self.results[name.lower()]['accuracy'],
                    'test_acc': accuracy
                }
                
                self.log(f"{name} - Train: {results[name]['train_acc']:.4f}, "
                        f"Test: {results[name]['test_acc']:.4f}")
                
            except Exception as e:
                self.log(f"Failed to train {name}: {e}")
                results[name] = {'train_acc': 0, 'test_acc': 0}
        
        return results
        
