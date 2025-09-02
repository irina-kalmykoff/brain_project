from brain_audio_decoder import BrainAudioDecoder
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import os
import scipy

class CustomBrainAudioDecoder(BrainAudioDecoder):
    """
    Extended version of BrainAudioDecoder with additional methods
    for experimentation and improvements.
    """
    
    def __init__(self, path_bids, path_output, path_results, **kwargs):
        """Initialize with parent class parameters and additional options"""
        super().__init__(path_bids, path_output, path_results, **kwargs)
        
        # Store additional models for comparison
        self.models = {
            'linear': self.model,  # The default LinearRegression model
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10),
            'svr': SVR(kernel='linear'),
            'mlp': MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)
        }
        
        # Results for each model
        self.model_results = {}
    
    def train_test_with_model(self, participant_id, model_name, save_audio=False):
        """
        Train and test with a specific model
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
        model_name : str
            Name of the model to use (from self.models)
        save_audio : bool
            Whether to save the reconstructed audio
            
        Returns:
        --------
        dict
            Dictionary containing results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        print(f"Training and testing {model_name} model for {participant_id}...")
        
        # Save current model
        original_model = self.model
        
        # Set model to use
        self.model = self.models[model_name]
        
        # Train and test
        results = self.train_test_model(participant_id, save_audio=save_audio)
        
        # Store results
        if participant_id not in self.model_results:
            self.model_results[participant_id] = {}
        
        self.model_results[participant_id][model_name] = results
        
        # Restore original model
        self.model = original_model
        
        return results
    
    def compare_models(self, participant_id, models_to_compare=None, save_audio=False):
        """
        Compare multiple models for a participant
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
        models_to_compare : list or None
            List of model names to compare. If None, compare all models.
        save_audio : bool
            Whether to save the reconstructed audio
            
        Returns:
        --------
        dict
            Dictionary containing comparison results
        """
        if models_to_compare is None:
            models_to_compare = list(self.models.keys())
        
        comparison = {}
        
        for model_name in models_to_compare:
            if (participant_id in self.model_results and 
                model_name in self.model_results[participant_id]):
                # Use existing results
                results = self.model_results[participant_id][model_name]
            else:
                # Train and test model
                results = self.train_test_with_model(
                    participant_id, 
                    model_name, 
                    save_audio=save_audio
                )
            
            # Calculate mean correlation
            mean_corr = np.mean(results['correlations'])
            comparison[model_name] = mean_corr
        
        return comparison
    
    def plot_model_comparison(self, participant_id=None, save_fig=True):
        """
        Plot model comparison
        
        Parameters:
        -----------
        participant_id : str or None
            Participant ID to plot for. If None, average across all participants.
        save_fig : bool
            Whether to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if participant_id is not None:
            # Check if we have results for this participant
            if participant_id not in self.model_results:
                print(f"No results found for {participant_id}. Running comparison...")
                comparison = self.compare_models(participant_id)
            else:
                # Calculate mean correlation for each model
                comparison = {}
                for model_name, results in self.model_results[participant_id].items():
                    comparison[model_name] = np.mean(results['correlations'])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            models = list(comparison.keys())
            correlations = list(comparison.values())
            
            # Sort by correlation
            sorted_indices = np.argsort(correlations)
            models = [models[i] for i in sorted_indices]
            correlations = [correlations[i] for i in sorted_indices]
            
            ax.barh(models, correlations)
            ax.set_xlabel('Mean Correlation')
            ax.set_title(f'Model Comparison for {participant_id}')
            
            if save_fig:
                plt.savefig(os.path.join(self.path_results, f'{participant_id}_model_comparison.png'))
            
            return fig
        else:
            # Average across all participants
            all_participants = [f'sub-{i:02d}' for i in range(1, 11)]
            avg_comparison = {}
            
            for participant_id in all_participants:
                comparison = self.compare_models(participant_id)
                
                for model_name, corr in comparison.items():
                    if model_name not in avg_comparison:
                        avg_comparison[model_name] = []
                    
                    avg_comparison[model_name].append(corr)
            
            # Calculate averages
            for model_name in avg_comparison:
                avg_comparison[model_name] = np.mean(avg_comparison[model_name])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            models = list(avg_comparison.keys())
            correlations = list(avg_comparison.values())
            
            # Sort by correlation
            sorted_indices = np.argsort(correlations)
            models = [models[i] for i in sorted_indices]
            correlations = [correlations[i] for i in sorted_indices]
            
            ax.barh(models, correlations)
            ax.set_xlabel('Mean Correlation')
            ax.set_title('Model Comparison (Average Across Participants)')
            
            if save_fig:
                plt.savefig(os.path.join(self.path_results, 'all_participants_model_comparison.png'))
            
            return fig
    
    def optimize_hyperparameters(self, participant_id, model_name, param_grid):
        """
        Optimize hyperparameters for a specific model
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
        model_name : str
            Name of the model to optimize
        param_grid : dict
            Dictionary of parameter grids to search
            
        Returns:
        --------
        dict
            Dictionary containing best parameters and results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Load features
        data, spectrogram, words, feature_names = self.load_features(participant_id)
        
        # Use first fold for optimization
        kf = KFold(self.n_folds, shuffle=False)
        train, test = next(kf.split(data))
        
        # Z-normalize using training data statistics
        mu = np.mean(data[train, :], axis=0)
        std = np.std(data[train, :], axis=0)
        train_data = (data[train, :] - mu) / std
        test_data = (data[test, :] - mu) / std
        
        # Fit PCA to training data
        self.pca.fit(train_data)
        
        # Transform data to component space
        train_data = np.dot(train_data, self.pca.components_[:self.n_components, :].T)
        test_data = np.dot(test_data, self.pca.components_[:self.n_components, :].T)
        
        # Initialize results
        best_params = {}
        best_score = -np.inf
        
        # Get base model class
        model_class = self.models[model_name].__class__
        
        # Generate parameter combinations
        import itertools
        param_combinations = list(itertools.product(*param_grid.values()))
        
        for params in param_combinations:
            # Create parameter dictionary
            param_dict = dict(zip(param_grid.keys(), params))
            
            # Create model with parameters
            model = model_class(**param_dict)
            
            # Fit model
            model.fit(train_data, spectrogram[train, :])
            
            # Predict spectrogram
            pred_spec = model.predict(test_data)
            
            # Calculate correlation
            correlations = []
            for spec_bin in range(spectrogram.shape[1]):
                r, _ = pearsonr(spectrogram[test, spec_bin], pred_spec[:, spec_bin])
                correlations.append(r)
            
            # Calculate mean correlation
            mean_corr = np.mean(correlations)
            
            # Update best parameters if better
            if mean_corr > best_score:
                best_score = mean_corr
                best_params = param_dict
        
        # Update model with best parameters
        self.models[model_name] = model_class(**best_params)
        
        return {
            'model_name': model_name,
            'best_params': best_params,
            'best_score': best_score
        }
    
    def feature_importance(self, participant_id, model_name='linear'):
        """
        Calculate feature importance for a specific model
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
        model_name : str
            Name of the model to use
            
        Returns:
        --------
        array
            Feature importance scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Load features
        data, spectrogram, words, feature_names = self.load_features(participant_id)
        
        # Z-normalize
        mu = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        data_norm = (data - mu) / std
        
        # Fit PCA
        self.pca.fit(data_norm)
        
        # Transform data
        data_pca = np.dot(data_norm, self.pca.components_[:self.n_components, :].T)
        
        # Fit model
        model = self.models[model_name]
        model.fit(data_pca, spectrogram)
        
        # Get feature importances
        if hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_).mean(axis=0)
        elif hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        else:
            # For other models, use permutation importance
            from sklearn.inspection import permutation_importance
            result = permutation_importance(
                model, data_pca, spectrogram, 
                n_repeats=10, random_state=42
            )
            importances = result.importances_mean
        
        # Map back to original feature space
        pca_components = self.pca.components_[:self.n_components, :]
        feature_importances = np.zeros(pca_components.shape[1])
        
        for i, importance in enumerate(importances):
            feature_importances += importance * np.abs(pca_components[i, :])
        
        return feature_importances, feature_names
    
    def plot_feature_importance(self, participant_id, model_name='linear', top_n=20, save_fig=True):
        """
        Plot feature importance
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
        model_name : str
            Name of the model to use
        top_n : int
            Number of top features to plot
        save_fig : bool
            Whether to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Calculate feature importance
        importances, feature_names = self.feature_importance(participant_id, model_name)
        
        # Sort features by importance
        sorted_indices = np.argsort(importances)[::-1]
        top_indices = sorted_indices[:top_n]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh([feature_names[i] for i in top_indices], importances[top_indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Features for {participant_id} using {model_name}')
        
        if save_fig:
            plt.savefig(os.path.join(
                self.path_results, 
                f'{participant_id}_{model_name}_feature_importance.png'
            ))
        
        return fig
    
    def custom_feature_extraction(self, eeg, eeg_sr, method='high_gamma'):
        """
        Custom feature extraction method
        
        Parameters:
        -----------
        eeg : array
            EEG time series
        eeg_sr : int
            Sampling rate of EEG
        method : str
            Feature extraction method
            
        Returns:
        --------
        array
            Extracted features
        """
        if method == 'high_gamma':
            # Use original high gamma extraction
            return extractHG(eeg, eeg_sr, windowLength=self.win_length, frameshift=self.frameshift)
        elif method == 'multi_band':
            # Example of a custom feature extraction method
            # Extract multiple frequency bands
            
            # Define frequency bands
            bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 70),
                'high_gamma': (70, 170)
            }
            
            # Initialize features
            num_windows = int(np.floor((eeg.shape[0] - self.win_length * eeg_sr) / (self.frameshift * eeg_sr)))
            all_features = np.zeros((num_windows, eeg.shape[1] * len(bands)))
            
            # Extract features for each band
            for i, (band_name, (low_freq, high_freq)) in enumerate(bands.items()):
                # Linear detrend
                data = scipy.signal.detrend(eeg, axis=0)
                
                # Filter for band
                sos = scipy.signal.iirfilter(
                    4, [low_freq / (eeg_sr / 2), high_freq / (eeg_sr / 2)],
                    btype='bandpass', output='sos'
                )
                data = scipy.signal.sosfiltfilt(sos, data, axis=0)
                
                # If high gamma, attenuate line noise
                if band_name == 'high_gamma':
                    # Attenuate first harmonic of line noise
                    sos = scipy.signal.iirfilter(
                        4, [98 / (eeg_sr / 2), 102 / (eeg_sr / 2)],
                        btype='bandstop', output='sos'
                    )
                    data = scipy.signal.sosfiltfilt(sos, data, axis=0)
                    
                    # Attenuate second harmonic of line noise
                    sos = scipy.signal.iirfilter(
                        4, [148 / (eeg_sr / 2), 152 / (eeg_sr / 2)],
                        btype='bandstop', output='sos'
                    )
                    data = scipy.signal.sosfiltfilt(sos, data, axis=0)
                
                # Get envelope
                data = np.abs(scipy.signal.hilbert(data, axis=0))
                
                # Extract windows
                start_idx = i * eeg.shape[1]
                end_idx = (i + 1) * eeg.shape[1]
                
                for win in range(num_windows):
                    start = int(np.floor((win * self.frameshift) * eeg_sr))
                    stop = int(np.floor(start + self.win_length * eeg_sr))
                    all_features[win, start_idx:end_idx] = np.mean(data[start:stop, :], axis=0)
            
            return all_features
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")
    
    def extract_features_with_custom_method(self, participant_id, method='high_gamma'):
        """
        Extract features with a custom method
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
        method : str
            Feature extraction method
            
        Returns:
        --------
        tuple
            (features, spectrogram, words, feature_names)
        """
        print(f"Extracting features for {participant_id} using {method} method...")
        
        # Load data
        io = NWBHDF5IO(
            os.path.join(self.path_bids, participant_id, 'ieeg', 
                         f'{participant_id}_task-wordProduction_ieeg.nwb'), 
            'r'
        )
        nwbfile = io.read()
        
        # Get EEG data
        eeg = nwbfile.acquisition['iEEG'].data[:]
        eeg_sr = 1024
        
        # Get audio data
        audio = nwbfile.acquisition['Audio'].data[:]
        audio_sr = 48000
        
        # Get word markers
        words = nwbfile.acquisition['Stimulus'].data[:]
        words = np.array(words, dtype=str)
        io.close()
        
        # Get channel info
        channels = pd.read_csv(
            os.path.join(self.path_bids, participant_id, 'ieeg', 
                         f'{participant_id}_task-wordProduction_channels.tsv'), 
            delimiter='\t'
        )
        channels = np.array(channels['name'])
        
        # Extract features with custom method
        feat = self.custom_feature_extraction(eeg, eeg_sr, method=method)
        
        # Stack features with temporal context
        feat = stackFeatures(feat, modelOrder=self.model_order, stepSize=self.step_size)
        
        # Process audio
        audio = scipy.signal.decimate(audio, int(audio_sr / self.target_sr))
        audio_sr = self.target_sr
        scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)
        
        # Extract mel spectrogram
        mel_spec = extractMelSpecs(
            scaled, 
            audio_sr, 
            windowLength=self.win_length, 
            frameshift=self.frameshift
        )
        
        # Align to EEG features
        words = downsampleLabels(
            words, 
            eeg_sr, 
            windowLength=self.win_length, 
            frameshift=self.frameshift
        )
        words = words[self.model_order * self.step_size:words.shape[0] - self.model_order * self.step_size]
        mel_spec = mel_spec[self.model_order * self.step_size:mel_spec.shape[0] - self.model_order * self.step_size, :]
        
        # Adjust length
        if mel_spec.shape[0] != feat.shape[0]:
            tLen = np.min([mel_spec.shape[0], feat.shape[0]])
            mel_spec = mel_spec[:tLen, :]
            feat = feat[:tLen, :]
        
        # Create feature names
        if method == 'high_gamma':
            feature_names = nameVector(channels[:, None], modelOrder=self.model_order)
        elif method == 'multi_band':
            # Create names for each band
            bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
            all_names = []
            
            for band in bands:
                for ch in channels:
                    all_names.append(f"{ch}_{band}")
                    
            feature_names = nameVector(np.array(all_names)[:, None], modelOrder=self.model_order)
        
        # Save features with method suffix
        np.save(os.path.join(self.path_output, f'{participant_id}_feat_{method}.npy'), feat)
        np.save(os.path.join(self.path_output, f'{participant_id}_spec_{method}.npy'), mel_spec)
        np.save(os.path.join(self.path_output, f'{participant_id}_procWords_{method}.npy'), words)
        np.save(os.path.join(self.path_output, f'{participant_id}_feat_names_{method}.npy'), feature_names)
        
        return feat, mel_spec, words, feature_names