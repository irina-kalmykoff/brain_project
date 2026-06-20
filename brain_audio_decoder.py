import os
import numpy as np
import pandas as pd
import scipy
import scipy.signal
import scipy.stats
import scipy.io.wavfile
import scipy.fftpack
from pynwb import NWBHDF5IO
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# Import from existing files
from extract_features import extractHG, stackFeatures, extractMelSpecs, downsampleLabels, nameVector
import reconstructWave as rW
import MelFilterBank as mel

# extract spectorgamm
from extract_features import extractMelSpecs

from dataset_config import Dutch30Config

class BrainAudioDecoder:
    """
    A class to encapsulate the brain-to-audio decoding pipeline.
    This class uses the existing code from the project and provides
    a more structured interface to experiment with improvements.
    """
    
    def __init__(self, path_bids, path_output, path_results, win_length=None, 
                 frameshift=None, model_order=4, step_size=5, target_sr=None, 
                 n_folds=10, n_components=50, n_random=1000, config: Dutch30Config = None):
        """
        Initialize the decoder with the given parameters.
        
        Parameters:
        -----------
        path_bids : str
            Path to the BIDS dataset
        path_output : str
            Path to save extracted features
        path_results : str
            Path to save results
        win_length : float
            Window length in seconds for feature extraction
        frameshift : float
            Frame shift in seconds for feature extraction
        model_order : int
            Number of temporal contexts to include
        step_size : int
            Step size for temporal context
        target_sr : int
            Target sampling rate for audio
        n_folds : int
            Number of folds for cross-validation
        n_components : int
            Number of PCA components to use
        n_random : int
            Number of random controls to generate
        """
        self.config = config if config is not None else Dutch30Config()
        self.path_bids = path_bids
        self.path_output = path_output
        self.path_results = path_results
        
        self.win_length = self.config.window_length
        self.frameshift = self.config.frameshift
        self.model_order = model_order
        self.step_size = step_size
        self.target_sr = self.config.audio_target_sr
        
        self.n_folds = n_folds
        self.n_components = n_components
        self.n_random = n_random
        
        # Ensure directories exist
        os.makedirs(self.path_output, exist_ok=True)
        os.makedirs(self.path_results, exist_ok=True)
        
        # Initialize model
        self.model = LinearRegression(n_jobs=5)
        self.pca = PCA()
        
        # Results storage
        self.results = {}
        self.participants = None

    def _load_participant_data(self, participant_id):
        """
        Load raw data for a participant.
        Returns tuple of (eeg, eeg_sr, audio, audio_sr, words, channels)
        """
        # Load data
        io = NWBHDF5IO(
            os.path.join(self.path_bids, participant_id, 'ieeg', 
                         f'{participant_id}_task-wordProduction_ieeg.nwb'), 
            'r'
        )
        nwbfile = io.read()
        
        # Get EEG data
        eeg = nwbfile.acquisition['iEEG'].data[:]
        eeg_sr = self.config.eeg_sr
        
        # Get audio data
        audio = nwbfile.acquisition['Audio'].data[:]
        audio_sr = self.config.audio_sr
        
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
        
        return eeg, eeg_sr, audio, audio_sr, words, channels

    def _process_audio(self, audio, audio_sr):
        """
        Process audio data.
        Returns processed audio and new sampling rate.
        """
        # Process audio
        audio = scipy.signal.decimate(audio, int(audio_sr / self.target_sr))
        audio_sr = self.target_sr
        scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)
        
        return scaled, audio_sr

    def _align_features(self, feat, mel_spec, words, eeg_sr):
        """
        Align features, mel spectrogram, and words.
        Returns aligned words, mel_spec, and feat.
        """
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
        
        return words, mel_spec, feat

    def _save_features(self, participant_id, feat, mel_spec, words, feature_names, method=None):
        """
        Save extracted features to disk.
        """
        # Add method suffix if provided
        suffix = f"_{method}" if method else ""
        
        np.save(os.path.join(self.path_output, f'{participant_id}_feat{suffix}.npy'), feat)
        np.save(os.path.join(self.path_output, f'{participant_id}_procWords{suffix}.npy'), words)
        np.save(os.path.join(self.path_output, f'{participant_id}_spec{suffix}.npy'), mel_spec)
        np.save(os.path.join(self.path_output, f'{participant_id}_feat_names{suffix}.npy'), feature_names)
    
    def extract_features_for_participant(self, participant_id):
        """
        Extract features for a single participant
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
            
        Returns:
        --------
        tuple
            (features, spectrogram, words, feature_names)
        """
        print(f"Extracting features for {participant_id}...")
    
        # Load data
        eeg, eeg_sr, audio, audio_sr, words, channels = self._load_participant_data(participant_id)
        
        # Extract high gamma features
        feat = extractHG(eeg, eeg_sr, windowLength=self.win_length, frameshift=self.frameshift)
        
        # Stack features with temporal context
        feat = stackFeatures(feat, modelOrder=self.model_order, stepSize=self.step_size)
        
        # Process audio
        scaled, audio_sr = self._process_audio(audio, audio_sr)
        
        # Save original audio
        scipy.io.wavfile.write(
            os.path.join(self.path_output, f'{participant_id}_orig_audio.wav'),
            audio_sr, 
            scaled
        )
        
        # Extract mel spectrogram
        mel_spec = extractMelSpecs(
            scaled, 
            audio_sr, 
            windowLength=self.win_length, 
            frameshift=self.frameshift
        )
        
        # Align features
        words, mel_spec, feat = self._align_features(feat, mel_spec, words, eeg_sr)
        
        # Create feature names
        feature_names = nameVector(channels[:, None], modelOrder=self.model_order)
        
        # Save features
        self._save_features(participant_id, feat, mel_spec, words, feature_names)
        
        return feat, mel_spec, words, feature_names
    
    def extract_features_all_participants(self):
        """Extract features for all participants in the dataset"""
        if self.participants is None:
            self.load_participants()
        
        results = {}
        for _, participant in enumerate(self.participants['participant_id']):
            results[participant] = self.extract_features_for_participant(participant)
        
        return results
 
    def load_participants(self):
        """Load participant information from BIDS dataset"""
                
        # Load participants from TSV file
        self.participants = pd.read_csv(
            os.path.join(self.path_bids, 'participants.tsv'), 
            delimiter='\t'
        )
        
        # Print for verification
        participant_ids = list(self.participants['participant_id'])
        print(f"Found {len(participant_ids)} participants:")
        for i, participant_id in enumerate(participant_ids):
            print(f"  {i+1}. {participant_id}")
        
        return self.participants
        
    def load_features(self, participant_id):
        """
        Load previously extracted features for a participant
        """
        features = np.load(os.path.join(self.path_output, f'{participant_id}_feat.npy'))
        spectrogram = np.load(os.path.join(self.path_output, f'{participant_id}_spec.npy'))
        words = np.load(os.path.join(self.path_output, f'{participant_id}_procWords.npy'))
        feature_names = np.load(os.path.join(self.path_output, f'{participant_id}_feat_names.npy'))
        
        return features, spectrogram, words, feature_names
    
    '''
    def train_test_model(self, participant_id, save_audio=True):
        """
        Train and test the model for a single participant
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
        save_audio : bool
            Whether to save the reconstructed audio
            
        Returns:
        --------
        dict
            Dictionary containing results
        """
        print(f"Training and testing model for {participant_id}...")
        
        # Load features
        data, spectrogram, words, feature_names = self.load_features(participant_id)
        
        # Initialize results
        results = {
            'correlations': np.zeros((self.n_folds, spectrogram.shape[1])),
            'explained_variance': np.zeros(self.n_folds),
            'random_correlations': np.zeros((self.n_random, spectrogram.shape[1]))
        }
        
        # Initialize empty spectrogram for reconstruction
        rec_spec = np.zeros(spectrogram.shape)
        
        # Cross-validation
        kf = KFold(self.n_folds, shuffle=False)
        for k, (train, test) in enumerate(kf.split(data)):
            # Z-normalize using training data statistics
            mu = np.mean(data[train, :], axis=0)
            std = np.std(data[train, :], axis=0)
            train_data = (data[train, :] - mu) / std
            test_data = (data[test, :] - mu) / std
            
            # Fit PCA to training data
            self.pca.fit(train_data)
            
            # Get explained variance
            results['explained_variance'][k] = np.sum(
                self.pca.explained_variance_ratio_[:self.n_components]
            )
            
            # Transform data to component space
            train_data = np.dot(train_data, self.pca.components_[:self.n_components, :].T)
            test_data = np.dot(test_data, self.pca.components_[:self.n_components, :].T)
            
            # Fit model
            self.model.fit(train_data, spectrogram[train, :])
            
            # Predict spectrogram
            rec_spec[test, :] = self.model.predict(test_data)
            
            # Evaluate reconstruction
            for spec_bin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print(f'{participant_id} has {np.sum(np.isnan(rec_spec))} broken samples in reconstruction')
                
                r, _ = pearsonr(spectrogram[test, spec_bin], rec_spec[test, spec_bin])
                results['correlations'][k, spec_bin] = r
        
        # Calculate mean correlation
        mean_corr = np.mean(results['correlations'])
        print(f'{participant_id} has mean correlation of {mean_corr:.4f}')
        
        # Random baseline
        for rand_round in range(self.n_random):
            # Choose random split point
            split_point = np.random.choice(
                np.arange(int(spectrogram.shape[0] * 0.1), int(spectrogram.shape[0] * 0.9))
            )
            
            # Shuffle spectrogram
            shuffled = np.concatenate((
                spectrogram[split_point:, :], 
                spectrogram[:split_point, :]
            ))
            
            # Calculate correlations
            for spec_bin in range(spectrogram.shape[1]):
                r, _ = pearsonr(spectrogram[:, spec_bin], shuffled[:, spec_bin])
                results['random_correlations'][rand_round, spec_bin] = r
        
        # Save reconstructed spectrogram
        np.save(
            os.path.join(self.path_results, f'{participant_id}_predicted_spec.npy'), 
            rec_spec
        )
        
        if save_audio:
            # Synthesize waveform using Griffin-Lim
            reconstructed_wav = self.create_audio(
                rec_spec, 
                audiosr=self.target_sr, 
                winLength=self.win_length, 
                frameshift=self.frameshift
            )
            
            scipy.io.wavfile.write(
                os.path.join(self.path_results, f'{participant_id}_predicted.wav'),
                int(self.target_sr),
                reconstructed_wav
            )
            
            # For comparison, synthesize original spectrogram
            orig_wav = self.create_audio(
                spectrogram, 
                audiosr=self.target_sr, 
                winLength=self.win_length, 
                frameshift=self.frameshift
            )
            
            scipy.io.wavfile.write(
                os.path.join(self.path_results, f'{participant_id}_orig_synthesized.wav'),
                int(self.target_sr),
                orig_wav
            )
        
        # Store results
        self.results[participant_id] = results
        return results
    
    def train_test_all_participants(self, save_audio=True):
        """
        Train and test the model for all participants
        
        Parameters:
        -----------
        save_audio : bool
            Whether to save the reconstructed audio
            
        Returns:
        --------
        dict
            Dictionary containing results for all participants
        """
        # Get all participant IDs
        participant_ids = [f'sub-{i:02d}' for i in range(1, 11)]
        
        # Initialize results arrays
        all_correlations = np.zeros((len(participant_ids), self.n_folds, 23))
        explained_variance = np.zeros((len(participant_ids), self.n_folds))
        random_control = np.zeros((len(participant_ids), self.n_random, 23))
        
        # Train and test for each participant
        for p_nr, participant_id in enumerate(participant_ids):
            results = self.train_test_model(participant_id, save_audio=save_audio)
            
            # Store results
            all_correlations[p_nr, :, :] = results['correlations']
            explained_variance[p_nr, :] = results['explained_variance']
            random_control[p_nr, :, :] = results['random_correlations']
        
        # Save aggregate results
        np.save(os.path.join(self.path_results, 'linearResults.npy'), all_correlations)
        np.save(os.path.join(self.path_results, 'randomResults.npy'), random_control)
        np.save(os.path.join(self.path_results, 'explainedVariance.npy'), explained_variance)
        
        return {
            'all_correlations': all_correlations,
            'explained_variance': explained_variance,
            'random_control': random_control
        }
    '''
    
    def create_audio(self, spectrogram, audiosr=16000, winLength=0.05, frameshift=0.01):
        """
        Create reconstructed audio waveform from spectrogram
        
        Parameters:
        -----------
        spectrogram : array
            Spectrogram of the audio
        audiosr : int
            Sampling rate of the audio
        winLength : float
            Window length in seconds
        frameshift : float
            Frame shift in seconds
            
        Returns:
        --------
        array
            Reconstructed audio waveform
        """
        mfb = mel.MelFilterBank(int((audiosr * winLength) / 2 + 1), spectrogram.shape[1], audiosr)
        nfolds = 10
        hop = int(spectrogram.shape[0] / nfolds)
        rec_audio = np.array([])
        for_reconstruction = mfb.fromLogMels(spectrogram)
        
        for w in range(0, spectrogram.shape[0], hop):
            spec = for_reconstruction[w:min(w + hop, for_reconstruction.shape[0]), :]
            rec = rW.reconstructWavFromSpectrogram(
                spec, 
                spec.shape[0] * spec.shape[1],
                fftsize=int(audiosr * winLength),
                overlap=int(winLength / frameshift)
            )
            rec_audio = np.append(rec_audio, rec)
            
        scaled = np.int16(rec_audio / np.max(np.abs(rec_audio)) * 32767)
        return scaled
    
    '''
    def evaluate_performance(self, participant_id=None):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        participant_id : str or None
            If provided, evaluate for a specific participant
            If None, evaluate for all participants
            
        Returns:
        --------
        dict
            Dictionary containing performance metrics
        """
        if participant_id is not None:
            # Evaluate for specific participant
            if participant_id not in self.results:
                print(f"No results found for {participant_id}. Running model...")
                self.train_test_model(participant_id)
            
            results = self.results[participant_id]
            
            # Calculate metrics
            mean_corr = np.mean(results['correlations'])
            mean_random = np.mean(results['random_correlations'])
            improvement = mean_corr - mean_random
            
            return {
                'participant_id': participant_id,
                'mean_correlation': mean_corr,
                'mean_random': mean_random,
                'improvement': improvement,
                'mean_explained_variance': np.mean(results['explained_variance'])
            }
        else:
            # Evaluate for all participants
            all_metrics = []
            participant_ids = [f'sub-{i:02d}' for i in range(1, 11)]
            
            for participant_id in participant_ids:
                metrics = self.evaluate_performance(participant_id)
                all_metrics.append(metrics)
            
            # Calculate average metrics
            avg_metrics = {
                'mean_correlation': np.mean([m['mean_correlation'] for m in all_metrics]),
                'mean_random': np.mean([m['mean_random'] for m in all_metrics]),
                'improvement': np.mean([m['improvement'] for m in all_metrics]),
                'mean_explained_variance': np.mean([m['mean_explained_variance'] for m in all_metrics])
            }
            
            return {
                'individual_metrics': all_metrics,
                'average_metrics': avg_metrics
            }
            '''
            
    def analyze_channels(self, participant_id):
        """
        Analyze and print information about channels for a specific participant.
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
            
        Returns:
        --------
        dict
            Dictionary containing channel information and groupings
        """
        
        # Load channels dataframe
        channels_df = pd.read_csv(
            os.path.join(self.path_bids, participant_id, 'ieeg', 
                        f'{participant_id}_task-wordProduction_channels.tsv'), 
            delimiter='\t'
        )
        
        # Print basic info
        print(f"Total channels for {participant_id}: {len(channels_df)}")
        
        # Get a count of channels by brain region
        if 'description' in channels_df.columns:
            brain_regions = channels_df['description'].value_counts()
            print("\nChannels by brain region:")
            print(brain_regions)
            
            # Look for speech-related areas
            speech_related_terms = ['front', 'temporal', 'sylvian', 'wernicke', 'broca', 
                                   'inferior', 'superior temporal', 'angular', 'supramarginal']
            speech_areas = []
            for i, row in channels_df.iterrows():
                desc = str(row['description']).lower()
                if any(term in desc for term in speech_related_terms):
                    speech_areas.append((row['name'], row['description']))
            
            print("\nPotential speech-related channels:")
            for name, desc in speech_areas:
                print(f"{name}: {desc}")
            
            # Check for left hemisphere channels
            left_hemisphere = channels_df[channels_df['name'].str.startswith('L') | 
                                          channels_df['description'].str.contains('left|lh', case=False)]
            print(f"\nNumber of left hemisphere channels: {len(left_hemisphere)}")
        else:
            print("No 'description' column found in channels dataframe")
        
        # Define groups of speech-relevant channels by region
        # Note: These are example groupings, adjust based on actual data
        speech_channel_groups = {
            'superior_temporal': ['RH6', 'RH7', 'RH8', 'RT7', 'RT8'],  # Auditory processing
            'middle_temporal': ['RH10', 'RT9', 'RT10', 'RT11'],        # Semantic processing
            'angular_gyrus': ['RP13', 'RP14'],                         # Reading/language integration
            'frontal': ['RF1', 'RF2', 'RF4', 'RF5', 'RM1', 'RM2',      # Executive aspects
                        'RQ1', 'RQ2', 'RQ4', 'RW10']
        }
    
    def get_channel_indices(channel_names, df):
        indices = []
        for name in channel_names:
            matching_rows = df[df['name'] == name]
            if not matching_rows.empty:
                indices.append(matching_rows.index[0])
        return indices
    
        # Get indices for each group
        speech_indices = {}
        print("\nSpeech-related channel groups:")
        for group_name, channel_names in speech_channel_groups.items():
            # Filter for existing channels
            existing_channels = [ch for ch in channel_names if ch in channels_df['name'].values]
            
            speech_indices[group_name] = get_channel_indices(existing_channels, channels_df)
            print(f"{group_name}: {len(speech_indices[group_name])} channels out of {len(channel_names)} defined")
            
            # Show which channels were found
            if len(speech_indices[group_name]) > 0:
                found_channels = [channels_df.iloc[idx]['name'] for idx in speech_indices[group_name]]
                print(f"  Found: {', '.join(found_channels)}")
            
            # Show which channels were not found
            missing_channels = set(channel_names) - set(channels_df.iloc[speech_indices[group_name]]['name'] if speech_indices[group_name] else [])
            if missing_channels:
                print(f"  Missing: {', '.join(missing_channels)}")
        
        # Combine all speech-related indices
        all_speech_indices = []
        for indices in speech_indices.values():
            all_speech_indices.extend(indices)
        print(f"\nTotal speech-related channels: {len(all_speech_indices)}")
        
        # Return a dictionary with useful information
        return {
            'channels_df': channels_df,
            'speech_areas': speech_areas if 'description' in channels_df.columns else [],
            'left_hemisphere': left_hemisphere if 'description' in channels_df.columns else [],
            'speech_indices': speech_indices,
            'all_speech_indices': all_speech_indices
        }
        
    def analyze_individual_channels(self, participant_id, windowLength=0.05, frameshift=0.01, 
                               modelOrder=4, stepSize=5, numComps=10, save_results=True):
        """
        Analyze each channel individually to see its contribution to speech reconstruction
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
        windowLength : float
            Window length in seconds for feature extraction
        frameshift : float
            Frame shift in seconds for feature extraction
        modelOrder : int
            Model order for stacking features
        stepSize : int
            Step size for stacking features
        numComps : int
            Number of PCA components to use
        save_results : bool
            Whether to save results to disk
        
        Returns:
        --------
        channel_results : dict
            Dictionary with channel names as keys and correlation values as values
        """
        
        print(f"Analyzing individual channel contributions for {participant_id}...")
        
        # Load data
        try:
            # Load EEG data
            io = NWBHDF5IO(
                os.path.join(self.path_bids, participant_id, 'ieeg', 
                           f'{participant_id}_task-wordProduction_ieeg.nwb'), 
                'r'
            )
            nwbfile = io.read()
            eeg_data = nwbfile.acquisition['iEEG'].data[:]
            eeg_sr = 1024
            io.close()
            
            # Load spectrogram
            spec_path = os.path.join(self.path_output, f'{participant_id}_spec.npy')
            if os.path.exists(spec_path):
                spectrogram = np.load(spec_path)
            else:
                # If spectrogram doesn't exist, we need to process the audio
                print("Spectrogram not found, extracting from audio...")
                io = NWBHDF5IO(
                    os.path.join(self.path_bids, participant_id, 'ieeg', 
                               f'{participant_id}_task-wordProduction_ieeg.nwb'), 
                    'r'
                )
                nwbfile = io.read()
                audio = nwbfile.acquisition['Audio'].data[:]
                audio_sr = 48000
                io.close()
                
                # Process audio
                target_SR = 16000
                audio = scipy.signal.decimate(audio, int(audio_sr / target_SR))
                audio_sr = target_SR
                scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)
                
                # Extract spectrogram
                spectrogram = extractMelSpecs(scaled, audio_sr, windowLength=windowLength, frameshift=frameshift)
                
                # Align to match EEG features (assuming typical processing)
                spectrogram = spectrogram[modelOrder*stepSize:, :]
            
            # Load channel information
            channels_df = pd.read_csv(
                os.path.join(self.path_bids, participant_id, 'ieeg', 
                            f'{participant_id}_task-wordProduction_channels.tsv'), 
                delimiter='\t'
            )
            
            # Setup for cross-validation
            nfolds = 5  # Using fewer folds for speed
            kf = KFold(nfolds, shuffle=False)
            est = LinearRegression()
            
            # Store results
            channel_results = {}
            
            # Loop through each channel
            for chan_idx in range(eeg_data.shape[1]):
                # Get channel name and region
                chan_name = channels_df.iloc[chan_idx]['name'] if chan_idx < len(channels_df) else f"Channel_{chan_idx}"
                chan_region = channels_df.iloc[chan_idx]['description'] if 'description' in channels_df.columns and chan_idx < len(channels_df) else "Unknown"
                    
                print(f"Processing {chan_name} ({chan_region})... ", end="", flush=True)
                
                # Extract single channel data
                single_chan_data = eeg_data[:, [chan_idx]]
                
                # Extract high-gamma features
                try:
                    feat = extractHG(single_chan_data, eeg_sr, windowLength=windowLength, frameshift=frameshift)
                    
                    # Stack features
                    feat = stackFeatures(feat, modelOrder=modelOrder, stepSize=stepSize)
                    
                    # Ensure the feature length matches the spectrogram
                    if feat.shape[0] > spectrogram.shape[0]:
                        feat = feat[:spectrogram.shape[0], :]
                    elif feat.shape[0] < spectrogram.shape[0]:
                        spectrogram = spectrogram[:feat.shape[0], :]
                    
                    # Initialize for cross-validation
                    rec_spec = np.zeros((feat.shape[0], spectrogram.shape[1]))
                    rs = np.zeros((nfolds, spectrogram.shape[1]))
                    
                    # Run cross-validation
                    for k, (train, test) in enumerate(kf.split(feat)):
                        # Skip if training set is too small
                        if len(train) < numComps + 1 or len(test) < 2:
                            continue
                            
                        # Normalize
                        mu = np.mean(feat[train,:], axis=0)
                        std = np.std(feat[train,:], axis=0)
                        std[std == 0] = 1  # Avoid division by zero
                        trainData = (feat[train,:] - mu) / std
                        testData = (feat[test,:] - mu) / std
                        
                        # Use fewer components for single channel data
                        n_components = min(numComps, trainData.shape[1], trainData.shape[0] - 1)
                        if n_components < 2:
                            # Skip if too few components
                            print("Skipping - insufficient data")
                            continue
                        
                        # PCA
                        try:
                            pca = PCA(n_components=n_components)
                            pca.fit(trainData)
                            trainData = pca.transform(trainData)
                            testData = pca.transform(testData)
                            
                            # Train model
                            est.fit(trainData, spectrogram[train, :])
                            rec_spec[test, :] = est.predict(testData)
                            
                            # Evaluate
                            for specBin in range(spectrogram.shape[1]):
                                r, p = pearsonr(spectrogram[test, specBin], rec_spec[test, specBin])
                                rs[k, specBin] = r
                        except Exception as e:
                            print(f"Error in PCA/regression: {e}")
                            continue
                    
                    # Calculate mean correlation
                    mean_corr = np.nanmean(rs)
                    channel_results[chan_name] = {
                        'correlation': mean_corr,
                        'region': chan_region,
                        'index': chan_idx,
                        'correlations_by_fold': rs
                    }
                    print(f"Correlation: {mean_corr:.6f}")
                    
                except Exception as e:
                    print(f"Error processing channel: {e}")
                    channel_results[chan_name] = {
                        'correlation': np.nan,
                        'region': chan_region,
                        'index': chan_idx,
                        'correlations_by_fold': np.full((nfolds, spectrogram.shape[1]), np.nan)
                    }
            
            # Save results if requested
            if save_results:
                os.makedirs(os.path.join(self.path_results, 'channel_analysis'), exist_ok=True)
                np.save(
                    os.path.join(self.path_results, 'channel_analysis', f'{participant_id}_channel_correlations.npy'),
                    channel_results
                )
            
            return channel_results
        
        except Exception as e:
            print(f"Error analyzing individual channels: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def analyze_channels_across_participants(self, participant_ids=None, **kwargs):
        """
        Analyze individual channel contributions across multiple participants

        """
        
        print("Analyzing channels across participants...")
        
        # If no participant IDs provided, analyze all participants
        if participant_ids is None:
            participant_ids = [f'sub-{i:02d}' for i in range(1, 11)]
        
        # Initialize results dictionary
        all_results = {}
        
        # Analyze each participant
        for participant_id in participant_ids:
            print(f"\nAnalyzing {participant_id}...")
            
            # Check if results already exist
            result_path = os.path.join(self.path_results, 'channel_analysis', f'{participant_id}_channel_correlations.npy')
            if os.path.exists(result_path):
                print(f"Loading existing results for {participant_id}...")
                try:
                    channel_results = np.load(result_path, allow_pickle=True).item()
                    all_results[participant_id] = channel_results
                    continue
                except Exception as e:
                    print(f"Error loading existing results: {e}. Recomputing...")
            
            # Analyze individual channels
            channel_results = self.analyze_individual_channels(participant_id, **kwargs)
            if channel_results:
                all_results[participant_id] = channel_results
        
        # Save combined results
        os.makedirs(os.path.join(self.path_results, 'channel_analysis'), exist_ok=True)
        np.save(
            os.path.join(self.path_results, 'channel_analysis', 'all_participants_channel_correlations.npy'),
            all_results
        )
        
        return all_results
        
    def extract_word_list(self, participant_id=None, words=None, verbose=False):
        """
        Extract a list of unique words from the words array

        """

        # Load words if not provided
        if words is None and participant_id is not None:
            words_path = os.path.join(self.path_output, f'{participant_id}_procWords.npy')
            if os.path.exists(words_path):
                words = np.load(words_path)
            else:
                if verbose:
                    print(f"Words file not found at {words_path}")
                return []
        
        if words is None:
            if verbose:
                print("No words provided or loaded")
            return []
        
        # Convert numbers 1-12 to Dutch words
        dutch_numbers = {
            '1': 'een', '2': 'twee', '3': 'drie', '4': 'vier', '5': 'vijf',
            '6': 'zes', '7': 'zeven', '8': 'acht', '9': 'negen', '10': 'tien', 
            '11': 'elf', '12': 'twaalf'
        }
        
        # Find unique words
        unique_words = set()
        current_word = ""
        
        for word in words:
            # Ensure word is a string
            if not isinstance(word, str):
                word = str(word).strip()
                
            # Skip empty strings
            if word == "":
                continue
            
            # Convert numbers to Dutch words
            if word in dutch_numbers:
                word = dutch_numbers[word]
            elif word.isdigit() and int(word) <= 12 and int(word) >= 1:
                word = dutch_numbers[word]
            
            # Detect word changes
            if word != current_word:
                unique_words.add(word)
                current_word = word
        
        # Convert to sorted list
        word_list = sorted(list(unique_words))
        
        if verbose:
            print(f"Found {len(word_list)} unique words:")
            print(word_list)
        
        return word_list