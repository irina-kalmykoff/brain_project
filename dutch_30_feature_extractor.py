import os
import json
import h5py
import pandas as pd
import numpy as np
from debugger import DebugMixin
from config import BIDS_PATH, OUTPUT_PATH, RESULTS_PATH, DUTCH_30_PATH, DUTCH_10_PATH, get_dataset_paths
from dataset_config import Dutch30Config

class Dutch30FeatureExtractor(DebugMixin):
    """Fixed version that properly handles patient categorization"""
    
    def __init__(self, config: Dutch30Config = None):
        
        # Initialize the DebugMixin
        super().__init__(class_name="Dutch30FeatureExtractor", debug_mode=False)
        
        try:
            self.paths_30 = get_dataset_paths('dutch30')
            self.results_dir = self.paths_30['results_path']
        except Exception as e:
            self.log(f"Warning: Could not get paths from config: {e}")
            # Fallback: use DUTCH_30_PATH directly
            self.results_dir = os.path.join(DUTCH_30_PATH, 'results')
            self.paths_30 = {'results_path': self.results_dir}
        
        self.config = config if config is not None else Dutch30Config()
        
        # Set data_dir - this is the critical attribute
        self.data_dir = os.path.join(DUTCH_30_PATH, 'raw')
        
        # Set sampling rate
        self.sampling_rate = self.config.eeg_sr
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set up HDF5 file paths
        self.raw_features_file = os.path.join(
            self.results_dir, 
            f'raw_features_{self.sampling_rate}Hz.h5'
        )
        self.padded_features_file = os.path.join(
            self.results_dir, 
            f'padded_features_{self.sampling_rate}Hz.h5'
        )
        self.split_file = os.path.join(
            self.results_dir, 
            'patient_split.json'
        )
        self.final_data_file = os.path.join(
            self.results_dir, 
            f'final_data_{self.sampling_rate}Hz.h5'
        )
        
        # Verify data directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"Data directory does not exist: {self.data_dir}\n"
                f"Please check your DUTCH_30_PATH in config.py"
            )
        
        # Print confirmation
        self.log(f"Dutch30FeatureExtractor initialized:")
        self.log(f"  Data dir: {self.data_dir}")
        self.log(f"  Results dir: {self.results_dir}")
        self.log(f"  Sampling rate: {self.sampling_rate} Hz")
    
    def load_patient_raw_data(self, patient_id):
        """
        Load raw data for a patient
        patient_id: 'sub-p21' -> converts to 'P21'
        """
        # Convert sub-p21 -> P21
        if patient_id.startswith('sub-'):
            file_prefix = patient_id.replace('sub-', '').upper()
        else:
            file_prefix = patient_id.upper()
        
        
        eeg_path = os.path.join(self.data_dir, f'{file_prefix}_sEEG.npy')
        stimuli_path = os.path.join(self.data_dir, f'{file_prefix}_stimuli.npy')
        audio_path = os.path.join(self.data_dir, f'{file_prefix}_audio.npy')
        channels_path = os.path.join(self.data_dir, f'{file_prefix}_electrode_locations.csv')
        
        eeg = np.load(eeg_path)
        stimuli = np.load(stimuli_path, allow_pickle=True)
        audio = np.load(audio_path)
        
        # Load channels from CSV instead of npy
        channels_df = pd.read_csv(channels_path)
        channels = channels_df['electrode_name_1'].values # consistent with electrode_name_2
        
        return {
            'eeg': eeg,
            'stimuli': stimuli,
            'audio': audio,
            'channels': channels,
            'eeg_sr': self.sampling_rate
        }
        
    def get_all_patients(self):
        """Get list of all patient IDs"""
        return [f'P{i:02d}' for i in range(1, 31)]
        
    def categorize_patients(self):
        """Categorize patients BEFORE feature extraction based on raw stimuli files"""
        
        words_only = []
        sentences_only = []
        mixed = []
        
        for i in range(1, 31):
            pt_id = f'P{i:02d}'
            stimuli_file = os.path.join(self.data_dir, f'{pt_id}_stimuli.npy')
            
            if os.path.exists(stimuli_file):
                stimuli = np.load(stimuli_file, allow_pickle=True)
                unique = np.unique(stimuli)
                
                # Filter out empty strings
                unique = [s for s in unique if s and str(s).strip()]
                
                has_words = any(' ' not in str(s).strip() and len(str(s).strip()) <= 20 
                               for s in unique)
                has_sentences = any(' ' in str(s).strip() for s in unique)
                
                if has_sentences and not has_words:
                    sentences_only.append(pt_id)
                elif has_words and not has_sentences:
                    words_only.append(pt_id)
                else:
                    mixed.append(pt_id)
        
        print(f"\nPatient categorization:")
        print(f"  Words only (P11-P20): {len(words_only)} - {words_only}")
        print(f"  Sentences only (P21-P30): {len(sentences_only)} - {sentences_only}")
        print(f"  Mixed (P01-P10): {len(mixed)} - {mixed}")
        
        return words_only, sentences_only, mixed
    
    def create_patient_split(self, train_ratio=0.8, val_ratio=0.1):
        """Split patients into train/val/test"""
        patients = self.get_all_patients()
        n = len(patients)
        
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        return {
            'train': patients[:n_train],  # P01-P24
            'val': patients[n_train:n_train+n_val],  # P25-P27
            'test': patients[n_train+n_val:]  # P28-P30
        }
    
    def extract_all_patients(self, patient_list=None, batch_size=32, force_reextract=False):
        """Extract and save to HDF5"""
        
        if patient_list is None:
            patient_list = [f'P{i:02d}' for i in range(1, 31)]
        
        if not force_reextract and os.path.exists(self.raw_features_file):
            print(f"Loading existing features from {self.raw_features_file}")
            return self.load_from_h5(self.raw_features_file)
        
        print("Extracting features from scratch...")
        
        # Save directly to HDF5 as we go
        with h5py.File(self.raw_features_file, 'w') as f:
            for batch_start in range(0, len(patient_list), batch_size):
                batch_end = min(batch_start + batch_size, len(patient_list))
                batch_patients = patient_list[batch_start:batch_end]
                
                print(f"\nBatch {batch_start//batch_size + 1}: patients {batch_start+1}-{batch_end}")
                
                for pt_id in batch_patients:
                    print(f"  {pt_id}...", end=" ")
                    
                    eeg_path = os.path.join(self.data_dir, f'{pt_id}_sEEG.npy')
                    stimuli_path = os.path.join(self.data_dir, f'{pt_id}_stimuli.npy')
                    
                    if not os.path.exists(eeg_path):
                        print("not found")
                        continue
                    
                    try:
                        patient_features = self._extract_patient(eeg_path, stimuli_path)
                        if patient_features:
                            # Save to HDF5
                            grp = f.create_group(pt_id)
                            grp.create_dataset('features', data=patient_features['features'], 
                                             compression='gzip')
                            grp.attrs['labels'] = '|'.join(patient_features['labels'])
                            grp.attrs['n_windows'] = patient_features['n_windows']
                            print(f"done ({patient_features['features'].shape})")
                    except Exception as e:
                        print(f"error: {e}")
                    
                    gc.collect()
        
        return self.load_from_h5(self.raw_features_file)
        
        # Save checkpoint
        print(f"\nSaving raw features to {self.raw_features_file}")
        with open(self.raw_features_file, 'wb') as f:
            pickle.dump(all_patient_data, f)
        
        return all_patient_data    
        
    def _extract_patient(self, eeg_path, stimuli_path, max_samples=500000):
        """Extract features with flexible sampling rate"""
        
        eeg = np.load(eeg_path)
        stimuli = np.load(stimuli_path, allow_pickle=True)
        
        # Handle different sampling rates - downsample if needed
        if self.sampling_rate != 1024 and eeg.shape[0] % 2 == 0:
            # Assuming original might be 2048 Hz
            print(f"(downsampling)...", end=" ")
            eeg = eeg[::2]  # Simple downsampling - could use scipy.signal.decimate for better quality
        
        if eeg.shape[0] > max_samples:
            # Process in chunks
            all_feat = []
            all_labels = []
            
            for chunk_start in range(0, eeg.shape[0], max_samples):
                chunk_end = min(chunk_start + max_samples, eeg.shape[0])
                
                eeg_chunk = eeg[chunk_start:chunk_end]
                stimuli_chunk = stimuli[chunk_start:chunk_end]
                
                feat = extractHG(eeg_chunk, self.sampling_rate)
                feat_stacked = stackFeatures(feat, modelOrder=4, stepSize=5)
                labels_chunk = downsampleLabels(stimuli_chunk, self.sampling_rate)
                
                min_len = min(len(feat_stacked), len(labels_chunk) - 8)
                if min_len > 0:
                    feat_stacked = feat_stacked[:min_len]
                    labels_chunk = labels_chunk[4:min_len+4]
                    
                    all_feat.append(feat_stacked)
                    all_labels.extend(labels_chunk)
                
                del eeg_chunk, stimuli_chunk, feat
                gc.collect()
            
            if all_feat:
                feat_stacked = np.vstack(all_feat)
                labels = all_labels
            else:
                return None
        else:
            feat = extractHG(eeg, self.sampling_rate)
            feat_stacked = stackFeatures(feat, modelOrder=4, stepSize=5)
            labels = downsampleLabels(stimuli, self.sampling_rate)
            
            min_len = min(len(feat_stacked), len(labels) - 8)
            feat_stacked = feat_stacked[:min_len]
            labels = labels[4:min_len+4]
        
        del eeg, stimuli
        gc.collect()
        
        return {
            'features': feat_stacked,
            'labels': labels,
            'n_windows': len(labels)
        }
    
    def _stratified_split(self, words_only, sentences_only, mixed, 
                         val_size=0.1, test_size=0.1, random_state=42):
        """Your stratified split logic"""
        np.random.seed(random_state)
        
        def split_category(patients, name):
            n = len(patients)
            n_test = max(1, int(n * test_size))
            n_val = max(1, int(n * val_size))
            n_train = n - n_test - n_val
            
            if n_train < 0:
                n_train = 0
                n_val = max(1, n - n_test)
            
            shuffled = np.random.permutation(patients)
            
            train = shuffled[:n_train].tolist() if n_train > 0 else []
            val = shuffled[n_train:n_train+n_val].tolist()
            test = shuffled[n_train+n_val:].tolist()
            
            print(f"  {name}: {n} → train:{len(train)}, val:{len(val)}, test:{len(test)}")
            return train, val, test
        
        w_train, w_val, w_test = split_category(words_only, "Words")
        s_train, s_val, s_test = split_category(sentences_only, "Sentences")
        m_train, m_val, m_test = split_category(mixed, "Mixed")
        
        return {
            'train': w_train + s_train + m_train,
            'val': w_val + s_val + m_val,
            'test': w_test + s_test + m_test,
            'distribution': {
                'train': {'words': len(w_train), 'sentences': len(s_train), 'mixed': len(m_train)},
                'val': {'words': len(w_val), 'sentences': len(s_val), 'mixed': len(m_val)},
                'test': {'words': len(w_test), 'sentences': len(s_test), 'mixed': len(m_test)}
            }
        }
        
    def _combine_data_efficient(self, patient_ids, h5_file):
        """Combine data efficiently from HDF5"""
        
        # First pass: count total size
        total_samples = 0
        feature_dim = None
        
        with h5py.File(h5_file, 'r') as f:
            for pt_id in patient_ids:
                if pt_id in f:
                    total_samples += f[pt_id]['features'].shape[0]
                    if feature_dim is None:
                        feature_dim = f[pt_id]['features'].shape[1]
        
        if total_samples == 0:
            return np.array([]), np.array([])
        
        # Allocate arrays
        X = np.zeros((total_samples, feature_dim), dtype=np.float32)
        y = []
        
        # Second pass: fill arrays
        idx = 0
        with h5py.File(h5_file, 'r') as f:
            for pt_id in patient_ids:
                if pt_id in f:
                    grp = f[pt_id]
                    n_samples = grp['features'].shape[0]
                    X[idx:idx+n_samples] = grp['features'][()]
                    y.extend(grp.attrs['labels'].split('|'))
                    idx += n_samples
        
        return X, np.array(y)

    def get_sampled_data(self, sample_fraction=0.1):
        """Get a smaller sample of data for testing"""
        
        split_info = self.create_patient_split()
        
        # Sample fewer patients
        n_train = max(1, int(len(split_info['train']) * sample_fraction))
        n_val = max(1, int(len(split_info['val']) * sample_fraction))
        n_test = max(1, int(len(split_info['test']) * sample_fraction))
        
        sampled_split = {
            'train': split_info['train'][:n_train],
            'val': split_info['val'][:n_val],
            'test': split_info['test'][:n_test]
        }
        
        print(f"Sampling {sample_fraction*100}% of data:")
        print(f"  Train: {n_train} patients (from {len(split_info['train'])})")
        print(f"  Val: {n_val} patients (from {len(split_info['val'])})")
        print(f"  Test: {n_test} patients (from {len(split_info['test'])})")
        
        # Check if padded features exist
        if not os.path.exists(self.padded_features_file):
            print("Padded features not found. Running full pipeline...")
            self.get_final_data()  # This will create the padded features
        
        # Now combine with the smaller patient lists
        X_train, y_train = self._combine_data_efficient(sampled_split['train'], self.padded_features_file)
        X_val, y_val = self._combine_data_efficient(sampled_split['val'], self.padded_features_file)
        X_test, y_test = self._combine_data_efficient(sampled_split['test'], self.padded_features_file)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_final_data(self, force_rebuild=False):
        """Memory-efficient final data preparation"""
        
        if not force_rebuild and os.path.exists(self.final_data_file):
            print(f"Loading final data from {self.final_data_file}")
            with h5py.File(self.final_data_file, 'r') as f:
                X_train = f['X_train'][()]
                y_train = [label.decode() if isinstance(label, bytes) else label 
                          for label in f['y_train'][()]]
                X_val = f['X_val'][()]
                y_val = [label.decode() if isinstance(label, bytes) else label 
                        for label in f['y_val'][()]]
                X_test = f['X_test'][()]
                y_test = [label.decode() if isinstance(label, bytes) else label 
                         for label in f['y_test'][()]]
                
                return (X_train, np.array(y_train)), (X_val, np.array(y_val)), (X_test, np.array(y_test))
        
        # Step 1: Create/load split
        split_info = self.create_patient_split()
        
        # Step 2: Extract features if needed
        if not os.path.exists(self.raw_features_file):
            self.extract_all_patients()
        
        # Step 3: Pad features if needed (doesn't load data back)
        if not os.path.exists(self.padded_features_file):
            patient_data = self.load_from_h5(self.raw_features_file)
            self.pad_to_max_dimension(patient_data)
            # Clear from memory after padding
            del patient_data
            gc.collect()
        
        # Step 4: Combine data efficiently from HDF5 file
        print("Combining train data...")
        X_train, y_train = self._combine_data_efficient(split_info['train'], self.padded_features_file)
        print(f"  Train: {X_train.shape}")
        
        print("Combining val data...")
        X_val, y_val = self._combine_data_efficient(split_info['val'], self.padded_features_file)
        print(f"  Val: {X_val.shape}")
        
        print("Combining test data...")
        X_test, y_test = self._combine_data_efficient(split_info['test'], self.padded_features_file)
        print(f"  Test: {X_test.shape}")
        
        # Save final data
        print("Saving final data...")
        with h5py.File(self.final_data_file, 'w') as f:
            f.create_dataset('X_train', data=X_train, compression='gzip')
            f.create_dataset('y_train', data=np.array(y_train, dtype='S'), compression='gzip')
            f.create_dataset('X_val', data=X_val, compression='gzip')
            f.create_dataset('y_val', data=np.array(y_val, dtype='S'), compression='gzip')
            f.create_dataset('X_test', data=X_test, compression='gzip')
            f.create_dataset('y_test', data=np.array(y_test, dtype='S'), compression='gzip')
        
        print(f"Saved to {self.final_data_file}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)