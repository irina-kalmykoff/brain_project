# data_adapter.py
class DataAdapter:
    """Unified interface for different data sources"""
    
    def __init__(self, dataset_type='dutch'):
        self.dataset_type = dataset_type
        
    def load_participant_data(self, participant_id):
        """Load data in a unified format regardless of source"""
        
        if self.dataset_type == 'dutch':
            # Your existing NWB loading code
            io = NWBHDF5IO(f'path/to/{participant_id}/ieeg/{participant_id}_task-wordProduction_ieeg.nwb', 'r')
            nwbfile = io.read()
            eeg = nwbfile.acquisition['iEEG'].data[:]
            audio = nwbfile.acquisition['Audio'].data[:]
            words = nwbfile.acquisition['Stimulus'].data[:]
            io.close()
            
        elif self.dataset_type == 'shared':
            # New numpy data loading
            pt_id = participant_id.replace('sub-', 'P')  # Convert naming
            eeg = np.load(f'SharedData/raw/{pt_id}_sEEG.npy')
            audio = np.load(f'SharedData/raw/{pt_id}_audio.npy')
            words = np.load(f'SharedData/raw/{pt_id}_stimuli.npy')
            
        return {
            'eeg': eeg,
            'audio': audio,
            'words': words,
            'sr_eeg': 1024,
            'sr_audio': 48000 if self.dataset_type == 'dutch' else 48000
        }