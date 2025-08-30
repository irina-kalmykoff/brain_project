import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pynwb import NWBHDF5IO
from scipy.io import wavfile
from IPython.display import Audio, display, clear_output
import librosa
import librosa.display
from ipywidgets import interact, interactive, fixed, widgets
import functools
from debugger import DebugMixin

class BrainAudioDecoderViz(DebugMixin):
    """
    A class for visualizing EEG data, audio, and model results from the BrainAudioDecoder.
    This class is responsible for all visualization-related functionality.
    """
    
    def __init__(self, decoder, path_results=None, debug_mode=False):
        """
        Initialize the visualization class with a reference to a BrainAudioDecoder.
        
        Parameters:
        -----------
        decoder : BrainAudioDecoder
            Reference to a BrainAudioDecoder instance
        path_results : str or None
            Path to save visualization results. If None, uses decoder.path_results
        debug_mode : bool
            Whether to enable debug mode
        """
        # Initialize the DebugMixin
        super().__init__(class_name="BrainAudioDecoderViz", debug_mode=debug_mode)
        if debug_mode is not None:
            self.DEBUG_MODE = debug_mode
        self.log(f"Initialized with DEBUG_MODE={self.DEBUG_MODE}")
        
        self.log(f"Initializing visualization with debug_mode={self.DEBUG_MODE}")
        
        self.decoder = decoder
        self.path_results = path_results if path_results is not None else decoder.path_results
        
        # Create results directory if it doesn't exist
        if self.path_results is not None:
            os.makedirs(self.path_results, exist_ok=True)
            self.debug(f"Created results directory: {self.path_results}")
    
    def plot_raw_eeg(self, participant_id=None, eeg_data=None, sample_rate=None, 
                    duration=5, channels_to_plot=10, start_time=0, save_fig=False):
        """
        Plot raw EEG data for visualization.
        
        Parameters:
        -----------
        participant_id : str or None
            If provided, load EEG data for this participant. Ignored if eeg_data is provided.
        eeg_data : array or None
            EEG data to plot. If None, data will be loaded from the participant_id.
        sample_rate : int or None
            Sampling rate of the EEG data. If None, default to 1024 Hz.
        duration : float
            Duration in seconds to plot.
        channels_to_plot : int
            Number of channels to plot.
        start_time : float
            Start time in seconds for the plot window.
        save_fig : bool
            Whether to save the figure to results directory.
        
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Load data if not provided
        if eeg_data is None and participant_id is not None:
            # Load from NWB file
            io = NWBHDF5IO(
                os.path.join(self.decoder.path_bids, participant_id, 'ieeg', 
                            f'{participant_id}_task-wordProduction_ieeg.nwb'), 
                'r'
            )
            nwbfile = io.read()
            eeg_data = nwbfile.acquisition['iEEG'].data[:]
            sample_rate = 1024  # Default sampling rate for this dataset
            io.close()
            
            # Load channel names if available
            try:
                channels = pd.read_csv(
                    os.path.join(self.decoder.path_bids, participant_id, 'ieeg', 
                                f'{participant_id}_task-wordProduction_channels.tsv'), 
                    delimiter='\t'
                )
                channel_names = channels['name'].values
            except:
                channel_names = None
        else:
            channel_names = None
            
        if sample_rate is None:
            sample_rate = 1024  # Default to 1024 Hz if not provided
        
        # Convert time values to samples
        start_sample = int(start_time * sample_rate)
        duration_samples = int(duration * sample_rate)
        end_sample = start_sample + duration_samples
        
        # Make sure we don't go out of bounds
        if end_sample > eeg_data.shape[0]:
            print(f"Warning: Requested duration exceeds data length. Adjusting end time.")
            end_sample = eeg_data.shape[0]
            duration_samples = end_sample - start_sample
        
        # Select a subset of channels if there are many
        if eeg_data.shape[1] > channels_to_plot:
            channel_indices = np.linspace(0, eeg_data.shape[1]-1, channels_to_plot, dtype=int)
        else:
            channel_indices = range(eeg_data.shape[1])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Get the data we want to plot
        plot_data = eeg_data[start_sample:end_sample, :]
        
        # Calculate a reasonable offset based on signal amplitude
        max_amp = np.max(np.abs(plot_data[:, channel_indices]))
        offset = max_amp * 2  # Adjust based on signal amplitude
        
        # Plot each channel
        for i, ch_idx in enumerate(channel_indices):
            ch_offset = i * offset
            ax.plot(
                np.arange(duration_samples) / sample_rate, 
                plot_data[:, ch_idx] + ch_offset,
                label=f'Channel {channel_names[ch_idx] if channel_names is not None else ch_idx}'
            )
        
        # Add annotations
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV) + offset')
        ax.set_title(f'Raw EEG Signals {f"for {participant_id}" if participant_id else ""}')
        ax.legend(loc='upper right')
        ax.grid(True)
        
        if save_fig and self.path_results is not None:
            save_path = os.path.join(
                self.path_results, 
                f'{"" if participant_id is None else participant_id + "_"}raw_eeg.png'
            )
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.tight_layout()
        
        return fig
    

    def plot_spectrogram(self, participant_id, original=True, predicted=False, 
                         side_by_side=True, log_scale=True, save_fig=False):
        """
        Plot the spectrogram for a participant.
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
        original : bool
            Whether to plot the original spectrogram
        predicted : bool
            Whether to plot the predicted spectrogram (if available)
        side_by_side : bool
            Whether to plot original and predicted spectrograms side by side
        log_scale : bool
            Whether to use log scale for spectrograms
        save_fig : bool
            Whether to save the figure to results directory
        
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Load original spectrogram
        orig_spec_path = os.path.join(self.decoder.path_output, f'{participant_id}_spec.npy')
        if not os.path.exists(orig_spec_path):
            print(f"Error: Original spectrogram file not found at {orig_spec_path}")
            return None
        
        orig_spec = np.load(orig_spec_path)
        
        # Load predicted spectrogram if requested
        pred_spec = None
        if predicted:
            pred_spec_path = os.path.join(self.decoder.path_results, f'{participant_id}_predicted_spec.npy')
            if os.path.exists(pred_spec_path):
                pred_spec = np.load(pred_spec_path)
            else:
                print(f"Warning: Predicted spectrogram file not found at {pred_spec_path}")
                predicted = False
        
        # Create figure
        if original and predicted and side_by_side:
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            
            # Original spectrogram
            im1 = axs[0].imshow(
                orig_spec.T, 
                aspect='auto', 
                origin='lower',
                norm='log' if log_scale else None
            )
            axs[0].set_title(f'Original Spectrogram ({participant_id})')
            axs[0].set_xlabel('Time Window')
            axs[0].set_ylabel('Mel Frequency Bin')
            plt.colorbar(im1, ax=axs[0], label='Log Mel Energy')
            
            # Predicted spectrogram
            im2 = axs[1].imshow(
                pred_spec.T, 
                aspect='auto', 
                origin='lower',
                norm='log' if log_scale else None
            )
            axs[1].set_title(f'Predicted Spectrogram ({participant_id})')
            axs[1].set_xlabel('Time Window')
            axs[1].set_ylabel('Mel Frequency Bin')
            plt.colorbar(im2, ax=axs[1], label='Log Mel Energy')
            
        elif original and predicted:
            # Plot one above the other
            fig, axs = plt.subplots(2, 1, figsize=(12, 10))
            
            # Original spectrogram
            im1 = axs[0].imshow(
                orig_spec.T, 
                aspect='auto', 
                origin='lower',
                norm='log' if log_scale else None
            )
            axs[0].set_title(f'Original Spectrogram ({participant_id})')
            axs[0].set_xlabel('Time Window')
            axs[0].set_ylabel('Mel Frequency Bin')
            plt.colorbar(im1, ax=axs[0], label='Log Mel Energy')
            
            # Predicted spectrogram
            im2 = axs[1].imshow(
                pred_spec.T, 
                aspect='auto', 
                origin='lower',
                norm='log' if log_scale else None
            )
            axs[1].set_title(f'Predicted Spectrogram ({participant_id})')
            axs[1].set_xlabel('Time Window')
            axs[1].set_ylabel('Mel Frequency Bin')
            plt.colorbar(im2, ax=axs[1], label='Log Mel Energy')
            
        else:
            # Just one spectrogram
            fig, ax = plt.subplots(figsize=(12, 6))
            if original:
                im = ax.imshow(
                    orig_spec.T, 
                    aspect='auto', 
                    origin='lower',
                    norm='log' if log_scale else None
                )
                ax.set_title(f'Original Spectrogram ({participant_id})')
            else:
                im = ax.imshow(
                    pred_spec.T, 
                    aspect='auto', 
                    origin='lower',
                    norm='log' if log_scale else None
                )
                ax.set_title(f'Predicted Spectrogram ({participant_id})')
            
            ax.set_xlabel('Time Window')
            ax.set_ylabel('Mel Frequency Bin')
            plt.colorbar(im, ax=ax, label='Log Mel Energy')
        
        plt.tight_layout()
        
        if save_fig and self.path_results is not None:
            save_name = f'{participant_id}_spectrogram'
            if original and predicted:
                save_name += '_comparison'
            elif predicted:
                save_name += '_predicted'
            save_path = os.path.join(self.path_results, f'{save_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    

    def play_audio(self, participant_id, original=True, predicted=False, reconstructed=False):
        """
        Play audio for a participant.
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
        original : bool
            Whether to play the original audio
        predicted : bool
            Whether to play the predicted audio (if available)
        reconstructed : bool
            Whether to play the reconstructed original audio (if available)
        """
        audio_files = []
        
        # Check for original audio
        if original:
            orig_audio_path = os.path.join(self.decoder.path_output, f'{participant_id}_orig_audio.wav')
            if os.path.exists(orig_audio_path):
                audio_files.append(('Original Audio', orig_audio_path))
            else:
                print(f"Warning: Original audio file not found at {orig_audio_path}")
        
        # Check for predicted audio
        if predicted:
            pred_audio_path = os.path.join(self.decoder.path_results, f'{participant_id}_predicted.wav')
            if os.path.exists(pred_audio_path):
                audio_files.append(('Predicted Audio', pred_audio_path))
            else:
                print(f"Warning: Predicted audio file not found at {pred_audio_path}")
        
        # Check for reconstructed original audio
        if reconstructed:
            recon_audio_path = os.path.join(self.decoder.path_results, f'{participant_id}_orig_synthesized.wav')
            if os.path.exists(recon_audio_path):
                audio_files.append(('Reconstructed Original Audio', recon_audio_path))
            else:
                print(f"Warning: Reconstructed audio file not found at {recon_audio_path}")
        
        # Play audio files
        for label, audio_path in audio_files:
            print(f"{label}:")
            display(Audio(audio_path))
    

    def plot_waveform(self, participant_id, original=True, predicted=False, 
                      reconstructed=False, side_by_side=True, save_fig=False):
        """
        Plot audio waveforms for a participant.
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
        original : bool
            Whether to plot the original audio waveform
        predicted : bool
            Whether to plot the predicted audio waveform (if available)
        reconstructed : bool
            Whether to plot the reconstructed original audio waveform (if available)
        side_by_side : bool
            Whether to plot waveforms side by side
        save_fig : bool
            Whether to save the figure to results directory
        
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        audio_files = []
        
        # Check for original audio
        if original:
            orig_audio_path = os.path.join(self.decoder.path_output, f'{participant_id}_orig_audio.wav')
            if os.path.exists(orig_audio_path):
                audio_files.append(('Original Audio', orig_audio_path))
            else:
                print(f"Warning: Original audio file not found at {orig_audio_path}")
                original = False
        
        # Check for predicted audio
        if predicted:
            pred_audio_path = os.path.join(self.decoder.path_results, f'{participant_id}_predicted.wav')
            if os.path.exists(pred_audio_path):
                audio_files.append(('Predicted Audio', pred_audio_path))
            else:
                print(f"Warning: Predicted audio file not found at {pred_audio_path}")
                predicted = False
        
        # Check for reconstructed original audio
        if reconstructed:
            recon_audio_path = os.path.join(self.decoder.path_results, f'{participant_id}_orig_synthesized.wav')
            if os.path.exists(recon_audio_path):
                audio_files.append(('Reconstructed Original Audio', recon_audio_path))
            else:
                print(f"Warning: Reconstructed audio file not found at {recon_audio_path}")
                reconstructed = False
        
        # Determine number of subplots
        n_plots = len(audio_files)
        if n_plots == 0:
            print("No audio files found.")
            return None
        
        # Create figure
        if side_by_side and n_plots > 1:
            fig, axs = plt.subplots(1, n_plots, figsize=(16, 4), sharey=True)
            if n_plots == 1:
                axs = [axs]  # Make it iterable
        else:
            fig, axs = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
            if n_plots == 1:
                axs = [axs]  # Make it iterable
        
        # Plot waveforms
        for i, (label, audio_path) in enumerate(audio_files):
            sr, data = wavfile.read(audio_path)
            
            # Convert to float for plotting
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32767.0
            
            time = np.arange(len(data)) / sr
            axs[i].plot(time, data)
            axs[i].set_title(label)
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel('Amplitude')
            axs[i].grid(True)
        
        plt.tight_layout()
        
        if save_fig and self.path_results is not None:
            save_name = f'{participant_id}_waveform'
            if n_plots > 1:
                save_name += '_comparison'
            save_path = os.path.join(self.path_results, f'{save_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig   
        

    def plot_channels_comparison(self, participant_id, channels_to_plot=None, duration_seconds=5, start_time=0, save_fig=False):
        """Plot selected EEG channels for comparison"""
        # Load EEG data from NWB file
        io = NWBHDF5IO(
            os.path.join(self.decoder.path_bids, participant_id, 'ieeg', 
                       f'{participant_id}_task-wordProduction_ieeg.nwb'), 
            'r'
        )
        nwbfile = io.read()
        eeg_data = nwbfile.acquisition['iEEG'].data[:]
        io.close()
        
        # Default sampling rate for this dataset
        sampling_rate = 1024
        
        # Convert time to samples
        start_sample = int(start_time * sampling_rate)
        samples_to_plot = int(duration_seconds * sampling_rate)
        end_sample = start_sample + samples_to_plot
        
        # Make sure we don't go out of bounds
        if end_sample > eeg_data.shape[0]:
            print(f"Warning: Requested duration exceeds data length. Adjusting end time.")
            end_sample = eeg_data.shape[0]
            samples_to_plot = end_sample - start_sample
        
        # If channels_to_plot is None, select default channels
        if channels_to_plot is None:
            # Plot channel 0 and a few other channels
            channels_to_plot = [0] + list(range(1, min(5, eeg_data.shape[1])))
        
        # Select data segment
        data_segment = eeg_data[start_sample:end_sample, :]
        
        # Try to load channel names
        try:
            channels_df = pd.read_csv(
                os.path.join(self.decoder.path_bids, participant_id, 'ieeg', 
                            f'{participant_id}_task-wordProduction_channels.tsv'), 
                delimiter='\t'
            )
            channel_names = channels_df['name'].values
        except:
            channel_names = None
            print("Could not load channel names, using indices instead.")
        
        # Create figure
        fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(15, 10), sharex=True)
        if len(channels_to_plot) == 1:
            axes = [axes]  # Make it a list for consistent indexing
        
        # Create time axis in seconds
        time_axis = np.arange(samples_to_plot) / sampling_rate
        
        # Plot each channel
        for i, chan_idx in enumerate(channels_to_plot):
            axes[i].plot(time_axis, data_segment[:, chan_idx])
            
            # Add channel name or index as title
            if channel_names is not None:
                axes[i].set_title(f"Channel {channel_names[chan_idx]}")
            else:
                axes[i].set_title(f"Channel {chan_idx}")
            
            axes[i].set_ylabel("Amplitude (μV)")
            axes[i].grid(True)
        
        # Add common x-label
        axes[-1].set_xlabel("Time (s)")
        
        # Add overall title
        plt.suptitle(f"Channel Comparison for {participant_id}", fontsize=16)
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig and self.path_results is not None:
            save_path = os.path.join(
                self.path_results, 
                f'{participant_id}_channel_comparison.png'
            )
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
        
   
    def plot_channel_correlations_across_participants(self, channel_name=None, region=None, 
                                                 top_n=10, sort_by='mean', save_fig=False):
        """
        Plot channel correlations across participants
        
        Parameters:
        -----------
        channel_name : str or None
            Channel name to visualize. If None, will use region or top channels.
        region : str or None
            Brain region to visualize channels from. Ignored if channel_name is provided.
        top_n : int
            Number of top channels to display if channel_name and region are None.
        sort_by : str
            How to sort channels, options: 'mean', 'variance', 'max'
        save_fig : bool
            Whether to save the figure to results directory
        
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        
        # Load combined results
        result_path = os.path.join(self.decoder.path_results, 'channel_analysis', 'all_participants_channel_correlations.npy')
        if not os.path.exists(result_path):
            print("No channel analysis results found. Please run analyze_channels_across_participants first.")
            return None
        
        all_results = np.load(result_path, allow_pickle=True).item()
        
        # Extract participant IDs and channel names
        participant_ids = list(all_results.keys())
        all_channels = set()
        for p_id in participant_ids:
            all_channels.update(all_results[p_id].keys())
        
        # Create a DataFrame for easier analysis
        data = []
        for p_id in participant_ids:
            for ch_name, ch_info in all_results[p_id].items():
                if 'correlation' in ch_info and not np.isnan(ch_info['correlation']):
                    data.append({
                        'participant_id': p_id,
                        'channel_name': ch_name,
                        'correlation': ch_info['correlation'],
                        'region': ch_info['region'] if 'region' in ch_info else 'Unknown'
                    })
        
        df = pd.DataFrame(data)
        
        # If no data, return
        if len(df) == 0:
            print("No valid data found.")
            return None
        
        # Determine which channels to visualize
        if channel_name is not None:
            # Specific channel
            channels_to_plot = [channel_name]
            title = f"Channel Correlation: {channel_name}"
        elif region is not None:
            # Channels from specific region
            region_df = df[df['region'].str.contains(region, case=False)]
            if len(region_df) == 0:
                print(f"No channels found in region: {region}")
                return None
            
            # Get channels with data from most participants
            channel_counts = region_df['channel_name'].value_counts()
            channels_to_plot = channel_counts.head(top_n).index.tolist()
            title = f"Channel Correlations for Region: {region}"
        else:
            # Get top channels by chosen metric
            channel_stats = df.groupby('channel_name')['correlation'].agg(['mean', 'std', 'max', 'count'])
            channel_stats['variance'] = channel_stats['std'] ** 2
            
            # Only consider channels present in at least half the participants
            min_participants = max(2, len(participant_ids) // 2)
            channel_stats = channel_stats[channel_stats['count'] >= min_participants]
            
            if sort_by == 'variance':
                # Sort by variance (highest variance first)
                top_channels = channel_stats.sort_values('variance', ascending=False).head(top_n)
            elif sort_by == 'max':
                # Sort by maximum correlation
                top_channels = channel_stats.sort_values('max', ascending=False).head(top_n)
            else:
                # Default: sort by mean correlation
                top_channels = channel_stats.sort_values('mean', ascending=False).head(top_n)
            
            channels_to_plot = top_channels.index.tolist()
            title = f"Top {top_n} Channels by {sort_by.capitalize()}"
        
        # Filter data for the selected channels
        plot_df = df[df['channel_name'].isin(channels_to_plot)]
        
        # Create figure based on visualization type
        plt.figure(figsize=(12, 6 + 0.3 * len(channels_to_plot)))
        
        # Create boxplot for each channel across participants
        sns.boxplot(x='correlation', y='channel_name', data=plot_df, orient='h', whis=[0, 100])
        
        # Add individual points
        sns.stripplot(x='correlation', y='channel_name', data=plot_df, 
                     orient='h', color='black', alpha=0.5, jitter=True)
        
        # Add mean correlation line
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Channel')
        plt.title(title)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig and self.path_results is not None:
            save_name = 'channel_correlations'
            if channel_name:
                save_name += f'_{channel_name}'
            elif region:
                save_name += f'_region_{region}'
            else:
                save_name += f'_top{top_n}_by_{sort_by}'
                
            save_path = os.path.join(self.path_results, f'{save_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return plt.gcf()

  
    def interactive_channel_analysis(self):
        """Create interactive widgets for channel analysis"""
        
        # Load results if they exist
        result_path = os.path.join(self.decoder.path_results, 'channel_analysis', 'all_participants_channel_correlations.npy')
        if os.path.exists(result_path):
            all_results = np.load(result_path, allow_pickle=True).item()
            
            # Extract channel names and regions
            all_channels = set()
            all_regions = set()
            
            for p_id in all_results:
                for ch_name, ch_info in all_results[p_id].items():
                    all_channels.add(ch_name)
                    if 'region' in ch_info:
                        region = ch_info['region']
                        all_regions.add(region)
            
            all_channels = sorted(list(all_channels))
            all_regions = sorted(list(all_regions))
            
            # Create dropdown for channel selection
            channel_dropdown = widgets.Dropdown(
                options=['All Channels'] + all_channels,
                value='All Channels',
                description='Channel:',
                disabled=False
            )
            
            # Create dropdown for region selection
            region_dropdown = widgets.Dropdown(
                options=['All Regions'] + all_regions,
                value='All Regions',
                description='Region:',
                disabled=False
            )
            
            # Create dropdown for metric
            metric_dropdown = widgets.Dropdown(
                options=['correlation', 'presence'],
                value='correlation',
                description='Metric:',
                disabled=False
            )
            
            # Create slider for top_n
            top_n_slider = widgets.IntSlider(
                value=10,
                min=5,
                max=30,
                step=5,
                description='Top N:',
                disabled=False
            )
            
            # Create dropdown for sort_by
            sort_dropdown = widgets.Dropdown(
                options=['mean', 'variance', 'max'],
                value='mean',
                description='Sort by:',
                disabled=False
            )
            
            # Create checkbox for clustering
            cluster_checkbox = widgets.Checkbox(
                value=True,
                description='Cluster Matrix',
                disabled=False
            )
            
            # Define interactive function for plotting
            def plot_channels(channel, region, sort_by, top_n, plot_type):
                if plot_type == 'Box Plot':
                    if channel != 'All Channels':
                        self.plot_channel_correlations_across_participants(channel_name=channel)
                    elif region != 'All Regions':
                        self.plot_channel_correlations_across_participants(region=region, top_n=top_n)
                    else:
                        self.plot_channel_correlations_across_participants(top_n=top_n, sort_by=sort_by)
                else:  # Matrix
                    if region != 'All Regions':
                        self.plot_channel_matrix(region=region, metric='correlation', cluster=True)
                    else:
                        self.plot_channel_matrix(metric='correlation', cluster=True)
            
            # Create interactive widget
            interactive_plot = interactive(
                plot_channels,
                channel=channel_dropdown,
                region=region_dropdown,
                sort_by=sort_dropdown,
                top_n=top_n_slider,
                plot_type=widgets.RadioButtons(
                    options=['Box Plot', 'Matrix'],
                    value='Box Plot',
                    description='Plot Type:',
                    disabled=False
                )
            )
            
            return interactive_plot
        else:
            print("No channel analysis results found. Please run analyze_channels_across_participants first.")
            return None
            
      
    def plot_channel_availability(self, region=None, min_participants=1, 
                            sort_by='count', save_fig=False):
        """
        Visualize which channels are available across participants
        
        Parameters:
        -----------
        region : str or None
            Filter channels by brain region (if None, show all channels)
        min_participants : int
            Minimum number of participants that must have a channel for it to be shown
        sort_by : str
            How to sort channels: 'count' (number of participants), 'name' (alphabetical),
            or 'availability' (percentage of participants)
        save_fig : bool
            Whether to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        
        # Load channel analysis results
        result_path = os.path.join(self.decoder.path_results, 'channel_analysis', 'all_participants_channel_correlations.npy')
        if not os.path.exists(result_path):
            # Try to find any channel information
            print("Channel analysis results not found. Looking for channel information...")
            
            # Check if we can get channel information from the raw data
            participant_ids = [f'sub-{i:02d}' for i in range(1, 11)]
            channel_info = {}
            
            for participant_id in participant_ids:
                try:
                    # Load channel information
                    channels_df = pd.read_csv(
                        os.path.join(self.decoder.path_bids, participant_id, 'ieeg', 
                                    f'{participant_id}_task-wordProduction_channels.tsv'), 
                        delimiter='\t'
                    )
                    
                    # Store channel names and regions
                    channel_info[participant_id] = {
                        name: {'region': region if 'description' in channels_df.columns else 'Unknown'}
                        for name, region in zip(
                            channels_df['name'], 
                            channels_df['description'] if 'description' in channels_df.columns else ['Unknown'] * len(channels_df)
                        )
                    }
                    print(f"Found {len(channel_info[participant_id])} channels for {participant_id}")
                except Exception as e:
                    print(f"Error loading channel information for {participant_id}: {e}")
            
            if not channel_info:
                print("No channel information found. Please run analyze_channels_across_participants first.")
                return None
            
            all_results = channel_info
        else:
            # Load results from file
            all_results = np.load(result_path, allow_pickle=True).item()
        
        # Extract participant IDs
        participant_ids = list(all_results.keys())
        n_participants = len(participant_ids)
        
        # Collect channel information
        channel_data = []
        
        for p_id in participant_ids:
            for ch_name, ch_info in all_results[p_id].items():
                # Extract region if available
                if isinstance(ch_info, dict) and 'region' in ch_info:
                    ch_region = ch_info['region']
                else:
                    ch_region = 'Unknown'
                    
                # Apply region filter if specified
                if region is not None and (ch_region == 'Unknown' or region.lower() not in ch_region.lower()):
                    continue
                    
                channel_data.append({
                    'participant_id': p_id,
                    'channel_name': ch_name,
                    'region': ch_region
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(channel_data)
        
        # Count occurrences of each channel
        channel_counts = df['channel_name'].value_counts().reset_index()
        channel_counts.columns = ['channel_name', 'count']
        
        # Calculate availability percentage
        channel_counts['availability'] = channel_counts['count'] / n_participants * 100
        
        # Filter by minimum number of participants
        channel_counts = channel_counts[channel_counts['count'] >= min_participants]
        
        # Add region information (using the most common region for each channel)
        channel_regions = df.groupby('channel_name')['region'].agg(
            lambda x: pd.Series.mode(x)[0] if len(pd.Series.mode(x)) > 0 else 'Unknown'
        ).reset_index()
        
        channel_counts = pd.merge(channel_counts, channel_regions, on='channel_name', how='left')
        
        # Sort based on the specified criterion
        if sort_by == 'name':
            channel_counts = channel_counts.sort_values('channel_name')
        elif sort_by == 'availability':
            channel_counts = channel_counts.sort_values('availability', ascending=False)
        else:  # default: sort by count
            channel_counts = channel_counts.sort_values('count', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, max(8, len(channel_counts) * 0.25)))
        
        # Create colormap based on regions
        unique_regions = channel_counts['region'].unique()
        region_to_color = {region: i for i, region in enumerate(unique_regions)}
        
        # Create barplot
        bars = ax.barh(
            channel_counts['channel_name'], 
            channel_counts['count'],
            color=[plt.cm.tab20(region_to_color[r] % 20) for r in channel_counts['region']]
        )
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width + 0.1
            percentage = channel_counts['availability'].iloc[i]
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f"{percentage:.1f}%", va='center')
        
        # Set axis limits
        ax.set_xlim(0, n_participants + 1)  # Add space for percentage labels
        
        # Add labels and title
        region_str = f" in {region} region" if region else ""
        ax.set_xlabel(f"Number of Participants (out of {n_participants})")
        ax.set_ylabel("Channel Name")
        ax.set_title(f"Channel Availability Across Participants{region_str}")
        
        # Add grid for readability
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add a legend for regions
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=plt.cm.tab20(region_to_color[r] % 20), label=r)
            for r in unique_regions
        ]
        ax.legend(handles=legend_elements, title="Brain Region", 
                  loc="lower right", bbox_to_anchor=(1, 0))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig and self.path_results is not None:
            save_name = 'channel_availability'
            if region:
                save_name += f'_{region}'
            
            save_path = os.path.join(self.path_results, f'{save_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_channel_matrix(self, participant_ids=None, min_correlation=0.0, save_fig=False):
        """
        Create a matrix visualization showing channel correlations across participants.
        Channels as rows, participants as columns.

        """
        
        # Close any existing figures
        plt.close('all')
        
        # Get all participants if not specified
        if participant_ids is None:
            participant_ids = [f'sub-{i:02d}' for i in range(1, 11)]
        
        # Load channel data for each participant
        all_channel_data = {}
        all_channels = set()
        
        for p_id in participant_ids:
            result_path = os.path.join(self.decoder.path_results, 'channel_analysis', 
                                     f'{p_id}_channel_correlations.npy')
            if os.path.exists(result_path):
                channel_results = np.load(result_path, allow_pickle=True).item()
                
                # Convert to DataFrame
                results_df = pd.DataFrame.from_dict(channel_results, orient='index')
                results_df.index.name = 'channel'
                results_df = results_df.reset_index()
                
                # Filter by correlation
                results_df = results_df[results_df['correlation'] >= min_correlation]
                
                if not results_df.empty:
                    all_channel_data[p_id] = results_df
                    all_channels.update(results_df['channel'])
        
        if not all_channel_data:
            print("No channel data found or all channels filtered out by correlation threshold")
            return None
        
        # Count how many participants have each channel
        channel_counts = {}
        for channel in all_channels:
            channel_counts[channel] = sum(1 for p_id in all_channel_data 
                                        if channel in all_channel_data[p_id]['channel'].values)
        
        # Sort channels by number of participants (descending)
        sorted_channels = sorted(channel_counts.keys(), key=lambda c: (-channel_counts[c], c))
        
        # Create correlation matrix (channels as rows, participants as columns)
        correlation_matrix = np.zeros((len(sorted_channels), len(participant_ids)))
        correlation_matrix.fill(np.nan)  # Fill with NaN for missing values
        
        for i, channel in enumerate(sorted_channels):
            for j, p_id in enumerate(participant_ids):
                if p_id in all_channel_data:
                    # Find this channel in the participant's data
                    channel_row = all_channel_data[p_id][all_channel_data[p_id]['channel'] == channel]
                    if not channel_row.empty:
                        correlation_matrix[i, j] = channel_row.iloc[0]['correlation']
        
        # Create DataFrame for seaborn
        matrix_df = pd.DataFrame(
            correlation_matrix, 
            index=sorted_channels, 
            columns=participant_ids
        )
        
        # Create figure
        fig_width = min(14, max(8, len(participant_ids) * 0.8))
        fig_height = min(20, max(8, len(sorted_channels) * 0.2))
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create heatmap with correlation values
        sns.heatmap(
            matrix_df, 
            cmap='RdBu_r',  # Red-Blue colormap, reversed (blue is positive)
            vmin=-0.5,      # Minimum correlation value
            vmax=0.5,       # Maximum correlation value
            center=0,       # Center the colormap at zero
            linewidths=0.5,
            linecolor='lightgray',
            ax=ax,
            cbar_kws={'label': 'Correlation'}
        )
        
        # Add channel counts on right
        channel_participant_counts = np.sum(~np.isnan(correlation_matrix), axis=1)
        for i, count in enumerate(channel_participant_counts):
            ax.text(
                len(participant_ids) + 0.5, i + 0.5, 
                f"{int(count)}/{len(participant_ids)}",
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=9,
                color='black'
            )
        
        # Add average correlation on far right
        for i, channel in enumerate(sorted_channels):
            channel_corrs = correlation_matrix[i, :]
            mean_corr = np.nanmean(channel_corrs)
            ax.text(
                len(participant_ids) + 2.0, i + 0.5,
                f"{mean_corr:.3f}",
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=9,
                color='black'
            )
        
        # Add channel counts per participant at bottom
        participant_channel_counts = np.sum(~np.isnan(correlation_matrix), axis=0)
        for j, count in enumerate(participant_channel_counts):
            ax.text(
                j + 0.5, len(sorted_channels) + 0.2, 
                int(count),
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=9,
                color='black'
            )
        
        # Add labels and title
        ax.set_title(f"Channel Correlation Matrix Across Participants (min. correlation: {min_correlation:.2f})")
        ax.set_xlabel("Participants")
        ax.set_ylabel("Channels (sorted by number of participants)")
        
        # Add annotation for totals
        ax.text(
            -0.1, -0.05, 
            f"Total: {len(sorted_channels)} channels across {len(participant_ids)} participants",
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold'
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig and self.path_results is not None:
            save_path = os.path.join(
                self.path_results, 
                f'channel_correlation_matrix_min{min_correlation:.2f}.png'
            )
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
        
        
    def show_channel_stats(self, participant_id):
        """
        Show simple text statistics about channel contributions for a participant
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
        """
        
        # Load channel results
        result_path = os.path.join(self.decoder.path_results, 'channel_analysis', 
                                  f'{participant_id}_channel_correlations.npy')
        if not os.path.exists(result_path):
            print(f"No channel analysis results found for {participant_id}. Please run analyze_individual_channels first.")
            return
        
        channel_results = np.load(result_path, allow_pickle=True).item()
        
        # Convert results to DataFrame
        results_df = pd.DataFrame.from_dict(channel_results, orient='index')
        results_df.index.name = 'channel'
        results_df = results_df.reset_index()
        
        # Remove any NaN values
        results_df = results_df.dropna(subset=['correlation'])
        
        # Sort by correlation
        results_df = results_df.sort_values('correlation', ascending=False)
        
        # Display top channels
        print(f"Top 10 most informative channels for {participant_id}:")
        print(results_df.head(10)[['channel', 'region', 'correlation']])
        
        # Calculate region statistics
        region_stats = results_df.groupby('region')['correlation'].agg(['mean', 'median', 'std', 'count'])
        region_stats = region_stats.sort_values('median', ascending=False)
        
        print("\nRegion statistics:")
        print(region_stats)
        
        return results_df
        
    def simple_participant_selector(self):
        """
        Create a simple dropdown to select participants and view channel statistics
        
        Returns:
        --------
        ipywidgets.Widget
            Simple dropdown widget for participant selection
        """
        
        # Get participant IDs
        participant_ids = [f'sub-{i:02d}' for i in range(1, 11)]
        
        # Create output widget
        output = widgets.Output()
        
        # Create dropdown for participant selection
        participant_dropdown = widgets.Dropdown(
            options=participant_ids,
            value=participant_ids[0],
            description='Participant:',
            disabled=False
        )
        
        # Define function for showing statistics
        def show_stats(change):
            with output:
                clear_output(wait=True)
                self.show_channel_stats(change.new)
        
        # Connect dropdown to function
        participant_dropdown.observe(show_stats, names='value')
        
        # Initial display
        with output:
            self.show_channel_stats(participant_ids[0])
        
        # Create and return the widget
        widget = widgets.VBox([participant_dropdown, output])
        return widget
        
        
    def visualize_common_words(self, participant_ids=None, min_frequency=1.0, min_participants=2, 
                         max_words=50, save_fig=False):
        """
        Visualize common words across participants
        
        """
        
        # Close any existing figures
        plt.close('all')
        
        # Get all participants if not specified
        if participant_ids is None:
            participant_ids = [f'sub-{i:02d}' for i in range(1, 11)]
        
        # Extract word lists for each participant
        all_words = {}
        
        for p_id in participant_ids:
            word_list = self.decoder.extract_word_list(participant_id=p_id)
            if word_list is not None:
                # Filter by minimum frequency
                word_list = word_list[word_list['frequency'] >= min_frequency]
                all_words[p_id] = word_list
        
        if not all_words:
            print("No word lists found")
            return None
        
        # Find words that appear in multiple participants
        word_participants = {}
        
        for p_id, word_list in all_words.items():
            for _, row in word_list.iterrows():
                word = row['word']
                if word not in word_participants:
                    word_participants[word] = {'participants': [], 'total_count': 0, 'avg_frequency': 0}
                
                word_participants[word]['participants'].append(p_id)
                word_participants[word]['total_count'] += row['count']
                word_participants[word]['avg_frequency'] += row['frequency']
        
        # Calculate average frequency
        for word in word_participants:
            n_participants = len(word_participants[word]['participants'])
            word_participants[word]['avg_frequency'] /= n_participants
        
        # Filter by minimum number of participants
        filtered_words = {
            word: info for word, info in word_participants.items() 
            if len(info['participants']) >= min_participants
        }
        
        if not filtered_words:
            print(f"No words found that appear in at least {min_participants} participants")
            return None
        
        # Create DataFrame for visualization
        common_words_df = pd.DataFrame({
            'word': list(filtered_words.keys()),
            'participants': [len(info['participants']) for info in filtered_words.values()],
            'total_count': [info['total_count'] for info in filtered_words.values()],
            'avg_frequency': [info['avg_frequency'] for info in filtered_words.values()]
        })
        
        # Sort by number of participants (descending), then by average frequency (descending)
        common_words_df = common_words_df.sort_values(
            ['participants', 'avg_frequency'], 
            ascending=[False, False]
        ).reset_index(drop=True)
        
        # Limit to max_words
        if len(common_words_df) > max_words:
            common_words_df = common_words_df.head(max_words)
        
        print(f"Found {len(common_words_df)} common words across participants")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, max(8, len(common_words_df) * 0.25)))
        
        # Create a color map based on number of participants
        max_participants = len(participant_ids)
        colors = plt.cm.viridis(common_words_df['participants'] / max_participants)
        
        # Create horizontal bar chart
        bars = ax.barh(
            common_words_df['word'], 
            common_words_df['avg_frequency'],
            color=colors
        )
        
        # Add participant count as text
        for i, bar in enumerate(bars):
            ax.text(
                bar.get_width() + 0.5, 
                bar.get_y() + bar.get_height()/2,
                f"{common_words_df.iloc[i]['participants']}/{len(participant_ids)} participants",
                va='center'
            )
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(1, max_participants))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Number of Participants')
        
        # Add labels and title
        ax.set_xlabel('Average Frequency (%)')
        ax.set_ylabel('Word')
        ax.set_title(f'Common Words Across Participants (min. {min_participants} participants)')
        
        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig and self.path_results is not None:
            save_path = os.path.join(
                self.path_results, 
                f'common_words_min{min_participants}_freq{min_frequency:.1f}.png'
            )
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig

    def visualize_word_matrix(self, participant_ids=None, save_fig=False):
        """
        Create a matrix visualization showing which words are used by which participants.
        Words as rows, participants as columns.
        
        """
        
        # Close any existing figures
        plt.close('all')
        
        # Get all participants if not specified
        if participant_ids is None:
            participant_ids = [f'sub-{i:02d}' for i in range(1, 11)]
        
        # Extract word lists for each participant
        all_words = {}
        all_unique_words = set()
        
        for p_id in participant_ids:
            word_list = self.decoder.extract_word_list(participant_id=p_id, verbose=False)
            all_words[p_id] = set(word_list)
            all_unique_words.update(word_list)
        
        if not all_words:
            print("No word lists found")
            return None
        
        # Count how many participants use each word
        word_counts = {}
        for word in all_unique_words:
            word_counts[word] = sum(1 for p_id in all_words if word in all_words[p_id])
        
        # Sort words by number of participants (descending)
        sorted_words = sorted(word_counts.keys(), key=lambda w: (-word_counts[w], w))
        
        # Create presence matrix (words as rows, participants as columns)
        presence_matrix = np.zeros((len(sorted_words), len(participant_ids)))
        
        for i, word in enumerate(sorted_words):
            for j, p_id in enumerate(participant_ids):
                if p_id in all_words and word in all_words[p_id]:
                    presence_matrix[i, j] = 1
        
        # Create DataFrame for seaborn
        presence_df = pd.DataFrame(
            presence_matrix, 
            index=sorted_words, 
            columns=participant_ids
        )
        
        # Create figure
        fig_width = min(14, max(8, len(participant_ids) * 0.8))
        fig_height = min(20, max(8, len(sorted_words) * 0.2))
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create heatmap
        sns.heatmap(
            presence_df, 
            cmap='Blues', 
            cbar=False, 
            linewidths=0.5,
            linecolor='lightgray',
            ax=ax
        )
        
        # Add word counts on right
        word_participant_counts = np.sum(presence_matrix, axis=1)
        for i, count in enumerate(word_participant_counts):
            ax.text(
                len(participant_ids) + 0.2, i + 0.5, 
                f"{int(count)}/{len(participant_ids)}",
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=9,
                color='black'
            )
        
        # Add participant word counts on bottom
        participant_word_counts = np.sum(presence_matrix, axis=0)
        for j, count in enumerate(participant_word_counts):
            ax.text(
                j + 0.5, len(sorted_words) + 0.2, 
                int(count),
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=9,
                color='black'
            )
        
        # Add labels and title
        ax.set_title(f"Word Usage Matrix Across Participants")
        ax.set_xlabel("Participants")
        ax.set_ylabel("Words (sorted by number of participants)")
        
        # Add annotation for totals
        ax.text(
            -0.1, -0.05, 
            f"Total: {len(sorted_words)} words across {len(participant_ids)} participants",
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold'
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig and self.path_results is not None:
            save_path = os.path.join(
                self.path_results, 
                f'word_matrix.png'
            )
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
        
        
    def interactive_word_matrix(self):
        """
        Create interactive widget for word matrix visualization
        
        """

        
        # Create output widget
        output = widgets.Output()
        
        # Create participant selection widget
        all_participants = [f'sub-{i:02d}' for i in range(1, 11)]
        
        participant_selector = widgets.SelectMultiple(
            options=all_participants,
            value=all_participants,  # Default to all participants
            description='Participants:',
            disabled=False,
            layout=widgets.Layout(width='300px', height='200px')
        )
        
        # Create update button
        update_button = widgets.Button(
            description='Update Matrix',
            button_style='primary',
            tooltip='Click to update the visualization',
            icon='refresh'
        )
        
        # Define update function
        def update_matrix(b):
            with output:
                clear_output(wait=True)
                selected_participants = participant_selector.value
                if selected_participants:
                    self.visualize_word_matrix(participant_ids=selected_participants)
                else:
                    print("Please select at least one participant")
        
        # Connect button to update function
        update_button.on_click(update_matrix)
        
        # Initial visualization
        with output:
            self.visualize_word_matrix()
        
        # Create and return widget
        widget = widgets.VBox([
            widgets.Label("Select participants to include:"),
            participant_selector,
            update_button,
            output
        ])
        
        return widget