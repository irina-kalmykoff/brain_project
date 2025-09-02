import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import welch
from scipy import signal
import mne

class PhonemeFeatureVisualizer:
    """
    Visualize different features of phoneme-related iEEG data.
    """
    
    def __init__(self, output_dir='./phoneme_visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def visualize_phoneme_features(self, phoneme_segments, phoneme_name, eeg_sr=1024, 
                               method='high_gamma', band=(70, 150)):
                                   
        
        # Create figure with 8 subplots instead of 6 - add pre/post filtering
        fig, axs = plt.subplots(4, 2, figsize=(15, 20))
        fig.suptitle(f"Phoneme '{phoneme_name}' - {method}", fontsize=16)
        
        # 1. Raw ERP (pre-filtering)
        self._plot_erp(axs[0, 0], phoneme_segments, eeg_sr)
        axs[0, 0].set_title("ERP Time Series (Raw)")
        
        # 2. Filtered ERP
        filtered_segments = []
        for segment in phoneme_segments:
            filtered = self._bandpass_filter(segment, eeg_sr, band[0], band[1])
            filtered_segments.append(filtered)
        self._plot_erp(axs[0, 1], filtered_segments, eeg_sr)
        axs[0, 1].set_title(f"ERP Time Series (Filtered {band[0]}-{band[1]} Hz)")
        
        # 3. Raw Power Spectrum
        self._plot_spectrum(axs[1, 0], phoneme_segments, eeg_sr)
        axs[1, 0].set_title("Power Spectrum (Raw)")
        
        # 4. Filtered Power Spectrum
        self._plot_spectrum(axs[1, 1], filtered_segments, eeg_sr)
        axs[1, 1].set_title(f"Power Spectrum (Filtered)")
        
        # Continue with other plots using filtered_segments...
        self._plot_amplitude(axs[2, 0], filtered_segments, eeg_sr)
        axs[2, 0].set_title("Amplitude Envelope")
        
        self._plot_phase(axs[2, 1], filtered_segments, eeg_sr)
        axs[2, 1].set_title("Phase Angle")
        
        self._plot_static_spectrum(axs[3, 0], phoneme_segments, eeg_sr, band)
        axs[3, 0].set_title("Static Spectrum (Band Focus)")
        
        self._plot_time_frequency(axs[3, 1], phoneme_segments[0], eeg_sr)
        axs[3, 1].set_title("Time-Frequency (Example)")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_dir, f"phoneme_{phoneme_name}_{method}.png"), dpi=300)
        plt.show()  # Add this to display
        plt.close()
        
    def _bandpass_filter(self, data, fs, low_freq, high_freq):
        """Apply bandpass filter to the data."""
        sos = signal.butter(4, [low_freq, high_freq], btype='bandpass', 
                           fs=fs, output='sos')
        return signal.sosfiltfilt(sos, data, axis=0)
    
    def _plot_erp(self, ax, segments, fs):
        """Plot ERP time series."""
        # Calculate average across segments and channels
        avg_segments = []
        for segment in segments:
            if segment.ndim > 1:
                # Average across channels
                avg_segments.append(np.mean(segment, axis=1))
            else:
                avg_segments.append(segment)
        
        # Standardize lengths by truncating to shortest
        min_length = min(len(seg) for seg in avg_segments)
        truncated = [seg[:min_length] for seg in avg_segments]
        
        # Calculate time axis
        time_axis = np.arange(min_length) / fs
        
        # Plot individual segments (faded)
        for i, segment in enumerate(truncated):
            ax.plot(time_axis, segment, alpha=0.2, color='gray')
        
        # Calculate and plot average
        average = np.mean(truncated, axis=0)
        ax.plot(time_axis, average, color='blue', linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _plot_amplitude(self, ax, segments, fs):
        """Plot amplitude envelope using Hilbert transform."""
        amp_segments = []
        for segment in segments:
            # Calculate Hilbert transform for each channel
            if segment.ndim > 1:
                # Process each channel
                channel_amps = []
                for ch in range(segment.shape[1]):
                    analytic = signal.hilbert(segment[:, ch])
                    channel_amps.append(np.abs(analytic))
                
                # Average across channels
                amp_segments.append(np.mean(channel_amps, axis=0))
            else:
                analytic = signal.hilbert(segment)
                amp_segments.append(np.abs(analytic))
        
        # Standardize lengths
        min_length = min(len(seg) for seg in amp_segments)
        truncated = [seg[:min_length] for seg in amp_segments]
        
        # Calculate time axis
        time_axis = np.arange(min_length) / fs
        
        # Plot individual segments (faded)
        for i, segment in enumerate(truncated):
            ax.plot(time_axis, segment, alpha=0.2, color='gray')
        
        # Calculate and plot average
        average = np.mean(truncated, axis=0)
        ax.plot(time_axis, average, color='red', linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude Envelope')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _plot_phase(self, ax, segments, fs):
        """Plot phase angle using Hilbert transform."""
        phase_segments = []
        for segment in segments:
            # Calculate Hilbert transform for each channel
            if segment.ndim > 1:
                # Process each channel
                channel_phases = []
                for ch in range(segment.shape[1]):
                    analytic = signal.hilbert(segment[:, ch])
                    channel_phases.append(np.angle(analytic))
                
                # Average across channels (be careful with circular mean)
                # Simple approximation: convert to complex, average, convert back
                complex_exp = np.mean([np.exp(1j * phase) for phase in channel_phases], axis=0)
                phase_segments.append(np.angle(complex_exp))
            else:
                analytic = signal.hilbert(segment)
                phase_segments.append(np.angle(analytic))
        
        # Standardize lengths
        min_length = min(len(seg) for seg in phase_segments)
        truncated = [seg[:min_length] for seg in phase_segments]
        
        # Calculate time axis
        time_axis = np.arange(min_length) / fs
        
        # Plot individual segments (faded)
        for i, segment in enumerate(truncated):
            ax.plot(time_axis, segment, alpha=0.2, color='gray')
        
        # For phase, we need to be careful with averaging due to circularity
        # Convert to complex numbers, average, then convert back to angle
        complex_exp = np.mean([np.exp(1j * phase) for phase in truncated], axis=0)
        average_phase = np.angle(complex_exp)
        
        ax.plot(time_axis, average_phase, color='green', linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Phase Angle (rad)')
        ax.set_ylim(-np.pi, np.pi)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _plot_spectrum(self, ax, segments, fs):
        """Plot frequency domain representation."""
        psd_segments = []
        
        for segment in segments:
            if segment.ndim > 1:
                # Process each channel
                channel_psds = []
                for ch in range(segment.shape[1]):
                    freqs, psd = welch(segment[:, ch], fs=fs, nperseg=min(256, len(segment)))
                    channel_psds.append(psd)
                
                # Average across channels
                psd_segments.append((freqs, np.mean(channel_psds, axis=0)))
            else:
                freqs, psd = welch(segment, fs=fs, nperseg=min(256, len(segment)))
                psd_segments.append((freqs, psd))
        
        # Plot individual segments (faded)
        for i, (freqs, psd) in enumerate(psd_segments):
            ax.semilogy(freqs, psd, alpha=0.2, color='gray')
        
        # Calculate and plot average
        # First, ensure all frequency axes are the same
        ref_freqs = psd_segments[0][0]
        aligned_psds = []
        
        for freqs, psd in psd_segments:
            if np.array_equal(freqs, ref_freqs):
                aligned_psds.append(psd)
            else:
                # Interpolate to match reference frequencies
                from scipy.interpolate import interp1d
                f = interp1d(freqs, psd, bounds_error=False, fill_value='extrapolate')
                aligned_psds.append(f(ref_freqs))
        
        average_psd = np.mean(aligned_psds, axis=0)
        ax.semilogy(ref_freqs, average_psd, color='purple', linewidth=2)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _plot_static_spectrum(self, ax, segments, fs, band):
        """Plot static power spectrum focused on the specified band."""
        band_low, band_high = band
        
        psd_segments = []
        for segment in segments:
            if segment.ndim > 1:
                # Process each channel
                channel_psds = []
                for ch in range(segment.shape[1]):
                    freqs, psd = welch(segment[:, ch], fs=fs, nperseg=min(256, len(segment)))
                    channel_psds.append(psd)
                
                # Average across channels
                psd_segments.append((freqs, np.mean(channel_psds, axis=0)))
            else:
                freqs, psd = welch(segment, fs=fs, nperseg=min(256, len(segment)))
                psd_segments.append((freqs, psd))
        
        # Calculate average PSD
        ref_freqs = psd_segments[0][0]
        aligned_psds = []
        
        for freqs, psd in psd_segments:
            if np.array_equal(freqs, ref_freqs):
                aligned_psds.append(psd)
            else:
                # Interpolate to match reference frequencies
                from scipy.interpolate import interp1d
                f = interp1d(freqs, psd, bounds_error=False, fill_value='extrapolate')
                aligned_psds.append(f(ref_freqs))
        
        average_psd = np.mean(aligned_psds, axis=0)
        
        # Focus on the band of interest
        band_mask = (ref_freqs >= band_low) & (ref_freqs <= band_high)
        band_freqs = ref_freqs[band_mask]
        band_psd = average_psd[band_mask]
        
        # Create bar plot
        ax.bar(band_freqs, band_psd, width=(band_freqs[1]-band_freqs[0]) if len(band_freqs) > 1 else 1)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_xlim(band_low, band_high)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _plot_time_frequency(self, ax, segment, fs):
        """Plot time-frequency representation using spectrogram."""
        if segment.ndim > 1:
            # Average across channels
            data = np.mean(segment, axis=1)
        else:
            data = segment
        
        # Calculate spectrogram
        f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=min(128, len(data)))
        
        # Plot spectrogram
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        plt.colorbar(im, ax=ax, label='Power/Frequency (dB/Hz)')
        
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
    
    def process_batches(self, train_data, method='high_gamma', band=(70, 150), eeg_sr=1024):
        """
        Process batches of training data to generate visualizations for each phoneme.
        
        Parameters:
        -----------
        train_data : dict
            Training data dictionary from your pipeline
        method : str
            Feature extraction method
        band : tuple
            Frequency band to analyze
        eeg_sr : int
            Sampling rate of the iEEG data
        """
        # Extract features and labels
        features = train_data.get('features', [])
        phoneme_labels = train_data.get('phoneme_labels', [])
        
        if not features or not phoneme_labels:
            print("Error: Missing features or phoneme labels in train data")
            return
        
        # Group features by phoneme
        phoneme_segments = {}
        for i, (feature, phoneme) in enumerate(zip(features, phoneme_labels)):
            if phoneme not in phoneme_segments:
                phoneme_segments[phoneme] = []
            
            # Skip empty or invalid features
            if feature is None or len(feature) == 0:
                continue
                
            phoneme_segments[phoneme].append(feature)
        
        # Process each phoneme
        for phoneme, segments in phoneme_segments.items():
            # Skip phonemes with too few examples
            if len(segments) < 3:
                print(f"Skipping phoneme '{phoneme}': only {len(segments)} examples")
                continue
                
            print(f"Visualizing features for phoneme '{phoneme}' ({len(segments)} examples)")
            self.visualize_phoneme_features(segments, phoneme, eeg_sr, method, band)
        
        print(f"Visualization complete. Results saved to: {self.output_dir}")