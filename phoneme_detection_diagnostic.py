import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
from scipy.signal import welch, hilbert
from scipy.fft import fft, fftfreq
from debugger import DebugMixin
from IPython.display import Audio, display
from scipy.ndimage import gaussian_filter1d



class Dutch30PhonemeDetectionDiagnostic(DebugMixin):
    """
    Diagnostic tool for Dutch30 phoneme detection with comprehensive visualizations
    """
    
    def __init__(self, pipeline):
        """
        Initialize with Dutch30Pipeline instance
        
        Args:
            pipeline: Dutch30Pipeline object with detector and phonetic_dict
        """
        # Initialize the DebugMixin
        super().__init__(class_name="Diagnostic Tool", debug_mode=False)
        self.pipeline = pipeline
        self.detector = pipeline.detector
        self.phonetic_dict = pipeline.phonetic_dict
        
    def visualize_word_analysis(self, participant_id, word_name, instance=0, save_path=None):
        """
        Comprehensive visualization of phoneme segmentation with EEG analysis.
        """

        
        # Get word segments
        if hasattr(self.pipeline, 'split_result'):
            word_segments = self.pipeline.split_result['word_segments_dict'][participant_id]
        else:
            word_segments = self.pipeline.segment_data_by_words(participant_id)
        
        # Check if word exists
        if word_name not in word_segments['words']:
            print(f"Word '{word_name}' not found for {participant_id}")
            print(f"Available words: {list(word_segments['words'].keys())[:20]}")
            return
        
        word_data = word_segments['words'][word_name]
        
        # Check instance
        if instance >= len(word_data['instances']):
            print(f"Instance {instance} not available. Word '{word_name}' has {len(word_data['instances'])} instances.")
            return
        
        # Get the instance
        inst = word_data['instances'][instance]
        
        eeg_segment = inst['eeg_segment']
        spec_segment = inst['spectrogram_segment']
        audio_segment = inst.get('audio_segment', None)
        
        self.log(f"Word: '{word_name}'")
        self.log(f"EEG shape: {eeg_segment.shape}")
        self.log(f"Spectrogram shape: {spec_segment.shape}")
        
        # Detect phoneme boundaries
        result = self.pipeline.detector.detect_boundaries(
            spectrogram=spec_segment,
            word=word_name,
            participant_id=participant_id,
            use_multifeature=True,         
            audio_segment=audio_segment, 
            audio_sr=self.pipeline.config.audio_sr 
        )
        
        boundaries = result['boundaries']
        boundary_times = result['boundary_times']
        expected_phonemes = self.pipeline.phonetic_dict.extract_phonemes(word_name)
        
        self.log(f"Expected phonemes: {expected_phonemes}")
        self.log(f"Detected segments: {len(boundaries)-1}")
        
        # Extract phoneme segments
        boundary_samples = np.round(boundary_times * self.pipeline.config.eeg_sr).astype(int)
        phoneme_eeg_segments = []
        phoneme_audio_segments = []
        
        for i in range(len(boundary_samples) - 1):
            start = boundary_samples[i]
            end = boundary_samples[i + 1]
            start = max(0, min(start, eeg_segment.shape[0]))
            end = max(0, min(end, eeg_segment.shape[0]))
            
            if start < end:
                phoneme_eeg_segments.append(eeg_segment[start:end])
                
                if audio_segment is not None:
                    audio_start = int(start * len(audio_segment) / eeg_segment.shape[0])
                    audio_end = int(end * len(audio_segment) / eeg_segment.shape[0])
                    phoneme_audio_segments.append(audio_segment[audio_start:audio_end])
        
        n_phonemes = len(phoneme_eeg_segments)
        
        # Compute band powers
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'low_gamma': (30, 70),
            'high_gamma': (70, 170)
        }
        
        band_powers = []
        for ph_eeg in phoneme_eeg_segments:
            n_channels = ph_eeg.shape[1]
            ph_band_powers = np.zeros((n_channels, len(bands)))
            
            for ch in range(n_channels):
                freqs, psd = welch(
                    ph_eeg[:, ch],
                    fs=self.pipeline.config.eeg_sr,
                    nperseg=min(256, ph_eeg.shape[0])
                )
                
                for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
                    band_mask = (freqs >= low) & (freqs < high)
                    if np.any(band_mask):
                        ph_band_powers[ch, band_idx] = np.mean(psd[band_mask])
            
            band_powers.append(ph_band_powers)
        
        # ===== PART 1: OVERVIEW PLOTS =====
        n_eeg_channels_to_show = min(4, eeg_segment.shape[1])
        
        fig_overview = plt.figure(figsize=(18, 22))
        gs = fig_overview.add_gridspec(6, 2, hspace=0.5, wspace=0.3)
        
        time_axis = np.arange(eeg_segment.shape[0]) / self.pipeline.config.eeg_sr
        
        # 1. Full word spectrogram with boundaries
        ax1 = fig_overview.add_subplot(gs[0, :])
        ax1.imshow(spec_segment.T, aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title(f"Word: '{word_name}' | Expected phonemes: {expected_phonemes}", 
                      fontweight='bold', fontsize=14)
        ax1.set_ylabel('Mel Frequency Bin')
        ax1.set_xlabel('Time (frames)')
        
        for i, boundary in enumerate(boundaries[1:-1], 1):
            ax1.axvline(boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
            label = expected_phonemes[i-1] if i-1 < len(expected_phonemes) else '?'
            ax1.text(boundary, spec_segment.shape[1] * 0.95, label,
                    ha='center', va='top', color='white', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        
        # 2. Energy contour + Enhanced distances
        ax2 = fig_overview.add_subplot(gs[1, :])
        energy = result.get('energy', np.sum(spec_segment**2, axis=1))
        enhanced = result['enhanced_distances']
        
        spec_time = np.linspace(0, time_axis[-1], len(energy))
        enhanced_time = np.linspace(0, time_axis[-1], len(enhanced))
        
        ax2_twin = ax2.twinx()
        ax2.plot(spec_time, energy, 'b-', linewidth=2, label='Energy')
        ax2_twin.plot(enhanced_time, enhanced, 'g-', linewidth=2, label='Enhanced Distance')
        
        for bt in boundary_times[1:-1]:
            ax2.axvline(bt, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax2.set_ylabel('Energy', color='b', fontweight='bold')
        ax2_twin.set_ylabel('Enhanced Distance', color='g', fontweight='bold')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('Energy Contour & Distance Metric', fontweight='bold')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(alpha=0.3)
        
        # 3. Raw EEG channels (stacked)
        ax_eeg = fig_overview.add_subplot(gs[2, :])
        
        colors_eeg = plt.cm.tab10(np.linspace(0, 1, n_eeg_channels_to_show))
        
        for ch_idx in range(n_eeg_channels_to_show):
            signal = eeg_segment[:, ch_idx]
            normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
            offset = ch_idx * 3
            ax_eeg.plot(time_axis, normalized + offset, linewidth=1, 
                       color=colors_eeg[ch_idx], label=f'Channel {ch_idx+1}')
        
        for bt in boundary_times[1:-1]:
            ax_eeg.axvline(bt, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax_eeg.set_xlabel('Time (s)')
        ax_eeg.set_ylabel('Normalized Amplitude (offset)', fontweight='bold')
        ax_eeg.set_title(f'Raw EEG - First {n_eeg_channels_to_show} Channels', fontweight='bold')
        ax_eeg.legend(loc='upper right', ncol=n_eeg_channels_to_show)
        ax_eeg.grid(alpha=0.3)
        ax_eeg.set_yticks([])
        
        # 4. Time-averaged power spectrum (across all channels)
        ax_psd = fig_overview.add_subplot(gs[3, :])
        
        all_psds = []
        for ch in range(eeg_segment.shape[1]):
            freqs, psd = welch(eeg_segment[:, ch], fs=self.pipeline.config.eeg_sr, nperseg=512)
            all_psds.append(psd)
        
        mean_psd = np.mean(all_psds, axis=0)
        std_psd = np.std(all_psds, axis=0)
        
        ax_psd.semilogy(freqs, mean_psd, 'b-', linewidth=2, label='Mean PSD')
        ax_psd.fill_between(freqs, mean_psd - std_psd, mean_psd + std_psd, 
                            alpha=0.3, color='blue', label='±1 SD')
        
        for band_name, (low, high) in bands.items():
            ax_psd.axvspan(low, high, alpha=0.1, label=band_name)
        
        ax_psd.set_xlabel('Frequency (Hz)')
        ax_psd.set_ylabel('Power Spectral Density')
        ax_psd.set_title('Time-Averaged Power Spectrum (across all channels)', fontweight='bold')
        ax_psd.legend(loc='upper right', fontsize=8, ncol=2)
        ax_psd.grid(alpha=0.3)
        ax_psd.set_xlim(0, 200)
        
        # 5. Amplitude envelope & Phase angle
        ax_env = fig_overview.add_subplot(gs[4, 0])
        
        analytic_signal = hilbert(eeg_segment[:, 0])
        amplitude_envelope = np.abs(analytic_signal)
        
        ax_env.plot(time_axis, eeg_segment[:, 0], 'b-', alpha=0.5, linewidth=0.5, label='Raw Signal')
        ax_env.plot(time_axis, amplitude_envelope, 'r-', linewidth=2, label='Amplitude Envelope')
        
        for bt in boundary_times[1:-1]:
            ax_env.axvline(bt, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax_env.set_xlabel('Time (s)')
        ax_env.set_ylabel('Amplitude')
        ax_env.set_title('Amplitude Envelope (Channel 1)', fontweight='bold')
        ax_env.legend()
        ax_env.grid(alpha=0.3)
        
        ax_phase = fig_overview.add_subplot(gs[4, 1])
        
        instantaneous_phase = np.angle(analytic_signal)
        
        ax_phase.plot(time_axis, instantaneous_phase, 'g-', linewidth=1)
        
        for bt in boundary_times[1:-1]:
            ax_phase.axvline(bt, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax_phase.set_xlabel('Time (s)')
        ax_phase.set_ylabel('Phase (radians)')
        ax_phase.set_title('Instantaneous Phase (Channel 1)', fontweight='bold')
        ax_phase.grid(alpha=0.3)
        ax_phase.set_ylim(-np.pi, np.pi)
        
        # 6. FFT spectrum (averaged across all channels) & Band power heatmap
        ax_fft = fig_overview.add_subplot(gs[5, 0])
        
        # Compute FFT for all channels and average
        all_ffts = []
        n_samples = eeg_segment.shape[0]
        
        for ch in range(eeg_segment.shape[1]):
            yf = fft(eeg_segment[:, ch])
            all_ffts.append(np.abs(yf))
        
        mean_fft = np.mean(all_ffts, axis=0)
        xf = fftfreq(n_samples, 1 / self.pipeline.config.eeg_sr)
        
        # Only positive frequencies
        positive_freq_idx = xf > 0
        
        ax_fft.semilogy(xf[positive_freq_idx], mean_fft[positive_freq_idx], 'b-', linewidth=1)
        ax_fft.set_xlabel('Frequency (Hz)')
        ax_fft.set_ylabel('Magnitude')
        ax_fft.set_title('FFT Spectrum (averaged across all channels)', fontweight='bold')
        ax_fft.grid(alpha=0.3)
        ax_fft.set_xlim(0, 200)
        
        ax_bp = fig_overview.add_subplot(gs[5, 1])
        
        band_power_summary = np.array([np.mean(bp, axis=0) for bp in band_powers])
        
        im = ax_bp.imshow(band_power_summary.T, aspect='auto', cmap='hot', origin='lower')
        ax_bp.set_yticks(range(len(bands)))
        ax_bp.set_yticklabels(bands.keys())
        ax_bp.set_xticks(range(n_phonemes))
        phoneme_labels = [expected_phonemes[i] if i < len(expected_phonemes) else '?' 
                          for i in range(n_phonemes)]
        ax_bp.set_xticklabels(phoneme_labels)
        ax_bp.set_xlabel('Phoneme')
        ax_bp.set_ylabel('Frequency Band')
        ax_bp.set_title('Average Band Power per Phoneme', fontweight='bold')
        plt.colorbar(im, ax=ax_bp, label='Power')
        
        plt.tight_layout()
        
        if save_path:
            fig_overview.savefig(save_path.replace('.png', '_overview.png'), dpi=150, bbox_inches='tight')
        
        plt.show()
        
        # ===== PART 2: INDIVIDUAL PHONEME PLOTS WITH AUDIO =====
        self.log("\n" + "="*70)
        self.log("INDIVIDUAL PHONEME ANALYSIS")
        self.log("="*70)
        
        for ph_idx in range(n_phonemes):
            label = expected_phonemes[ph_idx] if ph_idx < len(expected_phonemes) else '?'
            
            self.log(f"\n{'='*70}")
            self.log(f"PHONEME #{ph_idx+1}: '{label}'")
            self.log(f"Duration: {boundary_times[ph_idx+1] - boundary_times[ph_idx]:.3f}s")
            self.log(f"EEG samples: {phoneme_eeg_segments[ph_idx].shape[0]}")
            self.log(f"{'='*70}")
            
            # Create figure for this phoneme
            fig_ph = plt.figure(figsize=(16, 5))
            gs_ph = fig_ph.add_gridspec(1, 2, wspace=0.3)
            
            # Spectrogram
            ax_spec = fig_ph.add_subplot(gs_ph[0, 0])
            ph_start = boundaries[ph_idx]
            ph_end = boundaries[ph_idx + 1]
            ph_spec = spec_segment[ph_start:ph_end]
            
            ax_spec.imshow(ph_spec.T, aspect='auto', origin='lower', cmap='viridis')
            ax_spec.set_title(f"Phoneme #{ph_idx+1}: '{label}' [{boundary_times[ph_idx]:.2f}s - {boundary_times[ph_idx+1]:.2f}s]",
                             fontweight='bold', fontsize=12)
            ax_spec.set_ylabel('Mel Bin')
            ax_spec.set_xlabel('Time (frames)')
            
            # Band powers
            ax_bands = fig_ph.add_subplot(gs_ph[0, 1])
            
            bp_data = [band_powers[ph_idx][:, i] for i in range(len(bands))]
            bp = ax_bands.boxplot(bp_data, labels=list(bands.keys()), patch_artist=True)
            
            colors_bp = plt.cm.viridis(np.linspace(0, 1, len(bands)))
            for patch, color in zip(bp['boxes'], colors_bp):
                patch.set_facecolor(color)
            
            ax_bands.set_ylabel('Power')
            ax_bands.set_title(f"Band Power Distribution ({phoneme_eeg_segments[ph_idx].shape[1]} channels)",
                              fontweight='bold', fontsize=12)
            ax_bands.grid(axis='y', alpha=0.3)
            ax_bands.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                fig_ph.savefig(save_path.replace('.png', f'_phoneme_{ph_idx+1}.png'), dpi=150, bbox_inches='tight')
            
            plt.show()
            
            # Audio immediately after this phoneme's plots
            if ph_idx < len(phoneme_audio_segments):
                audio_seg = phoneme_audio_segments[ph_idx]
                duration = len(audio_seg) / self.pipeline.config.audio_sr
                self.log(f"Audio duration: {duration:.3f}s")
                display(Audio(audio_seg, rate=int(self.pipeline.config.audio_sr)))
            
            # Band power statistics
            self.log(f"\nBand powers (mean ± std across channels):")
            for band_idx, band_name in enumerate(bands.keys()):
                mean_power = np.mean(band_powers[ph_idx][:, band_idx])
                std_power = np.std(band_powers[ph_idx][:, band_idx])
                self.log(f"  {band_name:12s}: {mean_power:10.2e} ± {std_power:10.2e}")
        
        self.log("\n" + "="*70)
        self.log("ANALYSIS COMPLETE")
        self.log("="*70)

    
    def batch_diagnostic(self, participant_id, num_samples=5):
        """
        Quick diagnostic for multiple words from one patient
        """
        print(f"\n{'='*80}")
        print(f"BATCH DIAGNOSTIC: {participant_id}")
        print(f"{'='*80}")
        
        word_result = self.pipeline.segment_data_by_words(participant_id)
        
        words_list = word_result.get('words_list', [])
        specs_list = word_result.get('spectrogram_segments', [])
        issues_found = []
        
        for i in range(min(num_samples, len(word_result['words']))):
            word = words_list[i]
            spec = specs_list[i]
            
            boundary_result = self.detector.detect_boundaries(
                spectrogram=spec,
                word=word
            )
            
            expected = self.phonetic_dict.extract_phonemes(word)
            detected = len(boundary_result['segments'])
            
            self.log(f"\nWord {i}: '{word}'")
            self.log(f"  Expected: {expected} ({len(expected or [])} phonemes)")
            self.log(f"  Detected: {detected} segments")
            
            if expected and detected != len(expected):
                issue = f"MISMATCH: {word} - expected {len(expected)}, got {detected}"
                self.log(f" {issue}")
                issues_found.append(issue)
            elif not expected:
                self.log(f" Word not in dictionary")
                issues_found.append(f"Unknown word: {word}")
            else:
                self.log(f"Match")
        
        print(f"\n{'='*80}")
        print(f"Total issues: {len(issues_found)}")
        
        return issues_found
        
        
    def visualize_multifeature_analysis(self, participant_id, word_index=0):
        """Show contribution of each feature"""
        
        # Get word
        word_segments = self.pipeline.split_result['word_segments_dict'][participant_id]
        word = list(word_segments['words'].keys())[word_index]
        instance = word_segments['words'][word]['instances'][0]
        spec = instance['spectrogram_segment']
        audio = instance.get('audio_segment', None) 
        
        # Detect with multi-feature
        result = self.pipeline.detector.detect_boundaries(
            spec, 
            word=word, 
            use_multifeature=True,
            audio_segment=audio,               
            audio_sr=self.pipeline.config.audio_sr
        )
        
        feature_dict = result['feature_dict']
        boundaries = result['boundaries']
        
        # Plot each feature
        fig, axes = plt.subplots(len(feature_dict) + 1, 1, figsize=(14, 3*len(feature_dict)))
        
        # Spectrogram with boundaries
        axes[0].imshow(spec.T, aspect='auto', origin='lower', cmap='viridis')
        for b in boundaries[1:-1]:
            axes[0].axvline(b, color='red', linestyle='--', linewidth=2)
        axes[0].set_title(f"Word: '{word}'", fontweight='bold')
        
        # Individual features
        for idx, (name, values) in enumerate(feature_dict.items(), 1):
            axes[idx].plot(values, linewidth=2)
            axes[idx].set_title(f"{name.replace('_', ' ').title()}", fontweight='bold')
            axes[idx].set_ylabel('Normalized Score')
            axes[idx].grid(alpha=0.3)
            for b in boundaries[1:-1]:
                axes[idx].axvline(b, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

    def visualize_rms_boundaries(self, participant_id, word_name, instance=0, save_path=None):
        """Visualize RMS-based phoneme boundary detection"""
        
        # Get word segments
        if (hasattr(self.pipeline, 'split_result') and 
            'word_segments_dict' in self.pipeline.split_result and
            participant_id in self.pipeline.split_result['word_segments_dict']):
            word_segments = self.pipeline.split_result['word_segments_dict'][participant_id]
        else:
            word_segments = self.pipeline.segment_data_by_words(participant_id)
        
        # Check if word exists
        if word_name not in word_segments['words']:
            print(f"Word '{word_name}' not found for {participant_id}")
            return
        
        word_data = word_segments['words'][word_name]
        
        # Check instance
        if instance >= len(word_data['instances']):
            print(f"Instance {instance} not available. Word '{word_name}' has {len(word_data['instances'])} instances.")
            return
        
        # Get the instance
        inst = word_data['instances'][instance]
        
        spec = inst['spectrogram_segment']
        audio = inst.get('audio_segment', None)
        
        if audio is None:
            self.log("No audio available for this word")
            return
        
        # Detect boundaries with RMS
        result = self.pipeline.detector.detect_boundaries(
            spec, 
            word=word_name, 
            use_multifeature=False,  # Turn off multifeature when using RMS
            use_rms_boundaries=True,
            audio_segment=audio,
            audio_sr=self.pipeline.config.audio_sr
        )
        
        boundaries = result['boundaries']
        expected_phonemes = self.pipeline.phonetic_dict.extract_phonemes(word_name)
        
        # Compute RMS for visualization
        sr = self.pipeline.config.audio_sr
        hop_length = int(0.005 * sr)
        frame_length = int(0.020 * sr)
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        rms_smoothed = gaussian_filter1d(rms, sigma=2)
        rms_change = np.abs(np.gradient(rms_smoothed))
        rms_change_smoothed = gaussian_filter1d(rms_change, sigma=1.5)
        
        rms_time = np.arange(len(rms)) * hop_length / sr
        audio_time = np.arange(len(audio)) / sr
        
        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        
        # 1. Raw audio
        axes[0].plot(audio_time, audio, linewidth=0.5, alpha=0.7)
        axes[0].set_title(f"Word: '{word_name}' | Expected: {expected_phonemes} | Detected: {len(boundaries)-1} segments", 
                         fontweight='bold', fontsize=14)
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(alpha=0.3)
        
        boundary_times = boundaries * self.pipeline.config.frameshift
        for bt in boundary_times[1:-1]:
            axes[0].axvline(bt, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # 2. RMS envelope
        axes[1].plot(rms_time, rms_smoothed, linewidth=2, color='blue')
        axes[1].fill_between(rms_time, 0, rms_smoothed, alpha=0.3, color='blue')
        axes[1].set_title('RMS Envelope (smoothed)', fontweight='bold')
        axes[1].set_ylabel('RMS Power')
        axes[1].grid(alpha=0.3)
        
        for bt in boundary_times[1:-1]:
            axes[1].axvline(bt, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # 3. RMS change
        axes[2].plot(rms_time, rms_change_smoothed, linewidth=2, color='darkred')
        axes[2].fill_between(rms_time, 0, rms_change_smoothed, alpha=0.3, color='darkred')
        axes[2].set_title('RMS Change (Boundary Detection Signal)', fontweight='bold')
        axes[2].set_ylabel('RMS Change')
        axes[2].grid(alpha=0.3)
        
        median_val = np.median(rms_change_smoothed)
        mad = np.median(np.abs(rms_change_smoothed - median_val))
        threshold = median_val + 1.2 * mad
        axes[2].axhline(threshold, color='orange', linestyle=':', linewidth=2, label=f'Threshold: {threshold:.4f}')
        axes[2].legend()
        
        for bt in boundary_times[1:-1]:
            axes[2].axvline(bt, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # 4. Spectrogram
        axes[3].imshow(spec.T, aspect='auto', origin='lower', cmap='viridis')
        axes[3].set_title('Spectrogram with Detected Boundaries', fontweight='bold')
        axes[3].set_ylabel('Mel Bin')
        axes[3].set_xlabel('Time (frames)')
        
        for i, boundary in enumerate(boundaries[1:-1], 1):
            axes[3].axvline(boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
            label = expected_phonemes[i-1] if i-1 < len(expected_phonemes) else '?'
            axes[3].text(boundary, spec.shape[1] * 0.95, label,
                        ha='center', va='top', color='white', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
        # Summary
        self.log(f"\n{'='*70}")
        self.log(f"RMS-BASED BOUNDARY DETECTION: '{word_name}'")
        self.log(f"{'='*70}")
        self.log(f"Expected: {expected_phonemes} ({len(expected_phonemes)} phonemes)")
        self.log(f"Detected: {len(boundaries)-1} segments")
        self.log(f"Match: {'✓ PERFECT' if len(boundaries)-1 == len(expected_phonemes) else '✗ MISMATCH'}")