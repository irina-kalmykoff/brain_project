import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from debugger import DebugMixin
from scipy.fft import fft, fftfreq

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
        
    def visualize_word_analysis(self, participant_id, word_index=0, save_path=None):
        """
        Complete visualization for a specific word from a patient
        
        Args:
            participant_id: e.g., 'sub-p21'
            word_index: which word to visualize (default: 0 for first word)
            save_path: optional path to save figure
        """
        print(f"\n{'='*80}")
        print(f"WORD ANALYSIS: {participant_id}, Word #{word_index}")
        print(f"{'='*80}")
        
        # Get word segments
        word_result = self.pipeline.segment_data_by_words(participant_id)
        
        if word_index >= len(word_result['words']):
            self.log(f"Error: Only {len(word_result['words'])} words available")
            return
        
        word = word_result['words_list'][word_index]
        eeg_seg = word_result['eeg_segments'][word_index]
        spec_seg = word_result['spectrogram_segments'][word_index]
        
        self.log(f"Word: '{word}'")
        self.log(f"EEG shape: {eeg_seg.shape}")
        self.log(f"Spectrogram shape: {spec_seg.shape}")
        
        # Get phoneme boundaries
        boundary_result = self.detector.detect_boundaries(
            spectrogram=spec_seg,
            word=word,
            frameshift=0.01
        )
        
        expected_phonemes = self.phonetic_dict.extract_phonemes(word)
        self.log(f"Expected phonemes: {expected_phonemes}")
        self.log(f"Detected segments: {len(boundary_result['segments'])}")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(7, 2, hspace=0.4, wspace=0.3)
        
        # Calculate common time axes
        time_eeg = np.arange(eeg_seg.shape[0]) / 512
        time_spec = np.arange(spec_seg.shape[0]) * 0.01
        max_time = max(time_eeg[-1], time_spec[-1])
        
        # 1. ERP Time Series (multi-channel)
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot first few channels
        n_channels_to_plot = min(5, eeg_seg.shape[1])
        for ch in range(n_channels_to_plot):
            ax1.plot(time_eeg, eeg_seg[:, ch] + ch*50, label=f'Ch {ch}', alpha=0.7)
        
        # Mark phoneme boundaries on time axis
        for boundary in boundary_result['boundaries']:
            boundary_time = boundary * 0.01
            ax1.axvline(x=boundary_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax1.set_title(f"ERP Time Series - '{word}'")
        ax1.set_ylabel('Amplitude (μV, offset)')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, max_time])
        ax1.set_xlabel('')
        
        # 2. Spectrogram with Boundaries
        ax2 = fig.add_subplot(gs[1, :])
        ax2.imshow(spec_seg.T, aspect='auto', origin='lower', 
                   cmap='viridis', extent=[0, time_spec[-1], 0, spec_seg.shape[1]])
        for boundary in boundary_result['boundaries']:
            ax2.axvline(x=boundary*0.01, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax2.set_title('Spectrogram with Detected Phoneme Boundaries')
        ax2.set_ylabel('Frequency Bin')
        ax2.set_xlim([0, max_time])
        ax2.set_xlabel('')
        
        # 3. Power Spectrum (average across time)
        ax3 = fig.add_subplot(gs[2, 0])
        avg_power = np.mean(spec_seg, axis=0)
        freq_bins = np.arange(len(avg_power))
        ax3.plot(freq_bins, avg_power, color='blue')
        ax3.set_title('Power Spectrum (Time-Averaged)')
        ax3.set_xlabel('Frequency Bin')
        ax3.set_ylabel('Power')
        ax3.grid(True, alpha=0.3)
        
        # 4. Amplitude Envelope
        ax4 = fig.add_subplot(gs[2, 1])
        # Compute envelope from average across channels
        avg_eeg = np.mean(eeg_seg, axis=1)
        analytic_signal = signal.hilbert(avg_eeg)
        amplitude_envelope = np.abs(analytic_signal)
        
        ax4.plot(time_eeg, amplitude_envelope, color='green')
        for boundary in boundary_result['boundaries']:
            boundary_time = boundary * 0.01
            ax4.axvline(x=boundary_time, color='red', linestyle='--', alpha=0.5)
        ax4.set_title('Amplitude Envelope')
        ax4.set_ylabel('Amplitude')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, max_time])
        ax4.set_xlabel('')
        
        # 5. Static Spectrum (FFT of entire signal)
        ax5 = fig.add_subplot(gs[3, 0])
        fft_vals = np.abs(fft(avg_eeg))
        freqs = fftfreq(len(avg_eeg), 1/512)
        
        # Plot positive frequencies only
        pos_mask = freqs > 0
        ax5.plot(freqs[pos_mask], fft_vals[pos_mask], color='purple')
        ax5.set_title('Static Spectrum (FFT)')
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('Magnitude')
        ax5.set_xlim([0, 100])  # Focus on 0-100 Hz
        ax5.grid(True, alpha=0.3)
        
        # 6. Phase Angle
        ax6 = fig.add_subplot(gs[3, 1])
        phase = np.angle(analytic_signal)
        ax6.plot(time_eeg, phase, color='orange')
        for boundary in boundary_result['boundaries']:
            boundary_time = boundary * 0.01
            ax6.axvline(x=boundary_time, color='red', linestyle='--', alpha=0.5)
        ax6.set_title('Phase Angle')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Phase (radians)')
        ax6.grid(True, alpha=0.3)
        
        # 7. Enhanced Distances with Peaks
        ax7 = fig.add_subplot(gs[4, :])
        enhanced = boundary_result['enhanced_distances']
        ax7.plot(enhanced, label='Enhanced Distance', color='green', linewidth=2)
        
        # Mark detected boundaries
        for boundary in boundary_result['boundaries'][1:-1]:
            if 0 <= boundary-1 < len(enhanced):
                ax7.axvline(x=boundary-1, color='red', linestyle='--', alpha=0.7)
                ax7.plot(boundary-1, enhanced[boundary-1], 'ro', markersize=8)
        
        threshold = self.detector.peak_threshold * np.max(enhanced)
        ax7.axhline(y=threshold, color='orange', linestyle=':', 
                   label=f'Threshold ({threshold:.3f})', alpha=0.7)
        ax7.set_title('Enhanced Distances for Boundary Detection')
        ax7.set_xlabel('Frame')
        ax7.set_ylabel('Enhanced Distance')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Energy Contour (different from power spectrum)
        ax8 = fig.add_subplot(gs[5, :])
        # Energy per time frame (sum of squared spectrogram values)
        energy = np.sum(spec_seg**2, axis=1)
        time_spec = np.arange(len(energy)) * 0.01
        ax8.plot(time_spec, energy, color='purple', linewidth=2)
        for boundary in boundary_result['boundaries']:
            ax8.axvline(x=boundary*0.01, color='red', linestyle='--', alpha=0.5)
        ax8.set_title('Energy Contour (Frame Energy over Time)')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Energy')
        ax8.grid(True, alpha=0.3)
        
        # 9. Segment Lengths
        ax9 = fig.add_subplot(gs[6, :])
        segment_lengths = [seg.shape[0] for seg in boundary_result['segments']]
        colors = ['teal' if i < len(expected_phonemes or []) else 'orange' 
                 for i in range(len(segment_lengths))]
        ax9.bar(range(len(segment_lengths)), segment_lengths, color=colors)
        
        # Add phoneme labels if available
        if expected_phonemes:
            for i, ph in enumerate(expected_phonemes):
                if i < len(segment_lengths):
                    ax9.text(i, segment_lengths[i], ph, ha='center', va='bottom', fontsize=10)
        
        ax9.set_title('Detected Segment Lengths')
        ax9.set_xlabel('Segment Index')
        ax9.set_ylabel('Length (frames)')
        ax9.grid(True, alpha=0.3, axis='y')
        
        # Add info box
        info_text = (
            f"Word: '{word}'\n"
            f"Expected phonemes: {expected_phonemes or 'Unknown'} ({len(expected_phonemes or [])})\n"
            f"Detected segments: {len(boundary_result['segments'])}\n"
            f"Boundaries: {list(boundary_result['boundaries'])}\n"
            f"Match: {'✓' if len(boundary_result['segments']) == len(expected_phonemes or []) else '✗'}"
        )
        plt.figtext(0.02, 0.02, info_text, fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
        return fig, boundary_result
    
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
                word=word,
                frameshift=0.01
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
