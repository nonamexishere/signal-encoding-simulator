"""
Signal Visualization using Matplotlib
Provides plotting functions for all signal types
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Tuple, Optional, List


class SignalPlotter:
    """Signal visualization with matplotlib."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6), style: str = 'default'):
        """
        Initialize plotter.
        
        Args:
            figsize: Figure size (width, height)
            style: Matplotlib style
        """
        self.figsize = figsize
        self.style = style
        plt.style.use(style)
    
    def plot_digital_signal(self, t: np.ndarray, signal: np.ndarray,
                           title: str = "Digital Signal",
                           bits: Optional[str] = None) -> Figure:
        """
        Plot digital signal as a step waveform.
        
        Args:
            t: Time array
            signal: Signal amplitude array
            title: Plot title
            bits: Original binary string (for annotation)
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Step plot for digital signals
        ax.step(t, signal, where='post', linewidth=2, color='#2563eb')
        ax.fill_between(t, signal, step='post', alpha=0.3, color='#3b82f6')
        
        # Styling
        ax.set_xlabel('Time (bit periods)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        # Set y-axis limits with padding
        y_min, y_max = np.min(signal) - 0.3, np.max(signal) + 0.3
        ax.set_ylim(y_min, y_max)
        
        # Add bit annotations if provided
        if bits:
            n_bits = len(bits)
            bit_duration = t[-1] / n_bits if len(t) > 0 and n_bits > 0 else 1
            for i, bit in enumerate(bits):
                ax.annotate(bit, xy=(i * bit_duration + bit_duration/2, y_max - 0.1),
                           ha='center', fontsize=10, fontweight='bold', color='#dc2626')
        
        plt.tight_layout()
        return fig
    
    def plot_analog_signal(self, t: np.ndarray, signal: np.ndarray,
                          title: str = "Analog Signal",
                          color: str = '#2563eb') -> Figure:
        """
        Plot analog signal as a continuous waveform.
        
        Args:
            t: Time array
            signal: Signal amplitude array
            title: Plot title
            color: Line color
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(t, signal, linewidth=1.5, color=color)
        
        # Styling
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_comparison(self, t1: np.ndarray, signal1: np.ndarray,
                       t2: np.ndarray, signal2: np.ndarray,
                       title1: str = "Input Signal",
                       title2: str = "Output Signal",
                       signal1_type: str = 'analog',
                       signal2_type: str = 'digital') -> Figure:
        """
        Plot two signals side-by-side for comparison.
        
        Args:
            t1, signal1: First signal
            t2, signal2: Second signal
            title1, title2: Subplot titles
            signal1_type, signal2_type: 'analog' or 'digital'
            
        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5))
        
        # First signal
        if signal1_type == 'digital':
            ax1.step(t1, signal1, where='post', linewidth=2, color='#2563eb')
            ax1.fill_between(t1, signal1, step='post', alpha=0.3, color='#3b82f6')
        else:
            ax1.plot(t1, signal1, linewidth=1.5, color='#2563eb')
        
        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Amplitude', fontsize=11)
        ax1.set_title(title1, fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        # Second signal
        if signal2_type == 'digital':
            ax2.step(t2, signal2, where='post', linewidth=2, color='#16a34a')
            ax2.fill_between(t2, signal2, step='post', alpha=0.3, color='#22c55e')
        else:
            ax2.plot(t2, signal2, linewidth=1.5, color='#16a34a')
        
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel('Amplitude', fontsize=11)
        ax2.set_title(title2, fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_encoding_process(self, bits: str, t: np.ndarray, 
                              encoded: np.ndarray, algorithm: str) -> Figure:
        """
        Plot the encoding process with bit annotations.
        
        Args:
            bits: Binary input string
            t: Time array
            encoded: Encoded signal
            algorithm: Name of encoding algorithm
            
        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.3))
        
        # Input bits visualization
        n_bits = len(bits)
        bit_t = np.arange(n_bits + 1)
        bit_signal = [int(b) for b in bits] + [int(bits[-1])]
        
        ax1.step(bit_t, bit_signal, where='post', linewidth=2, color='#7c3aed')
        ax1.fill_between(bit_t, bit_signal, step='post', alpha=0.3, color='#8b5cf6')
        ax1.set_ylabel('Bit Value', fontsize=11)
        ax1.set_title('Input Binary Data', fontsize=13, fontweight='bold')
        ax1.set_yticks([0, 1])
        ax1.grid(True, alpha=0.3)
        
        # Add bit labels
        for i, bit in enumerate(bits):
            ax1.annotate(bit, xy=(i + 0.5, 0.5), ha='center', va='center',
                        fontsize=12, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round', facecolor='#7c3aed', alpha=0.8))
        
        # Encoded signal
        ax2.step(t, encoded, where='post', linewidth=2, color='#2563eb')
        ax2.fill_between(t, encoded, step='post', alpha=0.3, color='#3b82f6')
        ax2.set_xlabel('Time (bit periods)', fontsize=11)
        ax2.set_ylabel('Amplitude', fontsize=11)
        ax2.set_title(f'Encoded Signal ({algorithm})', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_modulation_process(self, bits: str, t: np.ndarray,
                                modulated: np.ndarray, algorithm: str) -> Figure:
        """
        Plot the modulation process for digital-to-analog.
        
        Args:
            bits: Binary input string
            t: Time array
            modulated: Modulated signal
            algorithm: Name of modulation algorithm
            
        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.3))
        
        # Input bits visualization
        n_bits = len(bits)
        bit_t = np.linspace(0, t[-1], n_bits + 1)
        bit_signal = [int(b) for b in bits] + [int(bits[-1])]
        
        ax1.step(bit_t, bit_signal, where='post', linewidth=2, color='#7c3aed')
        ax1.fill_between(bit_t, bit_signal, step='post', alpha=0.3, color='#8b5cf6')
        ax1.set_ylabel('Bit Value', fontsize=11)
        ax1.set_title('Input Binary Data', fontsize=13, fontweight='bold')
        ax1.set_yticks([0, 1])
        ax1.grid(True, alpha=0.3)
        
        # Modulated signal
        ax2.plot(t, modulated, linewidth=1, color='#2563eb')
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Amplitude', fontsize=11)
        ax2.set_title(f'Modulated Signal ({algorithm})', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        # Add vertical lines for bit boundaries
        bit_duration = t[-1] / n_bits
        for i in range(n_bits + 1):
            ax2.axvline(x=i * bit_duration, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_pcm_process(self, t_orig: np.ndarray, original: np.ndarray,
                        t_sampled: np.ndarray, quantized: np.ndarray,
                        bitstream: str) -> Figure:
        """
        Plot PCM encoding process with sampling and quantization.
        
        Args:
            t_orig: Original signal time array
            original: Original analog signal
            t_sampled: Sampled time points
            quantized: Quantized signal values
            bitstream: Resulting binary string
            
        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 2))
        
        # Original signal
        ax1.plot(t_orig, original, linewidth=1.5, color='#2563eb', label='Original')
        ax1.scatter(t_sampled, quantized, color='#dc2626', s=50, zorder=5, label='Samples')
        ax1.set_ylabel('Amplitude', fontsize=11)
        ax1.set_title('Sampling (Nyquist Theorem: fs ≥ 2fmax)', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Quantized signal
        ax2.plot(t_orig, original, linewidth=1, color='#2563eb', alpha=0.5, label='Original')
        ax2.stem(t_sampled, quantized, linefmt='g-', markerfmt='go', basefmt='k-', label='Quantized')
        ax2.set_ylabel('Amplitude', fontsize=11)
        ax2.set_title('Quantization', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Bitstream visualization (first 64 bits)
        display_bits = bitstream[:64]
        n_bits = len(display_bits)
        bit_signal = [int(b) for b in display_bits]
        
        ax3.bar(range(n_bits), bit_signal, width=0.8, color='#16a34a', alpha=0.7)
        ax3.set_ylabel('Bit Value', fontsize=11)
        ax3.set_xlabel(f'Bit Index (showing first {n_bits} of {len(bitstream)} bits)', fontsize=11)
        ax3.set_title('PCM Encoded Bitstream', fontsize=13, fontweight='bold')
        ax3.set_yticks([0, 1])
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_am_fm_process(self, t: np.ndarray, message: np.ndarray,
                          modulated: np.ndarray, algorithm: str) -> Figure:
        """
        Plot AM/FM modulation showing message and modulated signals.
        
        Args:
            t: Time array
            message: Original message signal
            modulated: Modulated signal
            algorithm: 'AM' or 'FM'
            
        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.3))
        
        # Message signal
        ax1.plot(t, message, linewidth=1.5, color='#7c3aed')
        ax1.set_ylabel('Amplitude', fontsize=11)
        ax1.set_title('Message Signal (Baseband)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        # Modulated signal
        ax2.plot(t, modulated, linewidth=0.5, color='#2563eb')
        
        if algorithm == 'AM':
            # Show envelope
            envelope_upper = 1 + 0.5 * (message / np.max(np.abs(message)) if np.max(np.abs(message)) > 0 else message)
            envelope_lower = -(1 + 0.5 * (message / np.max(np.abs(message)) if np.max(np.abs(message)) > 0 else message))
            ax2.plot(t, envelope_upper, 'r--', linewidth=1, alpha=0.7, label='Envelope')
            ax2.plot(t, envelope_lower, 'r--', linewidth=1, alpha=0.7)
        
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Amplitude', fontsize=11)
        ax2.set_title(f'{algorithm} Modulated Signal', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        if algorithm == 'AM':
            ax2.legend()
        
        plt.tight_layout()
        return fig
