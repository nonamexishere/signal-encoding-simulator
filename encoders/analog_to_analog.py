"""
Analog-to-Analog Modulation Techniques
Based on William Stallings' Signal Encoding Techniques

Implements: AM (Amplitude Modulation), FM (Frequency Modulation)
"""
import numpy as np
from typing import Tuple


class AnalogToAnalogModulator:
    """Analog-to-Analog modulation with AM and FM."""
    
    ALGORITHMS = ['AM', 'FM']
    
    def __init__(self, carrier_freq: float = 100.0, sample_rate: int = 10000):
        """
        Initialize modulator.
        
        Args:
            carrier_freq: Carrier frequency in Hz
            sample_rate: Samples per second
        """
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
    
    def modulate(self, t: np.ndarray, signal: np.ndarray, 
                 algorithm: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Modulate analog signal using specified algorithm.
        
        Args:
            t: Time array
            signal: Message signal array
            algorithm: Modulation algorithm name
            
        Returns:
            Tuple of (time_array, modulated_signal)
        """
        if algorithm == 'AM':
            return self._am(t, signal)
        elif algorithm == 'FM':
            return self._fm(t, signal)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _am(self, t: np.ndarray, signal: np.ndarray, 
            modulation_index: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        AM (Amplitude Modulation)
        
        s(t) = [1 + m * x(t)] * cos(2πfc*t)
        
        where:
        - m is the modulation index (0 < m <= 1)
        - x(t) is the normalized message signal
        - fc is the carrier frequency
        """
        # Normalize message signal to [-1, 1]
        sig_max = np.max(np.abs(signal))
        if sig_max > 0:
            normalized = signal / sig_max
        else:
            normalized = signal
        
        # Generate carrier
        carrier = np.cos(2 * np.pi * self.carrier_freq * t)
        
        # AM modulation: s(t) = [1 + m*x(t)] * cos(2πfc*t)
        modulated = (1 + modulation_index * normalized) * carrier
        
        return t, modulated
    
    def _fm(self, t: np.ndarray, signal: np.ndarray,
            freq_deviation: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        FM (Frequency Modulation)
        
        s(t) = cos(2π[fc + kf*x(t)]*t)
        
        Or more precisely:
        s(t) = cos(2π*fc*t + 2π*kf*∫x(τ)dτ)
        
        where:
        - kf is the frequency deviation constant
        - x(t) is the message signal
        """
        # Normalize message signal
        sig_max = np.max(np.abs(signal))
        if sig_max > 0:
            normalized = signal / sig_max
        else:
            normalized = signal
        
        # Calculate instantaneous phase
        # Phase = 2π*fc*t + 2π*kf*∫x(τ)dτ
        dt = t[1] - t[0] if len(t) > 1 else 1.0 / self.sample_rate
        integral = np.cumsum(normalized) * dt
        
        phase = 2 * np.pi * self.carrier_freq * t + 2 * np.pi * freq_deviation * integral
        modulated = np.cos(phase)
        
        return t, modulated


class AnalogToAnalogDemodulator:
    """Demodulator for analog-to-analog signals."""
    
    def __init__(self, carrier_freq: float = 100.0, sample_rate: int = 10000):
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
    
    def demodulate(self, t: np.ndarray, signal: np.ndarray,
                   algorithm: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Demodulate signal back to message signal.
        
        Args:
            t: Time array
            signal: Modulated signal
            algorithm: Original modulation algorithm
            
        Returns:
            Tuple of (time_array, demodulated_signal)
        """
        if algorithm == 'AM':
            return self._demod_am(t, signal)
        elif algorithm == 'FM':
            return self._demod_fm(t, signal)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _demod_am(self, t: np.ndarray, signal: np.ndarray,
                  modulation_index: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Demodulate AM using envelope detection.
        
        Simple approach: take absolute value and low-pass filter
        """
        # Envelope detection (simplified)
        envelope = np.abs(signal)
        
        # Simple moving average filter (low-pass)
        window_size = max(1, int(self.sample_rate / self.carrier_freq * 2))
        kernel = np.ones(window_size) / window_size
        
        # Pad for convolution
        padded = np.pad(envelope, (window_size//2, window_size//2), mode='edge')
        filtered = np.convolve(padded, kernel, mode='valid')[:len(envelope)]
        
        # Remove DC offset and normalize
        demodulated = (filtered - 1) / modulation_index
        
        return t, demodulated
    
    def _demod_fm(self, t: np.ndarray, signal: np.ndarray,
                  freq_deviation: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Demodulate FM using differentiation and envelope detection.
        
        The instantaneous frequency is the derivative of the phase.
        """
        # Differentiate to get frequency information
        dt = t[1] - t[0] if len(t) > 1 else 1.0 / self.sample_rate
        derivative = np.gradient(signal, dt)
        
        # Envelope of derivative
        envelope = np.abs(derivative)
        
        # Low-pass filter
        window_size = max(1, int(self.sample_rate / self.carrier_freq * 2))
        kernel = np.ones(window_size) / window_size
        padded = np.pad(envelope, (window_size//2, window_size//2), mode='edge')
        filtered = np.convolve(padded, kernel, mode='valid')[:len(envelope)]
        
        # Remove mean and normalize
        demodulated = filtered - np.mean(filtered)
        max_val = np.max(np.abs(demodulated))
        if max_val > 0:
            demodulated = demodulated / max_val
        
        return t, demodulated
