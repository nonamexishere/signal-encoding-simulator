import numpy as np
from typing import Tuple


class AnalogToAnalogModulator:
    
    ALGORITHMS = ['AM', 'FM']
    
    def __init__(self, carrier_freq: float = 100.0, sample_rate: int = 10000):
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
    
    def modulate(self, t: np.ndarray, signal: np.ndarray, 
                 algorithm: str) -> Tuple[np.ndarray, np.ndarray]:
        if algorithm == 'AM':
            return self._am(t, signal)
        elif algorithm == 'FM':
            return self._fm(t, signal)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _am(self, t: np.ndarray, signal: np.ndarray, 
            modulation_index: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        sig_max = np.max(np.abs(signal))
        if sig_max > 0:
            normalized = signal / sig_max
        else:
            normalized = signal
        
        carrier = np.cos(2 * np.pi * self.carrier_freq * t)
        
        modulated = (1 + modulation_index * normalized) * carrier
        
        return t, modulated
    
    def _fm(self, t: np.ndarray, signal: np.ndarray,
            freq_deviation: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
        sig_max = np.max(np.abs(signal))
        if sig_max > 0:
            normalized = signal / sig_max
        else:
            normalized = signal
        
        dt = t[1] - t[0] if len(t) > 1 else 1.0 / self.sample_rate
        integral = np.cumsum(normalized) * dt
        
        phase = 2 * np.pi * self.carrier_freq * t + 2 * np.pi * freq_deviation * integral
        modulated = np.cos(phase)
        
        return t, modulated


class AnalogToAnalogDemodulator:
    
    def __init__(self, carrier_freq: float = 100.0, sample_rate: int = 10000):
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
    
    def demodulate(self, t: np.ndarray, signal: np.ndarray,
                   algorithm: str) -> Tuple[np.ndarray, np.ndarray]:
        if algorithm == 'AM':
            return self._demod_am(t, signal)
        elif algorithm == 'FM':
            return self._demod_fm(t, signal)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _demod_am(self, t: np.ndarray, signal: np.ndarray,
                  modulation_index: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        envelope = np.abs(signal)
        
        window_size = max(1, int(self.sample_rate / self.carrier_freq * 2))
        kernel = np.ones(window_size) / window_size
        
        padded = np.pad(envelope, (window_size//2, window_size//2), mode='edge')
        filtered = np.convolve(padded, kernel, mode='valid')[:len(envelope)]
        
        demodulated = (filtered - 1) / modulation_index
        
        return t, demodulated
    
    def _demod_fm(self, t: np.ndarray, signal: np.ndarray,
                  freq_deviation: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
        dt = t[1] - t[0] if len(t) > 1 else 1.0 / self.sample_rate
        derivative = np.gradient(signal, dt)
        
        envelope = np.abs(derivative)
        
        window_size = max(1, int(self.sample_rate / self.carrier_freq * 2))
        kernel = np.ones(window_size) / window_size
        padded = np.pad(envelope, (window_size//2, window_size//2), mode='edge')
        filtered = np.convolve(padded, kernel, mode='valid')[:len(envelope)]
        
        demodulated = filtered - np.mean(filtered)
        max_val = np.max(np.abs(demodulated))
        if max_val > 0:
            demodulated = demodulated / max_val
        
        return t, demodulated
