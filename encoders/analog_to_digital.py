import numpy as np
from typing import Tuple, List


class AnalogToDigitalConverter:
    ALGORITHMS = ['PCM', 'Delta Modulation']
    
    def __init__(self, sample_rate: int = 1000, quantization_bits: int = 8):
        self.sample_rate = sample_rate
        self.quantization_bits = quantization_bits
        self.quantization_levels = 2 ** quantization_bits
    
    def convert(self, t: np.ndarray, signal: np.ndarray, 
                algorithm: str) -> Tuple[np.ndarray, str, np.ndarray]:
        if algorithm == 'PCM':
            return self._pcm(t, signal)
        elif algorithm == 'Delta Modulation':
            return self._delta_modulation(t, signal)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _pcm(self, t: np.ndarray, signal: np.ndarray) -> Tuple[np.ndarray, str, np.ndarray]:
        
        sig_min, sig_max = np.min(signal), np.max(signal)
        if sig_max - sig_min > 0:
            normalized = (signal - sig_min) / (sig_max - sig_min)
        else:
            normalized = np.zeros_like(signal)
        
        sample_interval = max(1, len(signal) // self.sample_rate)
        sample_indices = np.arange(0, len(signal), sample_interval)
        sampled_t = t[sample_indices]
        sampled_signal = normalized[sample_indices]
        
        quantized_indices = np.floor(sampled_signal * (self.quantization_levels - 1)).astype(int)
        quantized_indices = np.clip(quantized_indices, 0, self.quantization_levels - 1)
        
        quantized_signal = quantized_indices / (self.quantization_levels - 1)
        quantized_signal = quantized_signal * (sig_max - sig_min) + sig_min
        
        bitstream = ''.join([format(q, f'0{self.quantization_bits}b') for q in quantized_indices])
        
        return sampled_t, bitstream, quantized_signal
    
    def _delta_modulation(self, t: np.ndarray, signal: np.ndarray,
                          step_size: float = 0.1) -> Tuple[np.ndarray, str, np.ndarray]:
        sig_min, sig_max = np.min(signal), np.max(signal)
        sig_range = sig_max - sig_min if sig_max - sig_min > 0 else 1
        normalized = (signal - sig_min) / sig_range
        
        sample_interval = max(1, len(signal) // self.sample_rate)
        sample_indices = np.arange(0, len(signal), sample_interval)
        sampled_t = t[sample_indices]
        sampled_signal = normalized[sample_indices]
        
        bits = []
        staircase = [0.5] 
        
        for i, sample in enumerate(sampled_signal):
            if sample > staircase[-1]:
                bits.append('1')
                staircase.append(min(1.0, staircase[-1] + step_size))
            else:
                bits.append('0')
                staircase.append(max(0.0, staircase[-1] - step_size))
        
        # Convert staircase back to original scale
        staircase = np.array(staircase[1:])  # Remove initial value
        staircase = staircase * sig_range + sig_min
        
        bitstream = ''.join(bits)
        return sampled_t, bitstream, staircase


class AnalogToDigitalDecoder:
    """Decoder for analog-to-digital conversion."""
    
    def __init__(self, sample_rate: int = 1000, quantization_bits: int = 8):
        self.sample_rate = sample_rate
        self.quantization_bits = quantization_bits
        self.quantization_levels = 2 ** quantization_bits
    
    def decode(self, bitstream: str, algorithm: str, 
               signal_min: float = -1.0, signal_max: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode digital signal back to analog approximation.
        
        Args:
            bitstream: Binary string
            algorithm: Original conversion algorithm
            signal_min, signal_max: Original signal range
            
        Returns:
            Tuple of (time_array, reconstructed_signal)
        """
        if algorithm == 'PCM':
            return self._decode_pcm(bitstream, signal_min, signal_max)
        elif algorithm == 'Delta Modulation':
            return self._decode_dm(bitstream, signal_min, signal_max)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _decode_pcm(self, bitstream: str, sig_min: float, 
                    sig_max: float) -> Tuple[np.ndarray, np.ndarray]:
        """Decode PCM bitstream."""
        n_bits = self.quantization_bits
        n_samples = len(bitstream) // n_bits
        
        samples = []
        for i in range(n_samples):
            bits = bitstream[i*n_bits:(i+1)*n_bits]
            if len(bits) == n_bits:
                value = int(bits, 2) / (self.quantization_levels - 1)
                samples.append(value * (sig_max - sig_min) + sig_min)
        
        t = np.linspace(0, n_samples / self.sample_rate, n_samples)
        return t, np.array(samples)
    
    def _decode_dm(self, bitstream: str, sig_min: float, 
                   sig_max: float, step_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Decode Delta Modulation bitstream."""
        sig_range = sig_max - sig_min
        
        staircase = [0.5]  # Start at midpoint (normalized)
        for bit in bitstream:
            if bit == '1':
                staircase.append(min(1.0, staircase[-1] + step_size))
            else:
                staircase.append(max(0.0, staircase[-1] - step_size))
        
        staircase = np.array(staircase[1:])
        reconstructed = staircase * sig_range + sig_min
        
        t = np.linspace(0, len(reconstructed) / self.sample_rate, len(reconstructed))
        return t, reconstructed
