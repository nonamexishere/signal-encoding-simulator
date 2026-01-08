"""
Version A: Original Implementation
This is the baseline implementation without optimizations.
"""
import numpy as np
from typing import List, Tuple


def nrz_l_encode_v1(bits: List[int], samples_per_bit: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    NRZ-L encoding - Original version using loops.
    0 = High voltage (+1), 1 = Low voltage (-1)
    """
    levels = []
    for bit in bits:
        level = -1 if bit == 1 else 1
        levels.extend([level, level])
    
    total_samples = len(levels) * (samples_per_bit // 2)
    t = np.linspace(0, len(bits), total_samples)
    signal = np.zeros(total_samples)
    
    samples_per_half = samples_per_bit // 2
    for i, level in enumerate(levels):
        start = i * samples_per_half
        end = (i + 1) * samples_per_half
        signal[start:end] = level
    
    return t, signal


def manchester_encode_v1(bits: List[int], samples_per_bit: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Manchester encoding - Original version.
    0 = High-to-Low transition, 1 = Low-to-High transition
    """
    levels = []
    for bit in bits:
        if bit == 0:
            levels.extend([1, -1])
        else:
            levels.extend([-1, 1])
    
    total_samples = len(levels) * (samples_per_bit // 2)
    t = np.linspace(0, len(bits), total_samples)
    signal = np.zeros(total_samples)
    
    samples_per_half = samples_per_bit // 2
    for i, level in enumerate(levels):
        start = i * samples_per_half
        end = (i + 1) * samples_per_half
        signal[start:end] = level
    
    return t, signal


def ask_modulate_v1(bits: List[int], carrier_freq: float = 10.0, 
                   sample_rate: int = 1000, bit_duration: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    ASK modulation - Original version using loops.
    """
    samples_per_bit = int(sample_rate * bit_duration)
    total_samples = len(bits) * samples_per_bit
    t = np.linspace(0, len(bits) * bit_duration, total_samples)
    signal = np.zeros(total_samples)
    
    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = (i + 1) * samples_per_bit
        t_bit = t[start:end]
        
        if bit == 1:
            signal[start:end] = np.cos(2 * np.pi * carrier_freq * t_bit)
    
    return t, signal


def pcm_encode_v1(signal: np.ndarray, quantization_bits: int = 8) -> str:
    """
    PCM encoding - Original version.
    """
    quantization_levels = 2 ** quantization_bits
    
    # Normalize
    sig_min, sig_max = np.min(signal), np.max(signal)
    if sig_max - sig_min > 0:
        normalized = (signal - sig_min) / (sig_max - sig_min)
    else:
        normalized = np.zeros_like(signal)
    
    # Quantize
    quantized_indices = np.floor(normalized * (quantization_levels - 1)).astype(int)
    quantized_indices = np.clip(quantized_indices, 0, quantization_levels - 1)
    
    # Encode to binary - using loop
    bitstream = ''
    for q in quantized_indices:
        bitstream += format(q, f'0{quantization_bits}b')
    
    return bitstream


def am_modulate_v1(t: np.ndarray, message: np.ndarray, 
                  carrier_freq: float = 100.0, 
                  modulation_index: float = 0.5) -> np.ndarray:
    """
    AM modulation - Original version.
    """
    # Normalize
    sig_max = np.max(np.abs(message))
    if sig_max > 0:
        normalized = message / sig_max
    else:
        normalized = message
    
    # Generate carrier and modulate
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    modulated = (1 + modulation_index * normalized) * carrier
    
    return modulated


# For benchmarking
def run_all_v1(test_bits: List[int], test_signal: np.ndarray, t: np.ndarray):
    """Run all Version A implementations."""
    results = {}
    
    results['nrz_l'] = nrz_l_encode_v1(test_bits)
    results['manchester'] = manchester_encode_v1(test_bits)
    results['ask'] = ask_modulate_v1(test_bits)
    results['pcm'] = pcm_encode_v1(test_signal)
    results['am'] = am_modulate_v1(t, test_signal)
    
    return results
