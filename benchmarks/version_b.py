"""
Version B: Runtime Optimized Implementation
Optimized for speed using NumPy vectorization and minimizing loops.
"""
import numpy as np
from typing import List, Tuple


def nrz_l_encode_v2(bits: List[int], samples_per_bit: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    NRZ-L encoding - Vectorized version for better runtime.
    Uses NumPy broadcasting and repeat instead of loops.
    """
    bits_array = np.array(bits)
    
    # Vectorized level calculation: 0 -> +1, 1 -> -1
    levels = np.where(bits_array == 1, -1, 1)
    
    # Create signal using repeat (no loop)
    signal = np.repeat(levels, samples_per_bit)
    
    # Time array
    t = np.linspace(0, len(bits), len(signal))
    
    return t, signal


def manchester_encode_v2(bits: List[int], samples_per_bit: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Manchester encoding - Vectorized version.
    Uses NumPy operations for both halves of each bit.
    """
    bits_array = np.array(bits)
    n_bits = len(bits_array)
    samples_per_half = samples_per_bit // 2
    
    # First half: 0 -> +1, 1 -> -1
    first_half = np.where(bits_array == 0, 1, -1)
    # Second half: opposite
    second_half = -first_half
    
    # Interleave using column_stack and flatten
    levels = np.column_stack((first_half, second_half)).flatten()
    
    # Create signal using repeat
    signal = np.repeat(levels, samples_per_half)
    
    # Time array
    t = np.linspace(0, n_bits, len(signal))
    
    return t, signal


def ask_modulate_v2(bits: List[int], carrier_freq: float = 10.0,
                   sample_rate: int = 1000, bit_duration: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    ASK modulation - Vectorized version.
    Creates carrier once and applies mask for amplitude.
    """
    bits_array = np.array(bits)
    samples_per_bit = int(sample_rate * bit_duration)
    total_samples = len(bits) * samples_per_bit
    
    # Time array
    t = np.linspace(0, len(bits) * bit_duration, total_samples)
    
    # Generate full carrier
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    
    # Create amplitude mask using repeat (1 for bit=1, 0 for bit=0)
    amplitude_mask = np.repeat(bits_array, samples_per_bit)
    
    # Apply mask
    signal = carrier * amplitude_mask
    
    return t, signal


def pcm_encode_v2(signal: np.ndarray, quantization_bits: int = 8) -> str:
    """
    PCM encoding - Optimized version using vectorized operations.
    Uses numpy array operations instead of string concatenation.
    """
    quantization_levels = 2 ** quantization_bits
    
    # Normalize using broadcasting
    sig_min, sig_max = signal.min(), signal.max()
    sig_range = sig_max - sig_min
    
    if sig_range > 0:
        normalized = (signal - sig_min) / sig_range
    else:
        normalized = np.zeros_like(signal)
    
    # Vectorized quantization
    quantized_indices = np.floor(normalized * (quantization_levels - 1)).astype(np.int32)
    np.clip(quantized_indices, 0, quantization_levels - 1, out=quantized_indices)
    
    # Vectorized binary conversion using numpy
    # Create format string once
    fmt = f'{{:0{quantization_bits}b}}'
    
    # Use numpy vectorize for binary conversion (faster than loop for large arrays)
    binary_func = np.vectorize(lambda x: fmt.format(x))
    binary_array = binary_func(quantized_indices)
    
    # Join all at once
    bitstream = ''.join(binary_array)
    
    return bitstream


def am_modulate_v2(t: np.ndarray, message: np.ndarray,
                  carrier_freq: float = 100.0,
                  modulation_index: float = 0.5) -> np.ndarray:
    """
    AM modulation - Already vectorized, minor optimizations.
    Uses in-place operations where possible.
    """
    # Get max using numpy's optimized function
    max_abs = np.abs(message).max()
    
    # Normalize in-place style
    if max_abs > 0:
        normalized = message * (1.0 / max_abs)
    else:
        normalized = message
    
    # Pre-calculate constants
    angular_freq = 2 * np.pi * carrier_freq
    
    # Single expression for modulated signal
    modulated = (1 + modulation_index * normalized) * np.cos(angular_freq * t)
    
    return modulated


# For benchmarking
def run_all_v2(test_bits: List[int], test_signal: np.ndarray, t: np.ndarray):
    """Run all Version B implementations."""
    results = {}
    
    results['nrz_l'] = nrz_l_encode_v2(test_bits)
    results['manchester'] = manchester_encode_v2(test_bits)
    results['ask'] = ask_modulate_v2(test_bits)
    results['pcm'] = pcm_encode_v2(test_signal)
    results['am'] = am_modulate_v2(t, test_signal)
    
    return results
