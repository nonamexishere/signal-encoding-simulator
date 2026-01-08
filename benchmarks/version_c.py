"""
Version C: Memory and Readability Optimized Implementation
Optimized for memory usage and code clarity with better documentation.
"""
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class EncodingResult:
    """Result container for encoding operations."""
    time: np.ndarray
    signal: np.ndarray
    
    @property
    def memory_bytes(self) -> int:
        """Calculate memory usage."""
        return self.time.nbytes + self.signal.nbytes


def nrz_l_encode_v3(bits: List[int], samples_per_bit: int = 100) -> EncodingResult:
    """
    NRZ-L (Non-Return to Zero Level) Encoding
    
    Memory optimized version using generators and minimal intermediate arrays.
    
    Algorithm:
        - Binary 0 → High voltage (+1)
        - Binary 1 → Low voltage (-1)
    
    Args:
        bits: List of binary values (0 or 1)
        samples_per_bit: Resolution of the output waveform
        
    Returns:
        EncodingResult with time and signal arrays
    """
    n_bits = len(bits)
    total_samples = n_bits * samples_per_bit
    
    # Use float32 instead of float64 to save memory (50% reduction)
    signal = np.empty(total_samples, dtype=np.float32)
    
    # Direct assignment without intermediate arrays
    for i, bit in enumerate(bits):
        level = -1.0 if bit == 1 else 1.0
        start, end = i * samples_per_bit, (i + 1) * samples_per_bit
        signal[start:end] = level
    
    # Time array with float32
    time = np.linspace(0, n_bits, total_samples, dtype=np.float32)
    
    return EncodingResult(time=time, signal=signal)


def manchester_encode_v3(bits: List[int], samples_per_bit: int = 100) -> EncodingResult:
    """
    Manchester Encoding (IEEE 802.3 Standard)
    
    Memory-efficient implementation with clear documentation.
    
    Algorithm:
        - Binary 0 → High-to-Low transition at mid-bit
        - Binary 1 → Low-to-High transition at mid-bit
    
    The mid-bit transition serves as both data and clock signal,
    providing self-clocking capability.
    
    Args:
        bits: List of binary values
        samples_per_bit: Resolution per bit period
        
    Returns:
        EncodingResult with time and signal arrays
    """
    n_bits = len(bits)
    total_samples = n_bits * samples_per_bit
    samples_per_half = samples_per_bit // 2
    
    # Memory-efficient: single array, float32
    signal = np.empty(total_samples, dtype=np.float32)
    
    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        mid = start + samples_per_half
        end = (i + 1) * samples_per_bit
        
        if bit == 0:
            # High-to-Low transition
            signal[start:mid] = 1.0
            signal[mid:end] = -1.0
        else:
            # Low-to-High transition
            signal[start:mid] = -1.0
            signal[mid:end] = 1.0
    
    time = np.linspace(0, n_bits, total_samples, dtype=np.float32)
    
    return EncodingResult(time=time, signal=signal)


def ask_modulate_v3(bits: List[int], 
                   carrier_freq: float = 10.0,
                   sample_rate: int = 1000, 
                   bit_duration: float = 1.0) -> EncodingResult:
    """
    Amplitude Shift Keying (ASK) Modulation
    
    Memory-optimized with clear signal processing steps.
    
    Algorithm:
        s(t) = A(t) × cos(2πfc×t)
        where A(t) = 1 for binary 1, A(t) = 0 for binary 0
    
    Args:
        bits: Digital data to modulate
        carrier_freq: Carrier frequency in Hz
        sample_rate: Sampling rate in samples/second
        bit_duration: Duration of each bit in seconds
        
    Returns:
        EncodingResult with modulated signal
    """
    samples_per_bit = int(sample_rate * bit_duration)
    total_samples = len(bits) * samples_per_bit
    
    # Use float32 for memory efficiency
    time = np.linspace(0, len(bits) * bit_duration, total_samples, dtype=np.float32)
    
    # Generate carrier (computed on-the-fly, not stored separately)
    angular_frequency = np.float32(2 * np.pi * carrier_freq)
    
    # Single pass: compute modulated signal directly
    signal = np.empty(total_samples, dtype=np.float32)
    
    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = (i + 1) * samples_per_bit
        
        if bit == 1:
            signal[start:end] = np.cos(angular_frequency * time[start:end])
        else:
            signal[start:end] = 0.0
    
    return EncodingResult(time=time, signal=signal)


def pcm_encode_v3(signal: np.ndarray, quantization_bits: int = 8) -> str:
    """
    Pulse Code Modulation (PCM) Encoding
    
    Memory-efficient implementation with streaming binary conversion.
    
    Process:
        1. Normalization: Scale signal to [0, 1]
        2. Quantization: Map to discrete levels (2^n levels)
        3. Encoding: Convert each level to binary
    
    Args:
        signal: Analog signal samples
        quantization_bits: Bits per sample (determines resolution)
        
    Returns:
        Binary string representing the encoded signal
    """
    max_level = (1 << quantization_bits) - 1  # 2^n - 1
    
    # Normalize to [0, 1] range
    sig_min = signal.min()
    sig_range = signal.max() - sig_min
    
    if sig_range > 0:
        # Scale to [0, max_level] directly (no intermediate normalized array)
        quantized = np.floor((signal - sig_min) / sig_range * max_level)
    else:
        quantized = np.zeros_like(signal)
    
    # Clip and convert to integer
    quantized = np.clip(quantized, 0, max_level).astype(np.uint16)
    
    # Memory-efficient string building using list comprehension
    # (slightly faster than join with generator)
    bitstream = ''.join(format(q, f'0{quantization_bits}b') for q in quantized)
    
    return bitstream


def am_modulate_v3(time: np.ndarray, 
                  message: np.ndarray,
                  carrier_freq: float = 100.0,
                  modulation_index: float = 0.5) -> np.ndarray:
    """
    Amplitude Modulation (AM)
    
    Standard AM formula with memory optimization.
    
    Formula:
        s(t) = [1 + m × x(t)] × cos(2πfc×t)
    
    where:
        m = modulation index (0 < m ≤ 1)
        x(t) = normalized message signal
        fc = carrier frequency
    
    Args:
        time: Time array
        message: Message/baseband signal
        carrier_freq: Carrier frequency in Hz
        modulation_index: Depth of modulation (0-1)
        
    Returns:
        AM modulated signal
    """
    # Normalize message in-place if possible
    max_amplitude = np.abs(message).max()
    
    if max_amplitude > 0:
        # Compute modulation envelope directly
        envelope = 1.0 + modulation_index * (message / max_amplitude)
    else:
        envelope = np.ones_like(message)
    
    # Carrier and modulation in single expression
    # Using float32 for memory efficiency
    angular_freq = np.float32(2 * np.pi * carrier_freq)
    modulated = (envelope * np.cos(angular_freq * time)).astype(np.float32)
    
    return modulated


# For benchmarking
def run_all_v3(test_bits: List[int], test_signal: np.ndarray, t: np.ndarray):
    """Run all Version C implementations."""
    results = {}
    
    nrz_result = nrz_l_encode_v3(test_bits)
    results['nrz_l'] = (nrz_result.time, nrz_result.signal)
    
    manchester_result = manchester_encode_v3(test_bits)
    results['manchester'] = (manchester_result.time, manchester_result.signal)
    
    ask_result = ask_modulate_v3(test_bits)
    results['ask'] = (ask_result.time, ask_result.signal)
    
    results['pcm'] = pcm_encode_v3(test_signal)
    results['am'] = am_modulate_v3(t.astype(np.float32), test_signal)
    
    return results
