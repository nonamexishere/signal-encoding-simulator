"""
Digital-to-Digital Encoding Techniques
Based on William Stallings' Signal Encoding Techniques

Implements: NRZ-L, NRZI, Bipolar-AMI, Manchester, Differential Manchester, B8ZS, HDB3
"""
import numpy as np
from typing import List, Tuple


class DigitalToDigitalEncoder:
    """Digital-to-Digital line coding encoder with multiple algorithms."""
    
    ALGORITHMS = ['NRZ-L', 'NRZI', 'Bipolar-AMI', 'Manchester', 
                  'Differential Manchester', 'B8ZS', 'HDB3']
    
    def __init__(self, samples_per_bit: int = 100):
        """
        Initialize encoder.
        
        Args:
            samples_per_bit: Number of samples per bit for waveform generation
        """
        self.samples_per_bit = samples_per_bit
    
    def encode(self, data: str, algorithm: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode binary data using specified algorithm.
        
        Args:
            data: Binary string (e.g., '10110001')
            algorithm: Encoding algorithm name
            
        Returns:
            Tuple of (time_array, signal_array)
        """
        bits = [int(b) for b in data if b in '01']
        
        if algorithm == 'NRZ-L':
            return self._nrz_l(bits)
        elif algorithm == 'NRZI':
            return self._nrzi(bits)
        elif algorithm == 'Bipolar-AMI':
            return self._bipolar_ami(bits)
        elif algorithm == 'Manchester':
            return self._manchester(bits)
        elif algorithm == 'Differential Manchester':
            return self._differential_manchester(bits)
        elif algorithm == 'B8ZS':
            return self._b8zs(bits)
        elif algorithm == 'HDB3':
            return self._hdb3(bits)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _generate_waveform(self, levels: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate time and signal arrays from level list."""
        n_bits = len(levels) // 2 if len(levels) > 0 else 0
        if n_bits == 0:
            return np.array([0]), np.array([0])
        
        total_samples = len(levels) * (self.samples_per_bit // 2)
        t = np.linspace(0, n_bits, total_samples)
        signal = np.zeros(total_samples)
        
        samples_per_half = self.samples_per_bit // 2
        for i, level in enumerate(levels):
            start = i * samples_per_half
            end = (i + 1) * samples_per_half
            signal[start:end] = level
        
        return t, signal
    
    def _nrz_l(self, bits: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        NRZ-L (Non-Return to Zero Level)
        0 = High voltage (+V), 1 = Low voltage (-V)
        """
        levels = []
        for bit in bits:
            level = -1 if bit == 1 else 1
            levels.extend([level, level])  # Two halves per bit
        return self._generate_waveform(levels)
    
    def _nrzi(self, bits: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        NRZI (Non-Return to Zero Inverted)
        1 = Transition at beginning, 0 = No transition
        """
        levels = []
        current_level = 1  # Start with high
        for bit in bits:
            if bit == 1:
                current_level = -current_level  # Transition
            levels.extend([current_level, current_level])
        return self._generate_waveform(levels)
    
    def _bipolar_ami(self, bits: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bipolar-AMI (Alternate Mark Inversion)
        0 = Zero voltage, 1 = Alternating +V/-V
        """
        levels = []
        last_one_level = -1  # Will alternate
        for bit in bits:
            if bit == 0:
                levels.extend([0, 0])
            else:
                last_one_level = -last_one_level
                levels.extend([last_one_level, last_one_level])
        return self._generate_waveform(levels)
    
    def _manchester(self, bits: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Manchester Encoding (IEEE 802.3)
        0 = High-to-Low transition, 1 = Low-to-High transition
        """
        levels = []
        for bit in bits:
            if bit == 0:
                levels.extend([1, -1])  # High to Low
            else:
                levels.extend([-1, 1])  # Low to High
        return self._generate_waveform(levels)
    
    def _differential_manchester(self, bits: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Differential Manchester
        Always mid-bit transition
        0 = Transition at start, 1 = No transition at start
        """
        levels = []
        current_level = 1
        for bit in bits:
            if bit == 0:
                current_level = -current_level  # Transition at start
            # First half
            levels.append(current_level)
            # Mid-bit transition (always)
            current_level = -current_level
            levels.append(current_level)
        return self._generate_waveform(levels)
    
    def _b8zs(self, bits: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        B8ZS (Bipolar with 8-Zero Substitution) - North American Standard
        Replaces 8 consecutive zeros with: 000VB0VB
        V = Violation (same polarity as last pulse)
        B = Bipolar (opposite polarity)
        """
        # First apply substitution
        substituted = self._apply_b8zs_substitution(bits)
        
        # Then encode using modified AMI
        levels = []
        last_pulse = -1
        for symbol in substituted:
            if symbol == 0:
                levels.extend([0, 0])
            elif symbol == 1:
                last_pulse = -last_pulse
                levels.extend([last_pulse, last_pulse])
            elif symbol == 'V':
                # Violation - same polarity as last pulse
                levels.extend([last_pulse, last_pulse])
            elif symbol == 'B':
                # Bipolar - alternate
                last_pulse = -last_pulse
                levels.extend([last_pulse, last_pulse])
        return self._generate_waveform(levels)
    
    def _apply_b8zs_substitution(self, bits: List[int]) -> List:
        """Apply B8ZS substitution for 8 consecutive zeros."""
        result = []
        i = 0
        last_pulse = -1
        
        while i < len(bits):
            # Check for 8 consecutive zeros
            if i + 8 <= len(bits) and all(b == 0 for b in bits[i:i+8]):
                # Determine substitution based on last pulse polarity
                # Pattern: 000VB0VB
                if last_pulse == 1:
                    result.extend([0, 0, 0, 'V', 'B', 0, 'V', 'B'])
                else:
                    result.extend([0, 0, 0, 'V', 'B', 0, 'V', 'B'])
                i += 8
            else:
                if bits[i] == 1:
                    last_pulse = -last_pulse
                result.append(bits[i])
                i += 1
        return result
    
    def _hdb3(self, bits: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        HDB3 (High Density Bipolar 3) - European Standard
        Replaces 4 consecutive zeros based on number of 1s since last substitution:
        - Odd number of 1s: 000V
        - Even number of 1s: B00V
        """
        substituted = self._apply_hdb3_substitution(bits)
        
        levels = []
        last_pulse = -1
        for symbol in substituted:
            if symbol == 0:
                levels.extend([0, 0])
            elif symbol == 1:
                last_pulse = -last_pulse
                levels.extend([last_pulse, last_pulse])
            elif symbol == 'V':
                levels.extend([last_pulse, last_pulse])
            elif symbol == 'B':
                last_pulse = -last_pulse
                levels.extend([last_pulse, last_pulse])
        return self._generate_waveform(levels)
    
    def _apply_hdb3_substitution(self, bits: List[int]) -> List:
        """Apply HDB3 substitution for 4 consecutive zeros."""
        result = []
        i = 0
        ones_since_sub = 0
        
        while i < len(bits):
            if i + 4 <= len(bits) and all(b == 0 for b in bits[i:i+4]):
                if ones_since_sub % 2 == 1:  # Odd
                    result.extend([0, 0, 0, 'V'])
                else:  # Even
                    result.extend(['B', 0, 0, 'V'])
                    ones_since_sub += 1  # B counts as a pulse
                ones_since_sub = 0
                i += 4
            else:
                if bits[i] == 1:
                    ones_since_sub += 1
                result.append(bits[i])
                i += 1
        return result


class DigitalToDigitalDecoder:
    """Decoder for digital line codes."""
    
    def __init__(self, samples_per_bit: int = 100):
        self.samples_per_bit = samples_per_bit
    
    def decode(self, signal: np.ndarray, algorithm: str) -> str:
        """Decode signal back to binary string."""
        samples_per_half = self.samples_per_bit // 2
        n_levels = len(signal) // samples_per_half
        
        # Extract levels (sample middle of each half-bit)
        levels = []
        for i in range(n_levels):
            mid = i * samples_per_half + samples_per_half // 2
            if mid < len(signal):
                levels.append(signal[mid])
        
        if algorithm == 'NRZ-L':
            return self._decode_nrz_l(levels)
        elif algorithm == 'NRZI':
            return self._decode_nrzi(levels)
        elif algorithm == 'Bipolar-AMI':
            return self._decode_ami(levels)
        elif algorithm == 'Manchester':
            return self._decode_manchester(levels)
        elif algorithm == 'Differential Manchester':
            return self._decode_differential_manchester(levels)
        elif algorithm in ['B8ZS', 'HDB3']:
            return self._decode_ami(levels)  # Simplified
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _decode_nrz_l(self, levels: List[float]) -> str:
        """Decode NRZ-L signal."""
        bits = []
        for i in range(0, len(levels), 2):
            bits.append('1' if levels[i] < 0 else '0')
        return ''.join(bits)
    
    def _decode_nrzi(self, levels: List[float]) -> str:
        """Decode NRZI signal."""
        bits = []
        prev_level = 1
        for i in range(0, len(levels), 2):
            if abs(levels[i] - prev_level) > 0.5:
                bits.append('1')
            else:
                bits.append('0')
            prev_level = levels[i]
        return ''.join(bits)
    
    def _decode_ami(self, levels: List[float]) -> str:
        """Decode Bipolar-AMI signal."""
        bits = []
        for i in range(0, len(levels), 2):
            bits.append('0' if abs(levels[i]) < 0.5 else '1')
        return ''.join(bits)
    
    def _decode_manchester(self, levels: List[float]) -> str:
        """Decode Manchester signal."""
        bits = []
        for i in range(0, len(levels), 2):
            if i + 1 < len(levels):
                if levels[i] > 0 and levels[i+1] < 0:
                    bits.append('0')
                else:
                    bits.append('1')
        return ''.join(bits)
    
    def _decode_differential_manchester(self, levels: List[float]) -> str:
        """Decode Differential Manchester signal."""
        bits = []
        prev_end = 1
        for i in range(0, len(levels), 2):
            # Check if there's a transition at the start
            if abs(levels[i] - prev_end) > 0.5:
                bits.append('0')
            else:
                bits.append('1')
            if i + 1 < len(levels):
                prev_end = levels[i + 1]
        return ''.join(bits)
