"""
Digital-to-Analog Modulation Techniques
Based on William Stallings' Signal Encoding Techniques

Implements: ASK, BFSK, BPSK, DPSK, QAM
"""
import numpy as np
from typing import Tuple, List


class DigitalToAnalogModulator:
    ALGORITHMS = ['ASK', 'BFSK', 'BPSK', 'DPSK', 'QAM']
    
    def __init__(self, carrier_freq: float = 10.0, sample_rate: int = 1000, 
                 bit_duration: float = 1.0):
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.bit_duration = bit_duration
        self.samples_per_bit = int(sample_rate * bit_duration)
    
    def modulate(self, data: str, algorithm: str) -> Tuple[np.ndarray, np.ndarray]:
        bits = [int(b) for b in data if b in '01']
        
        if algorithm == 'ASK':
            return self._ask(bits)
        elif algorithm == 'BFSK':
            return self._bfsk(bits)
        elif algorithm == 'BPSK':
            return self._bpsk(bits)
        elif algorithm == 'DPSK':
            return self._dpsk(bits)
        elif algorithm == 'QAM':
            return self._qam(bits)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _ask(self, bits: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        total_samples = len(bits) * self.samples_per_bit
        t = np.linspace(0, len(bits) * self.bit_duration, total_samples)
        signal = np.zeros(total_samples)
        
        for i, bit in enumerate(bits):
            start = i * self.samples_per_bit
            end = (i + 1) * self.samples_per_bit
            t_bit = t[start:end]
            
            if bit == 1:
                signal[start:end] = np.cos(2 * np.pi * self.carrier_freq * t_bit)
            # bit == 0: signal stays 0
        
        return t, signal
    
    def _bfsk(self, bits: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        f1 = self.carrier_freq * 0.5  # Lower frequency for 0
        f2 = self.carrier_freq * 1.5  # Higher frequency for 1
        
        total_samples = len(bits) * self.samples_per_bit
        t = np.linspace(0, len(bits) * self.bit_duration, total_samples)
        signal = np.zeros(total_samples)
        
        for i, bit in enumerate(bits):
            start = i * self.samples_per_bit
            end = (i + 1) * self.samples_per_bit
            t_bit = t[start:end]
            
            freq = f2 if bit == 1 else f1
            signal[start:end] = np.cos(2 * np.pi * freq * t_bit)
        
        return t, signal
    
    def _bpsk(self, bits: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        total_samples = len(bits) * self.samples_per_bit
        t = np.linspace(0, len(bits) * self.bit_duration, total_samples)
        signal = np.zeros(total_samples)
        
        for i, bit in enumerate(bits):
            start = i * self.samples_per_bit
            end = (i + 1) * self.samples_per_bit
            t_bit = t[start:end]
            
            phase = np.pi if bit == 1 else 0
            signal[start:end] = np.cos(2 * np.pi * self.carrier_freq * t_bit + phase)
        
        return t, signal
    
    def _dpsk(self, bits: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        total_samples = len(bits) * self.samples_per_bit
        t = np.linspace(0, len(bits) * self.bit_duration, total_samples)
        signal = np.zeros(total_samples)
        
        current_phase = 0
        for i, bit in enumerate(bits):
            start = i * self.samples_per_bit
            end = (i + 1) * self.samples_per_bit
            t_bit = t[start:end]
            
            if bit == 1:
                current_phase += np.pi  # Phase shift for 1
            
            signal[start:end] = np.cos(2 * np.pi * self.carrier_freq * t_bit + current_phase)
        
        return t, signal
    
    def _qam(self, bits: List[int], constellation_size: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        # Pad bits to even length for 4-QAM
        if len(bits) % 2 != 0:
            bits = bits + [0]
        
        # 4-QAM constellation (2 bits per symbol)
        # 00 -> (1, 1), 01 -> (1, -1), 10 -> (-1, 1), 11 -> (-1, -1)
        constellation = {
            (0, 0): (1, 1),
            (0, 1): (1, -1),
            (1, 0): (-1, 1),
            (1, 1): (-1, -1)
        }
        
        n_symbols = len(bits) // 2
        samples_per_symbol = self.samples_per_bit * 2
        total_samples = n_symbols * samples_per_symbol
        t = np.linspace(0, n_symbols * self.bit_duration * 2, total_samples)
        signal = np.zeros(total_samples)
        
        for i in range(n_symbols):
            bit_pair = (bits[2*i], bits[2*i + 1])
            I, Q = constellation[bit_pair]
            
            start = i * samples_per_symbol
            end = (i + 1) * samples_per_symbol
            t_symbol = t[start:end]
            
            signal[start:end] = (I * np.cos(2 * np.pi * self.carrier_freq * t_symbol) + 
                                  Q * np.sin(2 * np.pi * self.carrier_freq * t_symbol))
        
        return t, signal / np.sqrt(2)  # Normalize


class DigitalToAnalogDemodulator:
    def __init__(self, carrier_freq: float = 10.0, sample_rate: int = 1000,
                 bit_duration: float = 1.0):
        self.carrier_freq = carrier_freq
        self.sample_rate = sample_rate
        self.bit_duration = bit_duration
        self.samples_per_bit = int(sample_rate * bit_duration)
    
    def demodulate(self, t: np.ndarray, signal: np.ndarray, algorithm: str) -> str:
        """Demodulate signal back to binary string."""
        if algorithm == 'ASK':
            return self._demod_ask(signal)
        elif algorithm == 'BFSK':
            return self._demod_bfsk(t, signal)
        elif algorithm == 'BPSK':
            return self._demod_bpsk(t, signal)
        elif algorithm == 'DPSK':
            return self._demod_dpsk(t, signal)
        elif algorithm == 'QAM':
            return self._demod_qam(t, signal)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _demod_ask(self, signal: np.ndarray) -> str:
        """Demodulate ASK by checking amplitude."""
        bits = []
        n_bits = len(signal) // self.samples_per_bit
        
        for i in range(n_bits):
            start = i * self.samples_per_bit
            end = (i + 1) * self.samples_per_bit
            amplitude = np.max(np.abs(signal[start:end]))
            bits.append('1' if amplitude > 0.5 else '0')
        
        return ''.join(bits)
    
    def _demod_bfsk(self, t: np.ndarray, signal: np.ndarray) -> str:
        """Demodulate BFSK by comparing correlation with f1 and f2."""
        f1 = self.carrier_freq * 0.5
        f2 = self.carrier_freq * 1.5
        bits = []
        n_bits = len(signal) // self.samples_per_bit
        
        for i in range(n_bits):
            start = i * self.samples_per_bit
            end = (i + 1) * self.samples_per_bit
            t_bit = t[start:end]
            sig_bit = signal[start:end]
            
            corr_f1 = np.sum(sig_bit * np.cos(2 * np.pi * f1 * t_bit))
            corr_f2 = np.sum(sig_bit * np.cos(2 * np.pi * f2 * t_bit))
            
            bits.append('1' if corr_f2 > corr_f1 else '0')
        
        return ''.join(bits)
    
    def _demod_bpsk(self, t: np.ndarray, signal: np.ndarray) -> str:
        """Demodulate BPSK using coherent detection."""
        bits = []
        n_bits = len(signal) // self.samples_per_bit
        
        for i in range(n_bits):
            start = i * self.samples_per_bit
            end = (i + 1) * self.samples_per_bit
            t_bit = t[start:end]
            sig_bit = signal[start:end]
            
            # Correlate with reference carrier
            corr = np.sum(sig_bit * np.cos(2 * np.pi * self.carrier_freq * t_bit))
            bits.append('0' if corr > 0 else '1')
        
        return ''.join(bits)
    
    def _demod_dpsk(self, t: np.ndarray, signal: np.ndarray) -> str:
        """Demodulate DPSK by detecting phase changes."""
        bits = []
        n_bits = len(signal) // self.samples_per_bit
        prev_phase = 0
        
        for i in range(n_bits):
            start = i * self.samples_per_bit
            end = (i + 1) * self.samples_per_bit
            t_bit = t[start:end]
            sig_bit = signal[start:end]
            
            # Estimate phase
            corr_cos = np.sum(sig_bit * np.cos(2 * np.pi * self.carrier_freq * t_bit))
            corr_sin = np.sum(sig_bit * np.sin(2 * np.pi * self.carrier_freq * t_bit))
            phase = np.arctan2(corr_sin, corr_cos)
            
            # Detect phase change
            phase_diff = abs(phase - prev_phase)
            bits.append('1' if phase_diff > np.pi/2 else '0')
            prev_phase = phase
        
        return ''.join(bits)
    
    def _demod_qam(self, t: np.ndarray, signal: np.ndarray) -> str:
        """Demodulate 4-QAM."""
        bits = []
        samples_per_symbol = self.samples_per_bit * 2
        n_symbols = len(signal) // samples_per_symbol
        
        for i in range(n_symbols):
            start = i * samples_per_symbol
            end = (i + 1) * samples_per_symbol
            t_sym = t[start:end]
            sig_sym = signal[start:end] * np.sqrt(2)
            
            # Correlate with I and Q carriers
            I = np.sum(sig_sym * np.cos(2 * np.pi * self.carrier_freq * t_sym))
            Q = np.sum(sig_sym * np.sin(2 * np.pi * self.carrier_freq * t_sym))
            
            # Decide based on sign
            bit1 = '1' if I < 0 else '0'
            bit2 = '1' if Q < 0 else '0'
            bits.extend([bit1, bit2])
        
        return ''.join(bits)
