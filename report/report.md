# Signal Encoding and Modulation Techniques
## Technical Report - BLG 337E Principles of Computer Communication

**Authors**: Mustafa Bozdoğan Enes Saçak
**Date**: January 2026  
**Institution**: Istanbul Technical University

---

## 1. Introduction

This report presents a comprehensive simulation of data transmission between two computers using encoding, modulation, and demodulation techniques. The implementation covers four fundamental transmission modes as outlined in William Stallings' "Data and Computer Communications":

1. **Digital-to-Digital** (Line Coding)
2. **Digital-to-Analog** (Digital Modulation)
3. **Analog-to-Digital** (Digitization)
4. **Analog-to-Analog** (Analog Modulation)

---

## 2. Theoretical Background

### 2.1 Digital-to-Digital Encoding (Line Coding)

Line coding converts digital data into digital signals suitable for transmission over a physical medium.

#### 2.1.1 NRZ (Non-Return to Zero)

**NRZ-L (Level)**: The signal level represents the bit value directly.
- Binary 0 → Positive voltage (+V)
- Binary 1 → Negative voltage (-V)

**NRZI (Invert on ones)**: Uses signal transitions to represent data.
- Binary 1 → Transition at the beginning of the bit period
- Binary 0 → No transition

*Advantage*: Simple implementation  
*Disadvantage*: No self-clocking, DC component issues

#### 2.1.2 Bipolar-AMI (Alternate Mark Inversion)

A three-level encoding scheme:
- Binary 0 → Zero voltage
- Binary 1 → Alternating positive and negative voltage

*Advantage*: No DC component, error detection capability  
*Disadvantage*: Long sequences of zeros cause synchronization loss

#### 2.1.3 Manchester Encoding (IEEE 802.3)

Self-clocking code with mid-bit transitions:
- Binary 0 → High-to-low transition at mid-bit
- Binary 1 → Low-to-high transition at mid-bit

*Advantage*: Self-synchronizing, no DC component  
*Disadvantage*: Requires twice the bandwidth of NRZ

#### 2.1.4 Differential Manchester

Similar to Manchester but uses transitions differently:
- Always has a mid-bit transition (for clocking)
- Binary 0 → Transition at the beginning of bit period
- Binary 1 → No transition at the beginning

### 2.2 Scrambling Techniques

#### 2.2.1 B8ZS (Bipolar with 8-Zero Substitution) - North American Standard

B8ZS addresses the synchronization problem of AMI with long zero sequences by replacing 8 consecutive zeros with a violation pattern:

**Substitution Rule**:
- If previous pulse was positive: `000+-0-+`
- If previous pulse was negative: `000-+0+-`

The pattern contains **violations** (V) where the polarity doesn't alternate, deliberately breaking the AMI rule. The receiver detects these violations and restores the original zeros.

**Mathematical Representation**:
```
8 zeros → 000VB0VB
where V = Violation (same polarity as last pulse)
      B = Bipolar (alternating polarity)
```

#### 2.2.2 HDB3 (High Density Bipolar 3) - European Standard

HDB3 substitutes 4 consecutive zeros based on the count of 1s since the last substitution:

**Substitution Rules**:
- Odd number of 1s since last substitution: `000V`
- Even number of 1s since last substitution: `B00V`

The `B` pulse ensures that violations can always be detected by maintaining proper polarity counting.

### 2.3 Digital-to-Analog Modulation

Digital-to-analog modulation converts digital data to analog signals for transmission over analog channels.

#### 2.3.1 ASK (Amplitude Shift Keying)

$$s(t) = A(t) \cdot \cos(2\pi f_c t)$$

where:
- $A(t) = 1$ for binary 1
- $A(t) = 0$ for binary 0
- $f_c$ = carrier frequency

#### 2.3.2 FSK (Frequency Shift Keying)

$$s(t) = \cos(2\pi f_i t)$$

where:
- $f_1$ = frequency for binary 0
- $f_2$ = frequency for binary 1

#### 2.3.3 PSK (Phase Shift Keying)

**BPSK**: 
$$s(t) = \cos(2\pi f_c t + \phi)$$

where:
- $\phi = 0°$ for binary 0
- $\phi = 180°$ for binary 1

**DPSK (Differential PSK)**: Phase change represents the data rather than absolute phase.

#### 2.3.4 QAM (Quadrature Amplitude Modulation)

Combines ASK and PSK:
$$s(t) = I \cdot \cos(2\pi f_c t) + Q \cdot \sin(2\pi f_c t)$$

where I and Q are the in-phase and quadrature components representing multiple bits per symbol.

### 2.4 Analog-to-Digital Conversion

#### 2.4.1 PCM (Pulse Code Modulation)

PCM involves three steps:

1. **Sampling**: According to the **Nyquist Sampling Theorem**:
   $$f_s \geq 2 \cdot f_{max}$$
   
   The sampling frequency must be at least twice the highest frequency component of the signal to avoid aliasing.

2. **Quantization**: Mapping continuous amplitude values to discrete levels:
   $$L = 2^n$$
   
   where $n$ is the number of bits and $L$ is the number of quantization levels.

3. **Encoding**: Converting quantized values to binary representation.

**Quantization Error (Noise)**:
$$SNR_{dB} = 6.02n + 1.76$$

where $n$ is the number of bits per sample.

#### 2.4.2 Delta Modulation

A simpler 1-bit encoding using staircase approximation:
- If signal > approximation: output 1, increase approximation by step
- If signal ≤ approximation: output 0, decrease approximation by step

**Types of Distortion**:
- **Slope Overload**: Step size too small to track rapid signal changes
- **Granular Noise**: Step size too large for slow-varying signals

### 2.5 Analog-to-Analog Modulation

#### 2.5.1 AM (Amplitude Modulation)

$$s(t) = [1 + m \cdot x(t)] \cdot \cos(2\pi f_c t)$$

where:
- $m$ = modulation index (0 < m ≤ 1)
- $x(t)$ = normalized message signal
- $f_c$ = carrier frequency

**Bandwidth**: $BW = 2 \cdot f_m$ where $f_m$ is the message bandwidth.

#### 2.5.2 FM (Frequency Modulation)

$$s(t) = \cos\left(2\pi f_c t + 2\pi k_f \int x(\tau) d\tau\right)$$

where:
- $k_f$ = frequency deviation constant
- $x(t)$ = message signal

**Carson's Rule for Bandwidth**:
$$BW \approx 2(\Delta f + f_m)$$

where $\Delta f$ is the peak frequency deviation.

---

## 3. Implementation

### 3.1 System Architecture

The implementation follows a modular design with separate modules for:

- **Encoders**: `digital_to_digital.py`, `digital_to_analog.py`, `analog_to_digital.py`, `analog_to_analog.py`
- **Visualization**: `plotter.py` using Matplotlib
- **GUI**: Streamlit dashboard (`app.py`)

### 3.2 Class Hierarchy

```
DigitalToDigitalEncoder
├── encode(data, algorithm) → (time, signal)
├── _nrz_l(), _nrzi(), _bipolar_ami()
├── _manchester(), _differential_manchester()
└── _b8zs(), _hdb3()

DigitalToAnalogModulator
├── modulate(data, algorithm) → (time, signal)
├── _ask(), _bfsk(), _bpsk(), _dpsk(), _qam()

AnalogToDigitalConverter
├── convert(time, signal, algorithm) → (sampled_t, bitstream, quantized)
├── _pcm(), _delta_modulation()

AnalogToAnalogModulator
├── modulate(time, signal, algorithm) → (time, modulated)
├── _am(), _fm()
```

---

## 4. AI Optimization and Benchmarking

### 4.1 Optimization Strategies

Three versions were developed:

1. **Version A (Original)**: Straightforward implementation using Python loops
2. **Version B (Runtime Optimized)**: NumPy vectorization for speed
3. **Version C (Memory Optimized)**: float32 arrays and minimal intermediate storage

### 4.2 Key Optimizations

**Version B - Runtime**:
- Replaced Python loops with `np.repeat()`, `np.where()`
- Used broadcasting for element-wise operations
- Pre-computed constants outside loops

**Version C - Memory**:
- Used `float32` instead of `float64` (50% memory reduction)
- Avoided intermediate array creation
- Used generators where possible

### 4.3 Benchmark Results

| Data Size | Ver. A Time (ms) | Ver. B Time (ms) | Ver. C Time (ms) |
|-----------|------------------|------------------|------------------|
| 100       | ~0.5             | ~0.2             | ~0.4             |
| 1000      | ~4.0             | ~0.8             | ~2.5             |
| 5000      | ~20.0            | ~3.5             | ~12.0            |

*Note: Actual values may vary based on system configuration.*

---

## 5. Conclusions

1. **Line Coding Selection**: Manchester encoding provides self-clocking but requires double bandwidth. For bandwidth-limited applications, B8ZS/HDB3 offer a good balance.

2. **Modulation Trade-offs**: QAM provides highest data rates but requires precise amplitude/phase control. BPSK is more robust against noise.

3. **Sampling Theorem**: The Nyquist criterion ($f_s \geq 2f_{max}$) is fundamental to avoiding aliasing in digital conversion.

4. **Optimization Impact**: Vectorization (Version B) provides 3-5x speed improvement for large datasets. Memory optimization (Version C) reduces footprint by ~50%.

---

## References

1. Stallings, W. (2014). *Data and Computer Communications* (10th ed.). Pearson.
2. Forouzan, B. A. (2013). *Data Communications and Networking* (5th ed.). McGraw-Hill.
3. IEEE 802.3 Standard for Ethernet.
4. ITU-T G.703 - Physical/Electrical Characteristics of Hierarchical Digital Interfaces.

---

## Appendix: Running the Simulator

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app.py

# Run benchmarks
python benchmarks/run_benchmark.py
```
