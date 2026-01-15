"""
Signal Encoding and Modulation Simulator
BLG 337E - Principles of Computer Communication
ITU - Prof. Dr. Abdül Halim Zaim

Streamlit Dashboard for interactive signal simulation
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Import encoders
from encoders.digital_to_digital import DigitalToDigitalEncoder, DigitalToDigitalDecoder
from encoders.digital_to_analog import DigitalToAnalogModulator, DigitalToAnalogDemodulator
from encoders.analog_to_digital import AnalogToDigitalConverter, AnalogToDigitalDecoder
from encoders.analog_to_analog import AnalogToAnalogModulator, AnalogToAnalogDemodulator

# Import visualization
from visualization.plotter import SignalPlotter


# Page configuration
st.set_page_config(
    page_title="Signal Simulator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - supports both light and dark mode
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .mode-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 1rem;
    }
    
    /* Metric card styling for both themes */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Force dark text on metric values for visibility */
    [data-testid="stMetric"] label {
        color: #1e293b !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 600;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #475569 !important;
    }
    
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📡 Signal Encoding & Modulation Simulator</h1>', unsafe_allow_html=True)
st.markdown("**BLG 337E** - Principles of Computer Communication | ITU", unsafe_allow_html=True)

# Initialize plotter
plotter = SignalPlotter(figsize=(12, 5))


def generate_analog_signal(freq: float = 5.0, duration: float = 1.0, 
                          sample_rate: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a sample analog sine wave signal."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * freq * t)
    return t, signal


def main():
    # Sidebar - Mode Selection
    st.sidebar.markdown("## 🎛️ Transmission Mode")
    
    mode = st.sidebar.radio(
        "Select Mode:",
        ["Digital → Digital", "Digital → Analog", "Analog → Digital", "Analog → Analog", "⚡ Benchmark"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Mode-specific UI
    if mode == "Digital → Digital":
        digital_to_digital_mode()
    elif mode == "Digital → Analog":
        digital_to_analog_mode()
    elif mode == "Analog → Digital":
        analog_to_digital_mode()
    elif mode == "Analog → Analog":
        analog_to_analog_mode()
    elif mode == "⚡ Benchmark":
        benchmark_mode()


def digital_to_digital_mode():
    """Digital-to-Digital encoding mode."""
    st.markdown("### 🔢 Digital-to-Digital Encoding")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Algorithm Selection")
        algorithm = st.selectbox(
            "Encoding Algorithm:",
            DigitalToDigitalEncoder.ALGORITHMS,
            index=0
        )
        
        # Algorithm descriptions
        algo_desc = {
            'NRZ-L': "**NRZ-L**: 0 = High voltage, 1 = Low voltage",
            'NRZI': "**NRZI**: 1 = Transition at start, 0 = No transition",
            'Bipolar-AMI': "**AMI**: 0 = Zero level, 1 = Alternating ±V",
            'Manchester': "**Manchester**: 0 = High→Low, 1 = Low→High (mid-bit)",
            'Differential Manchester': "**Diff. Manchester**: Always mid-bit transition; 0 = transition at start",
            'B8ZS': "**B8ZS**: AMI + substitution for 8 zeros (North American)",
            'HDB3': "**HDB3**: AMI + substitution for 4 zeros (European)"
        }
        st.info(algo_desc.get(algorithm, ""))
        
        st.markdown("#### Input Data")
        binary_input = st.text_input(
            "Binary Data:",
            value="10110001",
            help="Enter binary string (0s and 1s only)"
        )
        
        # Validate input
        if not all(c in '01' for c in binary_input):
            st.error("Please enter only 0s and 1s")
            return
        
        if len(binary_input) < 1:
            st.warning("Please enter at least 1 bit")
            return
        
        samples_per_bit = st.slider("Samples per bit:", 50, 200, 100)
    
    with col2:
        # Encode
        encoder = DigitalToDigitalEncoder(samples_per_bit=samples_per_bit)
        t, encoded = encoder.encode(binary_input, algorithm)
        
        # Plot
        fig = plotter.plot_encoding_process(binary_input, t, encoded, algorithm)
        st.pyplot(fig)
        plt.close(fig)
        
        # Decode
        decoder = DigitalToDigitalDecoder(samples_per_bit=samples_per_bit)
        decoded = decoder.decode(encoded, algorithm)
        
        # Results
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Input", binary_input)
        col_b.metric("Decoded", decoded)
        col_c.metric("Match", "✅ Yes" if decoded == binary_input else "❌ No")


def digital_to_analog_mode():
    """Digital-to-Analog modulation mode."""
    st.markdown("### 📶 Digital-to-Analog Modulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Algorithm Selection")
        algorithm = st.selectbox(
            "Modulation Algorithm:",
            DigitalToAnalogModulator.ALGORITHMS,
            index=0
        )
        
        algo_desc = {
            'ASK': "**ASK**: Carrier amplitude varies (0 = off, 1 = on)",
            'BFSK': "**BFSK**: Carrier frequency varies (0 = f₁, 1 = f₂)",
            'BPSK': "**BPSK**: Carrier phase varies (0 = 0°, 1 = 180°)",
            'DPSK': "**DPSK**: Phase change represents data",
            'QAM': "**QAM**: Combines ASK + PSK (4-QAM uses 2 bits/symbol)"
        }
        st.info(algo_desc.get(algorithm, ""))
        
        st.markdown("#### Input Data")
        binary_input = st.text_input(
            "Binary Data:",
            value="10101100",
            key="d2a_input"
        )
        
        if not all(c in '01' for c in binary_input):
            st.error("Please enter only 0s and 1s")
            return
        
        st.markdown("#### Parameters")
        carrier_freq = st.slider("Carrier Frequency (Hz):", 5, 50, 10)
        sample_rate = st.slider("Sample Rate:", 500, 2000, 1000)
        bit_duration = st.slider("Bit Duration (s):", 0.5, 2.0, 1.0, 0.1)
    
    with col2:
        # Modulate
        modulator = DigitalToAnalogModulator(
            carrier_freq=carrier_freq,
            sample_rate=sample_rate,
            bit_duration=bit_duration
        )
        t, modulated = modulator.modulate(binary_input, algorithm)
        
        # Plot
        fig = plotter.plot_modulation_process(binary_input, t, modulated, algorithm)
        st.pyplot(fig)
        plt.close(fig)
        
        # Demodulate
        demodulator = DigitalToAnalogDemodulator(
            carrier_freq=carrier_freq,
            sample_rate=sample_rate,
            bit_duration=bit_duration
        )
        decoded = demodulator.demodulate(t, modulated, algorithm)
        
        # Handle QAM padding
        compare_input = binary_input
        if algorithm == 'QAM' and len(binary_input) % 2 != 0:
            compare_input = binary_input + '0'
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Input", binary_input)
        col_b.metric("Decoded", decoded)
        col_c.metric("Match", "✅ Yes" if decoded == compare_input else "❌ No")


def analog_to_digital_mode():
    """Analog-to-Digital conversion mode."""
    st.markdown("### 🔊→🔢 Analog-to-Digital Conversion")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Algorithm Selection")
        algorithm = st.selectbox(
            "Conversion Algorithm:",
            AnalogToDigitalConverter.ALGORITHMS,
            index=0
        )
        
        algo_desc = {
            'PCM': "**PCM**: Sample → Quantize → Encode (Nyquist: fs ≥ 2fmax)",
            'Delta Modulation': "**DM**: 1-bit encoding using staircase approximation"
        }
        st.info(algo_desc.get(algorithm, ""))
        
        st.markdown("#### Signal Parameters")
        signal_freq = st.slider("Signal Frequency (Hz):", 1, 20, 5)
        duration = st.slider("Duration (s):", 0.5, 2.0, 1.0, 0.1)
        
        st.markdown("#### Conversion Parameters")
        sample_rate = st.slider("Sample Rate (Hz):", 50, 500, 200, key="a2d_sr")
        
        if algorithm == 'PCM':
            quant_bits = st.slider("Quantization Bits:", 2, 8, 4)
            st.caption(f"Levels: {2**quant_bits}")
        else:
            quant_bits = 1
            step_size = st.slider("Step Size:", 0.05, 0.3, 0.1, 0.01)
        
        # Nyquist check
        nyquist_freq = sample_rate / 2
        if signal_freq > nyquist_freq:
            st.warning(f"⚠️ Aliasing! fs={sample_rate} < 2×{signal_freq}={2*signal_freq}")
        else:
            st.success(f"✅ Nyquist satisfied: {sample_rate} ≥ 2×{signal_freq}")
    
    with col2:
        # Generate analog signal
        t, signal = generate_analog_signal(signal_freq, duration, 1000)
        
        # Convert
        converter = AnalogToDigitalConverter(
            sample_rate=sample_rate,
            quantization_bits=quant_bits
        )
        sampled_t, bitstream, quantized = converter.convert(t, signal, algorithm)
        
        # Plot
        if algorithm == 'PCM':
            fig = plotter.plot_pcm_process(t, signal, sampled_t, quantized, bitstream)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            
            ax1.plot(t, signal, 'b-', label='Original')
            ax1.set_title('Original Signal', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.plot(t, signal, 'b-', alpha=0.5, label='Original')
            ax2.step(sampled_t, quantized, 'g-', where='post', label='Staircase')
            ax2.set_title('Delta Modulation Staircase', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            bits_display = [int(b) for b in bitstream[:100]]
            ax3.bar(range(len(bits_display)), bits_display, width=0.8, color='#16a34a')
            ax3.set_title(f'Bitstream (first {len(bits_display)} of {len(bitstream)} bits)', fontweight='bold')
            ax3.set_yticks([0, 1])
            ax3.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Stats
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Samples", len(sampled_t))
        col_b.metric("Bits Generated", len(bitstream))
        col_c.metric("Bits/Sample", quant_bits if algorithm == 'PCM' else 1)


def analog_to_analog_mode():
    """Analog-to-Analog modulation mode."""
    st.markdown("### 🔊→📡 Analog-to-Analog Modulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Algorithm Selection")
        algorithm = st.selectbox(
            "Modulation Algorithm:",
            AnalogToAnalogModulator.ALGORITHMS,
            index=0
        )
        
        algo_desc = {
            'AM': "**AM**: s(t) = [1 + m·x(t)]·cos(2πfc·t)",
            'FM': "**FM**: s(t) = cos(2π[fc + kf·x(t)]·t)"
        }
        st.info(algo_desc.get(algorithm, ""))
        
        st.markdown("#### Message Signal")
        message_freq = st.slider("Message Frequency (Hz):", 1, 20, 5)
        duration = st.slider("Duration (s):", 0.2, 1.0, 0.5, 0.05)
        
        st.markdown("#### Carrier Parameters")
        carrier_freq = st.slider("Carrier Frequency (Hz):", 50, 500, 100)
        sample_rate = st.slider("Sample Rate:", 2000, 10000, 5000)
        
        if algorithm == 'AM':
            mod_index = st.slider("Modulation Index (m):", 0.1, 1.0, 0.5, 0.1)
        else:
            freq_dev = st.slider("Frequency Deviation:", 10, 100, 50)
    
    with col2:
        # Generate message signal
        t = np.linspace(0, duration, int(sample_rate * duration))
        message = np.sin(2 * np.pi * message_freq * t)
        
        # Modulate
        modulator = AnalogToAnalogModulator(
            carrier_freq=carrier_freq,
            sample_rate=sample_rate
        )
        _, modulated = modulator.modulate(t, message, algorithm)
        
        # Plot
        fig = plotter.plot_am_fm_process(t, message, modulated, algorithm)
        st.pyplot(fig)
        plt.close(fig)
        
        # Demodulate
        demodulator = AnalogToAnalogDemodulator(
            carrier_freq=carrier_freq,
            sample_rate=sample_rate
        )
        _, demodulated = demodulator.demodulate(t, modulated, algorithm)
        
        # Show demodulated
        st.markdown("#### Demodulated Signal")
        fig2, ax = plt.subplots(figsize=(12, 3))
        ax.plot(t, demodulated, 'g-', linewidth=1.5, label='Demodulated')
        ax.plot(t, message / np.max(np.abs(message)), 'b--', alpha=0.5, label='Original (normalized)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Demodulated Signal Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


def benchmark_mode():
    """Benchmark mode comparing Version A, B, and C implementations."""
    import timeit
    import tracemalloc
    
    st.markdown("### ⚡ AI Optimization Benchmark")
    st.markdown("""
    Compare three versions of the encoding algorithms:
    - **Version A (Original)**: Basic Python loops
    - **Version B (Google Gemini Pro)**: NumPy vectorization for speed
    - **Version C (OpenAI ChatGPT)**: float32 arrays for 50% memory reduction
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Test Configuration")
        
        test_algorithm = st.selectbox(
            "Algorithm to Test:",
            ["NRZ-L Encoding", "Manchester Encoding", "ASK Modulation", "PCM Conversion", "AM Modulation"],
            index=0
        )
        
        data_size = st.slider("Data Size (samples/bits):", 100, 5000, 1000, 100)
        iterations = st.slider("Benchmark Iterations:", 10, 100, 50)
        
        run_benchmark = st.button("🚀 Run Benchmark", type="primary")
        
        st.markdown("---")
        st.markdown("#### AI Tools Used")
        st.info("""
        **Version B**: Optimized by **Google Gemini Pro**
        - NumPy vectorization
        - Eliminated Python loops
        
        **Version C**: Optimized by **OpenAI ChatGPT**  
        - float32 instead of float64
        - Memory-efficient operations
        """)
    
    with col2:
        if run_benchmark:
            # Import benchmark versions
            from benchmarks.version_a import nrz_l_encode_v1, manchester_encode_v1, ask_modulate_v1, pcm_encode_v1, am_modulate_v1
            from benchmarks.version_b import nrz_l_encode_v2, manchester_encode_v2, ask_modulate_v2, pcm_encode_v2, am_modulate_v2
            from benchmarks.version_c import nrz_l_encode_v3, manchester_encode_v3, ask_modulate_v3, pcm_encode_v3, am_modulate_v3
            
            # Generate test data
            test_bits = [np.random.randint(0, 2) for _ in range(data_size)]
            t = np.linspace(0, 1, data_size)
            test_signal = np.sin(2 * np.pi * 5 * t)
            
            # Select functions based on algorithm
            if test_algorithm == "NRZ-L Encoding":
                funcs = [
                    ("Version A", lambda: nrz_l_encode_v1(test_bits)),
                    ("Version B", lambda: nrz_l_encode_v2(test_bits)),
                    ("Version C", lambda: nrz_l_encode_v3(test_bits)),
                ]
            elif test_algorithm == "Manchester Encoding":
                funcs = [
                    ("Version A", lambda: manchester_encode_v1(test_bits)),
                    ("Version B", lambda: manchester_encode_v2(test_bits)),
                    ("Version C", lambda: manchester_encode_v3(test_bits)),
                ]
            elif test_algorithm == "ASK Modulation":
                funcs = [
                    ("Version A", lambda: ask_modulate_v1(test_bits)),
                    ("Version B", lambda: ask_modulate_v2(test_bits)),
                    ("Version C", lambda: ask_modulate_v3(test_bits)),
                ]
            elif test_algorithm == "PCM Conversion":
                funcs = [
                    ("Version A", lambda: pcm_encode_v1(test_signal)),
                    ("Version B", lambda: pcm_encode_v2(test_signal)),
                    ("Version C", lambda: pcm_encode_v3(test_signal)),
                ]
            else:  # AM Modulation
                funcs = [
                    ("Version A", lambda: am_modulate_v1(t, test_signal)),
                    ("Version B", lambda: am_modulate_v2(t, test_signal)),
                    ("Version C", lambda: am_modulate_v3(t, test_signal)),
                ]
            
            # Run benchmarks
            results = []
            progress_bar = st.progress(0)
            
            for i, (name, func) in enumerate(funcs):
                # Time measurement
                timer = timeit.Timer(func)
                time_taken = (timer.timeit(number=iterations) / iterations) * 1000  # ms
                
                # Memory measurement
                tracemalloc.start()
                func()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_kb = peak / 1024
                
                results.append({
                    "Version": name,
                    "Time (ms)": round(time_taken, 3),
                    "Memory (KB)": round(memory_kb, 2)
                })
                
                progress_bar.progress((i + 1) / len(funcs))
            
            # Display results
            st.markdown("#### 📊 Benchmark Results")
            
            # Results table
            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Calculate improvements
            base_time = results[0]["Time (ms)"]
            base_memory = results[0]["Memory (KB)"]
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Version A (Baseline)", f"{results[0]['Time (ms)']} ms")
            
            with col_b:
                time_diff = ((base_time - results[1]["Time (ms)"]) / base_time) * 100
                st.metric("Version B (Runtime)", f"{results[1]['Time (ms)']} ms", 
                         delta=f"{time_diff:+.1f}%")
            
            with col_c:
                mem_diff = ((base_memory - results[2]["Memory (KB)"]) / base_memory) * 100
                st.metric("Version C (Memory)", f"{results[2]['Memory (KB)']} KB",
                         delta=f"{mem_diff:+.1f}% memory saved")
            
            # Bar chart
            st.markdown("#### Visual Comparison")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            versions = [r["Version"] for r in results]
            times = [r["Time (ms)"] for r in results]
            memories = [r["Memory (KB)"] for r in results]
            colors = ['#6366f1', '#22c55e', '#f59e0b']
            
            ax1.bar(versions, times, color=colors)
            ax1.set_ylabel('Time (ms)')
            ax1.set_title('Execution Time Comparison')
            ax1.grid(True, alpha=0.3, axis='y')
            
            ax2.bar(versions, memories, color=colors)
            ax2.set_ylabel('Memory (KB)')
            ax2.set_title('Memory Usage Comparison')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
        else:
            st.info("👆 Configure test parameters and click 'Run Benchmark' to compare versions.")
            
            # Show code snippets
            st.markdown("#### Code Comparison")
            
            tab1, tab2, tab3 = st.tabs(["Version A (Original)", "Version B (Runtime)", "Version C (Memory)"])
            
            with tab1:
                st.code('''
def nrz_l_encode_v1(bits, samples_per_bit=100):
    """Original version using Python loops."""
    levels = []
    for bit in bits:
        level = -1 if bit == 1 else 1
        levels.extend([level, level])
    
    # Generate signal using loop
    signal = np.zeros(total_samples)
    for i, level in enumerate(levels):
        signal[start:end] = level
    return t, signal
                ''', language='python')
            
            with tab2:
                st.code('''
def nrz_l_encode_v2(bits, samples_per_bit=100):
    """Vectorized version for better runtime."""
    bits_array = np.array(bits)
    
    # Vectorized: no loops!
    levels = np.where(bits_array == 1, -1, 1)
    signal = np.repeat(levels, samples_per_bit)
    
    return t, signal
                ''', language='python')
            
            with tab3:
                st.code('''
def nrz_l_encode_v3(bits, samples_per_bit=100):
    """Memory optimized with float32."""
    # Use float32 (50% less memory)
    signal = np.empty(total_samples, dtype=np.float32)
    
    for i, bit in enumerate(bits):
        level = -1.0 if bit == 1 else 1.0
        signal[start:end] = level
    
    time = np.linspace(0, n_bits, total_samples, 
                       dtype=np.float32)
    return EncodingResult(time=time, signal=signal)
                ''', language='python')


if __name__ == "__main__":
    main()
