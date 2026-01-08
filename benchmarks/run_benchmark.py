"""
Benchmark Runner
Compares Version A, B, and C implementations for runtime and memory usage.
"""
import timeit
import tracemalloc
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.version_a import run_all_v1
from benchmarks.version_b import run_all_v2
from benchmarks.version_c import run_all_v3


def measure_time(func, *args, iterations: int = 100) -> float:
    """Measure execution time using timeit."""
    timer = timeit.Timer(lambda: func(*args))
    total_time = timer.timeit(number=iterations)
    return (total_time / iterations) * 1000  # Convert to milliseconds


def measure_memory(func, *args) -> float:
    """Measure peak memory usage using tracemalloc."""
    tracemalloc.start()
    func(*args)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024  # Convert to KB


def run_benchmarks(data_sizes: list = None):
    """
    Run comprehensive benchmarks for all versions.
    
    Args:
        data_sizes: List of data sizes to test
    """
    if data_sizes is None:
        data_sizes = [100, 500, 1000, 5000]
    
    print("=" * 80)
    print("SIGNAL ENCODING & MODULATION BENCHMARK")
    print("Comparing: Version A (Original) | Version B (Runtime) | Version C (Memory)")
    print("=" * 80)
    print()
    
    results = []
    
    for size in data_sizes:
        print(f"\n{'='*60}")
        print(f"DATA SIZE: {size} samples/bits")
        print(f"{'='*60}")
        
        # Generate test data
        test_bits = [np.random.randint(0, 2) for _ in range(size)]
        t = np.linspace(0, 1, size)
        test_signal = np.sin(2 * np.pi * 5 * t)
        
        # Measure each version
        versions = {
            'Version A (Original)': run_all_v1,
            'Version B (Runtime)': run_all_v2,
            'Version C (Memory)': run_all_v3
        }
        
        print("\n{:<25} {:>15} {:>15}".format("Version", "Time (ms)", "Memory (KB)"))
        print("-" * 55)
        
        row = {'size': size}
        
        for name, func in versions.items():
            time_ms = measure_time(func, test_bits, test_signal, t, iterations=50)
            memory_kb = measure_memory(func, test_bits, test_signal, t)
            
            print("{:<25} {:>15.3f} {:>15.2f}".format(name, time_ms, memory_kb))
            
            row[f'{name}_time'] = time_ms
            row[f'{name}_memory'] = memory_kb
        
        results.append(row)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Calculate improvements
    for row in results:
        size = row['size']
        
        base_time = row['Version A (Original)_time']
        base_memory = row['Version A (Original)_memory']
        
        b_time = row['Version B (Runtime)_time']
        c_memory = row['Version C (Memory)_memory']
        
        time_improvement = ((base_time - b_time) / base_time) * 100 if base_time > 0 else 0
        memory_improvement = ((base_memory - c_memory) / base_memory) * 100 if base_memory > 0 else 0
        
        print(f"\nSize {size}:")
        print(f"  Version B runtime improvement: {time_improvement:.1f}%")
        print(f"  Version C memory improvement: {memory_improvement:.1f}%")
    
    return results


def generate_markdown_report(results: list) -> str:
    """Generate markdown report from benchmark results."""
    report = """# Benchmark Analysis Report

## Test Configuration
- **Iterations per measurement**: 50
- **Metrics**: Execution time (ms), Peak memory usage (KB)
- **Test data**: Random binary bits and sine wave signal

## Results

### Execution Time Comparison

| Data Size | Version A (ms) | Version B (ms) | Version C (ms) | B vs A (%) |
|-----------|----------------|----------------|----------------|------------|
"""
    
    for row in results:
        size = row['size']
        a_time = row['Version A (Original)_time']
        b_time = row['Version B (Runtime)_time']
        c_time = row['Version C (Memory)_time']
        improvement = ((a_time - b_time) / a_time) * 100 if a_time > 0 else 0
        
        report += f"| {size} | {a_time:.3f} | {b_time:.3f} | {c_time:.3f} | {improvement:+.1f}% |\n"
    
    report += """
### Memory Usage Comparison

| Data Size | Version A (KB) | Version B (KB) | Version C (KB) | C vs A (%) |
|-----------|----------------|----------------|----------------|------------|
"""
    
    for row in results:
        size = row['size']
        a_mem = row['Version A (Original)_memory']
        b_mem = row['Version B (Runtime)_memory']
        c_mem = row['Version C (Memory)_memory']
        improvement = ((a_mem - c_mem) / a_mem) * 100 if a_mem > 0 else 0
        
        report += f"| {size} | {a_mem:.2f} | {b_mem:.2f} | {c_mem:.2f} | {improvement:+.1f}% |\n"
    
    report += """
## Conclusions

1. **Version B (Runtime Optimized)**: Achieves significant speed improvements through NumPy vectorization, eliminating Python loops in favor of optimized C operations.

2. **Version C (Memory Optimized)**: Reduces memory footprint by using float32 instead of float64 (50% reduction) and avoiding intermediate array creation.

3. **Trade-offs**: Version B prioritizes speed but may use more memory for intermediate arrays. Version C balances memory efficiency with readability through comprehensive documentation.

## Recommendations

- For **large datasets**: Use Version B for fastest processing
- For **memory-constrained environments**: Use Version C
- For **production code**: Version C offers the best balance of efficiency and maintainability
"""
    
    return report


if __name__ == "__main__":
    results = run_benchmarks([100, 500, 1000, 5000])
    
    # Generate and save markdown report
    report = generate_markdown_report(results)
    
    report_path = os.path.join(os.path.dirname(__file__), '..', 'report', 'benchmark_results.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n\nReport saved to: {report_path}")
