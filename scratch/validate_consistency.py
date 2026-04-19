
import subprocess
import re
import collections
import numpy as np

def run_benchmark(seed):
    try:
        # Run with the specific seed
        cmd = ['python', 'benchmark.py', '--seed', str(seed), '--no-sim-latency']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse success rate
        match = re.search(r'V2 Success:\s+(\d+\.\d+)%', result.stdout)
        if match:
            return float(match.group(1)), result.stdout
        return None, result.stdout
    except Exception as e:
        print(f"Error running benchmark with seed {seed}: {e}")
        return None, ""

def main():
    seeds = [42, 123, 7, 999, 2024, 1, 13, 101, 888, 555] # 10 different seeds
    success_rates = []
    
    print(f"Starting {len(seeds)}-run statistical validation (Multi-Seed)...")
    for i, seed in enumerate(seeds):
        print(f"Run {i+1}/{len(seeds)} (Seed {seed})...", end="", flush=True)
        rate, output = run_benchmark(seed)
        if rate is not None:
            success_rates.append(rate)
            print(f" {rate}%")
        else:
            print(" Failed to parse output")
    
    if success_rates:
        print("\nStatistical Summary:")
        print(f"  Mean Success Rate: {np.mean(success_rates):.2f}%")
        print(f"  Std Deviation:     {np.std(success_rates):.2f}%")
        print(f"  Min/Max:           {min(success_rates)}% / {max(success_rates)}%")
        
        # Distribution
        print("\nDistribution:")
        counter = collections.Counter(success_rates)
        for rate, count in sorted(counter.items()):
            print(f"  {rate}%: {count} times")

if __name__ == "__main__":
    main()
