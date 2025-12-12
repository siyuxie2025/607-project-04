# Test parallelization speedup explicitly
import time
from joblib import Parallel, delayed
from simulation import SimulationStudy
from generators import TGenerator, TruncatedNormalGenerator

def run_one():
    """Run a single simulation"""
    config = {
        'n_sim': 1, 'K': 2, 'd': 10, 'T': 150,
        'q': 2, 'h': 0.5, 'tau': 0.5,
        'err_generator': TGenerator(df=2.25, scale=0.7),
        'context_generator': TruncatedNormalGenerator(0, 1),
    }
    study = SimulationStudy(**config)
    return study.run_simulation()

# Sequential
print("Sequential (10 runs):")
start = time.time()
for _ in range(10):
    run_one()
seq_time = time.time() - start
print(f"  Time: {seq_time:.2f}s")

# Parallel
print("\nParallel (10 runs, 4 cores):")
start = time.time()
Parallel(n_jobs=4)(delayed(run_one)() for _ in range(10))
par_time = time.time() - start
print(f"  Time: {par_time:.2f}s")
print(f"  Speedup: {seq_time/par_time:.2f}×")