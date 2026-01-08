# FlashCorr

**An Ultra-Low Latency Algorithm for Base Correlation Calibration in CDS Index Tranches**

FlashCorr is a high-performance, analytically-driven algorithm that calibrates base correlations for credit index tranches (e.g., CDX.NA.IG, iTraxx Europe) using the Gaussian copula in the Large Homogeneous Portfolio (LHP) approximation. By reformulating the expected tranche loss into a closed-form bivariate normal expression and exploiting analytical cancellations in the derivatives, FlashCorr achieves **over 100× speedup** compared to traditional numerical quadrature + finite-difference approaches while converging to machine precision in typically **2–3 iterations**.

## Key Features

- **Closed-form Expected Loss** using Drezner’s bivariate normal CDF (via QuantLib).
- **Exact analytical gradient** ("free gradient") derived from Plackett’s identity and a proven vanishing boundary term.
- **Variance-stabilizing transformation** (`z = arctanh(√ρ)`) for robust convergence.
- **Halley’s method** with analytical gradient + finite-difference Hessian for cubic convergence.
- Average **2.2 iterations** and **~5.5 μs** per calibration on a single Intel i9 core.
- Fully deterministic, no Monte Carlo noise.
- Apahce-2.0 licensed, production-ready C++17 implementation.

## Benchmark Summary

| Metric                  | Traditional (Quadrature + FD) | **FlashCorr**      |
|-------------------------|-------------------------------|--------------------|
| Gradient method         | Finite difference (bump)      | Analytical (Plackett) |
| Iteration scheme        | Secant / Brent                | Halley (cubic)     |
| Avg. iterations         | ~12.5                         | **~2.2**           |
| Time per solve          | ~650 μs                       | **~5.5 μs**        |
| Speedup                 | 1×                            | **>100×**          |

## Installation

FlashCorr depends on **QuantLib** (>= 1.28 recommended) for accurate bivariate normal evaluations.

```bash
# Clone the repository
git clone https://github.com/jialuechen/flashcorr.git
cd flashcorr

# Build with CMake (out-of-source build recommended)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)