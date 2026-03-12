# Echo State Networks (ESNs) from Scratch

Part of the "Build in Public" series deconstructing core AI architectures from first principles in pure PyTorch.

This project implements an Echo State Network (ESN), the most famous form of Reservoir Computing. It demonstrates how a random, sparse recurrent reservoir can completely bypass the need for Backpropagation Through Time (BPTT) and train near-instantly using purely closed-form linear algebra (Ridge Regression).

## Structure
*   `esn_layer.py`: The core reservoir. Features random sparse initialization, exact Spectral Radius scaling, state harvesting (the Echo State Property), and closed-form Tikhonov regularization (Ridge Regression) for the readout layer.
*   `esn_benchmark.py`: Generates the highly chaotic Mackey-Glass time series, trains the ESN, trains a standard BPTT PyTorch LSTM, and plots their comparative forecasting abilities and training speeds.

## Running the Code
1.  Make sure you have PyTorch and Matplotlib installed.
2.  Run the benchmark:
    ```bash
    python esn_benchmark.py
    ```
3.  Upon completion, it saves:
    *   `esn_vs_lstm_forecast.png`: The visual forecasting comparison.
    *   `benchmark_results.txt`: The raw training times and Mean Squared Errors.

## The Series
*   **Part 1: The End of BPTT** — The math behind the Echo State Property, Spectral Radius, and why reservoirs work.
*   **Part 2: Pure PyTorch Implementation** — Building the sparse initialized random reservoir and doing the closed-form Ridge Regression readout.
*   **Part 3: Predicting Chaos** — Benchmarking the ESN against an LSTM on a chaotic time series to showcase the massive 1000x training speedup.
