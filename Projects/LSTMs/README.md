# LSTMs from Scratch

Part of the "Build in Public" series deconstructing core AI architectures from first principles in pure PyTorch.

This project implements Long Short-Term Memory (LSTM) networks from the ground up—the architecture that solved the vanishing gradient problem and enabled learning long-range temporal dependencies.

## Structure

*   `lstm.py`: The core mathematical operations (LSTM cell with forget/input/output gates, multi-layer LSTM).
*   `lstm_architectures.py`: Complete architectures including LSTMClassifier, LSTMTagger, CharLSTM for text generation, and Seq2SeqLSTM for sequence-to-sequence tasks.
*   `train.py`: Training loop for sequence classification with long-range dependencies, along with visualization code to inspect gate activations.

## Running the Code

1.  Make sure you have PyTorch, NumPy, and Matplotlib installed.
2.  Run the training script (trains on synthetic sequence data with long-range dependencies):
    ```bash
    python train.py
    ```
    This will train the LSTM for 50 epochs.
3.  Upon completion, it saves:
    *   `train.log`: Training progression with loss and accuracy metrics.
    *   `training_curves.png`: Learning curves showing train/test loss and accuracy over epochs.
    *   `gate_activations.png`: Visualization of LSTM gate activations (input, forget, output, cell candidate) over time.

## The Series

*   **Part 1: The Math of Gated Recurrence** — Understanding the vanishing gradient problem, LSTM cell architecture, and how gates control information flow.
*   **Part 2: Pure PyTorch Implementation** — Building LSTM cells, multi-layer LSTMs, and sequence-to-sequence architectures from scratch.
*   **Part 3: Training and Analyzing Gates** — Benchmarking on long-range dependency tasks and visualizing how gates learn to remember and forget.
