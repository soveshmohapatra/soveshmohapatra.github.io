# RNNs from Scratch

Part of the "Build in Public" series deconstructing core AI architectures from first principles in pure PyTorch.

This project implements Recurrent Neural Networks (RNNs) from the ground up—the architecture designed to process sequential data by maintaining an evolving hidden state that captures temporal context.

## Structure

*   `rnn.py`: The core mathematical operations (RNN cell, multi-layer RNN, and activation functions).
*   `rnn_architectures.py`: Complete architectures including SequenceClassifier, SequenceTagger, CharRNN for text generation, and BidirectionalRNN.
*   `train.py`: Training loop for sequence classification, along with visualization code to inspect hidden state dynamics over time.

## Running the Code

1.  Make sure you have PyTorch, NumPy, and Matplotlib installed.
2.  Run the training script (trains on synthetic sequence data):
    ```bash
    python train.py
    ```
    This will train the RNN for 50 epochs.
3.  Upon completion, it saves:
    *   `train.log`: Training progression with loss and accuracy metrics.
    *   `training_curves.png`: Learning curves showing train/test loss and accuracy over epochs.
    *   `hidden_states.png`: Visualization of hidden unit activations evolving over time steps.

## The Series

*   **Part 1: The Math of Recurrence** — Understanding how hidden states capture temporal dependencies, backpropagation through time (BPTT), and the vanishing gradient problem.
*   **Part 2: Pure PyTorch Implementation** — Building RNN cells, multi-layer RNNs, and sequence-to-sequence architectures from scratch.
*   **Part 3: Training and Analyzing Dynamics** — Benchmarking on sequence tasks and visualizing how hidden states evolve to encode temporal information.
