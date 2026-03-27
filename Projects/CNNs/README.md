# CNNs from Scratch

Part of the "Build in Public" series deconstructing core AI architectures from first principles in pure PyTorch.

This project implements Convolutional Neural Networks (CNNs) from the ground up—the architecture that revolutionized computer vision by learning spatial hierarchies of features through learnable filters.

## Structure

*   `convolution.py`: The core mathematical operations (2D Convolution, Max Pooling, Average Pooling, and common activation functions).
*   `cnn.py`: Complete architectures including LeNet-5, SimpleCNN, and a deeper VGG-style CNN with reusable blocks.
*   `train.py`: Training loop for MNIST digit classification, along with visualization code to inspect learned filters and feature maps.

## Running the Code

1.  Make sure you have PyTorch, torchvision, NumPy, and Matplotlib installed.
2.  Run the training script (trains on MNIST):
    ```bash
    python train.py
    ```
    This will train the CNN for 10 epochs.
3.  Upon completion, it saves:
    *   `train.log`: Training progression with loss and accuracy metrics.
    *   `training_curves.png`: Learning curves showing train/test loss and accuracy over epochs.
    *   `filters.png`: Visualization of learned convolutional filters from the first layer.
    *   `feature_maps.png`: Feature map activations showing what the network detects in input images.

## The Series

*   **Part 1: The Math of Convolutions** — Understanding how learnable filters slide across images to detect edges, textures, and patterns through local connectivity and weight sharing.
*   **Part 2: Pure PyTorch Implementation** — Building convolutional layers, pooling operations, and complete CNN architectures from scratch.
*   **Part 3: Training and Visualizing Features** — Benchmarking on MNIST, analyzing learned filters, and visualizing hierarchical feature representations.
