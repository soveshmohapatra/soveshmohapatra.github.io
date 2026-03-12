import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import math

from esn_layer import EchoStateNetwork

# ---------------------------------------------------------
# 1. Mackey-Glass Time Series Generator
# ---------------------------------------------------------
def generate_mackey_glass(length=2000, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0):
    """
    Simulates the Mackey-Glass chaotic time series.
    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
    """
    history_length = int(tau / dt)
    # Initialize history with small random values between 0.5 and 1.5
    x = torch.rand(length + history_length) + 0.5 
    
    for t in range(history_length, length + history_length - 1):
        x_tau = x[t - history_length]
        dx = (beta * x_tau / (1.0 + x_tau ** n)) - (gamma * x[t])
        x[t + 1] = x[t] + dx * dt
        
    # Discard warmup
    return x[history_length:].unsqueeze(-1)


# ---------------------------------------------------------
# 2. Standard LSTM for Baseline Comparison
# ---------------------------------------------------------
class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


# ---------------------------------------------------------
# 3. Main Benchmark Execution
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Generating Mackey-Glass dataset...")
    data = generate_mackey_glass(length=3000)
    
    # Scale data to roughly [-1, 1] for better neural network behavior
    mean, std = data.mean(), data.std()
    data = (data - mean) / std
    
    # Split into train/test
    # We'll use the first 2000 steps to train, and forecast the next 1000
    TRAIN_LEN = 2000
    TEST_LEN = 1000
    
    train_data = data[:TRAIN_LEN]
    test_data = data[TRAIN_LEN:TRAIN_LEN+TEST_LEN]
    
    # The task: given data up to time t, predict data at time t+1
    # For sequence models, input is x[0:T-1], target is x[1:T]
    u_train = train_data[:-1]
    y_train = train_data[1:]
    
    u_test = test_data[:-1]
    y_test = test_data[1:]
    
    print("\n--- Echo State Network (ESN) ---")
    esn = EchoStateNetwork(
        input_size=1, 
        hidden_size=500,  # A large reservoir is computationally cheap!
        output_size=1, 
        spectral_radius=0.9,
        sparsity=0.05
    )
    
    # Measure ESN Training Time
    start_time = time.time()
    # Fit uses closed-form Ridge Regression. No gradient descent!
    esn.fit(u_train, y_train, washout=100)
    esn_train_time = time.time() - start_time
    
    # Evaluate ESN
    esn_train_pred, _ = esn(u_train)
    esn_test_pred, _ = esn(u_test)
    esn_train_mse = nn.functional.mse_loss(esn_train_pred[100:], y_train[100:]).item()
    esn_test_mse = nn.functional.mse_loss(esn_test_pred, y_test).item()
    
    print(f"ESN Training Time: {esn_train_time*1000:.2f} ms")
    print(f"ESN Train MSE: {esn_train_mse:.6f}")
    print(f"ESN Test MSE:  {esn_test_mse:.6f}")
    
    
    print("\n--- Standard LSTM (BPTT) ---")
    # For a fair comparison, we use a much smaller hidden size so parameters are comparable,
    # or we just show how slow BPTT is even on a small network.
    lstm = StandardLSTM(input_size=1, hidden_size=64, output_size=1)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)
    
    # LSTMs expect batch dimension
    lstm_u_train = u_train.unsqueeze(0)
    lstm_y_train = y_train.unsqueeze(0)
    lstm_u_test = u_test.unsqueeze(0)
    lstm_y_test = y_test.unsqueeze(0)
    
    # Measure LSTM Training Time
    start_time = time.time()
    epochs = 150
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = lstm(lstm_u_train)
        loss = nn.functional.mse_loss(pred[:, 100:], lstm_y_train[:, 100:])
        loss.backward()
        optimizer.step()
    lstm_train_time = time.time() - start_time
    
    # Evaluate LSTM
    lstm.eval()
    with torch.no_grad():
        lstm_train_pred = lstm(lstm_u_train).squeeze(0)
        lstm_test_pred = lstm(lstm_u_test).squeeze(0)
        lstm_train_mse = nn.functional.mse_loss(lstm_train_pred[100:], y_train[100:]).item()
        lstm_test_mse = nn.functional.mse_loss(lstm_test_pred, y_test).item()
        
    print(f"LSTM Training Time ({epochs} epochs): {lstm_train_time:.2f} seconds")
    print(f"LSTM Train MSE: {lstm_train_mse:.6f}")
    print(f"LSTM Test MSE:  {lstm_test_mse:.6f}")
    
    # ---------------------------------------------------------
    # 4. Save and Plot
    # ---------------------------------------------------------
    
    # Save Results to File for cross-checking
    with open("benchmark_results.txt", "w") as f:
        f.write("[ESN]\n")
        f.write(f"Train_Time_ms: {esn_train_time*1000:.2f}\n")
        f.write(f"Train_MSE: {esn_train_mse:.6f}\n")
        f.write(f"Test_MSE: {esn_test_mse:.6f}\n\n")
        f.write("[LSTM]\n")
        f.write(f"Train_Time_sec: {lstm_train_time:.2f}\n")
        f.write(f"Train_MSE: {lstm_train_mse:.6f}\n")
        f.write(f"Test_MSE: {lstm_test_mse:.6f}\n")
    print("\nSaved benchmark_results.txt")

    # Generate Plot
    plt.figure(figsize=(14, 8))
    
    # We'll plot the first 500 steps of the test set for clear visibility
    plot_len = 500
    t_axis = range(plot_len)
    
    plt.subplot(2, 1, 1)
    plt.plot(t_axis, y_test[:plot_len].numpy(), label="True Mackey-Glass", color='black', alpha=0.6, linewidth=2)
    plt.plot(t_axis, esn_test_pred[:plot_len].numpy(), label="ESN Prediction", color='#1565C0', linestyle='--', linewidth=2)
    plt.title(f"Echo State Network Forecast (MSE: {esn_test_mse:.4f} | Train Time: {esn_train_time*1000:.0f} ms)", fontweight='bold')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(t_axis, y_test[:plot_len].numpy(), label="True Mackey-Glass", color='black', alpha=0.6, linewidth=2)
    plt.plot(t_axis, lstm_test_pred[:plot_len].numpy(), label="LSTM Prediction", color='#EE4C2C', linestyle='--', linewidth=2)
    plt.title(f"Standard LSTM Forecast (MSE: {lstm_test_mse:.4f} | Train Time: {lstm_train_time:.1f} s)", fontweight='bold')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('esn_vs_lstm_forecast.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated esn_vs_lstm_forecast.png")
