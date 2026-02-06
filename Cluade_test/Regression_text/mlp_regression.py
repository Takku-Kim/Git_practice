"""
MLP Regression Model using PyTorch
This script trains a Multi-Layer Perceptron for regression on the provided dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load data
data = pd.read_csv('regression_data.csv')
print("=" * 50)
print("Data Overview")
print("=" * 50)
print(f"Shape: {data.shape}")
print(f"\nFirst 5 rows:\n{data.head()}")
print(f"\nStatistics:\n{data.describe()}")

# Prepare features and target
X = data[['feature1', 'feature2', 'feature3', 'feature4']].values
y = data['target'].values.reshape(-1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# Define MLP Model
class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPRegressor, self).__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Model configuration
input_size = 4
hidden_sizes = [64, 32, 16]
output_size = 1

model = MLPRegressor(input_size, hidden_sizes, output_size)
print("\n" + "=" * 50)
print("Model Architecture")
print("=" * 50)
print(model)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

# Training
num_epochs = 500
train_losses = []
test_losses = []

print("\n" + "=" * 50)
print("Training Progress")
print("=" * 50)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor).item()
        test_losses.append(test_loss)

    scheduler.step(test_loss)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Final Evaluation
model.eval()
with torch.no_grad():
    y_train_pred_scaled = model(X_train_tensor).numpy()
    y_test_pred_scaled = model(X_test_tensor).numpy()

    # Inverse transform predictions
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

# Calculate metrics
print("\n" + "=" * 50)
print("Model Performance")
print("=" * 50)

# Training metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print("\nTraining Set Metrics:")
print(f"  MSE:  {train_mse:.4f}")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE:  {train_mae:.4f}")
print(f"  R²:   {train_r2:.4f}")

# Test metrics
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTest Set Metrics:")
print(f"  MSE:  {test_mse:.4f}")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE:  {test_mae:.4f}")
print(f"  R²:   {test_r2:.4f}")

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Training and Test Loss
axes[0, 0].plot(train_losses, label='Train Loss', alpha=0.8)
axes[0, 0].plot(test_losses, label='Test Loss', alpha=0.8)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Training and Test Loss Over Epochs')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Actual vs Predicted (Test Set)
axes[0, 1].scatter(y_test, y_test_pred, alpha=0.7, edgecolors='black', linewidth=0.5)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].set_title(f'Actual vs Predicted (Test Set)\nR² = {test_r2:.4f}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Residual Plot
residuals = y_test.flatten() - y_test_pred.flatten()
axes[1, 0].scatter(y_test_pred, residuals, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Residual Plot')
axes[1, 0].grid(True, alpha=0.3)

# 4. Residual Distribution
axes[1, 1].hist(residuals, bins=10, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Residual Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'Residual Distribution\nMean: {residuals.mean():.4f}, Std: {residuals.std():.4f}')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mlp_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("Results saved to 'mlp_results.png'")
print("=" * 50)

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'model_config': {
        'input_size': input_size,
        'hidden_sizes': hidden_sizes,
        'output_size': output_size
    }
}, 'mlp_model.pth')
print("Model saved to 'mlp_model.pth'")
