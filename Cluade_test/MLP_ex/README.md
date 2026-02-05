# MLP Regression Model with PyTorch

A Multi-Layer Perceptron (MLP) regression model implemented using PyTorch.

## Project Structure

```
MLP_ex/
├── regression_data.csv    # Training dataset
├── mlp_regression.py      # Main training script
├── mlp_model.pth          # Saved model weights
├── mlp_results.png        # Performance visualization
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch
- pandas
- numpy
- matplotlib
- scikit-learn

Install dependencies:
```bash
pip install torch pandas numpy matplotlib scikit-learn
```

## Usage

```bash
python mlp_regression.py
```

## Model Architecture

```
Input (4 features) → Dense(64) → ReLU → Dropout(0.1)
                   → Dense(32) → ReLU → Dropout(0.1)
                   → Dense(16) → ReLU → Dropout(0.1)
                   → Dense(1) → Output
```

## Performance

| Metric | Training | Test |
|--------|----------|------|
| MSE    | 0.0434   | 0.0534 |
| RMSE   | 0.2083   | 0.2310 |
| MAE    | 0.1565   | 0.1753 |
| R²     | 0.9969   | 0.9939 |

## Results

The model achieves excellent performance with R² = 0.9939 on the test set.

![Results](mlp_results.png)
