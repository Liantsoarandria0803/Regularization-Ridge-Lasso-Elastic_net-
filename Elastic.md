# Elastic Net Regression with Synthetic Data

Elastic Net combines **L1 (Lasso)** and **L2 (Ridge)** regularization, making it a robust algorithm for regression problems where multicollinearity or feature selection is required.

---

## Features

- **Custom Elastic Net Implementation**: The model is implemented without using libraries like `scikit-learn` to provide a deeper understanding of the algorithm.
- **Synthetic Dataset**: The dataset is generated programmatically, simulating a real-world regression scenario.
- **Gradient Descent Optimization**: The model uses gradient descent to optimize weights and biases.
- **Visualization**: Includes a plot showing how the model fits the test data.

---

## Prerequisites

The following Python libraries are required to run this code:

- `numpy`: For numerical computations.
- `matplotlib`: For plotting and visualization.
- `scikit-learn`: For splitting the dataset into training and testing sets.

Install them using:
```bash
pip install numpy matplotlib scikit-learn
```

---

## Code Structure

### `ElasticRegression` Class

1. **Initialization (`__init__`)**:
   - `learning_rate`: Learning rate for gradient descent.
   - `iterations`: Number of gradient descent steps.
   - `l1_penality`: Coefficient for L1 regularization.
   - `l2_penality`: Coefficient for L2 regularization.

2. **Methods**:
   - `fit(X, Y)`: Trains the model on the given data.
   - `update_weights()`: Computes gradients and updates weights and bias.
   - `predict(X)`: Predicts the output for the given input features.

### Dataset
The dataset is synthesized as follows:
- `X`: Random values between 0 and 10 representing "Years of Experience".
- `Y`: Linear relationship with noise, \( Y = 2.5X + \text{random noise} + 30 \), simulating "Salary".

---



## Example Output

```
Predicted values: [54.23 72.11 66.34]
Real values    : [50.12 75.01 68.45]
Trained W      : 2.47
Trained b      : 31.22
```

The corresponding plot will show how well the regression line fits the test data.

---

## Visualization

The plot provides insight into the model's performance:

- **Blue Points**: Actual test data.
- **Orange Line**: Predicted regression line.

---

## Limitations

- The implementation assumes a simple single-feature dataset. For multi-feature datasets, additional scaling or preprocessing may be required.
- Performance depends on the choice of hyperparameters (`learning_rate`, `l1_penality`, `l2_penality`).

---

## Future Enhancements

- Extend the implementation to handle multi-feature datasets.
- Add support for feature scaling and normalization.
- Compare performance with `scikit-learn`'s ElasticNet for benchmarking.

---

## License

This project is open-source and available under the [MIT License](LICENSE).