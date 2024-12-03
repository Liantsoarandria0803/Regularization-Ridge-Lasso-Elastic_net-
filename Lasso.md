# Lasso Regression Implementation

 Lasso Regression is a linear regression method that uses \( L_1 \)-norm regularization to shrink some coefficients to zero, enabling feature selection and preventing overfitting.

## Table of Contents
1. [Introduction](#introduction)
2. [How It Works](#how-it-works)
3. [Code Explanation](#code-explanation)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Output](#output)
7. [License](#license)

---

## Introduction

Lasso Regression minimizes the sum of squared residuals and applies an \( L_1 \)-norm penalty to the coefficients:
\[
\text{Loss} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} |\beta_j|
\]

This penalty helps in feature selection by reducing some coefficients to exactly zero.

---

## How It Works

The implementation uses **coordinate descent**:
1. Iteratively optimizes each coefficient while keeping others fixed.
2. Applies a **soft-thresholding operator** to compute the coefficients.
3. Stops when the coefficients converge within a given tolerance.

---

## Code Explanation

### 1. **Initialization**
The class `LassoRegression` is initialized with:
- `alpha`: Regularization strength (\( \lambda \)).
- `max_iter`: Maximum number of iterations.
- `tol`: Tolerance level for convergence.

### 2. **Fitting the Model**
The `fit()` method:
- Centers the target variable \( y \) by subtracting its mean.
- Initializes coefficients to zero.
- Iteratively optimizes each coefficient using the soft-thresholding operator:
  \[
  \beta_j = \text{sign}(\rho_j) \cdot \max(0, |\rho_j| - \alpha) / \sum X_j^2
  \]

### 3. **Prediction**
The `predict()` method calculates predictions:
\[
\hat{y} = X \cdot \text{coef\_} + \text{intercept\_}
\]

### 4. **Loss Calculation**
The `calculate_loss()` method computes the mean squared error (MSE):
\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/lasso-regression.git
   cd lasso-regression
   ```
2. Install required dependencies:
   ```bash
   pip install numpy
   ```

---

## Usage

1. Prepare your data:
   ```python
   X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
   X = np.vstack((X, np.ones(X.shape[0]))).T  # Add a bias term
   y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])
   ```

2. Create and fit the model:
   ```python
   lasso = LassoRegression(alpha=1.0, max_iter=1000, tol=1e-4)
   lasso.fit(X, y)
   ```

3. Make predictions and calculate the loss:
   ```python
   predictions = lasso.predict(X)
   loss = LassoRegression.calculate_loss(y, predictions)
   ```

4. View the results:
   ```python
   print("Lasso Coefficients:", lasso.coef_)
   print("Intercept:", lasso.intercept_)
   print("Predictions:", predictions)
   print("Loss:", loss)
   ```

---

## Output

Example output for the given data:
```
Lasso Coefficients: [2. 0.]
Intercept: 10.0
Predictions: [ 2.  4.  6.  8. 10. 12. 14. 16. 18.]
Loss: 0.0
```

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute it as needed.

---

Feel free to contribute or raise issues for improvements.