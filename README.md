# Linear Regression with Regularization Techniques

This repository demonstrates **Linear Regression** using various regularization techniques to handle overfitting, feature selection, and multicollinearity. It includes implementations of:

1. **Ridge Regression**: Adds L2 regularization.
2. **Lasso Regression**: Adds L1 regularization.
3. **Elastic Net Regression**: Combines L1 and L2 regularization.

Each method is implemented from scratch in Python with detailed explanations and practical usage examples.

---

## Techniques Explained

### 1. Ridge Regression
- Adds a **penalty term proportional to the square of the coefficients (L2)**.
- Helps to reduce model complexity by shrinking coefficients closer to zero but not exactly zero.
- Formula:  
  \[
  J(\theta) = \text{MSE} + \lambda \sum \theta_j^2
  \]

### 2. Lasso Regression
- Adds a **penalty term proportional to the absolute value of the coefficients (L1)**.
- Performs both regularization and feature selection by driving some coefficients exactly to zero.
- Formula:  
  \[
  J(\theta) = \text{MSE} + \lambda \sum |\theta_j|
  \]

### 3. Elastic Net Regression
- Combines **L1 and L2 penalties**, balancing feature selection and coefficient shrinkage.
- Formula:  
  \[
  J(\theta) = \text{MSE} + \alpha \left( \rho \sum |\theta_j| + (1-\rho) \sum \theta_j^2 \right)
  \]
  Where:
  - \(\alpha\): Overall regularization strength.
  - \(\rho\): L1 ratio (0 = Ridge, 1 = Lasso).

---

## Code Structure

1. **Ridge Regression**:
   - Uses closed-form solution with regularization term.
   - Implemented in `ridge_regression.py`.

2. **Lasso Regression**:
   - Uses coordinate descent with the soft-thresholding operator.
   - Implemented in `lasso_regression.py`.

3. **Elastic Net Regression**:
   - Gradient descent optimization for combined penalties.
   - Implemented in `elastic_net_regression.py`.

---

## Prerequisites

Install the required libraries:
```bash
pip install numpy pandas matplotlib scikit-learn
```

---


## Outputs

### Ridge Regression
- Shrinks coefficients to avoid overfitting.
- Loss is minimized without driving coefficients to zero.

### Lasso Regression
- Performs feature selection by setting some coefficients to zero.
- Useful when some features are irrelevant.

### Elastic Net Regression
- Balances the strengths of Ridge and Lasso.
- Handles multicollinearity better than Lasso.

## Applications

- **Ridge Regression**: Ideal for datasets with multicollinearity.
- **Lasso Regression**: Useful for feature selection.
- **Elastic Net Regression**: Handles multicollinearity while performing feature selection.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Future Enhancements

- Add cross-validation for hyperparameter tuning.
- Extend the implementation to multi-target regression.
- Compare results with library implementations (`scikit-learn`).
