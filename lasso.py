import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None  # Model coefficients
        self.intercept_ = 0  # Intercept term

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = np.mean(y)
        y = y - self.intercept_
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            for j in range(n_features):
                # Calculate the residual without the current feature
                residual = y - X.dot(self.coef_) + X[:, j] * self.coef_[j]
                # Compute rho
                rho = np.dot(X[:, j], residual)

                # Soft-thresholding operator
                if rho < -self.alpha:
                    self.coef_[j] = (rho + self.alpha) / np.sum(X[:, j] ** 2)
                elif rho > self.alpha:
                    self.coef_[j] = (rho - self.alpha) / np.sum(X[:, j] ** 2)
                else:
                    self.coef_[j] = 0

            # Check convergence
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                break

    def predict(self, X):
        return X.dot(self.coef_) + self.intercept_

    @staticmethod
    def calculate_loss(y_true, y_pred): #least square
        return np.mean((y_true - y_pred) ** 2)

# Data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
X = np.vstack((X, np.ones(X.shape[0]))).T  # Add a bias term
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])

# Lasso Regression
lasso = LassoRegression(alpha=1.0, max_iter=1000, tol=1e-4)
lasso.fit(X, y)

# Predictions and Loss
predictions = lasso.predict(X)
loss = LassoRegression.calculate_loss(y, predictions)

print("Lasso Coefficients:", lasso.coef_)
print("Intercept:", lasso.intercept_)
print("Predictions:", predictions)
print("Loss:", loss)
