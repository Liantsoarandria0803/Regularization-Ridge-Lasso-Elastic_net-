import numpy as np

class LinearRegression:
    def __init__(self):
        self.parameters = None  # Coefficients of the model

    def fit(self, X, y):
        """
        Fit the linear regression model using the normal equation.
        """
        self.parameters = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return self.parameters

    def predict(self, X):
        """
        Predict using the fitted linear regression model.
        """
        if self.parameters is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return X.dot(self.parameters)

    @staticmethod
    def calculate_loss(predictions, y):
        """
        Calculate the mean squared error.
        """
        return np.sum((predictions - y) ** 2)

class RidgeRegression(LinearRegression):
    def __init__(self, lamb=1.0):
        """
        Initialize the ridge regression with a regularization parameter.
        """
        super().__init__()
        self.lamb = lamb

    def fit(self, X, y):
        """
        Fit the ridge regression model using the normal equation.
        """
        num_features = X.shape[1]
        identity_matrix = np.identity(num_features)
        self.parameters = np.linalg.inv(X.T.dot(X) + self.lamb * identity_matrix).dot(X.T).dot(y)
        return self.parameters

# Data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
X = np.vstack((X, np.ones(X.shape[0]))).T  
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])

# Linear Regression
print("Linear Regression:")
linear_model = LinearRegression()
paramL = linear_model.fit(X, y)
print("Coefficients:", paramL)
predictions_linear = linear_model.predict(X)
print("Predictions:", predictions_linear)
loss_linear = LinearRegression.calculate_loss(predictions_linear, y)
print("Loss:", loss_linear)

# Ridge Regression
print("\nRidge Regression:")
ridge_model = RidgeRegression(lamb=1.0)
paramR = ridge_model.fit(X, y)
print("Coefficients:", paramR)
predictions_ridge = ridge_model.predict(X)
print("Predictions:", predictions_ridge)
loss_ridge = RidgeRegression.calculate_loss(predictions_ridge, y)
print("Loss:", loss_ridge)
