import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Elastic Net Regression
class ElasticRegression:
    def __init__(self, learning_rate, iterations, l1_penality, l2_penality):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penality = l1_penality
        self.l2_penality = l2_penality

    # Function for model training
    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # Gradient descent learning
        for i in range(self.iterations):
            self.update_weights()

        return self

    # Helper function to update weights in gradient descent
    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = np.zeros(self.n)

        for j in range(self.n):
            if self.W[j] > 0:
                dW[j] = (
                    -2 * (self.X[:, j].dot(self.Y - Y_pred))
                    + self.l1_penality
                    + 2 * self.l2_penality * self.W[j]
                ) / self.m
            else:
                dW[j] = (
                    -2 * (self.X[:, j].dot(self.Y - Y_pred))
                    - self.l1_penality
                    + 2 * self.l2_penality * self.W[j]
                ) / self.m

        db = -2 * np.sum(self.Y - Y_pred) / self.m

        # Update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return self

    # Hypothetical function h(x)
    def predict(self, X):
        return X.dot(self.W) + self.b

# Driver Code
def main():
    # Generating synthetic dataset
    np.random.seed(0)
    X = np.random.rand(100, 1) * 10  # Random years of experience
    Y = 2.5 * X.flatten() + np.random.randn(100) * 5 + 30 

    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.33, random_state=42
    )

    # Model training
    model = ElasticRegression(
        iterations=1000, learning_rate=0.01, l1_penality=500, l2_penality=1
    )

    model.fit(X_train, Y_train)

    # Prediction on test set
    Y_pred = model.predict(X_test)

    print("Predicted values:", np.round(Y_pred[:3], 2))
    print("Real values    :", Y_test[:3])
    print("Trained W      :", round(model.W[0], 2))
    print("Trained b      :", round(model.b, 2))

    # Visualization on test set
    plt.scatter(X_test, Y_test, color="blue", label="Actual")
    plt.plot(X_test, Y_pred, color="orange", label="Predicted")
    plt.title("Salary vs Experience (Synthetic Data)")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
