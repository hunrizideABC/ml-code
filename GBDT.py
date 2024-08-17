import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class GBDT:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Gradient Boosting Decision Trees (GBDT) for binary classification
        :param n_estimators: Number of trees
        :param learning_rate: Learning rate
        :param max_depth: Maximum depth of each tree
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        """
        Train the GBDT model
        :param X: Training data, shape (n_samples, n_features)
        :param y: Training labels, shape (n_samples,)
        """
        n_samples = X.shape[0]
        # Initialize predictions with 0
        y_pred = np.zeros(n_samples)
        self.models = []

        for _ in range(self.n_estimators):
            # Calculate residuals
            residual = y - y_pred
            # Train decision tree model
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, residual)
            # Add model to the list
            self.models.append(model)
            # Update predictions
            y_pred += self.learning_rate * model.predict(X)

    def predict(self, X):
        """
        Predict using the trained model
        :param X: Test data, shape (n_samples, n_features)
        :return: Predicted labels, shape (n_samples,)
        """
        y_pred = np.mean([model.predict(X) for model in self.models], axis=0)
        # Convert predictions to binary class (0 or 1)
        return np.round(y_pred)

    def evaluate(self, X, y):
        """
        Evaluate the model
        :param X: Test data, shape (n_samples, n_features)
        :param y: Test labels, shape (n_samples,)
        :return: Evaluation metrics dictionary
        """
        y_pred = self.predict(X)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        return {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Accuracy': accuracy
        }


# Test GBDT implementation
if __name__ == "__main__":
    # Create simple dataset
    X_train = np.array([
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
        [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0],
    ])
    y_train = np.array([0, 1, 1, 1, 0, 1, 1, 1, 1])

    X_test = np.array([
        [1.0, 2.1],
        [5.0, 8.0],
        [9.0, 11.0]
    ])
    y_test = np.array([1, 0, 0])
    # Train and predict using GBDT model
    gbdt = GBDT(n_estimators=50, learning_rate=0.1, max_depth=3)
    gbdt.fit(X_train, y_train)
    # Evaluate the model
    metrics = gbdt.evaluate(X_test, y_test)
    print("Evaluation metrics:", metrics)
