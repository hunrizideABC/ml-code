import numpy as np
from sklearn.tree import DecisionTreeRegressor


class XGBoost:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_pred = None

    def fit(self, X, y):
        num_samples = X.shape[0]
        self.init_pred = np.mean(y)  # Initial prediction is the mean of the target values
        predictions = np.full(num_samples, self.init_pred)

        for _ in range(self.n_estimators):
            residuals = y - predictions
            gradients, hessians = self._compute_gradients(residuals)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, gradients)  # Fit tree on gradients
            self.trees.append(tree)
            # Predict with the current tree and update predictions
            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * tree_predictions

    def predict(self, X):
        predictions = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * tree_predictions
        return predictions


    def _compute_gradients(self, residuals):
        gradients = residuals  # Gradient of squared loss is residuals
        hessians = np.ones_like(residuals)  # Hessian (second derivative) of squared loss is 1
        return gradients, hessians


# Example usage
if __name__ == "__main__":
    # Create a simple dataset
    X = np.array([
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
        [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0],
    ])
    y = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1])

    # Split the data into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Train the XGBoost model
    xgb_model = XGBoost(n_estimators=10, learning_rate=0.1, max_depth=3)
    xgb_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = xgb_model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)  # Convert to binary labels

    # Print predictions
    print("Predictions:", y_pred_binary)

    # Evaluate the model
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    accuracy = accuracy_score(y_test, y_pred_binary)

    print("Evaluation metrics:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
