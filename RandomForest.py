import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import resample


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, max_features='auto', regression=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.regression = regression
        self.trees = []

    def fit(self, X, y):
        num_samples, num_features = X.shape
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            X_sample, y_sample = resample(X, y, n_samples=num_samples)

            # Determine the number of features to consider for each split
            if self.max_features == 'auto':
                max_features = int(np.sqrt(num_features))  # Default for classification
            elif self.max_features == 'sqrt':
                max_features = int(np.sqrt(num_features))
            elif self.max_features == 'log2':
                max_features = int(np.log2(num_features))
            else:
                max_features = self.max_features

            # Randomly select features
            feature_indices = np.random.choice(num_features, max_features, replace=False)
            X_sample = X_sample[:, feature_indices]

            # Fit a decision tree on the bootstrap sample
            if self.regression:
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
            else:
                tree = DecisionTreeClassifier(max_depth=self.max_depth)

            tree.fit(X_sample, y_sample)
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        # Aggregate predictions from all trees
        if self.regression:
            # For regression, average the predictions
            predictions = np.zeros(X.shape[0])
            for tree, feature_indices in self.trees:
                X_subset = X[:, feature_indices]
                predictions += tree.predict(X_subset)
            return predictions / len(self.trees)
        else:
            # For classification, use majority vote
            predictions = np.zeros((X.shape[0], len(self.trees)))
            for i, (tree, feature_indices) in enumerate(self.trees):
                X_subset = X[:, feature_indices]
                predictions[:, i] = tree.predict(X_subset)
            # Use majority voting for classification
            majority_vote = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)
            return majority_vote


# Example usage
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    # Create a simple dataset
    X = np.array([
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
        [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0], [9.0, 3.0],
    ])
    y = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the RandomForest model
    rf_model = RandomForest(n_estimators=10, max_depth=3, max_features='auto', regression=False)
    rf_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("Predictions:", y_pred)
    print("Evaluation metrics:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
