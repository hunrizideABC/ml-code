import numpy as np

class CARTNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        A node in the CART tree.
        :param feature: Feature index used for splitting
        :param threshold: Threshold value for splitting
        :param left: Left child node
        :param right: Right child node
        :param value: Value for leaf nodes
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class CART:
    def __init__(self, max_depth=None):
        """
        CART Tree
        :param max_depth: Maximum depth of the tree
        """
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        """
        Train the CART model
        :param X: Training data, shape (n_samples, n_features)
        :param y: Training labels, shape (n_samples,)
        """
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """
        Recursively build the tree
        :param X: Training data
        :param y: Training labels
        :param depth: Current depth of the tree
        :return: Tree node
        """
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)

        # If all labels are the same, return a leaf node
        if len(unique_labels) == 1:
            return CARTNode(value=unique_labels[0])

        # If max depth is reached, return a leaf node with the most common label
        if self.max_depth is not None and depth >= self.max_depth:
            most_common_label = np.bincount(y).argmax()
            return CARTNode(value=most_common_label)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            # No valid split found; return a leaf node with the most common label
            most_common_label = np.bincount(y).argmax()
            return CARTNode(value=most_common_label)

        # Recursively build the left and right subtrees
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        left_node = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_node = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return CARTNode(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

    def _best_split(self, X, y):
        """
        Find the best feature and threshold for splitting
        :param X: Training data
        :param y: Training labels
        :return: Best feature index and threshold
        """
        best_feature = None
        best_threshold = None
        best_score = float('inf')
        num_samples, num_features = X.shape
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold

                if len(np.unique(y[left_indices])) == 0 or len(np.unique(y[right_indices])) == 0:
                    continue

                score = self._gini_index(y[left_indices], y[right_indices])
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y, right_y):
        """
        Compute Gini index for a split
        :param left_y: Labels for the left split
        :param right_y: Labels for the right split
        :return: Gini index
        """
        left_size = len(left_y)
        right_size = len(right_y)
        total_size = left_size + right_size

        def gini(y):
            proportions = np.bincount(y) / len(y)
            return 1 - np.sum(proportions**2)

        left_gini = gini(left_y)
        right_gini = gini(right_y)

        return (left_size / total_size) * left_gini + (right_size / total_size) * right_gini

    def predict(self, X):
        """
        Predict using the trained model
        :param X: Test data, shape (n_samples, n_features)
        :return: Predicted labels, shape (n_samples,)
        """
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, node):
        """
        Predict a single sample
        :param x: Sample data
        :param node: Current tree node
        :return: Predicted label
        """
        if node.value is not None:
            return node.value

        if x[node.feature] < node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

# Test CART implementation
if __name__ == "__main__":
    # Create a simple dataset
    X_train = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [8.0, 8.0],
        [1.0, 0.6],
        [9.0, 11.0],
        [8.0, 2.0],
        [10.0, 2.0],
        [9.0, 3.0],
    ])
    y_train = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1])

    X_test = np.array([
        [1.0, 1.0],
        [7.0, 7.0],
    ])
    y_test = np.array([0, 1])

    # Train and predict using CART model
    cart = CART(max_depth=3)
    cart.fit(X_train, y_train)
    y_pred = cart.predict(X_test)

    # Print predictions
    print("Predictions:", y_pred)

    # Evaluate the model
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("Evaluation metrics:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
