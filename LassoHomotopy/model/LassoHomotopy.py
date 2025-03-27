import numpy as np
from scipy.linalg import pinvh

class LassoHomotopyModel:
    """
    A solver for linear regression with L1 regularization using the pathwise optimization approach.
    """

    def __init__(self, reg_param=1.0, convergence_threshold=1e-6, maximum_iterations=1000):
        """
        Initialize the sparse linear model solver.
        
        Args:
            reg_param: Strength of L1 regularization
            convergence_threshold: Tolerance for stopping condition
            maximum_iterations: Limit for optimization steps
        """
        self.reg_param = reg_param
        self.convergence_threshold = convergence_threshold
        self.maximum_iterations = maximum_iterations
        self.weights = None
        self.bias = None
        self.selected_features = []
        self.errors = None
        self.active_matrix = None

    def train(self, features, targets):
        """
        Fit the model to training data using the homotopy continuation method.
        
        Args:
            features: Input data matrix (samples x features)
            targets: Output values to predict
        Returns:
            Trained model results object
        """
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64).flatten()

        num_samples, num_features = features.shape

        # Include intercept term
        features = np.column_stack([np.ones(num_samples), features])

        # Initialize model parameters
        self.weights = np.zeros(num_features + 1)
        self.errors = targets.copy()

        current_iteration = 0
        while current_iteration < self.maximum_iterations:
            # Calculate feature correlations with residuals
            feature_correlations = features.T @ self.errors

            # Find most correlated feature
            strongest_feature = np.argmax(np.abs(feature_correlations))
            strongest_correlation = feature_correlations[strongest_feature]

            # Check stopping condition
            if np.abs(strongest_correlation) < self.reg_param + self.convergence_threshold:
                break

            # Update active feature set
            if strongest_feature not in self.selected_features:
                self.selected_features.append(strongest_feature)

            # Solve restricted least squares problem
            active_features = features[:, self.selected_features]
            self.active_matrix = active_features
            active_weights = pinvh(active_features.T @ active_features) @ (active_features.T @ targets)

            # Update model parameters
            for i, w in zip(self.selected_features, active_weights):
                self.weights[i] = w

            # Recompute residuals
            self.errors = targets - features @ self.weights

            # Check convergence
            if np.linalg.norm(self.errors) < self.convergence_threshold:
                break

            current_iteration += 1

        if current_iteration == self.maximum_iterations:
            print("Optimization did not converge within iteration limit.")

        # Separate bias term from feature weights
        self.bias = self.weights[0]
        self.weights = self.weights[1:]

        return LassoHomotopyResults(self.bias, self.weights)
    
    def incremental_update(self, new_feature, new_target):
        """
        Update model with additional data point.
        
        Args:
            new_feature: Additional feature vector
            new_target: Corresponding target value
        Returns:
            Updated model results
        """
        # Add intercept term
        new_feature = np.concatenate([[1], new_feature])
    
        # Verify dimensions match
        if self.errors.shape[0] != self.active_matrix.shape[0]:
            raise ValueError("Dimension inconsistency between residuals and active features")
    
        # Compute new correlation
        new_correlation = np.dot(new_feature, self.errors)
    
        if np.abs(new_correlation) >= self.reg_param:
            # Expand active set if needed
            self.selected_features.append(len(self.weights))
    
            # Update active feature matrix
            if self.active_matrix is None:
                self.active_matrix = new_feature.reshape(1, -1)
            else:
                self.active_matrix = np.row_stack([self.active_matrix, new_feature])
    
            # Solve expanded least squares problem
            updated_weights = pinvh(self.active_matrix.T @ self.active_matrix) @ (self.active_matrix.T @ np.append(self.errors, new_target))
    
            # Update model parameters
            for j, val in zip(self.selected_features, updated_weights):
                self.weights[j] = val
    
            # Update residuals
            self.errors = np.append(self.errors, new_target) - np.dot(self.active_matrix, updated_weights)
    
        return LassoHomotopyResults(self.bias, self.weights)

    def loo_cv_error(self, features, targets):
        """
        Compute leave-one-out cross-validation error.
        
        Args:
            features: Input feature matrix
            targets: Corresponding output values
        Returns:
            Average cross-validation error
        """
        sample_count = features.shape[0]
        squared_errors = []
    
        for i in range(sample_count):
            # Create leave-one-out dataset
            X_loo = np.delete(features, i, axis=0)
            y_loo = np.delete(targets, i)
    
            # Train model on reduced dataset
            temp_model = LassoHomotopyModel(reg_param=self.reg_param, convergence_threshold=self.convergence_threshold)
            trained_model = temp_model.train(X_loo, y_loo)
    
            # Predict left-out sample
            pred = trained_model.predict(features[i].reshape(1, -1))
    
            # Compute prediction error
            squared_errors.append((targets[i] - pred)**2)
    
        avg_error = np.mean(squared_errors)
        
        print(f"LOO-CV average squared error: {avg_error}")
        
        return avg_error

    def adjust_regularization(self):
        """
        Adapt regularization parameter based on current model performance.
        
        Returns:
            Updated regularization parameter
        """
        residual_magnitude = np.linalg.norm(self.errors)
        
        # Adaptive regularization adjustment
        self.reg_param *= 1.0 / (1.0 + residual_magnitude / np.linalg.norm(self.weights))
        
        return self.reg_param


class LassoHomotopyResults:
    def __init__(self, intercept, coefficients):
        """
        Container for model results.
        
        Args:
            intercept: Model bias term
            coefficients: Feature weights
        """
        self.intercept = intercept
        self.coefficients = coefficients

    def predict(self, X):
        """
        Generate predictions from input features.
        
        Args:
            X: Input feature matrix
        Returns:
            Model predictions
        """
        return np.dot(X, self.coefficients) + self.intercept
