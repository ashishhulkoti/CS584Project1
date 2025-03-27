import numpy as np
from model.LassoHomotopy import LassoHomotopyModel, LassoHomotopyResults

def load_dataset(file_path):
    """
    Load dataset from a CSV file.
    :param file_path: Path to the CSV file.
    :return: Features (X) and target (y).
    """
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Target
    return X, y

def test_sparse_solution_collinear_data():
    """
    Test that the model produces sparse solutions when trained on collinear data.
    """
    # Load collinear data
    X, y = load_dataset("collinear_data.csv")

    # Validate collinearity in the dataset
    correlation_matrix = np.corrcoef(X, rowvar=False)
    max_correlation = np.max(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
    print(f"Maximum correlation between features: {max_correlation}")
    assert max_correlation > 0.8, "Dataset does not contain highly correlated features!"

    # Initialize the model with higher regularization
    model = LassoHomotopyModel(reg_param=10.0, convergence_threshold=1e-6, maximum_iterations=1000)

    # Fit the model with collinear data
    results = model.train(X, y)

    # Check sparsity: count zero coefficients
    num_zero_coefficients = np.sum(np.isclose(results.coefficients, 0, atol=1e-4))
    print(f"Number of zero coefficients: {num_zero_coefficients} out of {len(results.coefficients)}")

    # Assert sparsity (expecting at least some coefficients to be zero)
    assert num_zero_coefficients > 0, "Model did not produce a sparse solution!"

def test_online_updates():
    """
    Test that the model correctly incorporates new observations without retraining from scratch.
    """
    # Load synthetic data
    X_train, y_train = load_dataset("small_test.csv")

    # Initialize the model
    model = LassoHomotopyModel(reg_param=1.0, convergence_threshold=1e-6)

    # Fit the model with initial data
    results = model.train(X_train[:80], y_train[:80])

    # Add new observations incrementally
    for i in range(80, len(X_train)):
        results = model.incremental_update(X_train[i], y_train[i])

    predictions = results.predict(X_train[80:])
    
    assert len(predictions) == len(X_train[80:]), "Number of predictions does not match number of new observations!"

def test_leave_one_out_cross_validation():
    """
    Test that leave-one-out cross-validation works as expected.
    """
    # Load synthetic data
    X, y = load_dataset("small_test.csv")

    # Initialize the model
    model = LassoHomotopyModel(reg_param=1.0, convergence_threshold=1e-6)

    # Perform leave-one-out cross-validation
    mean_error = model.loo_cv_error(X, y)

    # Assert that LOOCV executes without errors and returns a valid error value
    assert mean_error >= 0, "LOOCV returned an invalid error value!"

def test_regularization_updates():
    """
    Test that the regularization parameter (alpha) is dynamically updated based on residuals.
    """
    # Load synthetic data
    X_train, y_train = load_dataset("small_test.csv")

    # Initialize the model and fit it incrementally
    model = LassoHomotopyModel(reg_param=1.0, convergence_threshold=1e-6)
    
    initial_alpha = model.reg_param
    
    results = model.train(X_train[:80], y_train[:80])
    
    residual_error = np.mean((y_train[:80] - results.predict(X_train[:80])) ** 2)
    
    updated_alpha = model.adjust_regularization()
    
    assert updated_alpha != initial_alpha, "Regularization parameter was not updated!"

def test_predictions():
    """
    Test that predictions are reasonable and match expectations.
    """
    # Load synthetic test data
    X_test, _ = load_dataset("small_test.csv")
    
    # Use pre-trained coefficients for testing predictions
    test_coefficients = np.array([3.5, -2.1, 0.8])  # Example coefficients for testing purposes
    test_intercept = 1.2
    
    # Create model output with test parameters
    results = LassoHomotopyResults(test_intercept, test_coefficients)
    
    # Make predictions
    predictions = results.predict(X_test)
    
    # Basic validation
    assert len(predictions) == len(X_test), "Number of predictions doesn't match input size"
    assert not np.any(np.isnan(predictions)), "Predictions contain NaN values"
