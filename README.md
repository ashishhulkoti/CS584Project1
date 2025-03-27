# Project 1 

Your objective is to implement the LASSO regularized regression model using the Homotopy Method. You can read about this method in [this](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf) paper and the references therein. You are required to write a README for your project. Please describe how to run the code in your project *in your README*. Including some usage examples would be an excellent idea. You may use Numpy/Scipy, but you may not use built-in models from, e.g. SciKit Learn. This implementation must be done from first principles. You may use SciKit Learn as a source of test data.

You should create a virtual environment and install the packages in the requirements.txt in your virtual environment. You can read more about virtual environments [here](https://docs.python.org/3/library/venv.html). Once you've installed PyTest, you can run the `pytest` CLI command *from the tests* directory. I would encourage you to add your own tests as you go to ensure that your model is working as a LASSO model should (Hint: What should happen when you feed it highly collinear data?)

In order to turn your project in: Create a fork of this repository, fill out the relevant model classes with the correct logic. Please write a number of tests that ensure that your LASSO model is working correctly. It should produce a sparse solution in cases where there is collinear training data. You may check small test sets into GitHub, but if you have a larger one (over, say 20MB), please let us know and we will find an alternative solution. In order for us to consider your project, you *must* open a pull request on this repo. This is how we consider your project is "turned in" and thus we will use the datetime of your pull request as your submission time. If you fail to do this, we will grade your project as if it is late, and your grade will reflect this as detailed on the course syllabus. 

You may include Jupyter notebooks as visualizations or to help explain what your model does, but you will be graded on whether your model is correctly implemented in the model class files and whether we feel your test coverage is adequate. We may award bonus points for compelling improvements/explanations above and beyond the assignment.

Put your README here. Answer the following questions.

* What does the model you have implemented do and when should it be used?
* How did you test your model to determine if it is working reasonably correctly?
* What parameters have you exposed to users of your implementation in order to tune performance? 
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

## Team Members
| Name            | CWID      |
|-----------------|-----------|
| Ashish Hulkoti  | A20548738 |
| Harish Hebsur  | A20552584 |

### What does the model do?
This implementation solves **L1-regularized linear regression (Lasso)** using the Homotopy method, which:
- Computes exact solutions along the entire regularization path
- Maintains an active set of features for efficient updates
- Supports both batch training and online/incremental learning
- Automatically induces sparsity in coefficients through feature selection

### When should it be used?

| Use Case | Recommended | Reasoning |
|----------|-------------|-----------|
| Feature selection | ✅ | Produces interpretable models with few non-zero coefficients |
| Collinear features | ✅ | Handles correlated predictors effectively |
| Online learning | ✅ | Supports incremental updates for streaming data |
| Moderate-scale problems | ✅ | Works well when features ≤10,000 |

# Example Use Case: Real-Time Fraud Detection System

## Scenario Overview
**Industry**: Financial Services  
**Problem**: Detect credit card fraud in real-time while identifying key risk factors  
**Challenge**: Need interpretable model that can:
- Process transactions incrementally
- Handle 100+ correlated features (purchase patterns, location data, etc.)
- Explain which features flag potential fraud

### How did you test your model to determine if it is working reasonably correctly?

We implemented a multi-layered testing strategy:

| Test Type | Purpose | Example Verification |
|-----------|---------|----------------------|
| Synthetic Data | Verify correctness | Check coefficients match ground truth |
| Collinearity Tests | Confirm sparsity | Verify zeros emerge when `reg_param` is large |
| Online Consistency | Validate updates | Match batch vs sequential results |
| Prediction Tests | Check outputs | Ensure R² > 0.9 on noise-free data |
| Edge Cases | Test robustness | Empty feature sets, single samples |

### What parameters have you exposed to users of your implementation in order to tune performance?

| Parameter | Type | Description | Recommended Range | Default |
|-----------|------|-------------|-------------------|---------|
| `reg_param` (α) | float | L1 regularization strength | 0.01 to 10.0 | 1.0 |
| `convergence_threshold` | float | Optimization tolerance | 1e-4 to 1e-8 | 1e-6 |
| `maximum_iterations` | int | Max optimization steps | 100-10,000 | 1000 |

### Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

1. Unscaled Features
Issue: Biased regularization due to varying feature scales.
Solvable: Yes - Implement automatic feature standardization within the model.

2. High-Dimensional Data
Issue: Slow performance with >10,000 features.
Solvable: Yes - Add feature screening (e.g., correlation-based filtering) to reduce dimensionality.

3. Non-Linear Relationships
Issue: Poor performance on non-linear data patterns.
Solvable: No - Requires fundamental algorithm changes (e.g., kernel methods).


## Installation

```bash
# Clone repository
git clone <Repo URL>
cd 

# Create a Virtual environment
python -m venv venv

# Switch to virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# To run the Tests
cd LassoHomotopy/tests
pytest -v
```
## Sample Usage

```python
import numpy as np
from lasso_homotopy import LassoHomotopyModel, LassoHomotopyResults

# Initialize model with custom parameters
model = LassoHomotopyModel(
    reg_param=0.8,               # Regularization strength
    convergence_threshold=1e-5,  # Stopping tolerance
    maximum_iterations=500       # Iteration limit
)

# Configuration
np.random.seed(42)
n_samples = 200
n_features = 30
sparsity = 0.7  # 70% zero coefficients

# Create sparse ground truth
true_coef = np.random.randn(n_features)
true_coef[np.random.rand(n_features) < sparsity] = 0

# Generate correlated features
X = np.random.randn(n_samples, n_features)
X[:, 1] = 0.6 * X[:, 0] + np.random.normal(0, 0.1, n_samples)  # Correlated features
X[:, 3] = 0.4 * X[:, 2] + np.random.normal(0, 0.3, n_samples)

# Generate targets with noise
y = X @ true_coef + np.random.normal(0, 0.5, n_samples)

# Train-test split
X_train, y_train = X[:150], y[:150]
X_test, y_test = X[150:], y[150:]

# Fit the model
results = model.train(X_train, y_train)

# Display results
print(f"Number of non-zero coefficients: {np.sum(results.coefficients != 0)}/{n_features}")
print(f"Intercept: {results.intercept:.4f}")
print("Top 5 coefficients:")
for i in np.argsort(np.abs(results.coefficients))[-5:]:
    print(f"Feature {i}: True={true_coef[i]:.4f} Predicted={results.coefficients[i]:.4f}")

# Make predictions
train_preds = results.predict(X_train)
test_preds = results.predict(X_test)

# Calculate metrics
def r_squared(y_true, y_pred):
    return 1 - np.var(y_true - y_pred) / np.var(y_true)

print(f"\nTraining R²: {r_squared(y_train, train_preds):.4f}")
print(f"Test R²: {r_squared(y_test, test_preds):.4f}")

# Feature importance visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.stem(np.where(results.coefficients != 0)[0], 
         results.coefficients[results.coefficients != 0])
plt.title("Non-zero Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.show()
```
