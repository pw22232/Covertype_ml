import torch
import pymc as pm
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

size = 5

def load_data():
    train_data = pd.read_csv("regression_train.txt", sep='\s+', names=["x", "y"])
    test_data = pd.read_csv("regression_test.txt", sep='\s+', names=["x", "y"])

    X_train = train_data["x"].values.reshape(-1, 1)
    y_train = train_data["y"].values
    X_test = test_data["x"].values.reshape(-1, 1)
    y_test = test_data["y"].values

    return X_train, y_train, X_test, y_test

def linear_regression():
    # Train a Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_train_pred = linear_model.predict(X_train)
    y_test_pred = linear_model.predict(X_test)
    mse_linear = mean_squared_error(y_test, y_test_pred)
    print(f"Linear Regression MSE on Test Set: {mse_linear}")

    # Plot for training data
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, color="blue", label="Training Data", s=size)
    plt.plot(X_train, y_train_pred, color="red", label="Model Prediction")
    plt.title("Linear Regression on Training Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    # Plot for test data
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color="green", label="Test Data", s=size)
    plt.plot(X_test, y_test_pred, color="orange", label="Model Prediction")
    plt.title("Linear Regression on Test Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    return mse_linear


def neural_network():
    # Define the neural network architecture
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(1, 128)  # First hidden layer
            self.fc2 = nn.Linear(128, 64)  # Second hidden layer
            self.fc3 = nn.Linear(64, 32)  # Third hidden layer
            self.fc4 = nn.Linear(32, 1)  # Output layer

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    # Prepare PyTorch DataLoader
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    # Initialize model, loss function and optimizer
    nn_model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=0.01)

    # Training loop
    epochs = 10000
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = nn_model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

        # if (epoch+1) % 100 == 0:
        #     print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Test the neural network on test set
    y_train_pred_nn = nn_model(X_train_tensor).detach().numpy()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_pred_nn = nn_model(X_test_tensor).detach().numpy()
    mse_nn = mean_squared_error(y_test, y_test_pred_nn)
    print(f"Neural Network MSE on Test Set: {mse_nn}")

    # Plot for training data
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, color="blue", label="Training Data", s=size)
    plt.scatter(X_train, y_train_pred_nn, color="red", label="Model Prediction", s=size)
    plt.title("Neural Network on Training Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    # Plot for test data
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color="green", label="Test Data", s=size)
    plt.scatter(X_test, y_test_pred_nn, color="orange", label="Model Prediction", s=size)
    plt.title("Neural Network on Test Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def bayesian_method():
    X_train, y_train, X_test, y_test = load_data()
    X_train = X_train.flatten()  # Flatten for easy handling by PyMC

    with pm.Model() as linear_regression_model:
        # Prior distribution of model parameters
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)

        mu = alpha + beta * X_train
        # Likelihood function of observed data
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)
        # Sample from the posterior distribution
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)

    # Plot the posterior distributions of alpha, beta, and sigma.
    pm.plot_posterior(trace, var_names=["alpha", "beta", "sigma"])
    plt.tight_layout()
    plt.show()

    # Posterior predictive check
    with linear_regression_model:
        posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["y_obs"])

    # Calculate the predicted mean for each test point
    y_pred_mean = posterior_predictive['posterior_predictive']['y_obs'].mean(axis=(0, 1))  # Calculate the average of the first two dimensions

    # Calculate the confidence interval
    y_pred_lower = np.percentile(posterior_predictive['posterior_predictive']['y_obs'], 2.5, axis=(0, 1))
    y_pred_upper = np.percentile(posterior_predictive['posterior_predictive']['y_obs'], 97.5, axis=(0, 1))

    # Drawing
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test.flatten(), y_test, color='blue', label="Actual test data", s=size)
    plt.scatter(X_test.flatten(), y_pred_mean, color='red', label="Posterior predictive mean", s=size)
    plt.fill_between(
        X_test.flatten(),
        y_pred_lower,
        y_pred_upper,
        color="orange",
        alpha=0.3,
        label="95% confidence interval"
    )
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title("Posterior predictive check")
    plt.show()


if __name__ == "__main__":
    # load data
    X_train, y_train, X_test, y_test = load_data()
    # Code Task 10 + 13
    linear_regression()
    # Code Task 11 + 13
    neural_network()
    # Code Task 12
    bayesian_method()
