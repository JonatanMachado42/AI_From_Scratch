# Data creation 
import numpy as np
from src.preprocessing.scalers import MinMaxScaler

def generate_linear_regression_data(n_samples=200, noise_std=1.0):
    # Relação linear verdadeira: y = 3*x + 5
    X = np.random.uniform(-10, 10, size=(n_samples, 1))
    y = 3 * X[:, 0] + 5 + np.random.normal(0, noise_std, size=n_samples)
    return X, y

X, y = generate_linear_regression_data()
print(X[:5], y[:5])

print("---------------")

scaler = MinMaxScaler()

scaler.fit(X)
normalized_dataset = scaler.transform(X)

print(normalized_dataset[:5])