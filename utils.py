import numpy as np
import pandas as pd

"""
	Standardize the dataframe except for the target column
"""
def standarize(dataframe: pd.DataFrame, target: str) -> None:
	features = [x for x in dataframe.columns if x != target]
	df_features = dataframe[features]
	dataframe[features] = (df_features - df_features.mean()) / df_features.std()

"""
	Sigmoid activation function
		- Setting the value between 0 and 1
		- Using the formula: 1 / (1 + e^(-x))
		- Avoiding overflow by using the property of e^x = 1 / e^(-x), so multiplying by e^x
"""
def sigmoid(x: np.matrix) -> np.matrix:
	result = np.zeros((x.shape[0], x.shape[1]))
	for i in range(x.shape[1]):
		if x[0, i] >= 0:
			result[0, i] = 1 / (1 + np.exp(-x[0, i]))
		else:
			result[0, i] = np.exp(x[0, i]) / (1 + np.exp(x[0, i]))
	return result

"""
	Derivative of sigmoid function
"""
def sigmoid_prime(x: np.matrix) -> np.matrix:
	return np.multiply(x, (1 - x))

"""
	ReLU activation function
		- If the input value is a positive number, return it
		- Else, the case of 0 or negative number, return 0
"""
def relu(x: np.matrix) -> np.matrix:
	return np.maximum(0, x)

"""
	Derivative of ReLU activation function
"""
def relu_prime(x: np.matrix) -> np.matrix:
	return (x > 0).astype(int)

def softmax(x: np.ndarray) -> np.ndarray:
	x_exps = np.exp(x - np.max(x, axis=1, keepdims=True))
	return x_exps / np.sum(x_exps, axis=1, keepdims=True)

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
	return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	# Clip y_pred to prevent log(0)
	y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
	return -np.sum(y_true * np.log(y_pred_clipped)) / y_true.shape[0]