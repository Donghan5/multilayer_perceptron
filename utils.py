import numpy as np
import pandas as pd

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
	return (x_exps := np.exp(x)) / sum(x_exps)