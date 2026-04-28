import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
	"""
		Sigmoid activation function
			- Setting the value between 0 and 1
			- Using the formula: 1 / (1 + e^(-x))
			- Avoiding overflow by using the property of e^x = 1 / e^(-x), so multiplying by e^x
	"""
	x = np.clip(x, -500, 500)
	return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def sigmoid_prime(z: np.ndarray) -> np.ndarray:
	"""
		Derivative of sigmoid function
		compute -> sigmoid(z) * (1 - sigmoid(z))
	"""
	s = sigmoid(z)
	return np.multiply(s, (1 - s))

def relu(x: np.ndarray) -> np.ndarray:
	"""
		ReLU activation function
			- If the input value is a positive number, return it
			- Else, the case of 0 or negative number, return 0
	"""
	return np.maximum(0, x)

def relu_prime(x: np.ndarray) -> np.ndarray:
	"""
		Derivative of ReLU activation function
	"""
	return (x > 0).astype(int)

def softmax(x: np.ndarray) -> np.ndarray:
	x_exps = np.exp(x - np.max(x, axis=1, keepdims=True))
	return x_exps / np.sum(x_exps, axis=1, keepdims=True)

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""
	Compute the cross entropy loss
	y_true: true labels (one-hot encoded)
	y_pred: predicted probabilities
	"""
	epsilon = 1e-15
	y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
	return -np.sum(y_true * np.log(y_pred_clipped)) / y_true.shape[0]

def one_hot_encode(y: np.ndarray) -> np.ndarray:
	"""
		One-hot encode the target labels
	"""
	encoded = np.zeros((len(y), 2))
	for i, label in enumerate(y):
		if label == 'M':
			encoded[i] = [0, 1]
		else:
			encoded[i] = [1, 0]
	return encoded