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
		- 'M' (malignant) is encoded as [0, 1]
		- 'B' (benign) is encoded as [1, 0]
	"""
	encoded = np.zeros((len(y), 2), dtype=np.float64)
	for i, label in enumerate(y):
		if isinstance(label, str):
			label = label.strip()

		if label == 'B':
			encoded[i] = [1.0, 0.0]
		elif label == 'M':
			encoded[i] = [0.0, 1.0]
		else:
			raise ValueError(f"Invalid label at index {i}: {label!r}. Expected 'B' or 'M'.")
	return encoded

def validate_dataset(df, name="dataset"):
	"""
		Validate the dataset format:
		- Must have 32 columns
		- The second column must contain only 'B' or 'M' labels (case-insensitive, with optional whitespace)
		- The remaining columns must be numeric
	"""

	if df.shape[1] != 32:
		raise ValueError(
			f"{name}: expected 32 columns got {df.shape[1]}"
		)
	
	labels = df.iloc[:, 1].astype(str).str.strip()
	invalid = labels[~labels.isin(['B', 'M'])]

	if not invalid.empty:
		raise ValueError(
			f"{name}: invalid labels found at indices {invalid.index.tolist()}: {invalid.unique().tolist()}"
		)
	
	try:
		df.iloc[:, 2:].astype(float)
	except ValueError as e:
		raise ValueError(f"{name}: non-numeric values found in feature columns") from e