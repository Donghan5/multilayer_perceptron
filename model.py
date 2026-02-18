import numpy as np
import copy
import pandas as pd
from network import Network, NetworkConfig
from optimizer import Sgd, Adam

class Model:
	def __init__(self, config: NetworkConfig) -> None:
		self.config = config
		self.network = Network(config)
	
	def fit(self, x_train, y_train, learning_rate, epochs, batch_size, optimization="sgd", x_val=None, y_val=None, early_stopping_rounds=10):
		"""
		:param self: Class itself
		:param x_train: Training input data
		:param y_train: Training target data
		:param learning_rate: Learning rate for the optimizer
		:param epochs: Number of training epochs
		:param batch_size: Batch size for training
		:param optimization: Optimization algorithm to use (e.g., "sgd" or "adam")
		:param x_val: Validation input data
		:param y_val: Validation target data
		:param early_stopping_rounds: Number of epochs with no improvement to stop training
		"""
		if optimization == "adam":
			optimizer = Adam(learning_rate)
		else:
			optimizer = Sgd(learning_rate)
		
		n_samples = len(x_train)
		history = {'loss':[], 'val_loss':[], 'accuracy':[], 'val_accuracy':[]}

		# variable init to early stopping
		best_loss = float('inf')
		patience = 0
		best_weights = None
		best_biases = None

		for epoch in range(epochs):
			indices = np.arange(n_samples)
			np.random.shuffle(indices)

			if hasattr(x_train, 'iloc'):
				x_shuffled = x_train.iloc[indices].values
				y_shuffled = y_train.iloc[indices].values
			else:
				x_shuffled = x_train[indices]
				y_shuffled = y_train[indices]
			
			for i in range(0, n_samples, batch_size):
				x_batch = x_shuffled[i : i + batch_size]
				y_batch = y_shuffled[i : i + batch_size]

				self.network.forward(x_batch)
				nabla_w, nabla_b = self.network.backward(y_batch)
				optimizer.update(self.network, nabla_w, nabla_b)

			train_output = self.network.forward(x_train)
			train_loss = self.network.loss(y_train, train_output)
			train_preds = np.argmax(train_output, axis=1)
			train_true = np.argmax(y_train, axis=1)
			train_accuracy = np.mean(train_preds == train_true)

			history['loss'].append(train_loss)
			history['accuracy'].append(train_accuracy)

			log_msg = f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f}"
			if x_val is not None and y_val is not None:
				val_output = self.network.forward(x_val)
				val_loss = self.network.loss(y_val, val_output)
				val_preds = np.argmax(val_output, axis=1)
				val_true = np.argmax(y_val, axis=1)
				val_accuracy = np.mean(val_preds == val_true)

				history['val_loss'].append(val_loss)
				history['val_accuracy'].append(val_accuracy)
				log_msg += f" - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}"
			
				if val_loss < best_loss:
					best_loss = val_loss
					patience = 0
					best_weights = copy.deepcopy(self.network.weights)
					best_biases = copy.deepcopy(self.network.biases)
				else:
					patience += 1
					if patience >= early_stopping_rounds:
						print(f"\nEarly stopping at epoch {epoch + 1}")
						print(f"Restoring best weights from epoch {epoch + 1 - patience} (Loss: {best_loss:.4f})")
						self.network.weights = best_weights
						self.network.biases = best_biases
						break
			print(log_msg)

		return history