import numpy as np
import copy
import pandas as pd
from network import Network, NetworkConfig
from optimizer import Sgd, Adam
import pickle

class Model:
	def __init__(
		self,
		hidden_layer_sizes=[24, 24, 24],
		output_layer_size=2,
		activation="relu",
		output_activation="softmax",
		loss="cross_entropy",
		learning_rate=0.0314,
		epochs=100,
		batch_size=8,
		weight_init="HeUniform",
		random_seed=None,
		solver="sgd",
	):
		self.hidden_layer_sizes = hidden_layer_sizes
		self.output_layer_size = output_layer_size
		self.activation = activation
		self.output_activation = output_activation
		self.loss = loss
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.weight_init = weight_init
		self.random_seed = random_seed

		self.network = None

		self.solver = solver
	
	def fit(self, x_train, y_train, x_val=None, y_val=None):
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

		# Convert pandas dataframe to numpy array
		if hasattr(x_train, 'values'):
			x_train = x_train.values
		if hasattr(y_train, 'values'):
			y_train = y_train.values
		if x_val is not None and hasattr(x_val, 'values'):
			x_val = x_val.values
		if y_val is not None and hasattr(y_val, 'values'):
			y_val = y_val.values
		
		history = {'loss':[], 'val_loss':[], 'accuracy':[], 'val_accuracy':[]}

		if self.solver == "adam":
			optimizer = Adam(self.learning_rate)
		else:
			optimizer = Sgd(self.learning_rate)
	
		n_samples = len(x_train)

		input_size = x_train.shape[1]
		
		early_stopping_rounds = 10
		
		layers = [input_size] + self.hidden_layer_sizes + [self.output_layer_size]
		config = NetworkConfig(
			layers=layers,
			activation=self.activation,
			loss=self.loss,
			output_activation=self.output_activation
		)
		self.network = Network(config)

		self.mean_train = x_train.mean(axis=0)
		self.std_train = x_train.std(axis=0) + 1e-08

		x_train = (x_train - self.mean_train) / self.std_train
		if x_val is not None:
			x_val = (x_val - self.mean_train) / self.std_train
		
		# variable init to early stopping
		best_loss = float('inf')
		patience = 0
		best_weights = None
		best_biases = None

		for epoch in range(self.epochs):
			indices = np.arange(n_samples)
			np.random.shuffle(indices)

			x_shuffled = x_train[indices]
			y_shuffled = y_train[indices]
			
			for i in range(0, n_samples, self.batch_size):
				x_batch = x_shuffled[i : i + self.batch_size]
				y_batch = y_shuffled[i : i + self.batch_size]

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

			log_msg = f"Epoch {epoch + 1}/{self.epochs} - loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f}"
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

	def predict(self, X):
		if self.network:
			return self.network.forward(X)
		return None
	
	def save(self, filename):
		""" Save the model to a file """
		with open(filename, 'wb') as f:
			pickle.dump(self, f)
		print(f"Model saved to {filename}")