import numpy as np
import copy
from network import Network, NetworkConfig
from optimizer import Sgd, Adam

class Model:
	def __init__(
		self,
		hidden_layer_sizes=None,
		output_layer_size=2,
		activation="relu",
		output_activation="softmax",
		loss="cross_entropy",
		learning_rate=0.0314,
		epochs=100,
		batch_size=8,
		weights_initializer="heUniform",
		solver="adam",
		early_stopping_rounds=10,
		min_delta=1e-4
	):
		if hidden_layer_sizes is None:
			hidden_layer_sizes = [24, 24, 24]
		self.hidden_layer_sizes = hidden_layer_sizes
		self.output_layer_size = output_layer_size
		self.activation = activation
		self.output_activation = output_activation
		self.loss = loss
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.batch_size = batch_size
		self.weights_initializer = weights_initializer
		self.network = None
		if solver not in ("adam", "sgd"):
			raise ValueError("solver must be 'adam' or 'sgd'.")
		self.solver = solver
		self.early_stopping_rounds = early_stopping_rounds
		self.min_delta = min_delta
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
		
		if x_val is None or y_val is None:
			print("Warning: no validation data provided, early stopping disabled.")
		
		history = {'loss':[], 'val_loss':[], 'accuracy':[], 'val_accuracy':[]}

		if self.solver == "adam":
			optimizer = Adam(self.learning_rate)
		else:
			optimizer = Sgd(self.learning_rate)
	
		n_samples = len(x_train)

		input_size = x_train.shape[1]
		
		early_stopping_rounds = self.early_stopping_rounds
		
		layers = [input_size] + self.hidden_layer_sizes + [self.output_layer_size]
		config = NetworkConfig(
			layers=layers,
			activation=self.activation,
			loss=self.loss,
			output_activation=self.output_activation,
			weights_initializer=self.weights_initializer
		)
		self.network = Network(config)

		self.mean_train = x_train.mean(axis=0)
		self.std_train = x_train.std(axis=0) + 1e-08
	
		x_train = (x_train - self.mean_train) / self.std_train
		if x_val is not None:
			x_val = (x_val - self.mean_train) / self.std_train
		
		# variable init to early stopping
		best_loss = float('inf')
		best_epoch = 0
		patience = 0
		best_weights = None
		best_biases = None
		stopped_early = False

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

			log_msg = f"Epoch {epoch + 1}/{self.epochs} - loss: {train_loss:.6f} - accuracy: {train_accuracy:.6f}"
			if x_val is not None and y_val is not None:
				val_output = self.network.forward(x_val)
				val_loss = self.network.loss(y_val, val_output)
				val_preds = np.argmax(val_output, axis=1)
				val_true = np.argmax(y_val, axis=1)
				val_accuracy = np.mean(val_preds == val_true)

				history['val_loss'].append(val_loss)
				history['val_accuracy'].append(val_accuracy)

				if val_loss < best_loss - self.min_delta:
					best_loss = val_loss
					best_epoch = epoch + 1
					patience = 0
					best_weights = copy.deepcopy(self.network.weights)
					best_biases = copy.deepcopy(self.network.biases)
				else:
					patience += 1

				log_msg += (
					f" - val_loss: {val_loss:.6f}"
					f" - val_accuracy: {val_accuracy:.6f}"
					f" - best_loss: {best_loss:.6f}"
					f" - patience: {patience}/{early_stopping_rounds}"
				)

				if patience >= early_stopping_rounds:
					if best_weights is None or best_biases is None:
						raise RuntimeError("Early stopping triggered before any valid best weights were saved.")

					print(log_msg)
					print(f"\nEarly stopping at epoch {epoch + 1}")
					print(f"Restoring best weights from epoch {best_epoch} (Loss: {best_loss:.6f})")
					self.network.weights = best_weights
					self.network.biases = best_biases
					stopped_early = True
					break
			print(log_msg)

		if x_val is not None and y_val is not None and best_weights is not None and not stopped_early:
			self.network.weights = best_weights
			self.network.biases = best_biases
			print(f"\nRestoring best weights from epoch {best_epoch} (Loss: {best_loss:.6f})")

		return history

	def predict(self, X):
		""" Predict class probabilities for the input data """
		if self.network is None:
			return None
		# Standardize input using training mean and std
		X = np.asarray(X)

		# Handle single sample input (1D array) by reshaping it to 2D
		if X.ndim == 1:
			X = X.reshape(1, -1)
		X = (X - self.mean_train) / self.std_train
		return self.network.forward(X)

	def save(self, filename):
		""" Save the model to a file """
		if self.network is None:
			raise ValueError("Cannot save an untrained model.")
		
		payload = {
			"layers": np.array(self.network.config.layers, dtype=np.int64),
			"activation": np.array(self.network.config.activation),
			"output_activation": np.array(self.network.config.output_activation),
			"loss": np.array(self.network.config.loss),
			"weights_initializer": np.array(self.network.config.weights_initializer),
			"mean_train": self.mean_train,
			"std_train": self.std_train,
		}

		for i, (W, b) in enumerate(zip(self.network.weights, self.network.biases)):
			payload[f"W{i}"] = W
			payload[f"b{i}"] = b

		np.savez(filename, **payload)
		print(f"Model saved to {filename}")

	@classmethod
	def load(filename="model.npz"):
		data = np.load(filename, allow_pickle=False)

		layers = data["layers"].tolist()
		activation = data["activation"].item()
		output_activation = data["output_activation"].item()
		loss = data["loss"].item()
		weights_initializer = data["weights_initializer"].item()


		model = Model(
			hidden_layer_sizes=layers[1:-1],
			output_layer_size=layers[-1],
			activation=activation,
			output_activation=output_activation,
			loss=loss,
			weights_initializer=weights_initializer
		)

		config = NetworkConfig(
			layers=layers,
			activation=activation,
			output_activation=output_activation,
			loss=loss,
			weights_initializer=weights_initializer
		)

		model.network = Network(config)

		model.network.weights = [
			data[f"W{i}"] for i in range(len(layers) - 1)
		]
		model.network.biases = [
			data[f"b{i}"] for i in range(len(layers) - 1)
		]

		model.mean_train = data["mean_train"]
		model.std_train = data["std_train"]

		return model
