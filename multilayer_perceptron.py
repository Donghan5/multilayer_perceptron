import numpy as np
from optimizer import Sgd, Adam
from network import Network, NetworkConfig

"""
	Implement MLP (Multi layer perceptron)
"""

class Mlp:
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

		self.model = None

		match(solver):
			case "sgd":
				self.optimizer = Sgd(self.learning_rate)
			case "adam":
				self.optimizer = Adam(self.learning_rate)
			case _:
				raise ValueError(f"Unknown solver: {solver}")
		self._init_model()

	def _init_model(self):
		layers = []
		for i in range(len(self.hidden_layer_sizes)):
			if i == 0:
				layers.append(
					Dense(
						self.hidden_layer_sizes[i],
						activation=self.activation,
						weight_init=self.weight_init,
						random_seed=self.random_seed,
					)
				)
			else:
				layers.append(
					Dense(
						self.hidden_layer_sizes[i],
						activation=self.activation,
						weight_init=self.weight_init,
						random_seed=self.random_seed,
					)
				)
		layers.append(
			Dense(
				self.output_layer_size,
				activation=self.output_activation,
				weight_init=self.weight_init,
				random_seed=self.random_seed,
			)
		)
		self.model = Network(layers)

	def fit(self, X, y):
		self.model.fit(X, y, self.learning_rate, self.epochs, self.batch_size, self.solver)