import numpy as np
import pickle
from optimizer import Sgd, Adam
from network import Network, NetworkConfig
from model import Model

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

		self.solver = solver

	def fit(self, X, y):
		""" Fit the model to the data """
		if hasattr(X, 'values'):
			input_size = X.shape[1]
		else:
			input_size = X.shape[1]
		
		layers = [input_size] + self.hidden_layer_sizes + [self.output_layer_size]
		config = NetworkConfig(
			layers=layers,
			activation=self.activation,
			loss=self.loss,
			output_activation=self.output_activation
		)
		self.model = Model(config)

		self.model.fit(
			X, y,
			self.learning_rate,
			self.epochs,
			self.batch_size,
			self.solver
		)

	def predict(self, X):
		if self.model:
			return self.model.network.forward(X)
		return None
	
	def save(self, filename):
		""" Save the model to a file """
		with open(filename, 'wb') as f:
			pickle.dump(self, f)
		print(f"Model saved to {filename}")